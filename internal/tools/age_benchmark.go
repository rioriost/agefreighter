package tools

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/age"
)

type BenchmarkWorkload string

const (
	BenchmarkVertices BenchmarkWorkload = "vertices"
	BenchmarkEdges    BenchmarkWorkload = "edges"
)

type BenchmarkStrategy string

const (
	BenchmarkDirect     BenchmarkStrategy = "direct-text"
	BenchmarkStaged     BenchmarkStrategy = "staged-binary"
	BenchmarkRelational BenchmarkStrategy = "plain-relational"
)

type BulkBenchmarkOptions struct {
	DSN              string
	Workload         BenchmarkWorkload
	Strategy         BenchmarkStrategy
	Rows             int
	EndpointVertices int
	PropertyBytes    int
	OperationTimeout time.Duration
}

type BulkBenchmarkResult struct {
	Workload      BenchmarkWorkload `json:"workload"`
	Strategy      BenchmarkStrategy `json:"strategy"`
	Rows          int               `json:"rows"`
	PropertyBytes int               `json:"propertyBytes"`
	Elapsed       time.Duration     `json:"-"`
	ElapsedNanos  int64             `json:"elapsedNanos"`
	RowsPerSecond float64           `json:"rowsPerSecond"`
	WALBytes      int64             `json:"walBytes"`
}

var benchmarkSequence atomic.Uint64

func RunBulkBenchmark(
	ctx context.Context,
	options BulkBenchmarkOptions,
) (BulkBenchmarkResult, error) {
	if err := validateBulkBenchmarkOptions(options); err != nil {
		return BulkBenchmarkResult{}, err
	}
	monitor, err := pgxpool.New(ctx, options.DSN)
	if err != nil {
		return BulkBenchmarkResult{}, fmt.Errorf("open benchmark monitor connection: %w", err)
	}
	defer monitor.Close()
	if err := monitor.Ping(ctx); err != nil {
		return BulkBenchmarkResult{}, fmt.Errorf("connect benchmark monitor: %w", err)
	}

	var elapsed time.Duration
	var walBytes int64
	if options.Strategy == BenchmarkRelational {
		elapsed, walBytes, err = benchmarkRelational(ctx, monitor, options)
	} else {
		elapsed, walBytes, err = benchmarkAGE(ctx, monitor, options)
	}
	if err != nil {
		return BulkBenchmarkResult{}, err
	}
	return BulkBenchmarkResult{
		Workload:      options.Workload,
		Strategy:      options.Strategy,
		Rows:          options.Rows,
		PropertyBytes: options.PropertyBytes,
		Elapsed:       elapsed,
		ElapsedNanos:  elapsed.Nanoseconds(),
		RowsPerSecond: float64(options.Rows) / elapsed.Seconds(),
		WALBytes:      walBytes,
	}, nil
}

func validateBulkBenchmarkOptions(options BulkBenchmarkOptions) error {
	if strings.TrimSpace(options.DSN) == "" {
		return errors.New("benchmark DSN is required")
	}
	if options.Rows <= 0 {
		return errors.New("benchmark rows must be positive")
	}
	if options.PropertyBytes < 0 {
		return errors.New("property bytes cannot be negative")
	}
	if options.OperationTimeout <= 0 {
		return errors.New("operation timeout must be positive")
	}
	switch options.Workload {
	case BenchmarkVertices:
	case BenchmarkEdges:
		if options.EndpointVertices < 2 {
			return errors.New("edge benchmark requires at least two endpoint vertices")
		}
	default:
		return fmt.Errorf("unsupported benchmark workload %q", options.Workload)
	}
	switch options.Strategy {
	case BenchmarkDirect, BenchmarkStaged, BenchmarkRelational:
	default:
		return fmt.Errorf("unsupported benchmark strategy %q", options.Strategy)
	}
	return nil
}

func benchmarkAGE(
	ctx context.Context,
	monitor *pgxpool.Pool,
	options BulkBenchmarkOptions,
) (elapsed time.Duration, walBytes int64, resultErr error) {
	adapter, err := age.Open(ctx, options.DSN, age.PoolOptions{
		MaxConnections:   2,
		ConnectTimeout:   options.OperationTimeout,
		OperationTimeout: options.OperationTimeout,
	})
	if err != nil {
		return 0, 0, err
	}
	defer adapter.Close()

	graphName := benchmarkObjectName("af_bench")
	defer func() {
		resultErr = errors.Join(
			resultErr,
			dropBenchmarkGraph(adapter, graphName, options.OperationTimeout),
		)
	}()
	if err := adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		if err := transaction.CreateGraph(ctx, graphName); err != nil {
			return err
		}
		if err := transaction.CreateLabel(ctx, graphName, "Person", age.VertexLabel); err != nil {
			return err
		}
		if options.Workload == BenchmarkEdges {
			return transaction.CreateLabel(ctx, graphName, "KNOWS", age.EdgeLabel)
		}
		return nil
	}); err != nil {
		return 0, 0, fmt.Errorf("prepare AGE benchmark graph: %w", err)
	}

	person, err := adapter.LookupLabel(ctx, graphName, "Person")
	if err != nil {
		return 0, 0, err
	}
	properties := benchmarkProperties(options.PropertyBytes)

	var (
		vertexRows []age.VertexRow
		edgeRows   []age.EdgeRow
		label      age.LabelCatalog
	)
	switch options.Workload {
	case BenchmarkVertices:
		label = person
		vertexRows, err = prepareVertexRows(ctx, adapter, person, options.Rows, properties)
	case BenchmarkEdges:
		endpoints, setupErr := prepareVertexRows(
			ctx,
			adapter,
			person,
			options.EndpointVertices,
			properties,
		)
		if setupErr != nil {
			return 0, 0, setupErr
		}
		if err := adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
			_, err := transaction.CopyVertices(
				ctx,
				person,
				endpoints,
				age.DirectTextCopy,
			)
			return err
		}); err != nil {
			return 0, 0, fmt.Errorf("load benchmark endpoint vertices: %w", err)
		}
		label, err = adapter.LookupLabel(ctx, graphName, "KNOWS")
		if err == nil {
			edgeRows, err = prepareEdgeRows(
				ctx,
				adapter,
				label,
				endpoints,
				options.Rows,
				properties,
			)
		}
	}
	if err != nil {
		return 0, 0, err
	}

	beforeLSN, err := currentWALLSN(ctx, monitor)
	if err != nil {
		return 0, 0, err
	}
	started := time.Now()
	copyStrategy := age.DirectTextCopy
	if options.Strategy == BenchmarkStaged {
		copyStrategy = age.StagedBinaryCopy
	}
	err = adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		if options.Workload == BenchmarkVertices {
			_, err := transaction.CopyVertices(ctx, label, vertexRows, copyStrategy)
			if err != nil {
				return err
			}
		} else {
			_, err := transaction.CopyEdges(ctx, label, edgeRows, copyStrategy)
			if err != nil {
				return err
			}
		}
		if err := transaction.AnalyzeLabel(ctx, label); err != nil {
			return err
		}
		return transaction.VerifyLabelRows(ctx, label, int64(options.Rows))
	})
	if err != nil {
		return 0, 0, fmt.Errorf("run AGE benchmark COPY: %w", err)
	}
	elapsed = time.Since(started)
	walBytes, err = walBytesSince(ctx, monitor, beforeLSN)
	return elapsed, walBytes, err
}

func prepareVertexRows(
	ctx context.Context,
	adapter *age.Adapter,
	label age.LabelCatalog,
	count int,
	properties []byte,
) ([]age.VertexRow, error) {
	var block age.IDBlock
	if err := adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		var err error
		block, err = transaction.ReserveIDs(ctx, label, uint64(count))
		return err
	}); err != nil {
		return nil, fmt.Errorf("reserve benchmark vertex IDs: %w", err)
	}
	rows := make([]age.VertexRow, count)
	for index := range rows {
		id, err := block.GraphID(uint64(index))
		if err != nil {
			return nil, err
		}
		rows[index] = age.VertexRow{ID: id, Properties: properties}
	}
	return rows, nil
}

func prepareEdgeRows(
	ctx context.Context,
	adapter *age.Adapter,
	label age.LabelCatalog,
	endpoints []age.VertexRow,
	count int,
	properties []byte,
) ([]age.EdgeRow, error) {
	var block age.IDBlock
	if err := adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		var err error
		block, err = transaction.ReserveIDs(ctx, label, uint64(count))
		return err
	}); err != nil {
		return nil, fmt.Errorf("reserve benchmark edge IDs: %w", err)
	}
	rows := make([]age.EdgeRow, count)
	for index := range rows {
		id, err := block.GraphID(uint64(index))
		if err != nil {
			return nil, err
		}
		start := index % len(endpoints)
		end := (index*31 + 1) % len(endpoints)
		if end == start {
			end = (end + 1) % len(endpoints)
		}
		rows[index] = age.EdgeRow{
			ID:         id,
			StartID:    endpoints[start].ID,
			EndID:      endpoints[end].ID,
			Properties: properties,
		}
	}
	return rows, nil
}

func benchmarkRelational(
	ctx context.Context,
	pool *pgxpool.Pool,
	options BulkBenchmarkOptions,
) (elapsed time.Duration, walBytes int64, resultErr error) {
	tableName := benchmarkObjectName("af_rel")
	table := pgx.Identifier{"public", tableName}.Sanitize()
	vertexTableName := tableName + "_vertices"
	vertexTable := pgx.Identifier{"public", vertexTableName}.Sanitize()
	create := relationalBenchmarkDDL(options.Workload, table, vertexTable)
	defer func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), options.OperationTimeout)
		defer cancel()
		_, cleanupErr := pool.Exec(
			cleanupCtx,
			"DROP TABLE IF EXISTS "+table+", "+vertexTable,
		)
		if cleanupErr != nil {
			resultErr = errors.Join(
				resultErr,
				fmt.Errorf("drop relational benchmark tables: %w", cleanupErr),
			)
		}
	}()
	if _, err := pool.Exec(ctx, create); err != nil {
		return 0, 0, fmt.Errorf("create relational benchmark table: %w", err)
	}

	properties := benchmarkProperties(options.PropertyBytes)
	if options.Workload == BenchmarkEdges {
		endpoints := make([]age.VertexRow, options.EndpointVertices)
		for index := range endpoints {
			endpoints[index] = age.VertexRow{
				ID:         age.GraphID(index + 1),
				Properties: properties,
			}
		}
		copied, err := pool.CopyFrom(
			ctx,
			pgx.Identifier{"public", vertexTableName},
			[]string{"id", "properties"},
			pgx.CopyFromSlice(len(endpoints), func(index int) ([]any, error) {
				row := endpoints[index]
				return []any{int64(row.ID), string(row.Properties)}, nil
			}),
		)
		if err != nil {
			return 0, 0, fmt.Errorf("load relational benchmark endpoint vertices: %w", err)
		}
		if copied != int64(options.EndpointVertices) {
			return 0, 0, fmt.Errorf(
				"relational endpoint COPY wrote %d rows, expected %d",
				copied,
				options.EndpointVertices,
			)
		}
	}

	var (
		columns []string
		source  pgx.CopyFromSource
	)
	if options.Workload == BenchmarkVertices {
		rows := make([]age.VertexRow, options.Rows)
		for index := range rows {
			rows[index] = age.VertexRow{
				ID:         age.GraphID(index + 1),
				Properties: properties,
			}
		}
		columns = []string{"id", "properties"}
		source = pgx.CopyFromSlice(len(rows), func(index int) ([]any, error) {
			row := rows[index]
			return []any{int64(row.ID), string(row.Properties)}, nil
		})
	} else {
		rows := make([]age.EdgeRow, options.Rows)
		columns = []string{"id", "start_id", "end_id", "properties"}
		for index := range rows {
			start := index%options.EndpointVertices + 1
			end := (index*31+1)%options.EndpointVertices + 1
			if end == start {
				end = end%options.EndpointVertices + 1
			}
			rows[index] = age.EdgeRow{
				ID:         age.GraphID(index + 1),
				StartID:    age.GraphID(start),
				EndID:      age.GraphID(end),
				Properties: properties,
			}
		}
		source = pgx.CopyFromSlice(len(rows), func(index int) ([]any, error) {
			row := rows[index]
			return []any{
				int64(row.ID),
				int64(row.StartID),
				int64(row.EndID),
				string(row.Properties),
			}, nil
		})
	}

	beforeLSN, err := currentWALLSN(ctx, pool)
	if err != nil {
		return 0, 0, err
	}
	started := time.Now()
	transaction, err := pool.Begin(ctx)
	if err != nil {
		return 0, 0, fmt.Errorf("begin relational benchmark: %w", err)
	}
	copied, err := transaction.CopyFrom(
		ctx,
		pgx.Identifier{"public", tableName},
		columns,
		source,
	)
	if err == nil && copied != int64(options.Rows) {
		err = fmt.Errorf("relational COPY wrote %d rows, expected %d", copied, options.Rows)
	}
	if err == nil {
		_, err = transaction.Exec(ctx, "ANALYZE "+table)
	}
	if err == nil {
		var actual int
		err = transaction.QueryRow(ctx, "SELECT count(*) FROM "+table).Scan(&actual)
		if err == nil && actual != options.Rows {
			err = fmt.Errorf("relational table has %d rows, expected %d", actual, options.Rows)
		}
	}
	if err != nil {
		rollbackCtx, cancel := context.WithTimeout(
			context.WithoutCancel(ctx),
			options.OperationTimeout,
		)
		defer cancel()
		rollbackErr := transaction.Rollback(rollbackCtx)
		return 0, 0, errors.Join(err, rollbackErr)
	}
	if err := transaction.Commit(ctx); err != nil {
		return 0, 0, fmt.Errorf("commit relational benchmark: %w", err)
	}
	elapsed = time.Since(started)
	walBytes, err = walBytesSince(ctx, pool, beforeLSN)
	return elapsed, walBytes, err
}

func relationalBenchmarkDDL(
	workload BenchmarkWorkload,
	table string,
	vertexTable string,
) string {
	if workload == BenchmarkVertices {
		return fmt.Sprintf(
			`CREATE TABLE %s (
				id bigint PRIMARY KEY,
				properties jsonb NOT NULL
			)`,
			table,
		)
	}
	return fmt.Sprintf(
		`CREATE TABLE %s (
			id bigint PRIMARY KEY,
			properties jsonb NOT NULL
		);
		CREATE TABLE %s (
			id bigint PRIMARY KEY,
			start_id bigint NOT NULL,
			end_id bigint NOT NULL,
			properties jsonb NOT NULL
		)`,
		vertexTable,
		table,
	)
}

func currentWALLSN(ctx context.Context, pool *pgxpool.Pool) (string, error) {
	var lsn string
	if err := pool.QueryRow(ctx, "SELECT pg_current_wal_lsn()::text").Scan(&lsn); err != nil {
		return "", fmt.Errorf("read current WAL LSN: %w", err)
	}
	return lsn, nil
}

func walBytesSince(
	ctx context.Context,
	pool *pgxpool.Pool,
	before string,
) (int64, error) {
	var bytes int64
	if err := pool.QueryRow(
		ctx,
		"SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), $1::pg_lsn)::bigint",
		before,
	).Scan(&bytes); err != nil {
		return 0, fmt.Errorf("measure benchmark WAL bytes: %w", err)
	}
	return bytes, nil
}

func benchmarkProperties(bytes int) []byte {
	return []byte(`{"payload":"` + strings.Repeat("x", bytes) + `"}`)
}

func benchmarkObjectName(prefix string) string {
	return fmt.Sprintf(
		"%s_%d_%d",
		prefix,
		time.Now().UnixNano(),
		benchmarkSequence.Add(1),
	)
}

func dropBenchmarkGraph(
	adapter *age.Adapter,
	graphName string,
	timeout time.Duration,
) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	if err := adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		if _, err := transaction.LookupGraph(ctx, graphName); err != nil {
			if errors.Is(err, age.ErrCatalogEntryNotFound) {
				return nil
			}
			return err
		}
		return transaction.DropGraph(ctx, graphName, true)
	}); err != nil {
		return fmt.Errorf("drop benchmark graph %q: %w", graphName, err)
	}
	return nil
}
