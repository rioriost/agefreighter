package pggraph

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/meta"
)

type PoolOptions struct {
	MinConnections   int32
	MaxConnections   int32
	ConnectTimeout   time.Duration
	OperationTimeout time.Duration
}

type Adapter struct {
	pool             *pgxpool.Pool
	store            *meta.Store
	capabilities     Capabilities
	operationTimeout time.Duration
}

func Open(ctx context.Context, dsn string, options PoolOptions) (*Adapter, error) {
	if strings.TrimSpace(dsn) == "" {
		return nil, errors.New("PostgreSQL property graph connection string is required")
	}
	if options.MinConnections < 0 || options.MaxConnections <= 0 ||
		options.MinConnections > options.MaxConnections {
		return nil, errors.New("PostgreSQL property graph connection limits are invalid")
	}
	if options.ConnectTimeout <= 0 || options.OperationTimeout <= 0 {
		return nil, errors.New("PostgreSQL property graph timeouts must be positive")
	}
	capabilities, err := Probe(ctx, dsn)
	if err != nil {
		return nil, err
	}
	config, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("parse PostgreSQL property graph connection string: %w", err)
	}
	config.MinConns = options.MinConnections
	config.MaxConns = options.MaxConnections
	config.ConnConfig.ConnectTimeout = options.ConnectTimeout
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("create PostgreSQL property graph pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("connect to PostgreSQL property graph target: %w", err)
	}
	store, err := meta.New(pool)
	if err != nil {
		pool.Close()
		return nil, err
	}
	return &Adapter{
		pool: pool, store: store, capabilities: capabilities,
		operationTimeout: options.OperationTimeout,
	}, nil
}

func (adapter *Adapter) Close() {
	if adapter != nil && adapter.pool != nil {
		adapter.pool.Close()
	}
}

func (adapter *Adapter) Metadata() *meta.Store { return adapter.store }

func (adapter *Adapter) Capabilities() Capabilities { return adapter.capabilities }

func (adapter *Adapter) GraphExists(
	ctx context.Context,
	schema string,
	graph string,
) (bool, error) {
	if !validIdentifier(schema) || !validIdentifier(graph) {
		return false, errors.New("property graph schema and name are invalid")
	}
	var exists bool
	if err := adapter.pool.QueryRow(ctx, `SELECT EXISTS (
		SELECT 1
		FROM pg_catalog.pg_class c
		JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
		WHERE n.nspname = $1 AND c.relname = $2 AND c.relkind = 'g'
	)`, schema, graph).Scan(&exists); err != nil {
		return false, fmt.Errorf("inspect PostgreSQL property graph: %w", err)
	}
	return exists, nil
}

func (adapter *Adapter) ComputeDigests(
	ctx context.Context,
	jobID string,
	definition Definition,
) (DigestSet, error) {
	if err := meta.ValidateJobID(jobID); err != nil {
		return DigestSet{}, err
	}
	if _, err := definition.Fingerprint(); err != nil {
		return DigestSet{}, err
	}
	return computeDigests(ctx, adapter.pool, jobID, definition)
}

func (adapter *Adapter) Prepare(
	ctx context.Context,
	jobID string,
	definition Definition,
) (meta.PropertyGraphGeneration, error) {
	fingerprint, err := definition.Fingerprint()
	if err != nil {
		return meta.PropertyGraphGeneration{}, err
	}
	wanted, err := definitionMapping(jobID, definition, fingerprint)
	if err != nil {
		return meta.PropertyGraphGeneration{}, err
	}
	stored, err := adapter.store.GetPropertyGraph(ctx, jobID)
	if err == nil {
		if !reflect.DeepEqual(stored, wanted) {
			return meta.PropertyGraphGeneration{}, fmt.Errorf(
				"%w: stored PostgreSQL property graph definition changed",
				meta.ErrGenerationMismatch,
			)
		}
		if err := adapter.admitObjects(ctx, definition); err != nil {
			return meta.PropertyGraphGeneration{}, err
		}
		return stored, nil
	}
	if !errors.Is(err, meta.ErrNotFound) {
		return meta.PropertyGraphGeneration{}, err
	}

	tx, err := adapter.pool.Begin(ctx)
	if err != nil {
		return meta.PropertyGraphGeneration{}, fmt.Errorf("begin property graph creation: %w", err)
	}
	defer tx.Rollback(context.WithoutCancel(ctx))
	statements, err := definition.DDL()
	if err != nil {
		return meta.PropertyGraphGeneration{}, err
	}
	for _, statement := range statements {
		if _, err := tx.Exec(ctx, statement); err != nil {
			return meta.PropertyGraphGeneration{}, fmt.Errorf(
				"create PostgreSQL property graph objects: %w", err,
			)
		}
	}
	transactionStore, err := meta.New(tx)
	if err != nil {
		return meta.PropertyGraphGeneration{}, err
	}
	if err := transactionStore.RegisterPropertyGraph(ctx, wanted); err != nil {
		return meta.PropertyGraphGeneration{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return meta.PropertyGraphGeneration{}, fmt.Errorf("commit property graph creation: %w", err)
	}
	return wanted, nil
}

func definitionMapping(
	jobID string,
	definition Definition,
	fingerprint string,
) (meta.PropertyGraphGeneration, error) {
	definition = definition.normalized()
	labels := make([]meta.PropertyGraphLabel, 0,
		len(definition.Vertices)+len(definition.Edges))
	vertexByTable := make(map[string]string, len(definition.Vertices))
	for _, vertex := range definition.Vertices {
		vertexByTable[vertex.Table] = vertex.Label
		labels = append(labels, meta.PropertyGraphLabel{
			Name: vertex.Label, Kind: meta.VertexLabel, Table: vertex.Table,
		})
	}
	for _, edge := range definition.Edges {
		start, startOK := vertexByTable[edge.SourceTable]
		end, endOK := vertexByTable[edge.DestinationTable]
		if !startOK || !endOK {
			return meta.PropertyGraphGeneration{}, errors.New("property graph edge endpoint table is unknown")
		}
		labels = append(labels, meta.PropertyGraphLabel{
			Name: edge.Label, Kind: meta.EdgeLabel, Table: edge.Table,
			StartLabel: start, EndLabel: end,
		})
	}
	return meta.PropertyGraphGeneration{
		JobID: jobID, Schema: definition.Schema, Graph: definition.Graph,
		DefinitionFingerprint: fingerprint,
		State:                 meta.PropertyGraphLoading,
		Labels:                labels,
	}, nil
}

func (adapter *Adapter) admitObjects(ctx context.Context, definition Definition) error {
	var graphKind string
	err := adapter.pool.QueryRow(ctx, `
		SELECT c.relkind::text
		FROM pg_catalog.pg_class c
		JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
		WHERE n.nspname = $1 AND c.relname = $2`,
		definition.Schema, definition.Graph,
	).Scan(&graphKind)
	if errors.Is(err, pgx.ErrNoRows) || graphKind != "g" {
		return fmt.Errorf("%w: PostgreSQL property graph object is missing or changed", meta.ErrGenerationMismatch)
	}
	if err != nil {
		return fmt.Errorf("inspect PostgreSQL property graph object: %w", err)
	}
	for _, table := range appendTableNames(definition) {
		var kind string
		err := adapter.pool.QueryRow(ctx, `
			SELECT c.relkind::text
			FROM pg_catalog.pg_class c
			JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
			WHERE n.nspname = $1 AND c.relname = $2`,
			definition.Schema, table,
		).Scan(&kind)
		if err != nil && !errors.Is(err, pgx.ErrNoRows) {
			return fmt.Errorf("inspect property graph table %q: %w", table, err)
		}
		if errors.Is(err, pgx.ErrNoRows) || (kind != "r" && kind != "p") {
			return fmt.Errorf("%w: property graph table %q is missing or changed", meta.ErrGenerationMismatch, table)
		}
	}
	return nil
}

func appendTableNames(definition Definition) []string {
	names := make([]string, 0, len(definition.Vertices)+len(definition.Edges))
	for _, vertex := range definition.Vertices {
		names = append(names, vertex.Table)
	}
	for _, edge := range definition.Edges {
		names = append(names, edge.Table)
	}
	return names
}

// Finalize proves that every committed pipeline row exists in the relational
// target and that SQL/PGQ can read the first vertex label before atomically
// activating the mapping and committing the job.
func (adapter *Adapter) Finalize(
	ctx context.Context,
	jobID string,
	definition Definition,
	telemetry meta.ConnectorTelemetry,
) error {
	if _, err := definition.Fingerprint(); err != nil {
		return err
	}
	tx, err := adapter.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin PostgreSQL property graph finalization: %w", err)
	}
	defer tx.Rollback(context.WithoutCancel(ctx))
	if _, err := tx.Exec(ctx,
		"SET LOCAL search_path TO "+QuoteIdentifier(definition.Schema)+", pg_catalog",
	); err != nil {
		return fmt.Errorf("set property graph finalization search path: %w", err)
	}
	store, err := meta.New(tx)
	if err != nil {
		return err
	}
	job, err := store.GetJob(ctx, jobID)
	if err != nil {
		return err
	}
	var total int64
	for _, table := range appendTableNames(definition.normalized()) {
		var count int64
		if err := tx.QueryRow(ctx, "SELECT count(*) FROM "+
			qualifiedName(definition.Schema, table)).Scan(&count); err != nil {
			return fmt.Errorf("count PostgreSQL property graph table %q: %w", table, err)
		}
		total += count
	}
	if total != job.CommittedRows {
		return fmt.Errorf("PostgreSQL property graph row count is %d, expected %d",
			total, job.CommittedRows)
	}
	first := definition.normalized().Vertices[0]
	var graphCount int64
	query := fmt.Sprintf(`SELECT count(*) FROM GRAPH_TABLE (
		%s MATCH (v IS %s)
		COLUMNS (v.external_id AS external_id)
	)`, QuoteIdentifier(definition.Graph), QuoteIdentifier(first.Label))
	if err := tx.QueryRow(ctx, query).Scan(&graphCount); err != nil {
		return fmt.Errorf("query finalized PostgreSQL property graph: %w", err)
	}
	var tableCount int64
	if err := tx.QueryRow(ctx, "SELECT count(*) FROM "+
		qualifiedName(definition.Schema, first.Table)).Scan(&tableCount); err != nil {
		return fmt.Errorf("count property graph verification vertex table: %w", err)
	}
	if graphCount != tableCount {
		return fmt.Errorf("SQL/PGQ vertex count is %d, expected %d", graphCount, tableCount)
	}
	digests, err := computeDigests(ctx, tx, jobID, definition)
	if err != nil {
		return fmt.Errorf("compute final PostgreSQL property graph digests: %w", err)
	}
	if digests.Rows != job.CommittedRows {
		return fmt.Errorf("PostgreSQL property graph digest covered %d rows, expected %d",
			digests.Rows, job.CommittedRows)
	}
	if err := store.ReplacePropertyGraphDigests(
		ctx, jobID, digests.Ranges, digests.Root, digests.Rows, DigestRangeCount,
	); err != nil {
		return err
	}
	if err := store.ActivatePropertyGraph(ctx, jobID); err != nil {
		return err
	}
	if err := store.CompleteJobWithTelemetry(ctx, jobID, telemetry); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit PostgreSQL property graph finalization: %w", err)
	}
	return nil
}
