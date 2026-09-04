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
	mutationLock     *pgx.Conn
	mutationLockKey  string
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
	if adapter == nil || adapter.pool == nil {
		return
	}
	if adapter.mutationLock != nil {
		ctx, cancel := context.WithTimeout(context.Background(), adapter.operationTimeout)
		_, _ = adapter.mutationLock.Exec(ctx,
			`SELECT pg_catalog.pg_advisory_unlock(pg_catalog.hashtextextended($1, 0))`,
			adapter.mutationLockKey)
		cancel()
		_ = adapter.mutationLock.Close(context.Background())
		adapter.mutationLock = nil
	}
	adapter.pool.Close()
}

func (adapter *Adapter) Metadata() *meta.Store { return adapter.store }

func (adapter *Adapter) Capabilities() Capabilities { return adapter.capabilities }

func (adapter *Adapter) LockTarget(ctx context.Context, schema string, graph string) error {
	if adapter.mutationLock != nil {
		return errors.New("PostgreSQL property graph mutation lock is already held")
	}
	if !validIdentifier(schema) || !validIdentifier(graph) {
		return errors.New("property graph schema and name are invalid")
	}
	owner, err := pgx.ConnectConfig(ctx, adapter.pool.Config().ConnConfig.Copy())
	if err != nil {
		return fmt.Errorf("acquire property graph mutation lock connection: %w", err)
	}
	key := fmt.Sprintf("%d:%s%d:%s", len(schema), schema, len(graph), graph)
	var locked bool
	if err := owner.QueryRow(ctx,
		`SELECT pg_catalog.pg_try_advisory_lock(pg_catalog.hashtextextended($1, 0))`,
		key).Scan(&locked); err != nil {
		_ = owner.Close(context.Background())
		return fmt.Errorf("lock PostgreSQL property graph target: %w", err)
	}
	if !locked {
		_ = owner.Close(context.Background())
		return fmt.Errorf("%w: PostgreSQL property graph target %q.%q is being mutated",
			meta.ErrConflict, schema, graph)
	}
	adapter.mutationLock, adapter.mutationLockKey = owner, key
	return nil
}

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

func (adapter *Adapter) PrepareExisting(
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
				meta.ErrGenerationMismatch)
		}
		if err := adapter.admitObjects(ctx, definition); err != nil {
			return meta.PropertyGraphGeneration{}, err
		}
		return stored, nil
	}
	if !errors.Is(err, meta.ErrNotFound) {
		return meta.PropertyGraphGeneration{}, err
	}
	active, err := adapter.store.ActivePropertyGraph(ctx, definition.Schema, definition.Graph)
	if err != nil {
		return meta.PropertyGraphGeneration{}, fmt.Errorf("admit incremental property graph target: %w", err)
	}
	if active.DefinitionFingerprint != fingerprint || active.DigestRoot == "" {
		return meta.PropertyGraphGeneration{}, fmt.Errorf(
			"%w: active PostgreSQL property graph definition or digest baseline changed",
			meta.ErrGenerationMismatch)
	}
	if err := adapter.admitObjects(ctx, definition); err != nil {
		return meta.PropertyGraphGeneration{}, err
	}
	if err := adapter.store.RegisterPropertyGraph(ctx, wanted); err != nil {
		return meta.PropertyGraphGeneration{}, err
	}
	return wanted, nil
}

func (adapter *Adapter) PrepareReplace(
	ctx context.Context,
	jobID string,
	canonical Definition,
) (Definition, error) {
	canonicalFingerprint, err := canonical.Fingerprint()
	if err != nil {
		return Definition{}, err
	}
	active, err := adapter.store.ActivePropertyGraph(ctx, canonical.Schema, canonical.Graph)
	if err != nil {
		return Definition{}, fmt.Errorf("admit replacement property graph target: %w", err)
	}
	if active.DefinitionFingerprint != canonicalFingerprint || active.DigestRoot == "" {
		return Definition{}, fmt.Errorf("%w: replacement target definition or digest baseline changed",
			meta.ErrGenerationMismatch)
	}
	shadow, _, err := ReplacementDefinitions(canonical, jobID)
	if err != nil {
		return Definition{}, err
	}
	if _, err := adapter.Prepare(ctx, jobID, shadow); err != nil {
		return Definition{}, err
	}
	return shadow, nil
}

func (adapter *Adapter) PromoteReplace(
	ctx context.Context,
	jobID string,
	canonical Definition,
	shadow Definition,
	telemetry meta.ConnectorTelemetry,
) error {
	canonicalFingerprint, err := canonical.Fingerprint()
	if err != nil {
		return err
	}
	shadowFingerprint, err := shadow.Fingerprint()
	if err != nil {
		return err
	}
	wantedShadow, err := definitionMapping(jobID, shadow, shadowFingerprint)
	if err != nil {
		return err
	}
	_, backup, err := ReplacementDefinitions(canonical, jobID)
	if err != nil {
		return err
	}
	backupFingerprint, err := backup.Fingerprint()
	if err != nil {
		return err
	}
	tx, err := adapter.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin PostgreSQL property graph replacement: %w", err)
	}
	defer tx.Rollback(context.WithoutCancel(ctx))
	store, err := meta.New(tx)
	if err != nil {
		return err
	}
	job, err := store.GetJob(ctx, jobID)
	if err != nil {
		return err
	}
	storedShadow, err := store.GetPropertyGraph(ctx, jobID)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(storedShadow, wantedShadow) {
		return fmt.Errorf("%w: replacement shadow mapping changed", meta.ErrGenerationMismatch)
	}
	active, err := store.ActivePropertyGraph(ctx, canonical.Schema, canonical.Graph)
	if err != nil {
		return err
	}
	if active.JobID == jobID || active.DefinitionFingerprint != canonicalFingerprint {
		return fmt.Errorf("%w: replacement target mapping changed", meta.ErrGenerationMismatch)
	}
	total, digests, err := inspectFinalTarget(ctx, tx, jobID, shadow)
	if err != nil {
		return err
	}
	if total != job.CommittedRows || digests.Rows != total {
		return fmt.Errorf("replacement shadow contains %d rows with %d digested, expected %d",
			total, digests.Rows, job.CommittedRows)
	}
	if _, err := tx.Exec(ctx,
		"SET LOCAL search_path TO "+QuoteIdentifier(canonical.Schema)+", pg_catalog"); err != nil {
		return err
	}
	for _, graph := range []string{canonical.Graph, shadow.Graph} {
		if _, err := tx.Exec(ctx, "DROP PROPERTY GRAPH "+QuoteIdentifier(graph)); err != nil {
			return fmt.Errorf("drop property graph %q during replacement: %w", graph, err)
		}
	}
	if err := renameDefinitionTables(ctx, tx, canonical, backup); err != nil {
		return err
	}
	if err := renameDefinitionTables(ctx, tx, shadow, canonical); err != nil {
		return err
	}
	for _, definition := range []Definition{backup, canonical} {
		if _, err := tx.Exec(ctx, propertyGraphDDL(definition)); err != nil {
			return fmt.Errorf("create promoted property graph %q: %w", definition.Graph, err)
		}
	}
	backupMapping, err := definitionMapping(active.JobID, backup, backupFingerprint)
	if err != nil {
		return err
	}
	backupMapping.State = meta.PropertyGraphRetainedBackup
	if err := store.RelocatePropertyGraph(ctx, backupMapping); err != nil {
		return err
	}
	if err := store.ReplacePropertyGraphDigests(
		ctx, jobID, digests.Ranges, digests.Root, digests.Rows, DigestRangeCount); err != nil {
		return err
	}
	canonicalMapping, err := definitionMapping(jobID, canonical, canonicalFingerprint)
	if err != nil {
		return err
	}
	canonicalMapping.State = meta.PropertyGraphActive
	if err := store.RelocatePropertyGraph(ctx, canonicalMapping); err != nil {
		return err
	}
	if err := store.CompleteJobWithTelemetry(ctx, jobID, telemetry); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit PostgreSQL property graph replacement: %w", err)
	}
	return nil
}

func renameDefinitionTables(
	ctx context.Context,
	tx pgx.Tx,
	from Definition,
	to Definition,
) error {
	from = from.normalized()
	to = to.normalized()
	if len(from.Vertices) != len(to.Vertices) || len(from.Edges) != len(to.Edges) {
		return errors.New("property graph replacement definitions do not align")
	}
	for index := range from.Vertices {
		if from.Vertices[index].Label != to.Vertices[index].Label {
			return errors.New("property graph replacement vertex labels do not align")
		}
	}
	for index := range from.Edges {
		if from.Edges[index].Label != to.Edges[index].Label {
			return errors.New("property graph replacement edge labels do not align")
		}
	}
	for index := range from.Vertices {
		if _, err := tx.Exec(ctx, fmt.Sprintf("ALTER TABLE %s RENAME TO %s",
			qualifiedName(from.Schema, from.Vertices[index].Table),
			QuoteIdentifier(to.Vertices[index].Table))); err != nil {
			return fmt.Errorf("rename replacement vertex table %q: %w", from.Vertices[index].Label, err)
		}
	}
	for index := range from.Edges {
		if _, err := tx.Exec(ctx, fmt.Sprintf("ALTER TABLE %s RENAME TO %s",
			qualifiedName(from.Schema, from.Edges[index].Table),
			QuoteIdentifier(to.Edges[index].Table))); err != nil {
			return fmt.Errorf("rename replacement edge table %q: %w", from.Edges[index].Label, err)
		}
	}
	return nil
}

func (adapter *Adapter) CleanupReplace(
	ctx context.Context,
	jobID string,
	canonical Definition,
) error {
	active, err := adapter.store.GetPropertyGraph(ctx, jobID)
	if err != nil {
		return err
	}
	canonicalFingerprint, err := canonical.Fingerprint()
	if err != nil {
		return err
	}
	if active.State != meta.PropertyGraphActive || active.Schema != canonical.Schema ||
		active.Graph != canonical.Graph || active.DefinitionFingerprint != canonicalFingerprint {
		return fmt.Errorf("%w: active replacement mapping changed", meta.ErrGenerationMismatch)
	}
	_, backup, err := ReplacementDefinitions(canonical, jobID)
	if err != nil {
		return err
	}
	retained, err := adapter.store.PropertyGraphByTargetState(
		ctx, backup.Schema, backup.Graph, meta.PropertyGraphRetainedBackup)
	if errors.Is(err, meta.ErrNotFound) {
		return nil
	}
	if err != nil {
		return err
	}
	backupFingerprint, err := backup.Fingerprint()
	if err != nil {
		return err
	}
	if retained.DefinitionFingerprint != backupFingerprint {
		return fmt.Errorf("%w: retained replacement backup mapping changed", meta.ErrGenerationMismatch)
	}
	if err := adapter.admitObjects(ctx, backup); err != nil {
		return err
	}
	tx, err := adapter.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin PostgreSQL property graph backup cleanup: %w", err)
	}
	defer tx.Rollback(context.WithoutCancel(ctx))
	if _, err := tx.Exec(ctx,
		"SET LOCAL search_path TO "+QuoteIdentifier(backup.Schema)+", pg_catalog"); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, "DROP PROPERTY GRAPH "+QuoteIdentifier(backup.Graph)); err != nil {
		return fmt.Errorf("drop retained property graph backup: %w", err)
	}
	for index := len(backup.Edges) - 1; index >= 0; index-- {
		if _, err := tx.Exec(ctx, "DROP TABLE "+
			qualifiedName(backup.Schema, backup.Edges[index].Table)); err != nil {
			return fmt.Errorf("drop retained edge table: %w", err)
		}
	}
	for index := len(backup.Vertices) - 1; index >= 0; index-- {
		if _, err := tx.Exec(ctx, "DROP TABLE "+
			qualifiedName(backup.Schema, backup.Vertices[index].Table)); err != nil {
			return fmt.Errorf("drop retained vertex table: %w", err)
		}
	}
	store, err := meta.New(tx)
	if err != nil {
		return err
	}
	retained.State = meta.PropertyGraphSuperseded
	if err := store.RelocatePropertyGraph(ctx, retained); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit PostgreSQL property graph backup cleanup: %w", err)
	}
	return nil
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

func inspectFinalTarget(
	ctx context.Context,
	tx pgx.Tx,
	jobID string,
	definition Definition,
) (int64, DigestSet, error) {
	if _, err := tx.Exec(ctx,
		"SET LOCAL search_path TO "+QuoteIdentifier(definition.Schema)+", pg_catalog"); err != nil {
		return 0, DigestSet{}, fmt.Errorf("set property graph inspection search path: %w", err)
	}
	var total int64
	for _, table := range appendTableNames(definition.normalized()) {
		var count int64
		if err := tx.QueryRow(ctx, "SELECT count(*) FROM "+
			qualifiedName(definition.Schema, table)).Scan(&count); err != nil {
			return 0, DigestSet{}, fmt.Errorf("count PostgreSQL property graph table %q: %w", table, err)
		}
		total += count
	}
	first := definition.normalized().Vertices[0]
	var graphCount, tableCount int64
	query := fmt.Sprintf(`SELECT count(*) FROM GRAPH_TABLE (
		%s MATCH (v IS %s) COLUMNS (v.external_id AS external_id)
	)`, QuoteIdentifier(definition.Graph), QuoteIdentifier(first.Label))
	if err := tx.QueryRow(ctx, query).Scan(&graphCount); err != nil {
		return 0, DigestSet{}, fmt.Errorf("query finalized PostgreSQL property graph: %w", err)
	}
	if err := tx.QueryRow(ctx, "SELECT count(*) FROM "+
		qualifiedName(definition.Schema, first.Table)).Scan(&tableCount); err != nil {
		return 0, DigestSet{}, fmt.Errorf("count property graph verification vertex table: %w", err)
	}
	if graphCount != tableCount {
		return 0, DigestSet{}, fmt.Errorf("SQL/PGQ vertex count is %d, expected %d", graphCount, tableCount)
	}
	digests, err := computeDigests(ctx, tx, jobID, definition)
	if err != nil {
		return 0, DigestSet{}, fmt.Errorf("compute final PostgreSQL property graph digests: %w", err)
	}
	return total, digests, nil
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
	if (job.LoadMode == "create" || job.LoadMode == "replace") && total != job.CommittedRows {
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
	if digests.Rows != total {
		return fmt.Errorf("PostgreSQL property graph digest covered %d rows, expected %d",
			digests.Rows, total)
	}
	if err := store.ReplacePropertyGraphDigests(
		ctx, jobID, digests.Ranges, digests.Root, digests.Rows, DigestRangeCount,
	); err != nil {
		return err
	}
	var activateErr error
	if job.LoadMode == "append" || job.LoadMode == "upsert" {
		activateErr = store.ActivatePropertyGraphReplacing(
			ctx, jobID, definition.Schema, definition.Graph)
	} else {
		activateErr = store.ActivatePropertyGraph(ctx, jobID)
	}
	if activateErr != nil {
		return activateErr
	}
	if err := store.CompleteJobWithTelemetry(ctx, jobID, telemetry); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit PostgreSQL property graph finalization: %w", err)
	}
	return nil
}
