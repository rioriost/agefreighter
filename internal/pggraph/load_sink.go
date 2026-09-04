package pggraph

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/checkpoint"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	sinkcontract "github.com/rioriost/agefreighter/internal/sink"
	"github.com/rioriost/agefreighter/pkg/model"
)

type LoadSinkOptions struct {
	JobID           string
	Definition      Definition
	JobVerification *meta.JobVerification
	Mode            config.LoadMode
	AppendDuplicate config.AppendDuplicatePolicy
	PropertyMode    config.PropertyMode
}

type loadBinding struct {
	kind       meta.LabelKind
	table      string
	startLabel model.Label
	endLabel   model.Label
}

type LoadSink struct {
	adapter  *Adapter
	options  LoadSinkOptions
	bindings map[model.Label]loadBinding
	mu       sync.Mutex
	active   bool
}

type loadTransaction struct {
	sink      *LoadSink
	tx        pgx.Tx
	owner     *pgxpool.Conn
	lockKey   string
	metadata  sinkcontract.BatchMetadata
	wrote     bool
	finalized bool
}

type committedReplayTransaction struct {
	sink      *LoadSink
	metadata  sinkcontract.BatchMetadata
	wrote     bool
	finalized bool
}

func NewLoadSink(adapter *Adapter, options LoadSinkOptions) (*LoadSink, error) {
	if adapter == nil || adapter.pool == nil || adapter.store == nil {
		return nil, errors.New("PostgreSQL property graph adapter is required")
	}
	if err := meta.ValidateJobID(options.JobID); err != nil {
		return nil, err
	}
	if _, err := options.Definition.Fingerprint(); err != nil {
		return nil, err
	}
	if options.Mode == "" {
		options.Mode = config.LoadCreate
	}
	if options.PropertyMode == "" {
		options.PropertyMode = config.PropertiesReplace
	}
	if options.Mode == config.LoadAppend && options.AppendDuplicate == "" {
		options.AppendDuplicate = config.AppendDuplicateError
	}
	if options.Mode != config.LoadCreate && options.Mode != config.LoadReplace &&
		options.Mode != config.LoadAppend && options.Mode != config.LoadUpsert {
		return nil, fmt.Errorf("unsupported PostgreSQL property graph load mode %q", options.Mode)
	}
	if options.PropertyMode != config.PropertiesReplace &&
		options.PropertyMode != config.PropertiesMerge &&
		options.PropertyMode != config.PropertiesMergeDeleteNull {
		return nil, fmt.Errorf("unsupported PostgreSQL property graph property mode %q", options.PropertyMode)
	}
	if options.Mode == config.LoadAppend && options.AppendDuplicate != config.AppendDuplicateError &&
		options.AppendDuplicate != config.AppendDuplicateIgnoreIdentical {
		return nil, fmt.Errorf("unsupported PostgreSQL property graph append duplicate policy %q", options.AppendDuplicate)
	}
	if options.Mode != config.LoadAppend && options.AppendDuplicate != "" {
		return nil, errors.New("PostgreSQL property graph append duplicate policy requires append mode")
	}
	bindings := make(map[model.Label]loadBinding,
		len(options.Definition.Vertices)+len(options.Definition.Edges))
	vertexByTable := make(map[string]model.Label, len(options.Definition.Vertices))
	for _, vertex := range options.Definition.Vertices {
		label := model.Label(vertex.Label)
		bindings[label] = loadBinding{kind: meta.VertexLabel, table: vertex.Table}
		vertexByTable[vertex.Table] = label
	}
	for _, edge := range options.Definition.Edges {
		label := model.Label(edge.Label)
		bindings[label] = loadBinding{
			kind: meta.EdgeLabel, table: edge.Table,
			startLabel: vertexByTable[edge.SourceTable],
			endLabel:   vertexByTable[edge.DestinationTable],
		}
	}
	return &LoadSink{adapter: adapter, options: options, bindings: bindings}, nil
}

func (target *LoadSink) Begin(
	ctx context.Context,
	batch sinkcontract.BatchMetadata,
) (sinkcontract.Transaction, error) {
	if err := validateBatchMetadata(batch); err != nil {
		return nil, err
	}
	target.mu.Lock()
	if target.active {
		target.mu.Unlock()
		return nil, errors.New("PostgreSQL property graph sink already has an active transaction")
	}
	target.active = true
	target.mu.Unlock()
	release := func() { target.setInactive() }

	owner, err := target.adapter.pool.Acquire(ctx)
	if err != nil {
		release()
		return nil, fmt.Errorf("acquire PostgreSQL property graph load connection: %w", err)
	}
	lockKey := fmt.Sprintf("%d/%d", batch.ID, batch.Attempt)
	if _, err := owner.Exec(ctx, `SELECT pg_catalog.pg_advisory_lock(
		pg_catalog.hashtext($1), pg_catalog.hashtext($2))`,
		target.options.JobID, lockKey,
	); err != nil {
		owner.Release()
		release()
		return nil, fmt.Errorf("lock PostgreSQL property graph batch: %w", err)
	}
	ownerStore, err := meta.New(owner.Conn())
	if err != nil {
		ownerErr := target.releaseOwner(owner, lockKey)
		release()
		return nil, errors.Join(err, ownerErr)
	}
	started, err := ownerStore.StartBatch(ctx, meta.BatchAttempt{
		JobID: target.options.JobID, BatchID: batch.ID, Attempt: batch.Attempt,
		Rows: int64(batch.Rows), Bytes: batch.Bytes,
		First: targetPosition(batch.FirstPosition),
	})
	if err != nil {
		ownerErr := target.releaseOwner(owner, lockKey)
		release()
		return nil, errors.Join(fmt.Errorf(
			"start PostgreSQL property graph batch: %w", err), ownerErr)
	}
	if started.Status == meta.BatchCommitted {
		if err := target.releaseOwner(owner, lockKey); err != nil {
			release()
			return nil, err
		}
		return &committedReplayTransaction{sink: target, metadata: batch}, nil
	}
	if started.Status != meta.BatchRunning {
		ownerErr := target.releaseOwner(owner, lockKey)
		release()
		return nil, errors.Join(fmt.Errorf(
			"started PostgreSQL property graph batch has status %q", started.Status), ownerErr)
	}
	tx, err := owner.Begin(ctx)
	if err != nil {
		diagnosticErr := target.recordFailedBatch(ctx, batch,
			"begin PostgreSQL property graph transaction: "+err.Error())
		ownerErr := target.releaseOwner(owner, lockKey)
		release()
		return nil, errors.Join(fmt.Errorf(
			"begin PostgreSQL property graph transaction: %w", err), diagnosticErr, ownerErr)
	}
	return &loadTransaction{
		sink: target, tx: tx, owner: owner, lockKey: lockKey, metadata: batch,
	}, nil
}

func (target *LoadSink) setInactive() {
	target.mu.Lock()
	target.active = false
	target.mu.Unlock()
}

func (transaction *loadTransaction) Write(
	ctx context.Context,
	records []model.Record,
) error {
	if transaction.finalized {
		return errors.New("PostgreSQL property graph transaction is finalized")
	}
	if transaction.wrote {
		return errors.New("PostgreSQL property graph records were already written")
	}
	if len(records) != transaction.metadata.Rows {
		return fmt.Errorf("load batch contains %d records, expected %d",
			len(records), transaction.metadata.Rows)
	}
	transaction.wrote = true
	vertices := make(map[model.Label][]*model.Vertex)
	edges := make(map[model.Label][]*model.Edge)
	seenEdge := false
	for index, record := range records {
		switch record.Kind() {
		case model.RecordVertex:
			if seenEdge {
				return fmt.Errorf("vertex record %d follows an edge record", index)
			}
			vertices[record.Vertex.Label] = append(vertices[record.Vertex.Label], record.Vertex)
		case model.RecordEdge:
			seenEdge = true
			edges[record.Edge.Label] = append(edges[record.Edge.Label], record.Edge)
		default:
			return fmt.Errorf("load record %d is invalid", index)
		}
	}
	for index, label := range sortedRecordLabels(vertices) {
		if err := transaction.writeVertices(ctx, label, vertices[label], index); err != nil {
			return err
		}
	}
	for index, label := range sortedRecordLabels(edges) {
		if err := transaction.writeEdges(ctx, label, edges[label], len(vertices)+index); err != nil {
			return err
		}
	}
	return nil
}

func sortedRecordLabels[T any](groups map[model.Label][]T) []model.Label {
	labels := make([]model.Label, 0, len(groups))
	for label := range groups {
		labels = append(labels, label)
	}
	slices.Sort(labels)
	return labels
}

func (transaction *loadTransaction) writeVertices(
	ctx context.Context,
	label model.Label,
	vertices []*model.Vertex,
	group int,
) error {
	binding, ok := transaction.sink.bindings[label]
	if !ok || binding.kind != meta.VertexLabel {
		return fmt.Errorf("vertex label %q is not registered", label)
	}
	rows := make([][]any, len(vertices))
	seen := make(map[string]struct{}, len(vertices))
	for index, vertex := range vertices {
		if vertex.Namespace == "" || vertex.ExternalID == "" {
			return fmt.Errorf("vertex %d identity is empty", index)
		}
		key := string(vertex.Namespace) + "\x00" + string(vertex.ExternalID)
		if _, exists := seen[key]; exists {
			return fmt.Errorf("vertex %d duplicates external identity in batch", index)
		}
		seen[key] = struct{}{}
		properties, err := recordProperties(vertex.Properties, vertex.EncodedProperties)
		if err != nil {
			return fmt.Errorf("encode vertex %q properties: %w", vertex.ExternalID, err)
		}
		rangeID, digest, err := vertexRecordDigest(
			string(label), string(vertex.Namespace), string(vertex.ExternalID), properties,
		)
		if err != nil {
			return fmt.Errorf("digest vertex %q: %w", vertex.ExternalID, err)
		}
		rows[index] = []any{
			string(vertex.Namespace), string(vertex.ExternalID), string(properties),
			int16(rangeID), digest,
		}
	}
	if transaction.sink.options.Mode == config.LoadAppend ||
		transaction.sink.options.Mode == config.LoadUpsert {
		return transaction.writeIncrementalVertices(ctx, label, binding, rows, group)
	}
	_, err := transaction.tx.CopyFrom(
		ctx,
		pgx.Identifier{transaction.sink.options.Definition.Schema, binding.table},
		[]string{"source_namespace", "external_id", "properties", "digest_range", "source_digest"},
		pgx.CopyFromRows(rows),
	)
	if err != nil {
		return fmt.Errorf("copy PostgreSQL property graph vertices for %q: %w", label, err)
	}
	return nil
}

func (transaction *loadTransaction) writeIncrementalVertices(
	ctx context.Context,
	label model.Label,
	binding loadBinding,
	rows [][]any,
	group int,
) error {
	tempName := fmt.Sprintf("af_pgv_%d_%d_%d", transaction.metadata.ID,
		transaction.metadata.Attempt, group)
	if _, err := transaction.tx.Exec(ctx, fmt.Sprintf(`CREATE TEMP TABLE %s (
		ordinal bigint NOT NULL,
		source_namespace text NOT NULL,
		external_id text NOT NULL,
		properties jsonb NOT NULL,
		digest_range smallint NOT NULL,
		source_digest character(64) NOT NULL
	) ON COMMIT DROP`, QuoteIdentifier(tempName))); err != nil {
		return fmt.Errorf("create bounded vertex stage for %q: %w", label, err)
	}
	staged := make([][]any, len(rows))
	for index, row := range rows {
		staged[index] = append([]any{int64(index)}, row...)
	}
	if _, err := transaction.tx.CopyFrom(ctx, pgx.Identifier{tempName}, []string{
		"ordinal", "source_namespace", "external_id", "properties",
		"digest_range", "source_digest",
	}, pgx.CopyFromRows(staged)); err != nil {
		return fmt.Errorf("stage PostgreSQL property graph vertices for %q: %w", label, err)
	}
	table := qualifiedName(transaction.sink.options.Definition.Schema, binding.table)
	if transaction.sink.options.Mode == config.LoadAppend &&
		transaction.sink.options.AppendDuplicate == config.AppendDuplicateIgnoreIdentical {
		var conflicts int64
		if err := transaction.tx.QueryRow(ctx, fmt.Sprintf(`SELECT count(*)
			FROM %s staged JOIN %s target USING (source_namespace, external_id)
			WHERE staged.source_digest <> target.source_digest`,
			QuoteIdentifier(tempName), table)).Scan(&conflicts); err != nil {
			return fmt.Errorf("inspect append vertex duplicates for %q: %w", label, err)
		}
		if conflicts != 0 {
			return fmt.Errorf("append vertex label %q has %d conflicting duplicate identities", label, conflicts)
		}
	}
	statement := fmt.Sprintf(`INSERT INTO %s AS current (
		source_namespace, external_id, properties, digest_range, source_digest
	) SELECT source_namespace, external_id, properties, digest_range, source_digest
	  FROM %s ORDER BY ordinal`, table, QuoteIdentifier(tempName))
	switch transaction.sink.options.Mode {
	case config.LoadAppend:
		if transaction.sink.options.AppendDuplicate == config.AppendDuplicateIgnoreIdentical {
			statement += " ON CONFLICT (source_namespace, external_id) DO NOTHING"
		}
	case config.LoadUpsert:
		statement += fmt.Sprintf(` ON CONFLICT (source_namespace, external_id) DO UPDATE SET
			properties = %s,
			digest_range = EXCLUDED.digest_range,
			source_digest = EXCLUDED.source_digest`,
			propertyUpdateExpression("current", transaction.sink.options.PropertyMode))
	}
	if _, err := transaction.tx.Exec(ctx, statement); err != nil {
		return fmt.Errorf("write incremental PostgreSQL property graph vertices for %q: %w", label, err)
	}
	if transaction.sink.options.Mode == config.LoadUpsert &&
		transaction.sink.options.PropertyMode != config.PropertiesReplace {
		return transaction.refreshVertexDigests(ctx, label, binding, tempName)
	}
	return nil
}

func propertyUpdateExpression(table string, mode config.PropertyMode) string {
	switch mode {
	case config.PropertiesMerge:
		return table + ".properties || EXCLUDED.properties"
	case config.PropertiesMergeDeleteNull:
		return "jsonb_strip_nulls(" + table + ".properties || EXCLUDED.properties)"
	default:
		return "EXCLUDED.properties"
	}
}

func (transaction *loadTransaction) refreshVertexDigests(
	ctx context.Context,
	label model.Label,
	binding loadBinding,
	tempName string,
) error {
	table := qualifiedName(transaction.sink.options.Definition.Schema, binding.table)
	rows, err := transaction.tx.Query(ctx, fmt.Sprintf(`SELECT target.source_namespace,
		target.external_id, target.properties
		FROM %s target JOIN %s staged USING (source_namespace, external_id)
		ORDER BY staged.ordinal`, table, QuoteIdentifier(tempName)))
	if err != nil {
		return fmt.Errorf("read merged vertex properties for %q: %w", label, err)
	}
	defer rows.Close()
	batch := &pgx.Batch{}
	for rows.Next() {
		var namespace, externalID string
		var properties []byte
		if err := rows.Scan(&namespace, &externalID, &properties); err != nil {
			return err
		}
		rangeID, digest, err := vertexRecordDigest(string(label), namespace, externalID, properties)
		if err != nil {
			return err
		}
		batch.Queue(fmt.Sprintf(`UPDATE %s SET digest_range = $3, source_digest = $4
			WHERE source_namespace = $1 AND external_id = $2`, table),
			namespace, externalID, int16(rangeID), digest)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	results := transaction.tx.SendBatch(ctx, batch)
	defer results.Close()
	for range batch.Len() {
		if _, err := results.Exec(); err != nil {
			return fmt.Errorf("refresh merged vertex digest for %q: %w", label, err)
		}
	}
	return results.Close()
}

func (transaction *loadTransaction) writeEdges(
	ctx context.Context,
	label model.Label,
	edges []*model.Edge,
	group int,
) error {
	binding, ok := transaction.sink.bindings[label]
	if !ok || binding.kind != meta.EdgeLabel {
		return fmt.Errorf("edge label %q is not registered", label)
	}
	start := transaction.sink.bindings[binding.startLabel]
	end := transaction.sink.bindings[binding.endLabel]
	tempName := fmt.Sprintf("af_pgq_%d_%d_%d", transaction.metadata.ID,
		transaction.metadata.Attempt, group)
	rows := make([][]any, len(edges))
	seen := make(map[string]struct{}, len(edges))
	for index, edge := range edges {
		if edge.Namespace == "" || edge.ExternalID == "" || edge.Start.Namespace == "" ||
			edge.Start.ExternalID == "" || edge.End.Namespace == "" ||
			edge.End.ExternalID == "" {
			return fmt.Errorf("edge %d identity or endpoint is empty", index)
		}
		if edge.Start.Label != binding.startLabel || edge.End.Label != binding.endLabel {
			return fmt.Errorf("edge %d endpoint labels do not match mapping for %q", index, label)
		}
		key := string(edge.Namespace) + "\x00" + string(edge.ExternalID)
		if _, exists := seen[key]; exists {
			return fmt.Errorf("edge %d duplicates external identity in batch", index)
		}
		seen[key] = struct{}{}
		properties, err := recordProperties(edge.Properties, edge.EncodedProperties)
		if err != nil {
			return fmt.Errorf("encode edge %d properties: %w", index, err)
		}
		rangeID, digest, err := edgeRecordDigest(
			string(label), string(edge.Namespace), string(edge.ExternalID),
			string(binding.startLabel), string(edge.Start.Namespace), string(edge.Start.ExternalID),
			string(binding.endLabel), string(edge.End.Namespace), string(edge.End.ExternalID),
			properties,
		)
		if err != nil {
			return fmt.Errorf("digest edge %q: %w", edge.ExternalID, err)
		}
		rows[index] = []any{
			int64(index), string(edge.Namespace), string(edge.ExternalID),
			string(edge.Start.Namespace), string(edge.Start.ExternalID),
			string(edge.End.Namespace), string(edge.End.ExternalID), string(properties),
			int16(rangeID), digest,
		}
	}
	if _, err := transaction.tx.Exec(ctx, fmt.Sprintf(`CREATE TEMP TABLE %s (
		ordinal bigint NOT NULL,
		source_namespace text NOT NULL,
		external_id text NOT NULL,
		start_namespace text NOT NULL,
		start_external_id text NOT NULL,
		end_namespace text NOT NULL,
		end_external_id text NOT NULL,
		properties jsonb NOT NULL,
		digest_range smallint NOT NULL,
		source_digest character(64) NOT NULL
	) ON COMMIT DROP`, QuoteIdentifier(tempName))); err != nil {
		return fmt.Errorf("create bounded edge stage for %q: %w", label, err)
	}
	if _, err := transaction.tx.CopyFrom(ctx, pgx.Identifier{tempName}, []string{
		"ordinal", "source_namespace", "external_id", "start_namespace",
		"start_external_id", "end_namespace", "end_external_id", "properties",
		"digest_range", "source_digest",
	}, pgx.CopyFromRows(rows)); err != nil {
		return fmt.Errorf("stage PostgreSQL property graph edges for %q: %w", label, err)
	}
	schema := transaction.sink.options.Definition.Schema
	targetTable := qualifiedName(schema, binding.table)
	statement := fmt.Sprintf(`INSERT INTO %s AS current (
			source_namespace, external_id, start_id, end_id, properties,
		digest_range, source_digest
	)
	SELECT staged.source_namespace, staged.external_id,
	       source_vertex.id, destination_vertex.id, staged.properties,
	       staged.digest_range, staged.source_digest
	FROM %s staged
	JOIN %s source_vertex
	  ON source_vertex.source_namespace = staged.start_namespace
	 AND source_vertex.external_id = staged.start_external_id
	JOIN %s destination_vertex
	  ON destination_vertex.source_namespace = staged.end_namespace
	 AND destination_vertex.external_id = staged.end_external_id
	ORDER BY staged.ordinal`,
		targetTable, QuoteIdentifier(tempName),
		qualifiedName(schema, start.table), qualifiedName(schema, end.table))
	if transaction.sink.options.Mode == config.LoadAppend &&
		transaction.sink.options.AppendDuplicate == config.AppendDuplicateIgnoreIdentical {
		var conflicts int64
		if err := transaction.tx.QueryRow(ctx, fmt.Sprintf(`SELECT count(*)
			FROM %s staged JOIN %s target USING (source_namespace, external_id)
			WHERE staged.source_digest <> target.source_digest`,
			QuoteIdentifier(tempName), targetTable)).Scan(&conflicts); err != nil {
			return fmt.Errorf("inspect append edge duplicates for %q: %w", label, err)
		}
		if conflicts != 0 {
			return fmt.Errorf("append edge label %q has %d conflicting duplicate identities", label, conflicts)
		}
		statement += " ON CONFLICT (source_namespace, external_id) DO NOTHING"
	}
	if transaction.sink.options.Mode == config.LoadUpsert {
		statement += fmt.Sprintf(` ON CONFLICT (source_namespace, external_id) DO UPDATE SET
			start_id = EXCLUDED.start_id,
			end_id = EXCLUDED.end_id,
			properties = %s,
			digest_range = EXCLUDED.digest_range,
			source_digest = EXCLUDED.source_digest`,
			propertyUpdateExpression("current", transaction.sink.options.PropertyMode))
	}
	tag, err := transaction.tx.Exec(ctx, statement)
	if err != nil {
		return fmt.Errorf("insert PostgreSQL property graph edges for %q: %w", label, err)
	}
	if transaction.sink.options.Mode != config.LoadAppend ||
		transaction.sink.options.AppendDuplicate != config.AppendDuplicateIgnoreIdentical {
		if tag.RowsAffected() != int64(len(edges)) {
			return fmt.Errorf("edge label %q resolved %d of %d endpoints",
				label, tag.RowsAffected(), len(edges))
		}
	} else if tag.RowsAffected() > int64(len(edges)) {
		return fmt.Errorf("edge label %q wrote an invalid row count %d", label, tag.RowsAffected())
	}
	if transaction.sink.options.Mode == config.LoadUpsert &&
		transaction.sink.options.PropertyMode != config.PropertiesReplace {
		return transaction.refreshEdgeDigests(ctx, label, binding, start, end, tempName)
	}
	return nil
}

func (transaction *loadTransaction) refreshEdgeDigests(
	ctx context.Context,
	label model.Label,
	binding loadBinding,
	start loadBinding,
	end loadBinding,
	tempName string,
) error {
	schema := transaction.sink.options.Definition.Schema
	table := qualifiedName(schema, binding.table)
	rows, err := transaction.tx.Query(ctx, fmt.Sprintf(`SELECT target.source_namespace,
		target.external_id, source.source_namespace, source.external_id,
		destination.source_namespace, destination.external_id, target.properties
		FROM %s target
		JOIN %s staged USING (source_namespace, external_id)
		JOIN %s source ON source.id = target.start_id
		JOIN %s destination ON destination.id = target.end_id
		ORDER BY staged.ordinal`, table, QuoteIdentifier(tempName),
		qualifiedName(schema, start.table), qualifiedName(schema, end.table)))
	if err != nil {
		return fmt.Errorf("read merged edge properties for %q: %w", label, err)
	}
	defer rows.Close()
	batch := &pgx.Batch{}
	for rows.Next() {
		var namespace, externalID, startNamespace, startID, endNamespace, endID string
		var properties []byte
		if err := rows.Scan(&namespace, &externalID, &startNamespace, &startID,
			&endNamespace, &endID, &properties); err != nil {
			return err
		}
		rangeID, digest, err := edgeRecordDigest(string(label), namespace, externalID,
			string(binding.startLabel), startNamespace, startID,
			string(binding.endLabel), endNamespace, endID, properties)
		if err != nil {
			return err
		}
		batch.Queue(fmt.Sprintf(`UPDATE %s SET digest_range = $3, source_digest = $4
			WHERE source_namespace = $1 AND external_id = $2`, table),
			namespace, externalID, int16(rangeID), digest)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	results := transaction.tx.SendBatch(ctx, batch)
	defer results.Close()
	for range batch.Len() {
		if _, err := results.Exec(); err != nil {
			return fmt.Errorf("refresh merged edge digest for %q: %w", label, err)
		}
	}
	return results.Close()
}

func recordProperties(properties model.Properties, encoded []byte) ([]byte, error) {
	if len(encoded) != 0 {
		if !json.Valid(encoded) {
			return nil, errors.New("encoded properties are not valid JSON")
		}
		var object map[string]any
		if err := json.Unmarshal(encoded, &object); err != nil || object == nil {
			return nil, errors.New("encoded properties must be a JSON object")
		}
		return encoded, nil
	}
	return model.EncodeProperties(properties)
}

func (transaction *loadTransaction) Commit(
	ctx context.Context,
	state checkpoint.State,
) error {
	if transaction.finalized {
		return errors.New("PostgreSQL property graph transaction is finalized")
	}
	if !transaction.wrote {
		return transaction.abort(ctx, errors.New("PostgreSQL property graph transaction has not written records"))
	}
	if state.BatchID != transaction.metadata.ID ||
		state.Attempt != transaction.metadata.Attempt ||
		state.Phase != checkpoint.PhaseCommitted ||
		state.Position != transaction.metadata.LastPosition {
		return transaction.abort(ctx, errors.New("checkpoint does not match PostgreSQL property graph batch"))
	}
	store, err := meta.New(transaction.tx)
	if err != nil {
		return transaction.abort(ctx, err)
	}
	if err := store.CommitBatchWithLabelCountersAndVerification(
		ctx, transaction.sink.options.JobID, transaction.metadata.ID,
		transaction.metadata.Attempt, targetPosition(state.Position), 0, nil,
		transaction.sink.options.JobVerification,
	); err != nil {
		return transaction.abort(ctx, fmt.Errorf("commit PostgreSQL property graph checkpoint: %w", err))
	}
	transaction.finalized = true
	if err := transaction.tx.Commit(ctx); err != nil {
		ownerErr := transaction.sink.releaseOwner(transaction.owner, transaction.lockKey)
		transaction.sink.setInactive()
		return errors.Join(fmt.Errorf("commit PostgreSQL property graph transaction: %w", err), ownerErr)
	}
	transaction.sink.options.JobVerification = nil
	ownerErr := transaction.sink.releaseOwner(transaction.owner, transaction.lockKey)
	transaction.sink.setInactive()
	return ownerErr
}

func (transaction *loadTransaction) Rollback(ctx context.Context) error {
	if transaction.finalized {
		return nil
	}
	transaction.finalized = true
	rollbackErr := transaction.tx.Rollback(ctx)
	if errors.Is(rollbackErr, pgx.ErrTxClosed) {
		rollbackErr = nil
	}
	diagnosticErr := transaction.sink.recordFailedBatch(
		ctx, transaction.metadata, "PostgreSQL property graph transaction rolled back")
	ownerErr := transaction.sink.releaseOwner(transaction.owner, transaction.lockKey)
	transaction.sink.setInactive()
	return errors.Join(rollbackErr, diagnosticErr, ownerErr)
}

func (transaction *loadTransaction) abort(ctx context.Context, cause error) error {
	return errors.Join(cause, transaction.Rollback(ctx))
}

func (target *LoadSink) recordFailedBatch(
	ctx context.Context,
	batch sinkcontract.BatchMetadata,
	message string,
) error {
	diagnosticCtx, cancel := context.WithTimeout(
		context.WithoutCancel(ctx), target.adapter.operationTimeout)
	defer cancel()
	return target.adapter.store.RecordFailedBatch(diagnosticCtx, meta.BatchAttempt{
		JobID: target.options.JobID, BatchID: batch.ID, Attempt: batch.Attempt,
		Rows: int64(batch.Rows), Bytes: batch.Bytes,
		First: targetPosition(batch.FirstPosition),
	}, message)
}

func (target *LoadSink) releaseOwner(owner *pgxpool.Conn, lockKey string) error {
	ctx, cancel := context.WithTimeout(context.Background(), target.adapter.operationTimeout)
	defer cancel()
	var unlocked bool
	err := owner.QueryRow(ctx, `SELECT pg_catalog.pg_advisory_unlock(
		pg_catalog.hashtext($1), pg_catalog.hashtext($2))`,
		target.options.JobID, lockKey,
	).Scan(&unlocked)
	if err != nil || !unlocked {
		_ = owner.Conn().Close(ctx)
	}
	owner.Release()
	if err != nil {
		return fmt.Errorf("unlock PostgreSQL property graph batch: %w", err)
	}
	if !unlocked {
		return errors.New("PostgreSQL property graph batch lock was not held")
	}
	return nil
}

func validateBatchMetadata(batch sinkcontract.BatchMetadata) error {
	if batch.ID == 0 || batch.ID > math.MaxInt64 {
		return errors.New("load batch ID must be within 1..MaxInt64")
	}
	if batch.Attempt == 0 || batch.Rows <= 0 || batch.Bytes <= 0 {
		return errors.New("load batch attempt, rows, and bytes must be positive")
	}
	if batch.LastPosition.Token == "" {
		return errors.New("load batch last position token is required")
	}
	return nil
}

func targetPosition(position model.SourcePosition) meta.Position {
	return meta.Position{
		Resource: position.Resource, Line: position.Line,
		ByteOffset: position.Offset, Token: position.Token,
	}
}

func (transaction *committedReplayTransaction) Write(_ context.Context, records []model.Record) error {
	if transaction.finalized || transaction.wrote {
		return errors.New("committed PostgreSQL property graph replay is finalized or already written")
	}
	if len(records) != transaction.metadata.Rows {
		return errors.New("committed PostgreSQL property graph replay size changed")
	}
	transaction.wrote = true
	return nil
}

func (transaction *committedReplayTransaction) Commit(ctx context.Context, state checkpoint.State) error {
	if transaction.finalized || !transaction.wrote {
		return errors.New("committed PostgreSQL property graph replay is not ready")
	}
	stored, err := transaction.sink.adapter.store.GetBatch(
		ctx, transaction.sink.options.JobID, transaction.metadata.ID, transaction.metadata.Attempt)
	if err != nil || stored.Status != meta.BatchCommitted ||
		stored.Last != targetPosition(state.Position) {
		transaction.finalized = true
		transaction.sink.setInactive()
		return errors.Join(errors.New("stored PostgreSQL property graph checkpoint changed"), err)
	}
	transaction.finalized = true
	transaction.sink.setInactive()
	return nil
}

func (transaction *committedReplayTransaction) Rollback(context.Context) error {
	if !transaction.finalized {
		transaction.finalized = true
		transaction.sink.setInactive()
	}
	return nil
}
