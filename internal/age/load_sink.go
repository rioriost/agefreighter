package age

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

type LoadLabel struct {
	Catalog    LabelCatalog
	Generation meta.LabelGeneration
}

type LoadSinkOptions struct {
	JobID           string
	Graph           meta.GraphGeneration
	Labels          []LoadLabel
	MissingEndpoint config.MissingEndpointPolicy
}

type LoadSink struct {
	adapter     *Adapter
	diagnostics *meta.Store
	options     LoadSinkOptions
	labels      map[model.Label]LoadLabel
	mu          sync.Mutex
	active      bool
}

type loadTransaction struct {
	sink      *LoadSink
	tx        pgx.Tx
	owner     *pgxpool.Conn
	lockKey   string
	metadata  sinkcontract.BatchMetadata
	rejected  int64
	finalized bool
	wrote     bool
}

type committedReplayTransaction struct {
	sink      *LoadSink
	metadata  sinkcontract.BatchMetadata
	wrote     bool
	finalized bool
}

type vertexIdentityRow struct {
	label      LoadLabel
	namespace  model.Namespace
	externalID model.ExternalID
	graphID    GraphID
}

type stagedEdge struct {
	record     model.Edge
	label      LoadLabel
	properties []byte
	ordinal    int
}

type resolvedEdge struct {
	stagedEdge
	startID GraphID
	endID   GraphID
}

func NewLoadSink(
	ctx context.Context,
	adapter *Adapter,
	options LoadSinkOptions,
) (*LoadSink, error) {
	if ctx == nil {
		return nil, errors.New("load sink context is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if adapter == nil || adapter.pool == nil {
		return nil, errors.New("Apache AGE adapter is required")
	}
	if options.JobID == "" || options.Graph.ID <= 0 {
		return nil, errors.New("load job and graph generation are required")
	}
	switch options.MissingEndpoint {
	case config.MissingEndpointError, config.MissingEndpointQuarantine:
	default:
		return nil, fmt.Errorf(
			"unsupported missing endpoint policy %q",
			options.MissingEndpoint,
		)
	}
	if adapter.pool.Config().MaxConns < 2 {
		return nil, errors.New(
			"AGE loading requires at least two target connections",
		)
	}
	diagnostics, err := adapter.Metadata()
	if err != nil {
		return nil, err
	}
	storedGraph, err := diagnostics.AdmitGraphGeneration(
		ctx,
		options.JobID,
		options.Graph,
	)
	if err != nil {
		return nil, fmt.Errorf("admit load graph generation: %w", err)
	}
	options.Graph = storedGraph
	labels := make(map[model.Label]LoadLabel, len(options.Labels))
	for _, binding := range options.Labels {
		if err := validateLoadLabel(options.Graph, binding); err != nil {
			return nil, err
		}
		key := model.Label(binding.Catalog.LabelName)
		if _, exists := labels[key]; exists {
			return nil, fmt.Errorf("duplicate load label %q", key)
		}
		current := binding.Generation
		current.GraphGenerationID = options.Graph.ID
		storedGeneration, err := diagnostics.AdmitLabelGeneration(
			ctx,
			options.Graph.ID,
			current,
		)
		if err != nil {
			return nil, fmt.Errorf("admit load label %q: %w", key, err)
		}
		binding.Generation = storedGeneration
		labels[key] = binding
	}
	if len(labels) == 0 {
		return nil, errors.New("load sink requires at least one label")
	}
	return &LoadSink{
		adapter:     adapter,
		diagnostics: diagnostics,
		options:     options,
		labels:      labels,
	}, nil
}

func validateLoadLabel(graph meta.GraphGeneration, binding LoadLabel) error {
	catalog := binding.Catalog
	generation := binding.Generation
	if catalog.GraphName != graph.GraphName ||
		catalog.GraphOID != graph.GraphOID ||
		catalog.NamespaceOID != graph.NamespaceOID {
		return fmt.Errorf(
			"label %q does not belong to admitted graph generation",
			catalog.LabelName,
		)
	}
	expectedKind := meta.VertexLabel
	if catalog.Kind == EdgeLabel {
		expectedKind = meta.EdgeLabel
	} else if catalog.Kind != VertexLabel {
		return fmt.Errorf("label %q has unsupported kind %q", catalog.LabelName, catalog.Kind)
	}
	if generation.LabelName != catalog.LabelName ||
		generation.GraphGenerationID != graph.ID ||
		generation.Kind != expectedKind ||
		generation.GraphNamespaceOID != catalog.NamespaceOID ||
		generation.LabelID != catalog.LabelID ||
		generation.RelationOID != catalog.RelationOID ||
		generation.SequenceOID != catalog.SequenceOID {
		return fmt.Errorf(
			"label %q generation does not match AGE catalog",
			catalog.LabelName,
		)
	}
	return nil
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
		return nil, errors.New("load sink already has an active transaction")
	}
	target.active = true
	target.mu.Unlock()
	release := func() {
		target.mu.Lock()
		target.active = false
		target.mu.Unlock()
	}

	if err := target.adapter.acquireLoadSlot(ctx); err != nil {
		release()
		return nil, fmt.Errorf("reserve AGE load connections: %w", err)
	}
	owner, err := target.adapter.pool.Acquire(ctx)
	if err != nil {
		target.adapter.releaseLoadSlot()
		release()
		return nil, fmt.Errorf("acquire AGE load ownership connection: %w", err)
	}
	lockKey := fmt.Sprintf("%d/%d", batch.ID, batch.Attempt)
	if _, err := owner.Exec(
		ctx,
		`SELECT pg_catalog.pg_advisory_lock(
			pg_catalog.hashtext($1), pg_catalog.hashtext($2)
		)`,
		target.options.JobID,
		lockKey,
	); err != nil {
		owner.Release()
		target.adapter.releaseLoadSlot()
		release()
		return nil, fmt.Errorf("lock AGE load batch: %w", err)
	}
	releaseOwner := func() error {
		return target.releaseBatchOwner(owner, lockKey)
	}

	attempt := meta.BatchAttempt{
		JobID:   target.options.JobID,
		BatchID: batch.ID,
		Attempt: batch.Attempt,
		Rows:    int64(batch.Rows),
		Bytes:   batch.Bytes,
		First:   metaPosition(batch.FirstPosition),
	}
	stored, err := target.diagnostics.StartBatch(ctx, attempt)
	if err != nil {
		ownerErr := releaseOwner()
		release()
		return nil, errors.Join(
			fmt.Errorf("start AGE load batch: %w", err),
			ownerErr,
		)
	}
	if stored.Status == meta.BatchCommitted {
		if err := releaseOwner(); err != nil {
			release()
			return nil, err
		}
		return &committedReplayTransaction{
			sink:     target,
			metadata: batch,
		}, nil
	}
	if stored.Status != meta.BatchRunning {
		ownerErr := releaseOwner()
		release()
		return nil, errors.Join(
			fmt.Errorf("started AGE load batch has status %q", stored.Status),
			ownerErr,
		)
	}
	tx, err := owner.Begin(ctx)
	if err != nil {
		diagnosticCtx, cancel := context.WithTimeout(
			context.WithoutCancel(ctx),
			target.adapter.operationTimeout,
		)
		defer cancel()
		diagnosticErr := target.diagnostics.RecordFailedBatch(
			diagnosticCtx,
			attempt,
			"begin AGE load transaction: "+err.Error(),
		)
		ownerErr := releaseOwner()
		release()
		return nil, errors.Join(
			fmt.Errorf("begin AGE load transaction: %w", err),
			diagnosticErr,
			ownerErr,
		)
	}
	return &loadTransaction{
		sink:     target,
		tx:       tx,
		owner:    owner,
		lockKey:  lockKey,
		metadata: batch,
	}, nil
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

func (transaction *loadTransaction) Write(
	ctx context.Context,
	records []model.Record,
) error {
	if transaction.finalized {
		return errors.New("load transaction is finalized")
	}
	if transaction.wrote {
		return errors.New("load transaction records were already written")
	}
	if len(records) != transaction.metadata.Rows {
		return fmt.Errorf(
			"load batch contains %d records, expected %d",
			len(records),
			transaction.metadata.Rows,
		)
	}
	transaction.wrote = true

	vertexGroups := make(map[model.Label][]model.Vertex)
	edgeGroups := make(map[model.Label][]model.Edge)
	seenEdge := false
	for index, record := range records {
		switch record.Kind() {
		case model.RecordVertex:
			if seenEdge {
				return fmt.Errorf("vertex record %d follows an edge record", index)
			}
			vertexGroups[record.Vertex.Label] = append(
				vertexGroups[record.Vertex.Label],
				*record.Vertex,
			)
		case model.RecordEdge:
			seenEdge = true
			edgeGroups[record.Edge.Label] = append(
				edgeGroups[record.Edge.Label],
				*record.Edge,
			)
		default:
			return fmt.Errorf("load record %d is invalid", index)
		}
	}
	for _, label := range sortedLabels(vertexGroups) {
		if err := transaction.writeVertices(ctx, label, vertexGroups[label]); err != nil {
			return err
		}
	}
	for _, label := range sortedLabels(edgeGroups) {
		if err := transaction.writeEdges(ctx, label, edgeGroups[label]); err != nil {
			return err
		}
	}
	return nil
}

func sortedLabels[T any](groups map[model.Label][]T) []model.Label {
	labels := make([]model.Label, 0, len(groups))
	for label := range groups {
		labels = append(labels, label)
	}
	slices.Sort(labels)
	return labels
}

func (transaction *loadTransaction) writeVertices(
	ctx context.Context,
	labelName model.Label,
	vertices []model.Vertex,
) error {
	binding, ok := transaction.sink.labels[labelName]
	if !ok {
		return fmt.Errorf("vertex label %q is not registered", labelName)
	}
	if binding.Catalog.Kind != VertexLabel {
		return fmt.Errorf("label %q is not a vertex label", labelName)
	}
	propertiesByIndex := make([][]byte, len(vertices))
	seen := make(map[string]struct{}, len(vertices))
	for index, vertex := range vertices {
		if vertex.ExternalID == "" || vertex.Namespace == "" {
			return fmt.Errorf("vertex %d identity is empty", index)
		}
		key := string(vertex.Namespace) + "\x00" + string(vertex.ExternalID)
		if _, exists := seen[key]; exists {
			return fmt.Errorf("vertex %d duplicates external identity in batch", index)
		}
		seen[key] = struct{}{}
		properties, err := EncodeProperties(vertex.Properties)
		if err != nil {
			return fmt.Errorf("encode vertex %q properties: %w", vertex.ExternalID, err)
		}
		propertiesByIndex[index] = properties
	}
	block, err := (&Transaction{tx: transaction.tx}).ReserveIDs(
		ctx,
		binding.Catalog,
		uint64(len(vertices)),
	)
	if err != nil {
		return err
	}
	rows := make([]VertexRow, len(vertices))
	identities := make([]vertexIdentityRow, len(vertices))
	for index, vertex := range vertices {
		id, err := block.GraphID(uint64(index))
		if err != nil {
			return err
		}
		rows[index] = VertexRow{ID: id, Properties: propertiesByIndex[index]}
		identities[index] = vertexIdentityRow{
			label:      binding,
			namespace:  vertex.Namespace,
			externalID: vertex.ExternalID,
			graphID:    id,
		}
	}
	if _, err := (&Transaction{tx: transaction.tx}).CopyVertices(
		ctx,
		binding.Catalog,
		rows,
		StagedBinaryCopy,
	); err != nil {
		return err
	}
	return transaction.insertVertexIdentities(ctx, identities)
}

func (transaction *loadTransaction) insertVertexIdentities(
	ctx context.Context,
	rows []vertexIdentityRow,
) error {
	const stage = "agefreighter_vertex_identity_stage"
	if _, err := transaction.tx.Exec(
		ctx,
		`CREATE TEMP TABLE IF NOT EXISTS pg_temp.agefreighter_vertex_identity_stage (
			label_generation_id bigint NOT NULL,
			graph_namespace_oid oid NOT NULL,
			label_id integer NOT NULL,
			label_relation_oid oid NOT NULL,
			mapping_generation bigint NOT NULL,
			source_namespace text NOT NULL,
			external_id text NOT NULL,
			graph_id bigint NOT NULL
		) ON COMMIT DROP`,
	); err != nil {
		return fmt.Errorf("prepare vertex identity stage: %w", err)
	}
	if _, err := transaction.tx.Exec(
		ctx,
		`TRUNCATE pg_temp.agefreighter_vertex_identity_stage`,
	); err != nil {
		return fmt.Errorf("truncate vertex identity stage: %w", err)
	}
	copied, err := transaction.tx.CopyFrom(
		ctx,
		pgx.Identifier{"pg_temp", stage},
		[]string{
			"label_generation_id",
			"graph_namespace_oid",
			"label_id",
			"label_relation_oid",
			"mapping_generation",
			"source_namespace",
			"external_id",
			"graph_id",
		},
		pgx.CopyFromSlice(len(rows), func(index int) ([]any, error) {
			row := rows[index]
			generation := row.label.Generation
			return []any{
				generation.ID,
				generation.GraphNamespaceOID,
				int32(generation.LabelID),
				generation.RelationOID,
				int64(generation.MappingGeneration),
				string(row.namespace),
				string(row.externalID),
				int64(row.graphID),
			}, nil
		}),
	)
	if err != nil {
		return fmt.Errorf("COPY vertex identities: %w", err)
	}
	if copied != int64(len(rows)) {
		return fmt.Errorf("COPY staged %d vertex identities, expected %d", copied, len(rows))
	}
	tag, err := transaction.tx.Exec(
		ctx,
		`INSERT INTO agefreighter_meta.vertex_identity (
			graph_generation_id, label_generation_id, graph_namespace_oid,
			label_id, label_relation_oid, mapping_generation, label_kind,
			source_namespace, external_id, graph_id
		)
		SELECT $1, label_generation_id, graph_namespace_oid, label_id,
		       label_relation_oid, mapping_generation, 'v',
		       source_namespace, external_id, graph_id
		FROM pg_temp.agefreighter_vertex_identity_stage`,
		transaction.sink.options.Graph.ID,
	)
	if err != nil {
		return fmt.Errorf("insert vertex identities: %w", err)
	}
	_, err = requireAffectedRows("insert vertex identities", tag.RowsAffected(), len(rows))
	return err
}

func (transaction *loadTransaction) writeEdges(
	ctx context.Context,
	labelName model.Label,
	edges []model.Edge,
) error {
	binding, ok := transaction.sink.labels[labelName]
	if !ok {
		return fmt.Errorf("edge label %q is not registered", labelName)
	}
	if binding.Catalog.Kind != EdgeLabel {
		return fmt.Errorf("label %q is not an edge label", labelName)
	}
	staged := make([]stagedEdge, len(edges))
	seenIdentities := make(map[string]struct{}, len(edges))
	for index, edge := range edges {
		if edge.Namespace == "" ||
			edge.Start.Namespace == "" || edge.Start.Label == "" ||
			edge.Start.ExternalID == "" ||
			edge.End.Namespace == "" || edge.End.Label == "" ||
			edge.End.ExternalID == "" {
			return fmt.Errorf("edge %d identity or endpoint is empty", index)
		}
		if edge.Position.Token == "" {
			return fmt.Errorf("edge %d source position token is empty", index)
		}
		if edge.ExternalID != "" {
			key := string(edge.Namespace) + "\x00" + string(edge.ExternalID)
			if _, exists := seenIdentities[key]; exists {
				return fmt.Errorf("edge %d duplicates external identity in batch", index)
			}
			seenIdentities[key] = struct{}{}
		}
		properties, err := EncodeProperties(edge.Properties)
		if err != nil {
			return fmt.Errorf("encode edge %q properties: %w", edge.ExternalID, err)
		}
		staged[index] = stagedEdge{
			record: edge, label: binding, properties: properties, ordinal: index,
		}
	}
	resolved, missing, err := transaction.resolveEdges(ctx, staged)
	if err != nil {
		return err
	}
	if len(missing) > 0 {
		if transaction.sink.options.MissingEndpoint == config.MissingEndpointError {
			return fmt.Errorf(
				"%d edges have missing endpoints; first at %s:%d",
				len(missing),
				missing[0].Position.Resource,
				missing[0].Position.Line,
			)
		}
		if err := transaction.quarantineMissingEdges(ctx, missing); err != nil {
			return err
		}
		transaction.rejected += int64(len(missing))
	}
	if len(resolved) == 0 {
		return nil
	}
	block, err := (&Transaction{tx: transaction.tx}).ReserveIDs(
		ctx,
		binding.Catalog,
		uint64(len(resolved)),
	)
	if err != nil {
		return err
	}
	rows := make([]EdgeRow, len(resolved))
	for index := range resolved {
		id, err := block.GraphID(uint64(index))
		if err != nil {
			return err
		}
		rows[index] = EdgeRow{
			ID:         id,
			StartID:    resolved[index].startID,
			EndID:      resolved[index].endID,
			Properties: resolved[index].properties,
		}
	}
	if _, err := (&Transaction{tx: transaction.tx}).CopyEdges(
		ctx,
		binding.Catalog,
		rows,
		StagedBinaryCopy,
	); err != nil {
		return err
	}
	return transaction.insertEdgeIdentities(ctx, resolved, rows)
}

func (transaction *loadTransaction) resolveEdges(
	ctx context.Context,
	edges []stagedEdge,
) ([]resolvedEdge, []model.Edge, error) {
	const stage = "agefreighter_edge_reference_stage"
	if _, err := transaction.tx.Exec(
		ctx,
		`CREATE TEMP TABLE IF NOT EXISTS pg_temp.agefreighter_edge_reference_stage (
			ordinal integer PRIMARY KEY,
			start_namespace text NOT NULL,
			start_label_id integer NOT NULL,
			start_external_id text NOT NULL,
			end_namespace text NOT NULL,
			end_label_id integer NOT NULL,
			end_external_id text NOT NULL
		) ON COMMIT DROP`,
	); err != nil {
		return nil, nil, fmt.Errorf("prepare edge reference stage: %w", err)
	}
	if _, err := transaction.tx.Exec(
		ctx,
		`TRUNCATE pg_temp.agefreighter_edge_reference_stage`,
	); err != nil {
		return nil, nil, fmt.Errorf("truncate edge reference stage: %w", err)
	}
	copied, err := transaction.tx.CopyFrom(
		ctx,
		pgx.Identifier{"pg_temp", stage},
		[]string{
			"ordinal",
			"start_namespace",
			"start_label_id",
			"start_external_id",
			"end_namespace",
			"end_label_id",
			"end_external_id",
		},
		pgx.CopyFromSlice(len(edges), func(index int) ([]any, error) {
			edge := edges[index].record
			start, ok := transaction.sink.labels[edge.Start.Label]
			if !ok || start.Catalog.Kind != VertexLabel {
				return nil, fmt.Errorf(
					"edge %d start label %q is not a registered vertex label",
					index,
					edge.Start.Label,
				)
			}
			end, ok := transaction.sink.labels[edge.End.Label]
			if !ok || end.Catalog.Kind != VertexLabel {
				return nil, fmt.Errorf(
					"edge %d end label %q is not a registered vertex label",
					index,
					edge.End.Label,
				)
			}
			return []any{
				index,
				string(edge.Start.Namespace),
				int32(start.Catalog.LabelID),
				string(edge.Start.ExternalID),
				string(edge.End.Namespace),
				int32(end.Catalog.LabelID),
				string(edge.End.ExternalID),
			}, nil
		}),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("COPY edge endpoint references: %w", err)
	}
	if copied != int64(len(edges)) {
		return nil, nil, fmt.Errorf(
			"COPY staged %d edge references, expected %d",
			copied,
			len(edges),
		)
	}
	rows, err := transaction.tx.Query(
		ctx,
		`SELECT s.ordinal, start_identity.graph_id, end_identity.graph_id
		 FROM pg_temp.agefreighter_edge_reference_stage s
		 LEFT JOIN agefreighter_meta.vertex_identity start_identity
		   ON start_identity.graph_generation_id = $1
		  AND start_identity.source_namespace = s.start_namespace
		  AND start_identity.label_id = s.start_label_id
		  AND start_identity.external_id = s.start_external_id
		 LEFT JOIN agefreighter_meta.vertex_identity end_identity
		   ON end_identity.graph_generation_id = $1
		  AND end_identity.source_namespace = s.end_namespace
		  AND end_identity.label_id = s.end_label_id
		  AND end_identity.external_id = s.end_external_id
		 ORDER BY s.ordinal`,
		transaction.sink.options.Graph.ID,
	)
	if err != nil {
		return nil, nil, fmt.Errorf("resolve edge endpoints: %w", err)
	}
	defer rows.Close()
	resolved := make([]resolvedEdge, 0, len(edges))
	missing := make([]model.Edge, 0)
	for rows.Next() {
		var ordinal int
		var startID, endID *int64
		if err := rows.Scan(&ordinal, &startID, &endID); err != nil {
			return nil, nil, fmt.Errorf("scan resolved edge endpoint: %w", err)
		}
		if ordinal < 0 || ordinal >= len(edges) {
			return nil, nil, fmt.Errorf("resolved edge ordinal %d is out of range", ordinal)
		}
		if startID == nil || endID == nil {
			missing = append(missing, edges[ordinal].record)
			continue
		}
		start := GraphID(*startID)
		end := GraphID(*endID)
		if err := start.Validate(); err != nil {
			return nil, nil, fmt.Errorf("resolved start endpoint: %w", err)
		}
		if err := end.Validate(); err != nil {
			return nil, nil, fmt.Errorf("resolved end endpoint: %w", err)
		}
		resolved = append(resolved, resolvedEdge{
			stagedEdge: edges[ordinal],
			startID:    start,
			endID:      end,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("iterate resolved edge endpoints: %w", err)
	}
	if len(resolved)+len(missing) != len(edges) {
		return nil, nil, errors.New("edge endpoint resolution returned an incomplete result")
	}
	return resolved, missing, nil
}

func (transaction *loadTransaction) quarantineMissingEdges(
	ctx context.Context,
	edges []model.Edge,
) error {
	for _, edge := range edges {
		payload, err := json.Marshal(edge)
		if err != nil {
			return fmt.Errorf("encode missing endpoint quarantine record: %w", err)
		}
		_, err = transaction.sink.diagnostics.PutReject(
			ctx,
			meta.RejectRecord{
				JobID:   transaction.sink.options.JobID,
				BatchID: transaction.metadata.ID,
				Attempt: transaction.metadata.Attempt,
				Position: meta.Position{
					Resource:   edge.Position.Resource,
					Line:       edge.Position.Line,
					ByteOffset: edge.Position.Offset,
					Token:      edge.Position.Token,
				},
				ErrorClass:   "missing-endpoint",
				ErrorMessage: missingEndpointMessage(edge),
				Record:       payload,
			},
		)
		if err != nil {
			return fmt.Errorf("quarantine missing endpoint: %w", err)
		}
	}
	return nil
}

func missingEndpointMessage(edge model.Edge) string {
	return fmt.Sprintf(
		"edge %q endpoints %s/%s/%s -> %s/%s/%s could not both be resolved",
		edge.ExternalID,
		edge.Start.Namespace,
		edge.Start.Label,
		edge.Start.ExternalID,
		edge.End.Namespace,
		edge.End.Label,
		edge.End.ExternalID,
	)
}

func (transaction *loadTransaction) insertEdgeIdentities(
	ctx context.Context,
	edges []resolvedEdge,
	rows []EdgeRow,
) error {
	identityCount := 0
	for _, edge := range edges {
		if edge.record.ExternalID != "" {
			identityCount++
		}
	}
	if identityCount == 0 {
		return nil
	}
	const stage = "agefreighter_edge_identity_stage"
	if _, err := transaction.tx.Exec(
		ctx,
		`CREATE TEMP TABLE IF NOT EXISTS pg_temp.agefreighter_edge_identity_stage (
			source_namespace text NOT NULL,
			external_id text NOT NULL,
			graph_id bigint NOT NULL,
			start_graph_id bigint NOT NULL,
			end_graph_id bigint NOT NULL
		) ON COMMIT DROP`,
	); err != nil {
		return fmt.Errorf("prepare edge identity stage: %w", err)
	}
	if _, err := transaction.tx.Exec(
		ctx,
		`TRUNCATE pg_temp.agefreighter_edge_identity_stage`,
	); err != nil {
		return fmt.Errorf("truncate edge identity stage: %w", err)
	}
	source := make([]int, 0, identityCount)
	for index, edge := range edges {
		if edge.record.ExternalID != "" {
			source = append(source, index)
		}
	}
	copied, err := transaction.tx.CopyFrom(
		ctx,
		pgx.Identifier{"pg_temp", stage},
		[]string{
			"source_namespace",
			"external_id",
			"graph_id",
			"start_graph_id",
			"end_graph_id",
		},
		pgx.CopyFromSlice(len(source), func(index int) ([]any, error) {
			sourceIndex := source[index]
			edge := edges[sourceIndex].record
			row := rows[sourceIndex]
			return []any{
				string(edge.Namespace),
				string(edge.ExternalID),
				int64(row.ID),
				int64(row.StartID),
				int64(row.EndID),
			}, nil
		}),
	)
	if err != nil {
		return fmt.Errorf("COPY edge identities: %w", err)
	}
	if copied != int64(identityCount) {
		return fmt.Errorf("COPY staged %d edge identities, expected %d", copied, identityCount)
	}
	generation := edges[0].label.Generation
	tag, err := transaction.tx.Exec(
		ctx,
		`INSERT INTO agefreighter_meta.edge_identity (
			graph_generation_id, label_generation_id, graph_namespace_oid,
			label_id, label_relation_oid, mapping_generation, label_kind,
			source_namespace, external_id, graph_id, start_graph_id, end_graph_id
		)
		SELECT $1, $2, $3, $4, $5, $6, 'e',
		       source_namespace, external_id, graph_id, start_graph_id, end_graph_id
		FROM pg_temp.agefreighter_edge_identity_stage`,
		transaction.sink.options.Graph.ID,
		generation.ID,
		generation.GraphNamespaceOID,
		int32(generation.LabelID),
		generation.RelationOID,
		int64(generation.MappingGeneration),
	)
	if err != nil {
		return fmt.Errorf("insert edge identities: %w", err)
	}
	_, err = requireAffectedRows("insert edge identities", tag.RowsAffected(), identityCount)
	return err
}

func (transaction *loadTransaction) Commit(
	ctx context.Context,
	state checkpoint.State,
) error {
	if transaction.finalized {
		return errors.New("load transaction is finalized")
	}
	if !transaction.wrote {
		return transaction.abortKnown(ctx, errors.New("load transaction has not written records"))
	}
	if state.BatchID != transaction.metadata.ID ||
		state.Attempt != transaction.metadata.Attempt ||
		state.Phase != checkpoint.PhaseCommitted ||
		state.Position != transaction.metadata.LastPosition {
		return transaction.abortKnown(ctx, errors.New("checkpoint does not match load batch"))
	}
	store, err := meta.New(transaction.tx)
	if err != nil {
		return transaction.abortKnown(ctx, err)
	}
	if err := store.CommitBatch(
		ctx,
		transaction.sink.options.JobID,
		transaction.metadata.ID,
		transaction.metadata.Attempt,
		metaPosition(state.Position),
		transaction.rejected,
	); err != nil {
		return transaction.abortKnown(
			ctx,
			fmt.Errorf("commit AGE load checkpoint: %w", err),
		)
	}
	transaction.finalized = true
	if err := transaction.tx.Commit(ctx); err != nil {
		ownerErr := transaction.releaseOwner()
		transaction.sink.release()
		return errors.Join(
			fmt.Errorf("commit AGE load transaction: %w", err),
			ownerErr,
		)
	}
	ownerErr := transaction.releaseOwner()
	transaction.sink.release()
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
	attempt := meta.BatchAttempt{
		JobID:        transaction.sink.options.JobID,
		BatchID:      transaction.metadata.ID,
		Attempt:      transaction.metadata.Attempt,
		Rows:         int64(transaction.metadata.Rows),
		Bytes:        transaction.metadata.Bytes,
		RejectedRows: transaction.rejected,
		First:        metaPosition(transaction.metadata.FirstPosition),
	}
	diagnosticCtx, cancel := context.WithTimeout(
		context.WithoutCancel(ctx),
		transaction.sink.adapter.operationTimeout,
	)
	defer cancel()
	diagnosticErr := transaction.sink.diagnostics.RecordFailedBatch(
		diagnosticCtx,
		attempt,
		"AGE load transaction rolled back",
	)
	ownerErr := transaction.releaseOwner()
	transaction.sink.release()
	return errors.Join(rollbackErr, diagnosticErr, ownerErr)
}

func (transaction *loadTransaction) abortKnown(
	ctx context.Context,
	cause error,
) error {
	rollbackCtx, rollbackCancel := context.WithTimeout(
		context.WithoutCancel(ctx),
		transaction.sink.adapter.operationTimeout,
	)
	defer rollbackCancel()
	rollbackErr := transaction.Rollback(rollbackCtx)
	return errors.Join(cause, rollbackErr)
}

func (transaction *committedReplayTransaction) Write(
	_ context.Context,
	records []model.Record,
) error {
	if transaction.finalized {
		return errors.New("committed replay transaction is finalized")
	}
	if transaction.wrote {
		return errors.New("committed replay records were already accepted")
	}
	if len(records) != transaction.metadata.Rows {
		return fmt.Errorf(
			"committed replay contains %d records, expected %d",
			len(records),
			transaction.metadata.Rows,
		)
	}
	transaction.wrote = true
	return nil
}

func (transaction *committedReplayTransaction) Commit(
	ctx context.Context,
	state checkpoint.State,
) error {
	if transaction.finalized {
		return errors.New("committed replay transaction is finalized")
	}
	if !transaction.wrote {
		return transaction.fail(
			errors.New("committed replay has not accepted records"),
		)
	}
	if state.BatchID != transaction.metadata.ID ||
		state.Attempt != transaction.metadata.Attempt ||
		state.Phase != checkpoint.PhaseCommitted ||
		state.Position != transaction.metadata.LastPosition {
		return transaction.fail(
			errors.New("checkpoint does not match committed replay batch"),
		)
	}
	stored, err := transaction.sink.diagnostics.GetBatch(
		ctx,
		transaction.sink.options.JobID,
		transaction.metadata.ID,
		transaction.metadata.Attempt,
	)
	if err != nil {
		return transaction.fail(
			fmt.Errorf("read committed replay checkpoint: %w", err),
		)
	}
	if stored.Status != meta.BatchCommitted ||
		stored.Last != metaPosition(state.Position) {
		return transaction.fail(
			errors.New("stored checkpoint does not match committed replay"),
		)
	}
	transaction.finalized = true
	transaction.sink.release()
	return nil
}

func (transaction *committedReplayTransaction) fail(cause error) error {
	transaction.finalized = true
	transaction.sink.release()
	return cause
}

func (transaction *committedReplayTransaction) Rollback(context.Context) error {
	if transaction.finalized {
		return nil
	}
	transaction.finalized = true
	transaction.sink.release()
	return nil
}

func (target *LoadSink) release() {
	target.mu.Lock()
	target.active = false
	target.mu.Unlock()
}

func (transaction *loadTransaction) releaseOwner() error {
	return transaction.sink.releaseBatchOwner(
		transaction.owner,
		transaction.lockKey,
	)
}

func (target *LoadSink) releaseBatchOwner(
	owner *pgxpool.Conn,
	lockKey string,
) error {
	releaseCtx, cancel := context.WithTimeout(
		context.Background(),
		target.adapter.operationTimeout,
	)
	defer cancel()
	var unlocked bool
	err := owner.QueryRow(
		releaseCtx,
		`SELECT pg_catalog.pg_advisory_unlock(
			pg_catalog.hashtext($1), pg_catalog.hashtext($2)
		)`,
		target.options.JobID,
		lockKey,
	).Scan(&unlocked)
	if err != nil || !unlocked {
		_ = owner.Conn().Close(releaseCtx)
	}
	owner.Release()
	target.adapter.releaseLoadSlot()
	if err != nil {
		return fmt.Errorf("unlock AGE load batch: %w", err)
	}
	if !unlocked {
		return errors.New("AGE load batch ownership lock was not held")
	}
	return nil
}

func metaPosition(position model.SourcePosition) meta.Position {
	return meta.Position{
		Resource:   position.Resource,
		Line:       position.Line,
		ByteOffset: position.Offset,
		Token:      position.Token,
	}
}

var _ sinkcontract.Sink = (*LoadSink)(nil)
var _ sinkcontract.Transaction = (*loadTransaction)(nil)
var _ sinkcontract.Transaction = (*committedReplayTransaction)(nil)
