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
	"github.com/rioriost/agefreighter/internal/reject"
	sinkcontract "github.com/rioriost/agefreighter/internal/sink"
	"github.com/rioriost/agefreighter/pkg/model"
)

type LoadLabel struct {
	Catalog    LabelCatalog
	Generation meta.LabelGeneration
}

type LoadSinkOptions struct {
	JobID                   string
	Graph                   meta.GraphGeneration
	Labels                  []LoadLabel
	Mode                    config.LoadMode
	AppendDuplicate         config.AppendDuplicatePolicy
	PropertyMode            config.PropertyMode
	MissingEndpoint         config.MissingEndpointPolicy
	MaxDeferredEdges        int
	Quarantine              reject.Writer
	JobVerification         *meta.JobVerification
	CatalogAdmitted         bool
	DenseEndpointCacheBytes int64
}

type LoadSink struct {
	adapter       *Adapter
	diagnostics   *meta.Store
	options       LoadSinkOptions
	labels        map[model.Label]LoadLabel
	mu            sync.Mutex
	active        bool
	endpointCache *denseEndpointCache
}

type loadTransaction struct {
	sink            *LoadSink
	tx              pgx.Tx
	owner           *pgxpool.Conn
	lockKey         string
	metadata        sinkcontract.BatchMetadata
	rejected        int64
	labelCounters   map[int64]meta.BatchLabelCounter
	finalized       bool
	wrote           bool
	incrementalLock bool
	stageSequence   uint64
	denseIdentities []pendingDenseIdentity
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

type externalIdentity struct {
	namespace  model.Namespace
	externalID model.ExternalID
}

type stagedEdge struct {
	record       *model.Edge
	label        LoadLabel
	startLabelID uint16
	endLabelID   uint16
	properties   []byte
	ordinal      int
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
	if options.Mode == "" {
		options.Mode = config.LoadCreate
	}
	switch options.Mode {
	case config.LoadCreate, config.LoadReplace:
	case config.LoadAppend:
		switch options.AppendDuplicate {
		case config.AppendDuplicateError, config.AppendDuplicateIgnoreIdentical:
		default:
			return nil, fmt.Errorf(
				"unsupported append duplicate policy %q",
				options.AppendDuplicate,
			)
		}
	case config.LoadUpsert:
		switch options.PropertyMode {
		case config.PropertiesReplace,
			config.PropertiesMerge,
			config.PropertiesMergeDeleteNull:
		default:
			return nil, fmt.Errorf(
				"unsupported upsert property mode %q",
				options.PropertyMode,
			)
		}
	default:
		return nil, fmt.Errorf("unsupported load mode %q", options.Mode)
	}
	switch options.MissingEndpoint {
	case config.MissingEndpointError, config.MissingEndpointQuarantine:
	case config.MissingEndpointDefer:
		if !incrementalMode(options.Mode) || options.MaxDeferredEdges <= 0 {
			return nil, errors.New(
				"deferred endpoints require an incremental mode and positive capacity",
			)
		}
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
	if !options.CatalogAdmitted {
		storedGraph, err := diagnostics.AdmitGraphGeneration(
			ctx,
			options.JobID,
			options.Graph,
		)
		if err != nil {
			return nil, fmt.Errorf("admit load graph generation: %w", err)
		}
		options.Graph = storedGraph
	}
	labels := make(map[model.Label]LoadLabel, len(options.Labels))
	for _, binding := range options.Labels {
		if err := validateLoadLabel(options.Graph, binding); err != nil {
			return nil, err
		}
		key := model.Label(binding.Catalog.LabelName)
		if _, exists := labels[key]; exists {
			return nil, fmt.Errorf("duplicate load label %q", key)
		}
		if !options.CatalogAdmitted {
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
		}
		labels[key] = binding
	}
	if len(labels) == 0 {
		return nil, errors.New("load sink requires at least one label")
	}
	var endpointCache *denseEndpointCache
	if options.DenseEndpointCacheBytes > 0 {
		if incrementalMode(options.Mode) {
			return nil, errors.New(
				"dense endpoint cache is supported only for create and replace modes",
			)
		}
		endpointCache, err = newDenseEndpointCache(options.DenseEndpointCacheBytes)
		if err != nil {
			return nil, err
		}
		if err := endpointCache.load(ctx, adapter.pool, options.Graph.ID); err != nil {
			return nil, err
		}
	}
	return &LoadSink{
		adapter:       adapter,
		diagnostics:   diagnostics,
		options:       options,
		labels:        labels,
		endpointCache: endpointCache,
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
	incrementalLock := false
	if target.incremental() {
		var locked bool
		if err := owner.QueryRow(
			ctx,
			`SELECT pg_catalog.pg_try_advisory_lock(
				pg_catalog.hashtextextended(
					'agefreighter:graph-lifecycle:' || $1,
					$2
				)
			)`,
			target.options.Graph.GraphName,
			graphLifecycleLockSeed,
		).Scan(&locked); err != nil {
			ownerErr := target.releaseBatchOwner(owner, lockKey, false)
			release()
			return nil, errors.Join(
				fmt.Errorf("lock incremental AGE graph: %w", err),
				ownerErr,
			)
		}
		if !locked {
			ownerErr := target.releaseBatchOwner(owner, lockKey, false)
			release()
			return nil, errors.Join(meta.ErrIncrementalConflict, ownerErr)
		}
		incrementalLock = true
		if err := target.validateIncrementalGeneration(
			ctx,
			owner.Conn(),
		); err != nil {
			ownerErr := target.releaseBatchOwner(
				owner,
				lockKey,
				incrementalLock,
			)
			release()
			return nil, errors.Join(err, ownerErr)
		}
	}
	releaseOwner := func() error {
		return target.releaseBatchOwner(owner, lockKey, incrementalLock)
	}

	attempt := meta.BatchAttempt{
		JobID:   target.options.JobID,
		BatchID: batch.ID,
		Attempt: batch.Attempt,
		Rows:    int64(batch.Rows),
		Bytes:   batch.Bytes,
		First:   metaPosition(batch.FirstPosition),
	}
	ownerStore, err := meta.New(owner.Conn())
	if err != nil {
		ownerErr := releaseOwner()
		release()
		return nil, errors.Join(err, ownerErr)
	}
	stored, err := ownerStore.StartBatch(ctx, attempt)
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
	labelCounters := make(map[int64]meta.BatchLabelCounter, len(target.labels))
	for _, label := range target.labels {
		labelCounters[label.Generation.ID] = meta.BatchLabelCounter{
			LabelGenerationID: label.Generation.ID,
			Kind:              label.Generation.Kind,
		}
	}
	return &loadTransaction{
		sink:            target,
		tx:              tx,
		owner:           owner,
		lockKey:         lockKey,
		metadata:        batch,
		incrementalLock: incrementalLock,
		labelCounters:   labelCounters,
	}, nil
}

func (target *LoadSink) incremental() bool {
	return incrementalMode(target.options.Mode)
}

func (target *LoadSink) validateIncrementalGeneration(
	ctx context.Context,
	database *pgx.Conn,
) error {
	store, err := meta.New(database)
	if err != nil {
		return err
	}
	stored, err := store.GraphGenerationForJob(ctx, target.options.JobID)
	if err != nil {
		return err
	}
	live, err := lookupGraph(ctx, database, target.options.Graph.GraphName)
	if err != nil {
		return err
	}
	if stored.ID != target.options.Graph.ID ||
		stored.State != meta.GenerationActive ||
		stored.GraphName != live.Name ||
		stored.GraphOID != live.GraphOID ||
		stored.NamespaceOID != live.NamespaceOID {
		return fmt.Errorf(
			"%w: incremental graph generation is no longer active at %q",
			meta.ErrGenerationMismatch,
			target.options.Graph.GraphName,
		)
	}
	for _, binding := range target.labels {
		catalog, err := lookupLabel(
			ctx,
			database,
			live.Name,
			binding.Catalog.LabelName,
		)
		if err != nil {
			return err
		}
		current := binding
		current.Catalog = catalog
		if err := validateLoadLabel(stored, current); err != nil {
			return fmt.Errorf(
				"%w: %v",
				meta.ErrGenerationMismatch,
				err,
			)
		}
		generation := binding.Generation
		generation.GraphGenerationID = stored.ID
		if _, err := store.AdmitLabelGeneration(
			ctx,
			stored.ID,
			generation,
		); err != nil {
			return err
		}
	}
	return nil
}

func incrementalMode(mode config.LoadMode) bool {
	return mode == config.LoadAppend || mode == config.LoadUpsert
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

	vertexCounts := make(map[model.Label]int)
	edgeCounts := make(map[model.Label]int)
	seenEdge := false
	for index, record := range records {
		switch record.Kind() {
		case model.RecordVertex:
			if seenEdge {
				return fmt.Errorf("vertex record %d follows an edge record", index)
			}
			vertexCounts[record.Vertex.Label]++
		case model.RecordEdge:
			seenEdge = true
			edgeCounts[record.Edge.Label]++
		default:
			return fmt.Errorf("load record %d is invalid", index)
		}
	}
	for labelName, count := range vertexCounts {
		binding, ok := transaction.sink.labels[labelName]
		if !ok {
			return fmt.Errorf("vertex label %q is not registered", labelName)
		}
		transaction.addAccepted(binding, int64(count))
	}
	for labelName, count := range edgeCounts {
		binding, ok := transaction.sink.labels[labelName]
		if !ok {
			return fmt.Errorf("edge label %q is not registered", labelName)
		}
		transaction.addAccepted(binding, int64(count))
	}
	vertexGroups := make(map[model.Label][]*model.Vertex, len(vertexCounts))
	for label, count := range vertexCounts {
		vertexGroups[label] = make([]*model.Vertex, 0, count)
	}
	edgeGroups := make(map[model.Label][]*model.Edge, len(edgeCounts))
	for label, count := range edgeCounts {
		edgeGroups[label] = make([]*model.Edge, 0, count)
	}
	for _, record := range records {
		if record.Vertex != nil {
			vertexGroups[record.Vertex.Label] = append(
				vertexGroups[record.Vertex.Label],
				record.Vertex,
			)
		} else {
			edgeGroups[record.Edge.Label] = append(
				edgeGroups[record.Edge.Label],
				record.Edge,
			)
		}
	}
	for _, label := range sortedLabels(vertexGroups) {
		if err := transaction.writeVertices(ctx, label, vertexGroups[label]); err != nil {
			return err
		}
	}
	if transaction.sink.incremental() {
		if err := transaction.drainDeferredEdges(ctx); err != nil {
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

func (transaction *loadTransaction) nextStageSequence() uint64 {
	transaction.stageSequence++
	return transaction.stageSequence
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
	vertices []*model.Vertex,
) error {
	binding, ok := transaction.sink.labels[labelName]
	if !ok {
		return fmt.Errorf("vertex label %q is not registered", labelName)
	}
	if binding.Catalog.Kind != VertexLabel {
		return fmt.Errorf("label %q is not a vertex label", labelName)
	}
	propertiesByIndex := make([][]byte, len(vertices))
	seen := make(map[externalIdentity]struct{}, len(vertices))
	for index, vertex := range vertices {
		if vertex.ExternalID == "" || vertex.Namespace == "" {
			return fmt.Errorf("vertex %d identity is empty", index)
		}
		key := externalIdentity{
			namespace:  vertex.Namespace,
			externalID: vertex.ExternalID,
		}
		if _, exists := seen[key]; exists {
			return fmt.Errorf("vertex %d duplicates external identity in batch", index)
		}
		seen[key] = struct{}{}
		properties, err := loadProperties(vertex.Properties, vertex.EncodedProperties)
		if err != nil {
			return fmt.Errorf("encode vertex %q properties: %w", vertex.ExternalID, err)
		}
		propertiesByIndex[index] = properties
	}
	if transaction.sink.incremental() {
		return transaction.writeVerticesIncremental(
			ctx,
			binding,
			vertices,
			propertiesByIndex,
		)
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
	if transaction.sink.endpointCache != nil {
		pending, err := transaction.sink.endpointCache.prepare(identities)
		if err != nil {
			return err
		}
		transaction.denseIdentities = append(transaction.denseIdentities, pending...)
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
	if err := transaction.lockIdentityGeneration(ctx, rows[0].label); err != nil {
		return err
	}
	reader := &copyBinaryReader{
		rowCount: len(rows),
		rowAt: func(index int, output []byte) []byte {
			row := rows[index]
			generation := row.label.Generation
			output = appendBinaryInt16(output, 6)
			output = appendBinaryInt64Field(output, transaction.sink.options.Graph.ID)
			output = appendBinaryInt64Field(output, generation.ID)
			output = appendBinaryInt32Field(output, int32(generation.LabelID))
			output = appendBinaryTextField(output, row.namespace)
			output = appendBinaryTextField(output, row.externalID)
			return appendBinaryInt64Field(output, int64(row.graphID))
		},
	}
	copied, err := (&Transaction{tx: transaction.tx}).copyBinaryTable(
		ctx,
		pgx.Identifier{"agefreighter_meta", "vertex_identity"},
		[]string{
			"graph_generation_id",
			"label_generation_id",
			"label_id",
			"source_namespace",
			"external_id",
			"graph_id",
		},
		reader,
		len(rows),
	)
	if err != nil {
		return fmt.Errorf("COPY vertex identities into metadata: %w", err)
	}
	if copied != int64(len(rows)) {
		return fmt.Errorf("COPY wrote %d vertex identities, expected %d", copied, len(rows))
	}
	transaction.addCommitted(rows[0].label, int64(len(rows)))
	return nil
}

func (transaction *loadTransaction) lockIdentityGeneration(
	ctx context.Context,
	label LoadLabel,
) error {
	var admitted int64
	err := transaction.tx.QueryRow(
		ctx,
		`SELECT label_generation_id
		 FROM agefreighter_meta.label_generation
		 WHERE label_generation_id = $1
		   AND graph_generation_id = $2
		   AND label_id = $3
		   AND kind = $4
		 FOR KEY SHARE`,
		label.Generation.ID,
		transaction.sink.options.Graph.ID,
		int32(label.Generation.LabelID),
		string(label.Generation.Kind),
	).Scan(&admitted)
	if errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf(
			"label %q identity generation no longer matches the admitted catalog",
			label.Generation.LabelName,
		)
	}
	if err != nil {
		return fmt.Errorf(
			"lock label %q identity generation: %w",
			label.Generation.LabelName,
			err,
		)
	}
	return nil
}

func (transaction *loadTransaction) writeEdges(
	ctx context.Context,
	labelName model.Label,
	edges []*model.Edge,
) error {
	binding, ok := transaction.sink.labels[labelName]
	if !ok {
		return fmt.Errorf("edge label %q is not registered", labelName)
	}
	if binding.Catalog.Kind != EdgeLabel {
		return fmt.Errorf("label %q is not an edge label", labelName)
	}
	staged := make([]stagedEdge, len(edges))
	seenIdentities := make(map[externalIdentity]struct{}, len(edges))
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
			key := externalIdentity{
				namespace:  edge.Namespace,
				externalID: edge.ExternalID,
			}
			if _, exists := seenIdentities[key]; exists {
				return fmt.Errorf("edge %d duplicates external identity in batch", index)
			}
			seenIdentities[key] = struct{}{}
		}
		properties, err := loadProperties(edge.Properties, edge.EncodedProperties)
		if err != nil {
			return fmt.Errorf("encode edge %q properties: %w", edge.ExternalID, err)
		}
		staged[index] = stagedEdge{
			record: edge, label: binding,
			properties: properties,
			ordinal:    index,
		}
	}
	for index := range staged {
		edge := staged[index].record
		start, ok := transaction.sink.labels[edge.Start.Label]
		if !ok || start.Catalog.Kind != VertexLabel {
			return fmt.Errorf(
				"edge %d start label %q is not a registered vertex label",
				index,
				edge.Start.Label,
			)
		}
		end, ok := transaction.sink.labels[edge.End.Label]
		if !ok || end.Catalog.Kind != VertexLabel {
			return fmt.Errorf(
				"edge %d end label %q is not a registered vertex label",
				index,
				edge.End.Label,
			)
		}
		staged[index].startLabelID = start.Catalog.LabelID
		staged[index].endLabelID = end.Catalog.LabelID
	}
	resolved, missing, err := transaction.resolveEdges(ctx, staged)
	if err != nil {
		return err
	}
	if len(missing) > 0 {
		switch transaction.sink.options.MissingEndpoint {
		case config.MissingEndpointError:
			return fmt.Errorf(
				"%d edges have missing endpoints; first at %s:%d",
				len(missing),
				missing[0].record.Position.Resource,
				missing[0].record.Position.Line,
			)
		case config.MissingEndpointQuarantine:
			if err := transaction.quarantineMissingEdges(ctx, missing); err != nil {
				return err
			}
			transaction.rejected += int64(len(missing))
			transaction.addRejected(binding, int64(len(missing)))
		case config.MissingEndpointDefer:
			if err := transaction.deferMissingEdges(ctx, missing); err != nil {
				return err
			}
		default:
			return errors.New("unsupported missing endpoint policy")
		}
	}
	if len(resolved) == 0 {
		return nil
	}
	if transaction.sink.incremental() {
		return transaction.writeCurrentEdgesIncremental(ctx, binding, resolved)
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
) ([]resolvedEdge, []stagedEdge, error) {
	if transaction.sink.endpointCache != nil {
		resolved, complete := transaction.resolveEdgesDense(edges)
		if complete {
			return resolved, nil, nil
		}
	}
	stage := fmt.Sprintf(
		"agefreighter_edge_reference_stage_%d",
		edges[0].label.Generation.ID,
	)
	stageName := pgx.Identifier{"pg_temp", stage}.Sanitize()
	if _, err := transaction.tx.Exec(
		ctx,
		fmt.Sprintf(`CREATE TEMP TABLE IF NOT EXISTS %s (
			ordinal integer PRIMARY KEY,
			start_namespace text NOT NULL,
			start_label_id integer NOT NULL,
			start_external_id text NOT NULL,
			end_namespace text NOT NULL,
			end_label_id integer NOT NULL,
			end_external_id text NOT NULL
		) ON COMMIT DROP`, stageName),
	); err != nil {
		return nil, nil, fmt.Errorf("prepare edge reference stage: %w", err)
	}
	reader := &copyBinaryReader{
		rowCount: len(edges),
		rowAt: func(index int, output []byte) []byte {
			staged := edges[index]
			edge := staged.record
			output = appendBinaryInt16(output, 7)
			output = appendBinaryInt32Field(output, int32(index))
			output = appendBinaryTextField(output, edge.Start.Namespace)
			output = appendBinaryInt32Field(output, int32(staged.startLabelID))
			output = appendBinaryTextField(output, edge.Start.ExternalID)
			output = appendBinaryTextField(output, edge.End.Namespace)
			output = appendBinaryInt32Field(output, int32(staged.endLabelID))
			return appendBinaryTextField(output, edge.End.ExternalID)
		},
	}
	copied, err := (&Transaction{tx: transaction.tx}).copyBinaryTable(
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
		reader,
		len(edges),
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
		 FROM `+stageName+` s
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
	missing := make([]stagedEdge, 0)
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
			missing = append(missing, edges[ordinal])
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
	edges []stagedEdge,
) error {
	for _, staged := range edges {
		edge := *staged.record
		payload, err := json.Marshal(edge)
		if err != nil {
			return fmt.Errorf("encode missing endpoint quarantine record: %w", err)
		}
		message := missingEndpointMessage(edge)
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
				ErrorMessage: message,
				Record:       payload,
			},
		)
		if err != nil {
			return fmt.Errorf("quarantine missing endpoint: %w", err)
		}
		if transaction.sink.options.Quarantine != nil {
			record := model.EdgeRecord(edge)
			if err := transaction.sink.options.Quarantine.Write(
				ctx,
				reject.Rejection{
					Record:   &record,
					Position: edge.Position,
					Code:     "missing-endpoint",
					Message:  message,
				},
			); err != nil {
				return fmt.Errorf(
					"write missing endpoint quarantine record: %w",
					err,
				)
			}
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
		transaction.addCommitted(edges[0].label, int64(len(edges)))
		return nil
	}
	if err := transaction.lockIdentityGeneration(ctx, edges[0].label); err != nil {
		return err
	}
	source := make([]int, 0, identityCount)
	for index, edge := range edges {
		if edge.record.ExternalID != "" {
			source = append(source, index)
		}
	}
	reader := &copyBinaryReader{
		rowCount: len(source),
		rowAt: func(index int, output []byte) []byte {
			sourceIndex := source[index]
			edge := edges[sourceIndex].record
			row := rows[sourceIndex]
			generation := edges[sourceIndex].label.Generation
			output = appendBinaryInt16(output, 8)
			output = appendBinaryInt64Field(output, transaction.sink.options.Graph.ID)
			output = appendBinaryInt64Field(output, generation.ID)
			output = appendBinaryInt32Field(output, int32(generation.LabelID))
			output = appendBinaryTextField(output, edge.Namespace)
			output = appendBinaryTextField(output, edge.ExternalID)
			output = appendBinaryInt64Field(output, int64(row.ID))
			output = appendBinaryInt64Field(output, int64(row.StartID))
			return appendBinaryInt64Field(output, int64(row.EndID))
		},
	}
	copied, err := (&Transaction{tx: transaction.tx}).copyBinaryTable(
		ctx,
		pgx.Identifier{"agefreighter_meta", "edge_identity"},
		[]string{
			"graph_generation_id",
			"label_generation_id",
			"label_id",
			"source_namespace",
			"external_id",
			"graph_id",
			"start_graph_id",
			"end_graph_id",
		},
		reader,
		len(source),
	)
	if err != nil {
		return fmt.Errorf("COPY edge identities into metadata: %w", err)
	}
	if copied != int64(identityCount) {
		return fmt.Errorf("COPY wrote %d edge identities, expected %d", copied, identityCount)
	}
	transaction.addCommitted(edges[0].label, int64(len(edges)))
	return nil
}

func (transaction *loadTransaction) labelCounter(
	label LoadLabel,
) meta.BatchLabelCounter {
	counter := transaction.labelCounters[label.Generation.ID]
	counter.LabelGenerationID = label.Generation.ID
	counter.Kind = label.Generation.Kind
	return counter
}

func (transaction *loadTransaction) addAccepted(label LoadLabel, rows int64) {
	if transaction.labelCounters == nil {
		transaction.labelCounters = make(map[int64]meta.BatchLabelCounter)
	}
	counter := transaction.labelCounter(label)
	counter.AcceptedRows += rows
	transaction.labelCounters[label.Generation.ID] = counter
}

func (transaction *loadTransaction) addCommitted(label LoadLabel, rows int64) {
	if transaction.labelCounters == nil {
		transaction.labelCounters = make(map[int64]meta.BatchLabelCounter)
	}
	counter := transaction.labelCounter(label)
	counter.CommittedRows += rows
	transaction.labelCounters[label.Generation.ID] = counter
}

func (transaction *loadTransaction) addRejected(label LoadLabel, rows int64) {
	if transaction.labelCounters == nil {
		transaction.labelCounters = make(map[int64]meta.BatchLabelCounter)
	}
	counter := transaction.labelCounter(label)
	counter.RejectedRows += rows
	transaction.labelCounters[label.Generation.ID] = counter
}

func (transaction *loadTransaction) counters() []meta.BatchLabelCounter {
	ids := make([]int64, 0, len(transaction.labelCounters))
	for id := range transaction.labelCounters {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	values := make([]meta.BatchLabelCounter, 0, len(ids))
	for _, id := range ids {
		values = append(values, transaction.labelCounters[id])
	}
	return values
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
	if err := store.CommitBatchWithLabelCountersAndVerification(
		ctx,
		transaction.sink.options.JobID,
		transaction.metadata.ID,
		transaction.metadata.Attempt,
		metaPosition(state.Position),
		transaction.rejected,
		transaction.counters(),
		transaction.sink.options.JobVerification,
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
	if transaction.sink.endpointCache != nil {
		transaction.sink.endpointCache.apply(transaction.denseIdentities)
	}
	transaction.sink.options.JobVerification = nil
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
		transaction.incrementalLock,
	)
}

func (target *LoadSink) releaseBatchOwner(
	owner *pgxpool.Conn,
	lockKey string,
	incrementalLock bool,
) error {
	releaseCtx, cancel := context.WithTimeout(
		context.Background(),
		target.adapter.operationTimeout,
	)
	defer cancel()
	var incrementalErr error
	if incrementalLock {
		var unlocked bool
		err := owner.QueryRow(
			releaseCtx,
			`SELECT pg_catalog.pg_advisory_unlock(
				pg_catalog.hashtextextended(
					'agefreighter:graph-lifecycle:' || $1,
					$2
				)
			)`,
			target.options.Graph.GraphName,
			graphLifecycleLockSeed,
		).Scan(&unlocked)
		if err != nil {
			incrementalErr = fmt.Errorf("unlock incremental AGE graph: %w", err)
		} else if !unlocked {
			incrementalErr = errors.New("incremental AGE graph lock was not held")
		}
	}
	var unlocked bool
	err := owner.QueryRow(
		releaseCtx,
		`SELECT pg_catalog.pg_advisory_unlock(
			pg_catalog.hashtext($1), pg_catalog.hashtext($2)
		)`,
		target.options.JobID,
		lockKey,
	).Scan(&unlocked)
	if incrementalErr != nil || err != nil || !unlocked {
		_ = owner.Conn().Close(releaseCtx)
	}
	owner.Release()
	target.adapter.releaseLoadSlot()
	if err != nil {
		return errors.Join(
			incrementalErr,
			fmt.Errorf("unlock AGE load batch: %w", err),
		)
	}
	if !unlocked {
		return errors.Join(
			incrementalErr,
			errors.New("AGE load batch ownership lock was not held"),
		)
	}
	return incrementalErr
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
