package age

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/meta"
)

var ErrCatalogEntryNotFound = errors.New("Apache AGE catalog entry not found")

type GraphCatalog struct {
	Name         string
	GraphOID     uint32
	NamespaceOID uint32
}

type LabelKind byte

const (
	VertexLabel LabelKind = 'v'
	EdgeLabel   LabelKind = 'e'
)

func (kind LabelKind) String() string {
	switch kind {
	case VertexLabel:
		return "vertex"
	case EdgeLabel:
		return "edge"
	default:
		return "unknown"
	}
}

type LabelCatalog struct {
	GraphName    string
	LabelName    string
	GraphOID     uint32
	NamespaceOID uint32
	LabelID      uint16
	Kind         LabelKind
	RelationOID  uint32
	SequenceOID  uint32
	SequenceName string
}

func lookupGraph(
	ctx context.Context,
	database databaseExecutor,
	name string,
) (GraphCatalog, error) {
	if err := ValidateGraphName(name); err != nil {
		return GraphCatalog{}, err
	}
	var graph GraphCatalog
	err := database.QueryRow(
		ctx,
		`SELECT name::text, graphid::oid, namespace::oid
		 FROM ag_catalog.ag_graph
		 WHERE name = $1::name`,
		name,
	).Scan(&graph.Name, &graph.GraphOID, &graph.NamespaceOID)
	if errors.Is(err, pgx.ErrNoRows) {
		return GraphCatalog{}, fmt.Errorf("%w: graph %q", ErrCatalogEntryNotFound, name)
	}
	if err != nil {
		return GraphCatalog{}, fmt.Errorf("lookup graph %q: %w", name, err)
	}
	if graph.GraphOID != graph.NamespaceOID {
		return GraphCatalog{}, fmt.Errorf(
			"graph %q catalog mismatch: graph OID %d, namespace OID %d",
			name,
			graph.GraphOID,
			graph.NamespaceOID,
		)
	}
	return graph, nil
}

func lookupLabel(
	ctx context.Context,
	database databaseExecutor,
	graphName string,
	labelName string,
) (LabelCatalog, error) {
	if err := ValidateGraphName(graphName); err != nil {
		return LabelCatalog{}, err
	}
	if err := ValidateLabelName(labelName); err != nil {
		return LabelCatalog{}, err
	}

	var (
		label      LabelCatalog
		labelID    int32
		kind       string
		relationNS uint32
	)
	err := database.QueryRow(
		ctx,
		`SELECT
			g.name::text,
			l.name::text,
			g.graphid::oid,
			g.namespace::oid,
			l.id::integer,
			l.kind::text,
			l.relation::oid,
			r.relnamespace::oid,
			s.oid,
			l.seq_name::text
		FROM ag_catalog.ag_graph g
		JOIN ag_catalog.ag_label l ON l.graph = g.graphid
		JOIN pg_class r ON r.oid = l.relation
		JOIN pg_class s
		  ON s.relnamespace = g.namespace
		 AND s.relname = l.seq_name
		 AND s.relkind = 'S'
		WHERE g.name = $1::name
		  AND l.name = $2::name`,
		graphName,
		labelName,
	).Scan(
		&label.GraphName,
		&label.LabelName,
		&label.GraphOID,
		&label.NamespaceOID,
		&labelID,
		&kind,
		&label.RelationOID,
		&relationNS,
		&label.SequenceOID,
		&label.SequenceName,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return LabelCatalog{}, fmt.Errorf(
			"%w: label %q in graph %q",
			ErrCatalogEntryNotFound,
			labelName,
			graphName,
		)
	}
	if err != nil {
		return LabelCatalog{}, fmt.Errorf(
			"lookup label %q in graph %q: %w",
			labelName,
			graphName,
			err,
		)
	}
	if labelID <= 0 || labelID > int32(MaxLabelID) {
		return LabelCatalog{}, fmt.Errorf("label %q has invalid ID %d", labelName, labelID)
	}
	label.LabelID = uint16(labelID)
	if len(kind) != 1 ||
		(LabelKind(kind[0]) != VertexLabel && LabelKind(kind[0]) != EdgeLabel) {
		return LabelCatalog{}, fmt.Errorf("label %q has invalid kind %q", labelName, kind)
	}
	label.Kind = LabelKind(kind[0])
	if label.GraphOID != label.NamespaceOID || relationNS != label.NamespaceOID {
		return LabelCatalog{}, fmt.Errorf(
			"label %q catalog mismatch: graph=%d namespace=%d relation_namespace=%d",
			labelName,
			label.GraphOID,
			label.NamespaceOID,
			relationNS,
		)
	}
	return label, nil
}

func (transaction *Transaction) RegisterCreatedLabel(
	ctx context.Context,
	graphGenerationID int64,
	graphName string,
	labelName string,
	expectedKind LabelKind,
) (LoadLabel, error) {
	if graphGenerationID <= 0 {
		return LoadLabel{}, errors.New("graph generation ID must be positive")
	}
	if err := ValidateGraphName(graphName); err != nil {
		return LoadLabel{}, err
	}
	if err := ValidateLabelName(labelName); err != nil {
		return LoadLabel{}, err
	}
	if expectedKind != VertexLabel && expectedKind != EdgeLabel {
		return LoadLabel{}, fmt.Errorf("invalid label kind %d", expectedKind)
	}
	var (
		value      LoadLabel
		labelID    int32
		kind       string
		relationNS uint32
	)
	err := transaction.tx.QueryRow(
		ctx,
		`WITH catalog AS MATERIALIZED (
			SELECT
				graph.name::text AS graph_name,
				label.name::text AS label_name,
				graph.graphid::oid AS graph_oid,
				graph.namespace::oid AS namespace_oid,
				label.id::integer AS label_id,
				label.kind::text AS kind,
				label.relation::oid AS relation_oid,
				relation.relnamespace::oid AS relation_namespace_oid,
				sequence.oid AS sequence_oid,
				label.seq_name::text AS sequence_name
			FROM ag_catalog.ag_graph graph
			JOIN ag_catalog.ag_label label ON label.graph = graph.graphid
			JOIN pg_class relation ON relation.oid = label.relation
			JOIN pg_class sequence
			  ON sequence.relnamespace = graph.namespace
			 AND sequence.relname = label.seq_name
			 AND sequence.relkind = 'S'
			WHERE graph.name = $1::name
			  AND label.name = $2::name
		),
		inserted AS (
			INSERT INTO agefreighter_meta.label_generation (
				graph_generation_id, label_name, kind, graph_namespace_oid,
				label_id, relation_oid, sequence_oid, mapping_generation
			)
			SELECT $3, catalog.label_name, catalog.kind, catalog.namespace_oid,
			       catalog.label_id, catalog.relation_oid, catalog.sequence_oid, 1
			FROM catalog
			JOIN agefreighter_meta.graph_generation generation
			  ON generation.graph_generation_id = $3
			 AND generation.namespace_oid = catalog.namespace_oid
			WHERE catalog.kind = $4
			  AND catalog.graph_oid = catalog.namespace_oid
			  AND catalog.relation_namespace_oid = catalog.namespace_oid
			RETURNING label_generation_id, created_at, updated_at
		)
		SELECT
			catalog.graph_name, catalog.label_name, catalog.graph_oid,
			catalog.namespace_oid, catalog.label_id, catalog.kind,
			catalog.relation_oid, catalog.relation_namespace_oid,
			catalog.sequence_oid, catalog.sequence_name,
			inserted.label_generation_id, inserted.created_at, inserted.updated_at
		FROM catalog
		CROSS JOIN inserted`,
		pgx.QueryExecModeExec,
		graphName,
		labelName,
		graphGenerationID,
		string(byte(expectedKind)),
	).Scan(
		&value.Catalog.GraphName,
		&value.Catalog.LabelName,
		&value.Catalog.GraphOID,
		&value.Catalog.NamespaceOID,
		&labelID,
		&kind,
		&value.Catalog.RelationOID,
		&relationNS,
		&value.Catalog.SequenceOID,
		&value.Catalog.SequenceName,
		&value.Generation.ID,
		&value.Generation.CreatedAt,
		&value.Generation.UpdatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return LoadLabel{}, fmt.Errorf(
			"%w: created label %q in graph %q",
			meta.ErrGenerationMismatch,
			labelName,
			graphName,
		)
	}
	if err != nil {
		return LoadLabel{}, fmt.Errorf(
			"register created label %q in graph %q: %w",
			labelName,
			graphName,
			err,
		)
	}
	if labelID <= 0 || labelID > int32(MaxLabelID) ||
		len(kind) != 1 || LabelKind(kind[0]) != expectedKind ||
		value.Catalog.GraphOID != value.Catalog.NamespaceOID ||
		relationNS != value.Catalog.NamespaceOID {
		return LoadLabel{}, fmt.Errorf(
			"%w: invalid created label %q catalog",
			meta.ErrGenerationMismatch,
			labelName,
		)
	}
	value.Catalog.LabelID = uint16(labelID)
	value.Catalog.Kind = expectedKind
	value.Generation.GraphGenerationID = graphGenerationID
	value.Generation.LabelName = labelName
	value.Generation.Kind = meta.LabelKind(expectedKind)
	value.Generation.GraphNamespaceOID = value.Catalog.NamespaceOID
	value.Generation.LabelID = value.Catalog.LabelID
	value.Generation.RelationOID = value.Catalog.RelationOID
	value.Generation.SequenceOID = value.Catalog.SequenceOID
	value.Generation.MappingGeneration = 1
	return value, nil
}

func (transaction *Transaction) RegisterCreatedGraph(
	ctx context.Context,
	jobID string,
	graphName string,
	replacesGraphOID uint32,
	generation uint64,
) (meta.GraphGeneration, error) {
	if err := meta.ValidateJobID(jobID); err != nil {
		return meta.GraphGeneration{}, err
	}
	if err := ValidateGraphName(graphName); err != nil {
		return meta.GraphGeneration{}, err
	}
	if generation == 0 {
		return meta.GraphGeneration{}, errors.New("graph generation must be positive")
	}
	var value meta.GraphGeneration
	err := transaction.tx.QueryRow(ctx, `
		WITH catalog AS MATERIALIZED (
			SELECT name::text AS graph_name, graphid::oid AS graph_oid,
			       namespace::oid AS namespace_oid
			FROM ag_catalog.ag_graph
			WHERE name = $2::name
			  AND graphid = namespace
		),
		inserted AS (
			INSERT INTO agefreighter_meta.graph_generation (
				job_id, graph_name, graph_oid, namespace_oid,
				replaces_graph_oid, generation, state
			)
			SELECT $1::uuid, catalog.graph_name, catalog.graph_oid,
			       catalog.namespace_oid, NULLIF($3, 0)::oid, $4, 'loading'
			FROM catalog
			RETURNING graph_generation_id, job_id::text, graph_name,
			          graph_oid, namespace_oid, COALESCE(replaces_graph_oid, 0),
			          generation, state, created_at, updated_at
		),
		bound AS (
			UPDATE agefreighter_meta.load_job job
			SET graph_generation_id = inserted.graph_generation_id,
			    updated_at = clock_timestamp()
			FROM inserted
			WHERE job.job_id = $1::uuid
			  AND job.graph_generation_id IS NULL
			RETURNING inserted.*
		)
		SELECT *
		FROM bound`,
		pgx.QueryExecModeExec,
		jobID,
		graphName,
		replacesGraphOID,
		generation,
	).Scan(
		&value.ID,
		&value.JobID,
		&value.GraphName,
		&value.GraphOID,
		&value.NamespaceOID,
		&value.ReplacesGraphOID,
		&value.Generation,
		&value.State,
		&value.CreatedAt,
		&value.UpdatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return meta.GraphGeneration{}, fmt.Errorf(
			"%w: created graph %q",
			meta.ErrGenerationMismatch,
			graphName,
		)
	}
	if err != nil {
		return meta.GraphGeneration{}, fmt.Errorf(
			"register created graph %q: %w",
			graphName,
			err,
		)
	}
	return value, nil
}

func (adapter *Adapter) LookupGraph(
	ctx context.Context,
	name string,
) (GraphCatalog, error) {
	return lookupGraph(ctx, adapter.pool, name)
}

func (adapter *Adapter) LookupLabel(
	ctx context.Context,
	graphName string,
	labelName string,
) (LabelCatalog, error) {
	return lookupLabel(ctx, adapter.pool, graphName, labelName)
}

func (transaction *Transaction) LookupGraph(
	ctx context.Context,
	name string,
) (GraphCatalog, error) {
	return lookupGraph(ctx, transaction.tx, name)
}

func (transaction *Transaction) LookupLabel(
	ctx context.Context,
	graphName string,
	labelName string,
) (LabelCatalog, error) {
	return lookupLabel(ctx, transaction.tx, graphName, labelName)
}
