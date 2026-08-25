package age

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
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
