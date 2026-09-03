package pggraph

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/meta"
)

type LabelInspection struct {
	Name          string
	Kind          meta.LabelKind
	Rows          int64
	PrimaryKeys   int64
	UniqueKeys    int64
	ForeignKeys   int64
	MissingStarts int64
	MissingEnds   int64
}

type Inspection struct {
	Mapping         meta.PropertyGraphGeneration
	Labels          []LabelInspection
	Rows            int64
	RejectedRows    int64
	DirectedMatches int64
	UndirectedMatch int64
	Digests         DigestSet
}

func (inspection Inspection) Validate() error {
	if inspection.Mapping.State != meta.PropertyGraphActive {
		return fmt.Errorf("%w: property graph mapping is not active", ErrIntegrity)
	}
	for _, label := range inspection.Labels {
		if label.PrimaryKeys != 1 || label.UniqueKeys < 1 {
			return fmt.Errorf("%w: property graph label %q is missing identity constraints",
				ErrIntegrity, label.Name)
		}
		if label.Kind == meta.EdgeLabel {
			if label.ForeignKeys != 2 {
				return fmt.Errorf("%w: property graph edge label %q has %d endpoint foreign keys",
					ErrIntegrity, label.Name, label.ForeignKeys)
			}
			if label.MissingStarts != 0 || label.MissingEnds != 0 {
				return fmt.Errorf("%w: property graph edge label %q has missing endpoints",
					ErrIntegrity, label.Name)
			}
		}
	}
	if inspection.DirectedMatches > inspection.UndirectedMatch {
		return fmt.Errorf("%w: undirected SQL/PGQ match count is below directed count", ErrIntegrity)
	}
	return nil
}

func (adapter *Adapter) Inspect(
	ctx context.Context,
	jobID string,
	definition Definition,
) (Inspection, error) {
	if err := meta.ValidateJobID(jobID); err != nil {
		return Inspection{}, err
	}
	fingerprint, err := definition.Fingerprint()
	if err != nil {
		return Inspection{}, err
	}
	mapping, err := adapter.store.GetPropertyGraph(ctx, jobID)
	if err != nil {
		return Inspection{}, err
	}
	if mapping.Schema != definition.Schema || mapping.Graph != definition.Graph ||
		mapping.DefinitionFingerprint != fingerprint {
		return Inspection{}, fmt.Errorf("%w: stored property graph mapping changed",
			meta.ErrGenerationMismatch)
	}
	if err := adapter.admitObjects(ctx, definition); err != nil {
		return Inspection{}, err
	}
	result := Inspection{Mapping: mapping}
	definition = definition.normalized()
	for _, vertex := range definition.Vertices {
		label, err := adapter.inspectTable(ctx, definition.Schema, vertex.Table,
			vertex.Label, meta.VertexLabel, "", "")
		if err != nil {
			return Inspection{}, err
		}
		result.Labels = append(result.Labels, label)
		result.Rows += label.Rows
	}
	for _, edge := range definition.Edges {
		label, err := adapter.inspectTable(ctx, definition.Schema, edge.Table,
			edge.Label, meta.EdgeLabel, edge.SourceTable, edge.DestinationTable)
		if err != nil {
			return Inspection{}, err
		}
		result.Labels = append(result.Labels, label)
		result.Rows += label.Rows
	}
	job, err := adapter.store.GetJob(ctx, jobID)
	if err != nil {
		return Inspection{}, err
	}
	result.RejectedRows = job.RejectedRows + job.SourceRejectedRows
	if len(definition.Edges) > 0 {
		result.DirectedMatches, result.UndirectedMatch, err = adapter.inspectGraphQueries(ctx, definition)
		if err != nil {
			return Inspection{}, err
		}
	}
	result.Digests, err = adapter.ComputeDigests(ctx, jobID, definition)
	if err != nil {
		return Inspection{}, err
	}
	return result, nil
}

func (adapter *Adapter) inspectTable(
	ctx context.Context,
	schema string,
	table string,
	label string,
	kind meta.LabelKind,
	startTable string,
	endTable string,
) (LabelInspection, error) {
	result := LabelInspection{Name: label, Kind: kind}
	qualified := qualifiedName(schema, table)
	if err := adapter.pool.QueryRow(ctx,
		"SELECT count(*) FROM "+qualified).Scan(&result.Rows); err != nil {
		return LabelInspection{}, fmt.Errorf("count property graph label %q: %w", label, err)
	}
	if err := adapter.pool.QueryRow(ctx, `
		SELECT count(*) FILTER (WHERE contype = 'p'),
		       count(*) FILTER (WHERE contype = 'u'),
		       count(*) FILTER (WHERE contype = 'f')
		FROM pg_catalog.pg_constraint
		WHERE conrelid = pg_catalog.to_regclass($1)`, qualified,
	).Scan(&result.PrimaryKeys, &result.UniqueKeys, &result.ForeignKeys); err != nil {
		return LabelInspection{}, fmt.Errorf("inspect constraints for %q: %w", label, err)
	}
	if kind == meta.EdgeLabel {
		query := fmt.Sprintf(`SELECT
			count(*) FILTER (WHERE source.id IS NULL),
			count(*) FILTER (WHERE destination.id IS NULL)
		FROM %s edge
		LEFT JOIN %s source ON source.id = edge.start_id
		LEFT JOIN %s destination ON destination.id = edge.end_id`,
			qualified, qualifiedName(schema, startTable), qualifiedName(schema, endTable))
		if err := adapter.pool.QueryRow(ctx, query).Scan(
			&result.MissingStarts, &result.MissingEnds,
		); err != nil {
			return LabelInspection{}, fmt.Errorf("inspect endpoints for %q: %w", label, err)
		}
	}
	return result, nil
}

func (adapter *Adapter) inspectGraphQueries(
	ctx context.Context,
	definition Definition,
) (int64, int64, error) {
	edge := definition.Edges[0]
	labels := make(map[string]string, len(definition.Vertices))
	for _, vertex := range definition.Vertices {
		labels[vertex.Table] = vertex.Label
	}
	tx, err := adapter.pool.Begin(ctx)
	if err != nil {
		return 0, 0, fmt.Errorf("begin SQL/PGQ inspection: %w", err)
	}
	defer tx.Rollback(context.WithoutCancel(ctx))
	if _, err := tx.Exec(ctx,
		"SET LOCAL search_path TO "+QuoteIdentifier(definition.Schema)+", pg_catalog",
	); err != nil {
		return 0, 0, fmt.Errorf("set SQL/PGQ inspection search path: %w", err)
	}
	query := func(pattern string) (int64, error) {
		statement := fmt.Sprintf(`SELECT count(*) FROM GRAPH_TABLE (
			%s MATCH %s COLUMNS (edge.external_id AS external_id)
		)`, QuoteIdentifier(definition.Graph), pattern)
		var count int64
		if err := tx.QueryRow(ctx, statement).Scan(&count); err != nil {
			return 0, err
		}
		return count, nil
	}
	directedPattern := fmt.Sprintf("(source IS %s)-[edge IS %s]->(destination IS %s)",
		QuoteIdentifier(labels[edge.SourceTable]), QuoteIdentifier(edge.Label),
		QuoteIdentifier(labels[edge.DestinationTable]))
	undirectedPattern := fmt.Sprintf("(source IS %s)-[edge IS %s]-(destination IS %s)",
		QuoteIdentifier(labels[edge.SourceTable]), QuoteIdentifier(edge.Label),
		QuoteIdentifier(labels[edge.DestinationTable]))
	directed, err := query(directedPattern)
	if err != nil {
		return 0, 0, fmt.Errorf("execute directed SQL/PGQ check: %w", err)
	}
	undirected, err := query(undirectedPattern)
	if err != nil {
		return 0, 0, fmt.Errorf("execute undirected SQL/PGQ check: %w", err)
	}
	if err := tx.Commit(ctx); err != nil && !errors.Is(err, pgx.ErrTxClosed) {
		return 0, 0, fmt.Errorf("finish SQL/PGQ inspection: %w", err)
	}
	return directed, undirected, nil
}
