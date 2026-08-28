package age

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

type incrementalVertexDecision struct {
	ordinal    int
	namespace  model.Namespace
	externalID model.ExternalID
	graphID    GraphID
	properties []byte
	isNew      bool
}

type incrementalEdgeDecision struct {
	ordinal    int
	namespace  model.Namespace
	externalID *model.ExternalID
	graphID    GraphID
	startID    GraphID
	endID      GraphID
	properties []byte
	isNew      bool
}

type deferredEdgeGroup struct {
	binding         LoadLabel
	mode            config.LoadMode
	appendDuplicate config.AppendDuplicatePolicy
	propertyMode    config.PropertyMode
	edges           []resolvedEdge
	ids             []int64
}

func (transaction *loadTransaction) deferMissingEdges(
	ctx context.Context,
	edges []stagedEdge,
) error {
	edges, err := transaction.filterDeferredDuplicates(ctx, edges)
	if err != nil {
		return err
	}
	if len(edges) == 0 {
		return nil
	}
	rows := make([][]any, len(edges))
	appendDuplicate := deferredAppendDuplicate(
		transaction.sink.options.Mode,
		transaction.sink.options.AppendDuplicate,
	)
	for index, staged := range edges {
		edge := staged.record
		var externalID any
		if edge.ExternalID != "" {
			externalID = string(edge.ExternalID)
		}
		rows[index] = []any{
			transaction.sink.options.Graph.ID,
			staged.label.Generation.ID,
			int32(staged.label.Generation.LabelID),
			transaction.sink.options.JobID,
			string(edge.Namespace),
			externalID,
			string(edge.Start.Namespace),
			int32(staged.startLabelID),
			string(edge.Start.ExternalID),
			string(edge.End.Namespace),
			int32(staged.endLabelID),
			string(edge.End.ExternalID),
			string(staged.properties),
			string(transaction.sink.options.Mode),
			string(appendDuplicate),
			string(transaction.sink.options.PropertyMode),
			edge.Position.Resource,
			edge.Position.Line,
			edge.Position.Offset,
			edge.Position.Token,
		}
	}
	if _, err := transaction.tx.CopyFrom(
		ctx,
		pgx.Identifier{"agefreighter_meta", "deferred_edge"},
		[]string{
			"graph_generation_id", "label_generation_id", "label_id", "job_id",
			"source_namespace", "external_id",
			"start_namespace", "start_label_id", "start_external_id",
			"end_namespace", "end_label_id", "end_external_id",
			"properties", "load_mode", "append_duplicate", "property_mode",
			"resource", "line", "byte_offset", "resume_token",
		},
		pgx.CopyFromRows(rows),
	); err != nil {
		return fmt.Errorf("persist deferred edges: %w", err)
	}
	var count int64
	if err := transaction.tx.QueryRow(
		ctx,
		`SELECT COUNT(*)
		 FROM agefreighter_meta.deferred_edge
		 WHERE graph_generation_id = $1`,
		transaction.sink.options.Graph.ID,
	).Scan(&count); err != nil {
		return fmt.Errorf("count deferred edges: %w", err)
	}
	if count > int64(transaction.sink.options.MaxDeferredEdges) {
		return fmt.Errorf(
			"deferred edge capacity exceeded: %d rows exceeds limit %d",
			count,
			transaction.sink.options.MaxDeferredEdges,
		)
	}
	return nil
}

func deferredAppendDuplicate(
	mode config.LoadMode,
	policy config.AppendDuplicatePolicy,
) config.AppendDuplicatePolicy {
	if mode == config.LoadUpsert && policy == "" {
		return config.AppendDuplicateError
	}
	return policy
}

func (transaction *loadTransaction) filterDeferredDuplicates(
	ctx context.Context,
	edges []stagedEdge,
) ([]stagedEdge, error) {
	if transaction.sink.options.Mode != config.LoadAppend {
		return edges, nil
	}
	filtered := make([]stagedEdge, 0, len(edges))
	for _, staged := range edges {
		edge := staged.record
		if edge.ExternalID == "" {
			filtered = append(filtered, staged)
			continue
		}
		var existingGraphID int64
		err := transaction.tx.QueryRow(
			ctx,
			`SELECT graph_id
			 FROM agefreighter_meta.edge_identity
			 WHERE graph_generation_id = $1
			   AND label_generation_id = $2
			   AND source_namespace = $3
			   AND external_id = $4`,
			transaction.sink.options.Graph.ID,
			staged.label.Generation.ID,
			edge.Namespace,
			edge.ExternalID,
		).Scan(&existingGraphID)
		if err == nil {
			if transaction.sink.options.AppendDuplicate ==
				config.AppendDuplicateError {
				return nil, fmt.Errorf(
					"append duplicate edge identity %q",
					edge.ExternalID,
				)
			}
			return nil, fmt.Errorf(
				"append edge %q with missing endpoints conflicts with an existing edge",
				edge.ExternalID,
			)
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return nil, fmt.Errorf("match existing edge identity: %w", err)
		}
		rows, err := transaction.tx.Query(
			ctx,
			`SELECT
				start_namespace, start_label_id, start_external_id,
				end_namespace, end_label_id, end_external_id,
				properties
			 FROM agefreighter_meta.deferred_edge
			 WHERE graph_generation_id = $1
			   AND label_generation_id = $2
			   AND source_namespace = $3
			   AND external_id = $4
			 ORDER BY deferred_edge_id
			 FOR UPDATE`,
			transaction.sink.options.Graph.ID,
			staged.label.Generation.ID,
			edge.Namespace,
			edge.ExternalID,
		)
		if err != nil {
			return nil, fmt.Errorf("match deferred edge identity: %w", err)
		}
		matched := false
		for rows.Next() {
			matched = true
			var (
				startNamespace  model.Namespace
				startLabelID    int32
				startExternalID model.ExternalID
				endNamespace    model.Namespace
				endLabelID      int32
				endExternalID   model.ExternalID
				properties      string
			)
			if err := rows.Scan(
				&startNamespace,
				&startLabelID,
				&startExternalID,
				&endNamespace,
				&endLabelID,
				&endExternalID,
				&properties,
			); err != nil {
				rows.Close()
				return nil, fmt.Errorf("scan deferred edge identity: %w", err)
			}
			if transaction.sink.options.AppendDuplicate ==
				config.AppendDuplicateError {
				rows.Close()
				return nil, fmt.Errorf(
					"append duplicate deferred edge identity %q",
					edge.ExternalID,
				)
			}
			sameProperties, err := equalJSON(
				[]byte(properties),
				staged.properties,
			)
			if err != nil {
				rows.Close()
				return nil, fmt.Errorf(
					"compare deferred edge %q properties: %w",
					edge.ExternalID,
					err,
				)
			}
			if startNamespace != edge.Start.Namespace ||
				startLabelID != int32(staged.startLabelID) ||
				startExternalID != edge.Start.ExternalID ||
				endNamespace != edge.End.Namespace ||
				endLabelID != int32(staged.endLabelID) ||
				endExternalID != edge.End.ExternalID ||
				!sameProperties {
				rows.Close()
				return nil, fmt.Errorf(
					"append deferred edge %q conflicts with an existing deferred edge",
					edge.ExternalID,
				)
			}
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, fmt.Errorf("iterate deferred edge identities: %w", err)
		}
		rows.Close()
		if !matched {
			filtered = append(filtered, staged)
		}
	}
	return filtered, nil
}

func (transaction *loadTransaction) drainDeferredEdges(ctx context.Context) error {
	labelByGeneration := make(map[int64]LoadLabel)
	labelGenerationIDs := make([]int64, 0)
	for _, binding := range transaction.sink.labels {
		if binding.Catalog.Kind != EdgeLabel {
			continue
		}
		labelByGeneration[binding.Generation.ID] = binding
		labelGenerationIDs = append(labelGenerationIDs, binding.Generation.ID)
	}
	if len(labelGenerationIDs) == 0 {
		return nil
	}
	slices.Sort(labelGenerationIDs)

	for {
		rows, err := transaction.tx.Query(
			ctx,
			`SELECT
				d.deferred_edge_id, d.label_generation_id,
				d.source_namespace, d.external_id, d.properties,
				d.load_mode, d.append_duplicate, d.property_mode,
				d.resource, d.line, d.byte_offset, d.resume_token,
				start_identity.graph_id, end_identity.graph_id
			 FROM agefreighter_meta.deferred_edge d
			 JOIN agefreighter_meta.vertex_identity start_identity
			   ON start_identity.graph_generation_id = d.graph_generation_id
			  AND start_identity.source_namespace = d.start_namespace
			  AND start_identity.label_id = d.start_label_id
			  AND start_identity.external_id = d.start_external_id
			 JOIN agefreighter_meta.vertex_identity end_identity
			   ON end_identity.graph_generation_id = d.graph_generation_id
			  AND end_identity.source_namespace = d.end_namespace
			  AND end_identity.label_id = d.end_label_id
			  AND end_identity.external_id = d.end_external_id
			 WHERE d.graph_generation_id = $1
			   AND d.label_generation_id = ANY($2::bigint[])
			   AND (
			     d.external_id IS NULL OR NOT EXISTS (
			       SELECT 1
			       FROM agefreighter_meta.deferred_edge earlier
			       WHERE earlier.graph_generation_id = d.graph_generation_id
			         AND earlier.label_generation_id = d.label_generation_id
			         AND earlier.source_namespace = d.source_namespace
			         AND earlier.external_id = d.external_id
			         AND earlier.deferred_edge_id < d.deferred_edge_id
			     )
			   )
			 ORDER BY d.deferred_edge_id
			 LIMIT 1000
			 FOR UPDATE OF d SKIP LOCKED`,
			transaction.sink.options.Graph.ID,
			labelGenerationIDs,
		)
		if err != nil {
			return fmt.Errorf("select resolvable deferred edges: %w", err)
		}
		groups := make([]deferredEdgeGroup, 0)
		for rows.Next() {
			var (
				deferredID        int64
				labelGenerationID int64
				namespace         model.Namespace
				externalID        *string
				properties        string
				mode              config.LoadMode
				appendDuplicate   config.AppendDuplicatePolicy
				propertyMode      config.PropertyMode
				position          model.SourcePosition
				startID           int64
				endID             int64
			)
			if err := rows.Scan(
				&deferredID,
				&labelGenerationID,
				&namespace,
				&externalID,
				&properties,
				&mode,
				&appendDuplicate,
				&propertyMode,
				&position.Resource,
				&position.Line,
				&position.Offset,
				&position.Token,
				&startID,
				&endID,
			); err != nil {
				rows.Close()
				return fmt.Errorf("scan resolvable deferred edge: %w", err)
			}
			binding, ok := labelByGeneration[labelGenerationID]
			if !ok {
				rows.Close()
				return fmt.Errorf(
					"deferred edge label generation %d is not admitted",
					labelGenerationID,
				)
			}
			groups = append(groups, deferredEdgeGroup{
				binding:         binding,
				mode:            mode,
				appendDuplicate: appendDuplicate,
				propertyMode:    propertyMode,
			})
			group := &groups[len(groups)-1]
			edge := model.Edge{
				Label:     model.Label(binding.Catalog.LabelName),
				Namespace: namespace,
				Position:  position,
			}
			if externalID != nil {
				edge.ExternalID = model.ExternalID(*externalID)
			}
			group.edges = append(group.edges, resolvedEdge{
				stagedEdge: stagedEdge{
					record:     &edge,
					label:      binding,
					properties: []byte(properties),
					ordinal:    len(group.edges),
				},
				startID: GraphID(startID),
				endID:   GraphID(endID),
			})
			group.ids = append(group.ids, deferredID)
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return fmt.Errorf("iterate resolvable deferred edges: %w", err)
		}
		rows.Close()
		if len(groups) == 0 {
			return nil
		}
		for _, group := range groups {
			originalMode := transaction.sink.options.Mode
			originalDuplicate := transaction.sink.options.AppendDuplicate
			originalPropertyMode := transaction.sink.options.PropertyMode
			transaction.sink.options.Mode = group.mode
			transaction.sink.options.AppendDuplicate = group.appendDuplicate
			transaction.sink.options.PropertyMode = group.propertyMode
			err := transaction.writeEdgesIncremental(
				ctx,
				group.binding,
				group.edges,
			)
			transaction.sink.options.Mode = originalMode
			transaction.sink.options.AppendDuplicate = originalDuplicate
			transaction.sink.options.PropertyMode = originalPropertyMode
			if err != nil {
				return fmt.Errorf("drain deferred edges: %w", err)
			}
			if _, err := transaction.tx.Exec(
				ctx,
				`DELETE FROM agefreighter_meta.deferred_edge
				 WHERE deferred_edge_id = ANY($1::bigint[])`,
				group.ids,
			); err != nil {
				return fmt.Errorf("delete drained deferred edges: %w", err)
			}
		}
	}
}

func (transaction *loadTransaction) writeVerticesIncremental(
	ctx context.Context,
	binding LoadLabel,
	vertices []*model.Vertex,
	properties [][]byte,
) error {
	inputName := fmt.Sprintf(
		"agefreighter_vertex_incremental_input_%d_%d",
		binding.Generation.ID,
		transaction.nextStageSequence(),
	)
	inputTable := pgx.Identifier{"pg_temp", inputName}.Sanitize()
	if _, err := transaction.tx.Exec(
		ctx,
		fmt.Sprintf(`CREATE TEMP TABLE %s (
			ordinal integer PRIMARY KEY,
			source_namespace text NOT NULL,
			external_id text NOT NULL,
			properties text NOT NULL
		) ON COMMIT DROP`, inputTable),
	); err != nil {
		return fmt.Errorf("prepare incremental vertex input: %w", err)
	}
	inputRows := make([][]any, len(vertices))
	for index, vertex := range vertices {
		inputRows[index] = []any{
			index,
			string(vertex.Namespace),
			string(vertex.ExternalID),
			string(properties[index]),
		}
	}
	if _, err := transaction.tx.CopyFrom(
		ctx,
		pgx.Identifier{"pg_temp", inputName},
		[]string{"ordinal", "source_namespace", "external_id", "properties"},
		pgx.CopyFromRows(inputRows),
	); err != nil {
		return fmt.Errorf("stage incremental vertices: %w", err)
	}

	labelTable := pgx.Identifier{
		binding.Catalog.GraphName,
		binding.Catalog.LabelName,
	}.Sanitize()
	rows, err := transaction.tx.Query(
		ctx,
		fmt.Sprintf(
			`SELECT
				s.ordinal, s.source_namespace, s.external_id, s.properties,
				i.graph_id, target.properties::text
			 FROM %s s
			 LEFT JOIN agefreighter_meta.vertex_identity i
			   ON i.graph_generation_id = $1
			  AND i.label_generation_id = $2
			  AND i.source_namespace = s.source_namespace
			  AND i.external_id = s.external_id
			 LEFT JOIN %s target
			   ON target.id = i.graph_id::text::graphid
			 ORDER BY s.ordinal`,
			inputTable,
			labelTable,
		),
		transaction.sink.options.Graph.ID,
		binding.Generation.ID,
	)
	if err != nil {
		return fmt.Errorf("match incremental vertices: %w", err)
	}
	defer rows.Close()

	decisions := make([]incrementalVertexDecision, 0, len(vertices))
	for rows.Next() {
		var (
			decision           incrementalVertexDecision
			incoming           string
			existingGraphID    *int64
			existingProperties *string
		)
		if err := rows.Scan(
			&decision.ordinal,
			&decision.namespace,
			&decision.externalID,
			&incoming,
			&existingGraphID,
			&existingProperties,
		); err != nil {
			return fmt.Errorf("scan incremental vertex match: %w", err)
		}
		decision.properties = []byte(incoming)
		if existingGraphID == nil {
			decision.isNew = true
		} else {
			if existingProperties == nil {
				return fmt.Errorf(
					"vertex identity %q points to a missing AGE row",
					decision.externalID,
				)
			}
			decision.graphID = GraphID(*existingGraphID)
			if err := decision.graphID.Validate(); err != nil {
				return fmt.Errorf("stored vertex identity: %w", err)
			}
			effective, err := transaction.resolveExistingProperties(
				[]byte(*existingProperties),
				decision.properties,
			)
			if err != nil {
				return fmt.Errorf(
					"resolve duplicate vertex %q: %w",
					decision.externalID,
					err,
				)
			}
			decision.properties = effective
		}
		decisions = append(decisions, decision)
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate incremental vertex matches: %w", err)
	}
	if len(decisions) != len(vertices) {
		return errors.New("incremental vertex match returned an incomplete result")
	}
	if err := assignVertexDecisionIDs(ctx, transaction.tx, binding, decisions); err != nil {
		return err
	}
	return transaction.applyIncrementalVertices(ctx, binding, decisions)
}

func assignVertexDecisionIDs(
	ctx context.Context,
	tx pgx.Tx,
	binding LoadLabel,
	decisions []incrementalVertexDecision,
) error {
	var unseen uint64
	for _, decision := range decisions {
		if decision.isNew {
			unseen++
		}
	}
	if unseen == 0 {
		return nil
	}
	block, err := (&Transaction{tx: tx}).ReserveIDs(ctx, binding.Catalog, unseen)
	if err != nil {
		return err
	}
	var offset uint64
	for index := range decisions {
		if !decisions[index].isNew {
			continue
		}
		id, err := block.GraphID(offset)
		if err != nil {
			return err
		}
		decisions[index].graphID = id
		offset++
	}
	return nil
}

func (transaction *loadTransaction) applyIncrementalVertices(
	ctx context.Context,
	binding LoadLabel,
	decisions []incrementalVertexDecision,
) error {
	if err := transaction.lockIdentityGeneration(ctx, binding); err != nil {
		return err
	}
	stageName := fmt.Sprintf(
		"agefreighter_vertex_incremental_decision_%d_%d",
		binding.Generation.ID,
		transaction.nextStageSequence(),
	)
	stageTable := pgx.Identifier{"pg_temp", stageName}.Sanitize()
	if _, err := transaction.tx.Exec(
		ctx,
		fmt.Sprintf(`CREATE TEMP TABLE %s (
			ordinal integer PRIMARY KEY,
			source_namespace text NOT NULL,
			external_id text NOT NULL,
			graph_id bigint NOT NULL,
			properties text NOT NULL,
			is_new boolean NOT NULL
		) ON COMMIT DROP`, stageTable),
	); err != nil {
		return fmt.Errorf("prepare incremental vertex decisions: %w", err)
	}
	rows := make([][]any, len(decisions))
	for index, decision := range decisions {
		rows[index] = []any{
			decision.ordinal,
			string(decision.namespace),
			string(decision.externalID),
			int64(decision.graphID),
			string(decision.properties),
			decision.isNew,
		}
	}
	if _, err := transaction.tx.CopyFrom(
		ctx,
		pgx.Identifier{"pg_temp", stageName},
		[]string{
			"ordinal", "source_namespace", "external_id",
			"graph_id", "properties", "is_new",
		},
		pgx.CopyFromRows(rows),
	); err != nil {
		return fmt.Errorf("stage incremental vertex decisions: %w", err)
	}
	labelTable := pgx.Identifier{
		binding.Catalog.GraphName,
		binding.Catalog.LabelName,
	}.Sanitize()
	inserted, err := transaction.tx.Exec(
		ctx,
		fmt.Sprintf(
			`INSERT INTO %s (id, properties)
			 SELECT graph_id::text::graphid, properties::agtype
			 FROM %s
			 WHERE is_new
			 ORDER BY ordinal`,
			labelTable,
			stageTable,
		),
	)
	if err != nil {
		return fmt.Errorf("insert incremental vertices: %w", err)
	}
	expectedNew := countNewVertices(decisions)
	if inserted.RowsAffected() != expectedNew {
		return fmt.Errorf(
			"inserted %d incremental vertices, expected %d",
			inserted.RowsAffected(),
			expectedNew,
		)
	}
	if transaction.sink.options.Mode == config.LoadUpsert {
		if _, err := transaction.tx.Exec(
			ctx,
			fmt.Sprintf(
				`UPDATE %s target
				 SET properties = decision.properties::agtype
				 FROM %s decision
				 WHERE NOT decision.is_new
				   AND target.id = decision.graph_id::text::graphid`,
				labelTable,
				stageTable,
			),
		); err != nil {
			return fmt.Errorf("update incremental vertices: %w", err)
		}
	}
	if expectedNew == 0 {
		return nil
	}
	if _, err := transaction.tx.Exec(
		ctx,
		fmt.Sprintf(
			`INSERT INTO agefreighter_meta.vertex_identity (
				graph_generation_id, label_generation_id, label_id,
				source_namespace, external_id, graph_id
			 )
			 SELECT $1, $2, $3,
			        source_namespace, external_id, graph_id
			 FROM %s
			 WHERE is_new`,
			stageTable,
		),
		transaction.sink.options.Graph.ID,
		binding.Generation.ID,
		int32(binding.Generation.LabelID),
	); err != nil {
		return fmt.Errorf("insert incremental vertex identities: %w", err)
	}
	transaction.addCommitted(binding, expectedNew)
	return nil
}

func countNewVertices(decisions []incrementalVertexDecision) int64 {
	var count int64
	for _, decision := range decisions {
		if decision.isNew {
			count++
		}
	}
	return count
}

func (transaction *loadTransaction) writeEdgesIncremental(
	ctx context.Context,
	binding LoadLabel,
	edges []resolvedEdge,
) error {
	inputName := fmt.Sprintf(
		"agefreighter_edge_incremental_input_%d_%d",
		binding.Generation.ID,
		transaction.nextStageSequence(),
	)
	inputTable := pgx.Identifier{"pg_temp", inputName}.Sanitize()
	if _, err := transaction.tx.Exec(
		ctx,
		fmt.Sprintf(`CREATE TEMP TABLE %s (
			ordinal integer PRIMARY KEY,
			source_namespace text NOT NULL,
			external_id text,
			start_graph_id bigint NOT NULL,
			end_graph_id bigint NOT NULL,
			properties text NOT NULL
		) ON COMMIT DROP`, inputTable),
	); err != nil {
		return fmt.Errorf("prepare incremental edge input: %w", err)
	}
	inputRows := make([][]any, len(edges))
	for index, edge := range edges {
		var externalID any
		if edge.record.ExternalID != "" {
			externalID = string(edge.record.ExternalID)
		}
		inputRows[index] = []any{
			index,
			string(edge.record.Namespace),
			externalID,
			int64(edge.startID),
			int64(edge.endID),
			string(edge.properties),
		}
	}
	if _, err := transaction.tx.CopyFrom(
		ctx,
		pgx.Identifier{"pg_temp", inputName},
		[]string{
			"ordinal", "source_namespace", "external_id",
			"start_graph_id", "end_graph_id", "properties",
		},
		pgx.CopyFromRows(inputRows),
	); err != nil {
		return fmt.Errorf("stage incremental edges: %w", err)
	}
	labelTable := pgx.Identifier{
		binding.Catalog.GraphName,
		binding.Catalog.LabelName,
	}.Sanitize()
	rows, err := transaction.tx.Query(
		ctx,
		fmt.Sprintf(
			`SELECT
				s.ordinal, s.source_namespace, s.external_id,
				s.start_graph_id, s.end_graph_id, s.properties,
				i.graph_id, i.start_graph_id, i.end_graph_id,
				target.properties::text
			 FROM %s s
			 LEFT JOIN agefreighter_meta.edge_identity i
			   ON s.external_id IS NOT NULL
			  AND i.graph_generation_id = $1
			  AND i.label_generation_id = $2
			  AND i.source_namespace = s.source_namespace
			  AND i.external_id = s.external_id
			 LEFT JOIN %s target
			   ON target.id = i.graph_id::text::graphid
			 ORDER BY s.ordinal`,
			inputTable,
			labelTable,
		),
		transaction.sink.options.Graph.ID,
		binding.Generation.ID,
	)
	if err != nil {
		return fmt.Errorf("match incremental edges: %w", err)
	}
	defer rows.Close()

	decisions := make([]incrementalEdgeDecision, 0, len(edges))
	for rows.Next() {
		var (
			decision           incrementalEdgeDecision
			incoming           string
			graphID            *int64
			existingStartID    *int64
			existingEndID      *int64
			existingProperties *string
		)
		if err := rows.Scan(
			&decision.ordinal,
			&decision.namespace,
			&decision.externalID,
			&decision.startID,
			&decision.endID,
			&incoming,
			&graphID,
			&existingStartID,
			&existingEndID,
			&existingProperties,
		); err != nil {
			return fmt.Errorf("scan incremental edge match: %w", err)
		}
		decision.properties = []byte(incoming)
		if transaction.sink.options.Mode == config.LoadUpsert &&
			decision.externalID == nil {
			return errors.New("upsert edge external identity is required")
		}
		if graphID == nil {
			decision.isNew = true
		} else {
			if existingStartID == nil || existingEndID == nil ||
				existingProperties == nil {
				return errors.New("edge identity points to a missing AGE row")
			}
			decision.graphID = GraphID(*graphID)
			if err := decision.graphID.Validate(); err != nil {
				return fmt.Errorf("stored edge identity: %w", err)
			}
			if transaction.sink.options.Mode == config.LoadAppend &&
				(GraphID(*existingStartID) != decision.startID ||
					GraphID(*existingEndID) != decision.endID) {
				return fmt.Errorf(
					"append edge %q conflicts with existing endpoints",
					*decision.externalID,
				)
			}
			effective, err := transaction.resolveExistingProperties(
				[]byte(*existingProperties),
				decision.properties,
			)
			if err != nil {
				return fmt.Errorf("resolve duplicate edge properties: %w", err)
			}
			decision.properties = effective
		}
		decisions = append(decisions, decision)
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate incremental edge matches: %w", err)
	}
	if len(decisions) != len(edges) {
		return errors.New("incremental edge match returned an incomplete result")
	}
	if err := assignEdgeDecisionIDs(ctx, transaction.tx, binding, decisions); err != nil {
		return err
	}
	return transaction.applyIncrementalEdges(ctx, binding, decisions)
}

func (transaction *loadTransaction) writeCurrentEdgesIncremental(
	ctx context.Context,
	binding LoadLabel,
	edges []resolvedEdge,
) error {
	ready := make([]resolvedEdge, 0, len(edges))
	deferred := make([]stagedEdge, 0)
	for _, edge := range edges {
		if edge.record.ExternalID == "" {
			ready = append(ready, edge)
			continue
		}
		var deferredID int64
		err := transaction.tx.QueryRow(
			ctx,
			`SELECT deferred_edge_id
			 FROM agefreighter_meta.deferred_edge
			 WHERE graph_generation_id = $1
			   AND label_generation_id = $2
			   AND source_namespace = $3
			   AND external_id = $4
			 ORDER BY deferred_edge_id
			 LIMIT 1
			 FOR UPDATE`,
			transaction.sink.options.Graph.ID,
			binding.Generation.ID,
			edge.record.Namespace,
			edge.record.ExternalID,
		).Scan(&deferredID)
		if errors.Is(err, pgx.ErrNoRows) {
			ready = append(ready, edge)
			continue
		}
		if err != nil {
			return fmt.Errorf("match pending edge identity: %w", err)
		}
		if transaction.sink.options.Mode == config.LoadAppend {
			if transaction.sink.options.AppendDuplicate ==
				config.AppendDuplicateError {
				return fmt.Errorf(
					"append duplicate pending edge identity %q",
					edge.record.ExternalID,
				)
			}
			return fmt.Errorf(
				"append edge %q conflicts with an older pending edge",
				edge.record.ExternalID,
			)
		}
		deferred = append(deferred, edge.stagedEdge)
	}
	if len(deferred) > 0 {
		if err := transaction.deferMissingEdges(ctx, deferred); err != nil {
			return fmt.Errorf("queue upsert behind pending edge: %w", err)
		}
	}
	if len(ready) == 0 {
		return nil
	}
	return transaction.writeEdgesIncremental(ctx, binding, ready)
}

func assignEdgeDecisionIDs(
	ctx context.Context,
	tx pgx.Tx,
	binding LoadLabel,
	decisions []incrementalEdgeDecision,
) error {
	var unseen uint64
	for _, decision := range decisions {
		if decision.isNew {
			unseen++
		}
	}
	if unseen == 0 {
		return nil
	}
	block, err := (&Transaction{tx: tx}).ReserveIDs(ctx, binding.Catalog, unseen)
	if err != nil {
		return err
	}
	var offset uint64
	for index := range decisions {
		if !decisions[index].isNew {
			continue
		}
		id, err := block.GraphID(offset)
		if err != nil {
			return err
		}
		decisions[index].graphID = id
		offset++
	}
	return nil
}

func (transaction *loadTransaction) applyIncrementalEdges(
	ctx context.Context,
	binding LoadLabel,
	decisions []incrementalEdgeDecision,
) error {
	if err := transaction.lockIdentityGeneration(ctx, binding); err != nil {
		return err
	}
	stageName := fmt.Sprintf(
		"agefreighter_edge_incremental_decision_%d_%d",
		binding.Generation.ID,
		transaction.nextStageSequence(),
	)
	stageTable := pgx.Identifier{"pg_temp", stageName}.Sanitize()
	if _, err := transaction.tx.Exec(
		ctx,
		fmt.Sprintf(`CREATE TEMP TABLE %s (
			ordinal integer PRIMARY KEY,
			source_namespace text NOT NULL,
			external_id text,
			graph_id bigint NOT NULL,
			start_graph_id bigint NOT NULL,
			end_graph_id bigint NOT NULL,
			properties text NOT NULL,
			is_new boolean NOT NULL
		) ON COMMIT DROP`, stageTable),
	); err != nil {
		return fmt.Errorf("prepare incremental edge decisions: %w", err)
	}
	rows := make([][]any, len(decisions))
	for index, decision := range decisions {
		var externalID any
		if decision.externalID != nil {
			externalID = string(*decision.externalID)
		}
		rows[index] = []any{
			decision.ordinal,
			string(decision.namespace),
			externalID,
			int64(decision.graphID),
			int64(decision.startID),
			int64(decision.endID),
			string(decision.properties),
			decision.isNew,
		}
	}
	if _, err := transaction.tx.CopyFrom(
		ctx,
		pgx.Identifier{"pg_temp", stageName},
		[]string{
			"ordinal", "source_namespace", "external_id", "graph_id",
			"start_graph_id", "end_graph_id", "properties", "is_new",
		},
		pgx.CopyFromRows(rows),
	); err != nil {
		return fmt.Errorf("stage incremental edge decisions: %w", err)
	}
	labelTable := pgx.Identifier{
		binding.Catalog.GraphName,
		binding.Catalog.LabelName,
	}.Sanitize()
	inserted, err := transaction.tx.Exec(
		ctx,
		fmt.Sprintf(
			`INSERT INTO %s (id, start_id, end_id, properties)
			 SELECT
			    graph_id::text::graphid,
			    start_graph_id::text::graphid,
			    end_graph_id::text::graphid,
			    properties::agtype
			 FROM %s
			 WHERE is_new
			 ORDER BY ordinal`,
			labelTable,
			stageTable,
		),
	)
	if err != nil {
		return fmt.Errorf("insert incremental edges: %w", err)
	}
	expectedNew := countNewEdges(decisions)
	if inserted.RowsAffected() != expectedNew {
		return fmt.Errorf(
			"inserted %d incremental edges, expected %d",
			inserted.RowsAffected(),
			expectedNew,
		)
	}
	if transaction.sink.options.Mode == config.LoadUpsert {
		if _, err := transaction.tx.Exec(
			ctx,
			fmt.Sprintf(
				`UPDATE %s target
				 SET start_id = decision.start_graph_id::text::graphid,
				     end_id = decision.end_graph_id::text::graphid,
				     properties = decision.properties::agtype
				 FROM %s decision
				 WHERE NOT decision.is_new
				   AND target.id = decision.graph_id::text::graphid`,
				labelTable,
				stageTable,
			),
		); err != nil {
			return fmt.Errorf("update incremental edges: %w", err)
		}
	}
	if expectedNew > 0 {
		if _, err := transaction.tx.Exec(
			ctx,
			fmt.Sprintf(
				`INSERT INTO agefreighter_meta.edge_identity (
					graph_generation_id, label_generation_id, label_id,
					source_namespace, external_id, graph_id,
					start_graph_id, end_graph_id
				 )
				 SELECT $1, $2, $3,
				        source_namespace, external_id, graph_id,
				        start_graph_id, end_graph_id
				 FROM %s
				 WHERE is_new
				   AND external_id IS NOT NULL`,
				stageTable,
			),
			transaction.sink.options.Graph.ID,
			binding.Generation.ID,
			int32(binding.Generation.LabelID),
		); err != nil {
			return fmt.Errorf("insert incremental edge identities: %w", err)
		}
	}
	if transaction.sink.options.Mode == config.LoadUpsert {
		if _, err := transaction.tx.Exec(
			ctx,
			fmt.Sprintf(
				`UPDATE agefreighter_meta.edge_identity identity
				 SET start_graph_id = decision.start_graph_id,
				     end_graph_id = decision.end_graph_id
				 FROM %s decision
				 WHERE NOT decision.is_new
				   AND identity.graph_generation_id = $1
				   AND identity.label_generation_id = $2
				   AND identity.source_namespace = decision.source_namespace
				   AND identity.external_id = decision.external_id`,
				stageTable,
			),
			transaction.sink.options.Graph.ID,
			binding.Generation.ID,
		); err != nil {
			return fmt.Errorf("update incremental edge identities: %w", err)
		}
	}
	transaction.addCommitted(binding, expectedNew)
	return nil
}

func countNewEdges(decisions []incrementalEdgeDecision) int64 {
	var count int64
	for _, decision := range decisions {
		if decision.isNew {
			count++
		}
	}
	return count
}

func (transaction *loadTransaction) resolveExistingProperties(
	existing []byte,
	incoming []byte,
) ([]byte, error) {
	switch transaction.sink.options.Mode {
	case config.LoadAppend:
		switch transaction.sink.options.AppendDuplicate {
		case config.AppendDuplicateError:
			return nil, errors.New("append duplicate identity")
		case config.AppendDuplicateIgnoreIdentical:
			equal, err := equalJSON(existing, incoming)
			if err != nil {
				return nil, err
			}
			if !equal {
				return nil, errors.New("append duplicate has conflicting properties")
			}
			return existing, nil
		default:
			return nil, errors.New("unsupported append duplicate policy")
		}
	case config.LoadUpsert:
		switch transaction.sink.options.PropertyMode {
		case config.PropertiesReplace:
			return incoming, nil
		case config.PropertiesMerge:
			return mergeJSONObject(existing, incoming, false)
		case config.PropertiesMergeDeleteNull:
			return mergeJSONObject(existing, incoming, true)
		default:
			return nil, errors.New("unsupported upsert property mode")
		}
	default:
		return nil, errors.New("incremental property resolution requires append or upsert")
	}
}

func equalJSON(left, right []byte) (bool, error) {
	leftValue, err := decodeJSON(left)
	if err != nil {
		return false, fmt.Errorf("decode existing properties: %w", err)
	}
	rightValue, err := decodeJSON(right)
	if err != nil {
		return false, fmt.Errorf("decode incoming properties: %w", err)
	}
	leftCanonical, err := json.Marshal(leftValue)
	if err != nil {
		return false, fmt.Errorf("canonicalize existing properties: %w", err)
	}
	rightCanonical, err := json.Marshal(rightValue)
	if err != nil {
		return false, fmt.Errorf("canonicalize incoming properties: %w", err)
	}
	return bytes.Equal(leftCanonical, rightCanonical), nil
}

func decodeJSON(value []byte) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(value))
	decoder.UseNumber()
	var decoded any
	if err := decoder.Decode(&decoded); err != nil {
		return nil, err
	}
	if err := decoder.Decode(new(any)); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, errors.New("multiple JSON values")
		}
		return nil, err
	}
	return decoded, nil
}

func mergeJSONObject(
	existing []byte,
	incoming []byte,
	deleteNull bool,
) ([]byte, error) {
	var current map[string]json.RawMessage
	if err := json.Unmarshal(existing, &current); err != nil {
		return nil, fmt.Errorf("decode existing property object: %w", err)
	}
	var updates map[string]json.RawMessage
	if err := json.Unmarshal(incoming, &updates); err != nil {
		return nil, fmt.Errorf("decode incoming property object: %w", err)
	}
	if current == nil || updates == nil {
		return nil, errors.New("properties must be JSON objects")
	}
	for key, value := range updates {
		if deleteNull && bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			delete(current, key)
			continue
		}
		current[key] = value
	}
	merged, err := json.Marshal(current)
	if err != nil {
		return nil, fmt.Errorf("encode merged property object: %w", err)
	}
	return merged, nil
}
