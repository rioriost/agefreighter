package rangedigest

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	fixturemodel "github.com/rioriost/agefreighter/production-simulation/internal/fixture"
)

func TargetManifest(
	ctx context.Context,
	dsn string,
	manifestPath string,
	jobID string,
	rangeRows int64,
) (Manifest, error) {
	if ctx == nil {
		return Manifest{}, errors.New("context is required")
	}
	if dsn == "" || jobID == "" {
		return Manifest{}, errors.New("target DSN and job ID are required")
	}
	fixtureManifest, err := fixturemodel.Verify(manifestPath)
	if err != nil {
		return Manifest{}, err
	}
	builder, err := newRangeBuilder(rangeRows)
	if err != nil {
		return Manifest{}, err
	}
	connection, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return Manifest{}, fmt.Errorf("connect to target: %w", err)
	}
	defer connection.Close(context.Background())
	if _, err := connection.Exec(ctx, `
		SET search_path = ag_catalog, "$user", public`); err != nil {
		return Manifest{}, fmt.Errorf("initialize AGE session: %w", err)
	}

	var status, graph string
	var generationID int64
	err = connection.QueryRow(ctx, `
		SELECT job.status, generation.graph_generation_id, generation.graph_name
		FROM agefreighter_meta.load_job job
		JOIN agefreighter_meta.graph_generation generation
		  ON generation.graph_generation_id = job.graph_generation_id
		WHERE job.job_id = $1::uuid`, jobID).Scan(&status, &generationID, &graph)
	if err != nil {
		return Manifest{}, fmt.Errorf("resolve committed target generation: %w", err)
	}
	if status != "committed" {
		return Manifest{}, fmt.Errorf("target job status is %q, expected committed", status)
	}

	for _, spec := range fixtureManifest.Plan.VertexSpecs {
		labelGeneration, err := resolveLabelGeneration(ctx, connection, generationID, spec.Label, "v")
		if err != nil {
			return Manifest{}, err
		}
		if err := builder.begin("v", spec.Label); err != nil {
			return Manifest{}, err
		}
		// P3 may use Neo4j internal IDs for migration-time correlation. Canonical
		// fixture identity is the visible external_id property, so the independent
		// digest must not substitute vertex_identity.external_id for graph content.
		rows, err := connection.Query(ctx, fmt.Sprintf(`
			SELECT physical.properties::text
			FROM agefreighter_meta.vertex_identity identity
			JOIN %s physical
			  ON physical.id = identity.graph_id::text::ag_catalog.graphid
			WHERE identity.graph_generation_id = $1
			  AND identity.label_generation_id = $2
			ORDER BY identity.graph_id`,
			pgx.Identifier{graph, spec.Label}.Sanitize(),
		), generationID, labelGeneration)
		if err != nil {
			return Manifest{}, fmt.Errorf("query target vertex %q: %w", spec.Label, err)
		}
		count, err := digestTargetVertices(ctx, rows, spec.Label, builder)
		if err != nil {
			return Manifest{}, err
		}
		if count != spec.Count {
			return Manifest{}, fmt.Errorf("target vertex %q rows=%d expected=%d", spec.Label, count, spec.Count)
		}
		if err := builder.end(); err != nil {
			return Manifest{}, err
		}
	}

	for _, spec := range fixtureManifest.Plan.EdgeSpecs {
		labelGeneration, err := resolveLabelGeneration(ctx, connection, generationID, spec.Type, "e")
		if err != nil {
			return Manifest{}, err
		}
		if err := builder.begin("e", spec.Type); err != nil {
			return Manifest{}, err
		}
		// Read visible endpoint identities from the referenced AGE vertices. The
		// operational vertex identity can intentionally be a Neo4j internal ID.
		rows, err := connection.Query(ctx, fmt.Sprintf(`
			SELECT physical.properties::text,
			       start_physical.properties::text,
			       end_physical.properties::text
			FROM agefreighter_meta.edge_identity identity
			JOIN %s physical
			  ON physical.id = identity.graph_id::text::ag_catalog.graphid
			JOIN %s start_physical
			  ON start_physical.id = identity.start_graph_id::text::ag_catalog.graphid
			JOIN %s end_physical
			  ON end_physical.id = identity.end_graph_id::text::ag_catalog.graphid
			WHERE identity.graph_generation_id = $1
			  AND identity.label_generation_id = $2
			ORDER BY identity.graph_id`,
			pgx.Identifier{graph, spec.Type}.Sanitize(),
			pgx.Identifier{graph, spec.Start}.Sanitize(),
			pgx.Identifier{graph, spec.End}.Sanitize(),
		), generationID, labelGeneration)
		if err != nil {
			return Manifest{}, fmt.Errorf("query target edge %q: %w", spec.Type, err)
		}
		count, err := digestTargetEdges(ctx, rows, spec.Type, builder)
		if err != nil {
			return Manifest{}, err
		}
		if count != spec.Count {
			return Manifest{}, fmt.Errorf("target edge %q rows=%d expected=%d", spec.Type, count, spec.Count)
		}
		if err := builder.end(); err != nil {
			return Manifest{}, err
		}
	}

	return builder.result("apache-age", fixtureManifest.RootSHA256, graph, jobID), nil
}

func resolveLabelGeneration(
	ctx context.Context,
	connection *pgx.Conn,
	graphGeneration int64,
	name string,
	kind string,
) (int64, error) {
	rows, err := connection.Query(ctx, `
		SELECT label_generation_id
		FROM agefreighter_meta.label_generation
		WHERE graph_generation_id = $1
		  AND label_name = $2
		  AND kind = $3
		ORDER BY mapping_generation`, graphGeneration, name, kind)
	if err != nil {
		return 0, err
	}
	defer rows.Close()
	var values []int64
	for rows.Next() {
		var value int64
		if err := rows.Scan(&value); err != nil {
			return 0, err
		}
		values = append(values, value)
	}
	if err := rows.Err(); err != nil {
		return 0, err
	}
	if len(values) != 1 {
		return 0, fmt.Errorf("target label %q/%s has %d generations, expected one", name, kind, len(values))
	}
	return values[0], nil
}

func digestTargetVertices(
	ctx context.Context,
	rows pgx.Rows,
	label string,
	builder *rangeBuilder,
) (int64, error) {
	defer rows.Close()
	var count int64
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return count, err
		}
		var rawProperties string
		if err := rows.Scan(&rawProperties); err != nil {
			return count, err
		}
		key, canonical, err := canonicalTargetVertex(label, rawProperties)
		if err != nil {
			return count, fmt.Errorf("target vertex %q row %d: %w", label, count+1, err)
		}
		if err := builder.add(key, canonical); err != nil {
			return count, err
		}
		count++
	}
	return count, rows.Err()
}

func digestTargetEdges(
	ctx context.Context,
	rows pgx.Rows,
	label string,
	builder *rangeBuilder,
) (int64, error) {
	defer rows.Close()
	var count int64
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return count, err
		}
		var rawProperties, startProperties, endProperties string
		if err := rows.Scan(&rawProperties, &startProperties, &endProperties); err != nil {
			return count, err
		}
		key, canonical, err := canonicalTargetEdge(
			label, rawProperties, startProperties, endProperties,
		)
		if err != nil {
			return count, fmt.Errorf("target edge %q row %d: %w", label, count+1, err)
		}
		if err := builder.add(key, canonical); err != nil {
			return count, err
		}
		count++
	}
	return count, rows.Err()
}

func canonicalTargetVertex(label, rawProperties string) (int64, []byte, error) {
	properties, encoded, err := canonicalJSONProperties(rawProperties)
	if err != nil {
		return 0, nil, err
	}
	key, err := integerProperty(properties, "source_key")
	if err != nil {
		return 0, nil, err
	}
	externalID, err := stringProperty(properties, "external_id")
	if err != nil {
		return 0, nil, err
	}
	return key, vertexLine(label, key, externalID, encoded), nil
}

func canonicalTargetEdge(
	label string,
	rawProperties string,
	startRawProperties string,
	endRawProperties string,
) (int64, []byte, error) {
	properties, encoded, err := canonicalJSONProperties(rawProperties)
	if err != nil {
		return 0, nil, err
	}
	key, err := integerProperty(properties, "source_key")
	if err != nil {
		return 0, nil, err
	}
	externalID, err := stringProperty(properties, "relationship_id")
	if err != nil {
		return 0, nil, err
	}
	startProperties, _, err := canonicalJSONProperties(startRawProperties)
	if err != nil {
		return 0, nil, fmt.Errorf("decode start vertex: %w", err)
	}
	startExternalID, err := stringProperty(startProperties, "external_id")
	if err != nil {
		return 0, nil, fmt.Errorf("start vertex: %w", err)
	}
	endProperties, _, err := canonicalJSONProperties(endRawProperties)
	if err != nil {
		return 0, nil, fmt.Errorf("decode end vertex: %w", err)
	}
	endExternalID, err := stringProperty(endProperties, "external_id")
	if err != nil {
		return 0, nil, fmt.Errorf("end vertex: %w", err)
	}
	return key, edgeLine(
		label, key, externalID, startExternalID, endExternalID, encoded,
	), nil
}
