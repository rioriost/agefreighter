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
		rows, err := connection.Query(ctx, fmt.Sprintf(`
			SELECT identity.external_id, physical.properties::text
			FROM agefreighter_meta.vertex_identity identity
			JOIN %s physical
			  ON physical.id = identity.graph_id::text::ag_catalog.graphid
			WHERE identity.graph_generation_id = $1
			  AND identity.label_generation_id = $2
			ORDER BY identity.external_id`,
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
		rows, err := connection.Query(ctx, fmt.Sprintf(`
			SELECT identity.external_id,
			       start_identity.external_id,
			       end_identity.external_id,
			       physical.properties::text
			FROM agefreighter_meta.edge_identity identity
			JOIN %s physical
			  ON physical.id = identity.graph_id::text::ag_catalog.graphid
			JOIN agefreighter_meta.vertex_identity start_identity
			  ON start_identity.graph_generation_id = identity.graph_generation_id
			 AND start_identity.graph_id = identity.start_graph_id
			JOIN agefreighter_meta.vertex_identity end_identity
			  ON end_identity.graph_generation_id = identity.graph_generation_id
			 AND end_identity.graph_id = identity.end_graph_id
			WHERE identity.graph_generation_id = $1
			  AND identity.label_generation_id = $2
			ORDER BY identity.external_id`,
			pgx.Identifier{graph, spec.Type}.Sanitize(),
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
		var identity, rawProperties string
		if err := rows.Scan(&identity, &rawProperties); err != nil {
			return count, err
		}
		properties, encoded, err := canonicalJSONProperties(rawProperties)
		if err != nil {
			return count, fmt.Errorf("target vertex %q row %d: %w", label, count+1, err)
		}
		key, err := integerProperty(properties, "source_key")
		if err != nil {
			return count, err
		}
		externalID, err := stringProperty(properties, "external_id")
		if err != nil {
			return count, err
		}
		if externalID != identity {
			return count, fmt.Errorf("target vertex %q identity does not match properties", label)
		}
		if err := builder.add(key, vertexLine(label, key, identity, encoded)); err != nil {
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
		var identity, startIdentity, endIdentity, rawProperties string
		if err := rows.Scan(&identity, &startIdentity, &endIdentity, &rawProperties); err != nil {
			return count, err
		}
		properties, encoded, err := canonicalJSONProperties(rawProperties)
		if err != nil {
			return count, fmt.Errorf("target edge %q row %d: %w", label, count+1, err)
		}
		key, err := integerProperty(properties, "source_key")
		if err != nil {
			return count, err
		}
		externalID, err := stringProperty(properties, "relationship_id")
		if err != nil {
			return count, err
		}
		if externalID != identity {
			return count, fmt.Errorf("target edge %q identity does not match properties", label)
		}
		if err := builder.add(key, edgeLine(
			label, key, identity, startIdentity, endIdentity, encoded,
		)); err != nil {
			return count, err
		}
		count++
	}
	return count, rows.Err()
}
