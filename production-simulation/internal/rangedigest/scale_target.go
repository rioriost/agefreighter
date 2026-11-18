package rangedigest

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
)

// ScaleTargetManifest streams a source-key-ordered, read-only snapshot. SQL
// resolves physical endpoints, avoiding a client-sized vertex index or sort.
// Target sort/temp-storage costs still belong in the phase promotion gate.
func ScaleTargetManifest(ctx context.Context, dsn, manifestPath, jobID string, rangeRows int64, gremlin bool) (Manifest, error) {
	if ctx == nil || dsn == "" || jobID == "" {
		return Manifest{}, errors.New("context, target DSN and job ID are required")
	}
	f, err := fixture.Verify(manifestPath)
	if err != nil {
		return Manifest{}, err
	}
	switch f.Plan.Phase {
	case fixture.PhaseP0, fixture.PhaseP1, fixture.PhaseP2, fixture.PhaseP3:
	default:
		return Manifest{}, errors.New("scale target digest requires an approved P0-P3 base fixture")
	}
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return Manifest{}, err
	}
	defer conn.Close(context.Background())
	return scaleTargetSnapshot(ctx, conn, f, jobID, rangeRows, gremlin)
}

func scaleTargetSnapshot(ctx context.Context, conn *pgx.Conn, f fixture.Manifest, jobID string, rangeRows int64, gremlin bool) (Manifest, error) {
	builder, err := newRangeBuilder(rangeRows)
	if err != nil {
		return Manifest{}, err
	}
	tx, err := conn.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly})
	if err != nil {
		return Manifest{}, err
	}
	defer tx.Rollback(context.Background())
	if _, err := tx.Exec(ctx, `SET LOCAL search_path=ag_catalog,public; SET LOCAL work_mem='16MB'; SET LOCAL max_parallel_workers_per_gather=0; SET LOCAL statement_timeout='60min'; SET LOCAL lock_timeout='60s'`); err != nil {
		return Manifest{}, err
	}
	var status, graph string
	var generation int64
	if err := tx.QueryRow(ctx, `SELECT j.status,g.graph_name,g.graph_generation_id
		FROM agefreighter_meta.load_job j JOIN agefreighter_meta.graph_generation g USING(graph_generation_id)
		WHERE j.job_id=$1::uuid`, jobID).Scan(&status, &graph, &generation); err != nil {
		return Manifest{}, err
	}
	if status != "committed" {
		return Manifest{}, errors.New("scale target job must be committed")
	}
	if err := gremlinTargetCatalog(ctx, tx, graph, generation, f.Plan); err != nil {
		return Manifest{}, err
	}
	vertexGenerations := make(map[string]int64, len(f.Plan.VertexSpecs))
	for _, spec := range f.Plan.VertexSpecs {
		label, err := resolveLabelGeneration(ctx, tx, generation, spec.Label, "v")
		if err != nil {
			return Manifest{}, err
		}
		vertexGenerations[spec.Label] = label
		if err := gremlinIdentityCount(ctx, tx, "vertex_identity", generation, label, spec.Count); err != nil {
			return Manifest{}, err
		}
		if err := builder.begin("v", spec.Label); err != nil {
			return Manifest{}, err
		}
		rows, err := tx.Query(ctx, fmt.Sprintf(`SELECT p.properties::text,i.external_id,i.source_namespace
			FROM ONLY %s p LEFT JOIN agefreighter_meta.vertex_identity i
			ON i.graph_id=p.id::text::bigint AND i.graph_generation_id=$1 AND i.label_generation_id=$2
			ORDER BY (p.properties::text::jsonb->>'source_key')::bigint`,
			pgx.Identifier{graph, spec.Label}.Sanitize()), generation, label)
		if err != nil {
			return Manifest{}, err
		}
		err = digestScaleRows(ctx, rows, builder, spec.Count, func(row pgx.Rows) (int64, []byte, error) {
			var raw, id, namespace string
			if err := row.Scan(&raw, &id, &namespace); err != nil {
				return 0, nil, err
			}
			if namespace == "" {
				return 0, nil, errors.New("vertex source namespace is missing")
			}
			if gremlin {
				return canonicalGremlinTargetVertex(spec.Label, raw, id)
			}
			return canonicalTargetVertex(spec.Label, raw)
		})
		if err != nil {
			return Manifest{}, fmt.Errorf("scale vertex %s: %w", spec.Label, err)
		}
		if err := builder.end(); err != nil {
			return Manifest{}, err
		}
	}
	for _, spec := range f.Plan.EdgeSpecs {
		label, err := resolveLabelGeneration(ctx, tx, generation, spec.Type, "e")
		if err != nil {
			return Manifest{}, err
		}
		if err := gremlinIdentityCount(ctx, tx, "edge_identity", generation, label, spec.Count); err != nil {
			return Manifest{}, err
		}
		if err := builder.begin("e", spec.Type); err != nil {
			return Manifest{}, err
		}
		rows, err := tx.Query(ctx, fmt.Sprintf(`SELECT p.properties::text,i.external_id,i.source_namespace,
			p.start_id::text::bigint,p.end_id::text::bigint,i.start_graph_id,i.end_graph_id,
			s.properties::text::jsonb->>'external_id',d.properties::text::jsonb->>'external_id',
			si.external_id,di.external_id,si.source_namespace,di.source_namespace
			FROM ONLY %s p
			LEFT JOIN agefreighter_meta.edge_identity i ON i.graph_id=p.id::text::bigint
				AND i.graph_generation_id=$1 AND i.label_generation_id=$2
			LEFT JOIN ONLY %s s ON s.id=p.start_id
			LEFT JOIN ONLY %s d ON d.id=p.end_id
			LEFT JOIN agefreighter_meta.vertex_identity si ON si.graph_id=p.start_id::text::bigint
				AND si.graph_generation_id=$1 AND si.label_generation_id=$3
			LEFT JOIN agefreighter_meta.vertex_identity di ON di.graph_id=p.end_id::text::bigint
				AND di.graph_generation_id=$1 AND di.label_generation_id=$4
			ORDER BY (p.properties::text::jsonb->>'source_key')::bigint`,
			pgx.Identifier{graph, spec.Type}.Sanitize(),
			pgx.Identifier{graph, spec.Start}.Sanitize(),
			pgx.Identifier{graph, spec.End}.Sanitize()),
			generation, label, vertexGenerations[spec.Start], vertexGenerations[spec.End])
		if err != nil {
			return Manifest{}, err
		}
		err = digestScaleRows(ctx, rows, builder, spec.Count, func(row pgx.Rows) (int64, []byte, error) {
			var raw, id, namespace, startID, endID, storedStart, storedEnd, startNamespace, endNamespace string
			var start, end, metaStart, metaEnd int64
			if err := row.Scan(&raw, &id, &namespace, &start, &end, &metaStart, &metaEnd,
				&startID, &endID, &storedStart, &storedEnd, &startNamespace, &endNamespace); err != nil {
				return 0, nil, err
			}
			if start != metaStart || end != metaEnd || namespace == "" ||
				startNamespace != namespace || endNamespace != namespace {
				return 0, nil, errors.New("physical/retained edge endpoint or namespace differs")
			}
			if gremlin {
				if _, err := gremlinStoredID(storedStart, startID); err != nil {
					return 0, nil, err
				}
				if _, err := gremlinStoredID(storedEnd, endID); err != nil {
					return 0, nil, err
				}
				return canonicalGremlinTargetEdge(spec.Type, raw, id, storedStart, storedEnd)
			}
			return canonicalTargetEdge(spec.Type, raw, startID, endID)
		})
		if err != nil {
			return Manifest{}, fmt.Errorf("scale edge %s: %w", spec.Type, err)
		}
		if err := builder.end(); err != nil {
			return Manifest{}, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return Manifest{}, err
	}
	result := builder.result("apache-age", f.RootSHA256, graph, jobID)
	if gremlin {
		result.CanonicalVersion = GremlinCanonicalVersion
	}
	return result, nil
}

func digestScaleRows(ctx context.Context, rows pgx.Rows, builder *rangeBuilder, expected int64,
	decode func(pgx.Rows) (int64, []byte, error),
) error {
	defer rows.Close()
	var count int64
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return err
		}
		if count >= expected {
			return errors.New("physical row count exceeds fixture")
		}
		key, canonical, err := decode(rows)
		if err != nil {
			return err
		}
		if err := builder.add(key, canonical); err != nil {
			return err
		}
		count++
	}
	if err := rows.Err(); err != nil {
		return err
	}
	if count != expected {
		return fmt.Errorf("physical row count %d differs from expected %d", count, expected)
	}
	return nil
}
