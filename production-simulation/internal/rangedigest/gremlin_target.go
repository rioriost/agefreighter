package rangedigest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
)

// P1GremlinTargetManifest reads stored composite identities and physical edge
// endpoints. It never reconstructs observed identities from the expected fixture.
func P1GremlinTargetManifest(ctx context.Context, dsn, manifestPath, jobID string, rangeRows int64) (Manifest, error) {
	if ctx == nil || dsn == "" || jobID == "" {
		return Manifest{}, errors.New("context, target DSN and job ID are required")
	}
	f, err := fixture.Verify(manifestPath)
	if err != nil {
		return Manifest{}, err
	}
	if f.Plan.Phase != fixture.PhaseP1 {
		return Manifest{}, errors.New("Gremlin target digest is restricted to P1")
	}
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return Manifest{}, err
	}
	defer conn.Close(context.Background())
	return gremlinTargetSnapshot(ctx, conn, f, jobID, rangeRows)
}

// A single read-only snapshot prevents vertices and edges coming from different
// database states. The internal entry point also admits tiny SQL-contract tests.
func gremlinTargetSnapshot(ctx context.Context, conn *pgx.Conn, f fixture.Manifest, jobID string, rangeRows int64) (Manifest, error) {
	if f.Plan.Phase != fixture.PhaseP1 && f.Plan.Phase != fixture.PhaseTiny {
		return Manifest{}, errors.New("unsupported bounded fixture")
	}
	builder, err := newRangeBuilder(rangeRows)
	if err != nil {
		return Manifest{}, err
	}
	tx, err := conn.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly})
	if err != nil {
		return Manifest{}, err
	}
	defer tx.Rollback(context.Background())
	if _, err := tx.Exec(ctx, `SET LOCAL search_path = ag_catalog, "$user", public; SET LOCAL statement_timeout = '20min'; SET LOCAL max_parallel_workers_per_gather = 0`); err != nil {
		return Manifest{}, err
	}
	var status, graph string
	var generation int64
	if err := tx.QueryRow(ctx, `SELECT j.status,g.graph_generation_id,g.graph_name FROM agefreighter_meta.load_job j JOIN agefreighter_meta.graph_generation g USING(graph_generation_id) WHERE j.job_id=$1::uuid`, jobID).Scan(&status, &generation, &graph); err != nil {
		return Manifest{}, err
	}
	if status != "committed" {
		return Manifest{}, errors.New("target job must be committed")
	}
	if err := gremlinTargetCatalog(ctx, tx, graph, generation, f.Plan); err != nil {
		return Manifest{}, err
	}
	endpoints := newGremlinEndpointIndex(f.Plan.VertexTotal)
	for _, v := range f.Plan.VertexSpecs {
		labelGeneration, err := resolveLabelGeneration(ctx, tx, generation, v.Label, "v")
		if err != nil {
			return Manifest{}, err
		}
		if err := gremlinIdentityCount(ctx, tx, "vertex_identity", generation, labelGeneration, v.Count); err != nil {
			return Manifest{}, err
		}
		if err := builder.begin("v", v.Label); err != nil {
			return Manifest{}, err
		}
		rows, err := tx.Query(ctx, fmt.Sprintf(`SELECT p.id::text::bigint,p.properties::text,i.external_id,i.source_namespace
   FROM ONLY %s p LEFT JOIN agefreighter_meta.vertex_identity i
   ON i.graph_id=p.id::text::bigint AND i.graph_generation_id=$1 AND i.label_generation_id=$2`, pgx.Identifier{graph, v.Label}.Sanitize()), generation, labelGeneration)
		if err != nil {
			return Manifest{}, err
		}
		sink := targetSink(builder, true, v.Count)
		count, err := digestGremlinVertices(ctx, rows, v.Label, sink, endpoints)
		if err != nil {
			return Manifest{}, err
		}
		if count != v.Count {
			return Manifest{}, errors.New("physical vertex count differs")
		}
		if err := finishTargetSink(ctx, sink); err != nil {
			return Manifest{}, err
		}
		if err := builder.end(); err != nil {
			return Manifest{}, err
		}
	}
	for _, e := range f.Plan.EdgeSpecs {
		labelGeneration, err := resolveLabelGeneration(ctx, tx, generation, e.Type, "e")
		if err != nil {
			return Manifest{}, err
		}
		if err := gremlinIdentityCount(ctx, tx, "edge_identity", generation, labelGeneration, e.Count); err != nil {
			return Manifest{}, err
		}
		if err := builder.begin("e", e.Type); err != nil {
			return Manifest{}, err
		}
		rows, err := tx.Query(ctx, fmt.Sprintf(`SELECT p.properties::text,i.external_id,i.source_namespace,
   p.start_id::text::bigint,p.end_id::text::bigint,i.start_graph_id,i.end_graph_id
   FROM ONLY %s p LEFT JOIN agefreighter_meta.edge_identity i
   ON i.graph_id=p.id::text::bigint AND i.graph_generation_id=$1 AND i.label_generation_id=$2`, pgx.Identifier{graph, e.Type}.Sanitize()), generation, labelGeneration)
		if err != nil {
			return Manifest{}, err
		}
		sink := targetSink(builder, true, e.Count)
		count, err := digestGremlinEdges(ctx, rows, e, sink, endpoints)
		if err != nil {
			return Manifest{}, err
		}
		if count != e.Count {
			return Manifest{}, errors.New("physical edge count differs")
		}
		if err := finishTargetSink(ctx, sink); err != nil {
			return Manifest{}, err
		}
		if err := builder.end(); err != nil {
			return Manifest{}, err
		}
	}
	result := builder.result("apache-age", f.RootSHA256, graph, jobID)
	result.CanonicalVersion = GremlinCanonicalVersion
	if err := tx.Commit(ctx); err != nil {
		return Manifest{}, err
	}
	return result, nil
}

func gremlinIdentityCount(ctx context.Context, tx pgx.Tx, table string, generation, label, count int64) error {
	var actual int64
	if err := tx.QueryRow(ctx, fmt.Sprintf(`SELECT count(*) FROM %s WHERE graph_generation_id=$1 AND label_generation_id=$2`, pgx.Identifier{"agefreighter_meta", table}.Sanitize()), generation, label).Scan(&actual); err != nil {
		return err
	}
	if actual != count {
		return errors.New("retained identity count differs")
	}
	return nil
}

func gremlinTargetCatalog(ctx context.Context, tx pgx.Tx, graph string, generation int64, plan fixture.Plan) error {
	want := map[string]string{"_ag_label_vertex": "v", "_ag_label_edge": "e"}
	for _, v := range plan.VertexSpecs {
		want[v.Label] = "v"
	}
	for _, e := range plan.EdgeSpecs {
		want[e.Type] = "e"
	}
	rows, err := tx.Query(ctx, `SELECT l.name,l.kind::text FROM ag_catalog.ag_label l JOIN ag_catalog.ag_graph g ON l.graph=g.graphid WHERE g.name=$1`, graph)
	if err != nil {
		return err
	}
	seen := map[string]bool{}
	for rows.Next() {
		var name, kind string
		if err := rows.Scan(&name, &kind); err != nil {
			rows.Close()
			return err
		}
		if seen[name] || want[name] != kind {
			rows.Close()
			return errors.New("unexpected physical graph label")
		}
		seen[name] = true
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return err
	}
	if len(seen) != len(want) {
		return errors.New("incomplete physical graph labels")
	}
	for _, root := range []string{"_ag_label_vertex", "_ag_label_edge"} {
		var count int64
		if err := tx.QueryRow(ctx, fmt.Sprintf(`SELECT count(*) FROM ONLY %s`, pgx.Identifier{graph, root}.Sanitize())).Scan(&count); err != nil {
			return err
		}
		if count != 0 {
			return errors.New("unexpected unlabeled graph records")
		}
	}
	var labels int64
	if err := tx.QueryRow(ctx, `SELECT count(*) FROM agefreighter_meta.label_generation WHERE graph_generation_id=$1`, generation).Scan(&labels); err != nil {
		return err
	}
	if labels != int64(len(plan.VertexSpecs)+len(plan.EdgeSpecs)) {
		return errors.New("unexpected retained label generations")
	}
	for _, item := range []struct {
		table string
		count int64
	}{{"vertex_identity", plan.VertexTotal}, {"edge_identity", plan.EdgeTotal}} {
		var count int64
		if err := tx.QueryRow(ctx, fmt.Sprintf(`SELECT count(*) FROM %s WHERE graph_generation_id=$1`, pgx.Identifier{"agefreighter_meta", item.table}.Sanitize()), generation).Scan(&count); err != nil {
			return err
		}
		if count != item.count {
			return errors.New("unexpected generation-wide identity count")
		}
	}
	return nil
}

type gremlinEndpoint struct{ label, namespace, id string }
type gremlinEndpointIndex struct {
	byGraphID      map[int64]gremlinEndpoint
	maxRows, bytes int64
}

func newGremlinEndpointIndex(maxRows int64) *gremlinEndpointIndex {
	return &gremlinEndpointIndex{byGraphID: map[int64]gremlinEndpoint{}, maxRows: maxRows}
}
func (i *gremlinEndpointIndex) add(graphID int64, e gremlinEndpoint) error {
	size := int64(len(e.id) + len(e.namespace) + len(e.label))
	if graphID <= 0 || e.namespace == "" || e.label == "" || size > 1024 || i.maxRows < 1 || i.maxRows > 1_600_000 || int64(len(i.byGraphID)) >= i.maxRows || size > 256*1024*1024-i.bytes {
		return errors.New("Gremlin endpoint index bound or identity invalid")
	}
	if _, exists := i.byGraphID[graphID]; exists {
		return errors.New("duplicate physical vertex ID")
	}
	i.byGraphID[graphID] = e
	i.bytes += size
	return nil
}
func (i *gremlinEndpointIndex) lookup(graphID int64, label, namespace string) (string, error) {
	e, ok := i.byGraphID[graphID]
	if !ok || e.label != label || e.namespace != namespace {
		return "", errors.New("physical endpoint absent or label/namespace differs")
	}
	return e.id, nil
}

func gremlinStoredID(raw, visible string) ([2]string, error) {
	var result [2]string
	var values []string
	if len(raw) > 512 || json.Unmarshal([]byte(raw), &values) != nil || len(values) != 2 || values[0] == "" || values[1] == "" || values[1] != visible {
		return result, errors.New("invalid retained composite identity")
	}
	canonical, _ := json.Marshal(values)
	if string(canonical) != raw {
		return result, errors.New("noncanonical retained composite identity")
	}
	copy(result[:], values)
	return result, nil
}
func canonicalGremlinTargetVertex(label, raw, id string) (int64, []byte, error) {
	props, encoded, err := canonicalJSONProperties(raw)
	if err != nil {
		return 0, nil, err
	}
	key, err := integerProperty(props, "source_key")
	if err != nil {
		return 0, nil, err
	}
	visible, err := stringProperty(props, "external_id")
	if err != nil {
		return 0, nil, err
	}
	if _, err := gremlinStoredID(id, visible); err != nil {
		return 0, nil, err
	}
	return key, vertexLine(label, key, id, encoded), nil
}
func canonicalGremlinTargetEdge(label, raw, id, start, end string) (int64, []byte, error) {
	props, encoded, err := canonicalJSONProperties(raw)
	if err != nil {
		return 0, nil, err
	}
	key, err := integerProperty(props, "source_key")
	if err != nil {
		return 0, nil, err
	}
	visible, err := stringProperty(props, "relationship_id")
	if err != nil {
		return 0, nil, err
	}
	pair, err := gremlinStoredID(id, visible)
	if err != nil {
		return 0, nil, err
	}
	var startPair []string
	if json.Unmarshal([]byte(start), &startPair) != nil || len(startPair) != 2 || pair[0] != startPair[0] {
		return 0, nil, errors.New("edge/source partition differs")
	}
	return key, edgeLine(label, key, id, start, end, encoded), nil
}

func digestGremlinVertices(ctx context.Context, rows pgx.Rows, label string, sink canonicalSink, index *gremlinEndpointIndex) (int64, error) {
	defer rows.Close()
	var count int64
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return count, err
		}
		var graphID int64
		var raw, id, namespace string
		if err := rows.Scan(&graphID, &raw, &id, &namespace); err != nil {
			return count, err
		}
		key, line, err := canonicalGremlinTargetVertex(label, raw, id)
		if err != nil {
			return count, err
		}
		if err := index.add(graphID, gremlinEndpoint{label, namespace, id}); err != nil {
			return count, err
		}
		if err := sink.add(key, line); err != nil {
			return count, err
		}
		count++
	}
	return count, rows.Err()
}
func digestGremlinEdges(ctx context.Context, rows pgx.Rows, spec fixture.EdgeSpec, sink canonicalSink, index *gremlinEndpointIndex) (int64, error) {
	defer rows.Close()
	var count int64
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return count, err
		}
		var raw, id, namespace string
		var start, end, metaStart, metaEnd int64
		if err := rows.Scan(&raw, &id, &namespace, &start, &end, &metaStart, &metaEnd); err != nil {
			return count, err
		}
		if start != metaStart || end != metaEnd {
			return count, errors.New("physical/retained edge endpoints differ")
		}
		startID, err := index.lookup(start, spec.Start, namespace)
		if err != nil {
			return count, err
		}
		endID, err := index.lookup(end, spec.End, namespace)
		if err != nil {
			return count, err
		}
		key, line, err := canonicalGremlinTargetEdge(spec.Type, raw, id, startID, endID)
		if err != nil {
			return count, err
		}
		if err := sink.add(key, line); err != nil {
			return count, err
		}
		count++
	}
	return count, rows.Err()
}

func CompareGremlinTarget(expected, actual Manifest) (Comparison, error) {
	return compareManifests(expected, actual, GremlinCanonicalVersion, "apache-age")
}
