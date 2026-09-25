package rangedigest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
)

func TestGremlinStoredIdentityAndEndpoints(t *testing.T) {
	for _, raw := range []string{`["p","v"]`, `["東京","v"]`} {
		if _, err := gremlinStoredID(raw, "v"); err != nil {
			t.Fatal(err)
		}
	}
	for _, raw := range []string{`"v"`, `["p","wrong"]`, `["","v"]`, `["p","v","extra"]`, `["p", "v"]`, `[null,"v"]`, `["p","v"] {}`} {
		if _, err := gremlinStoredID(raw, "v"); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
	index := newGremlinEndpointIndex(2)
	if err := index.add(10, gremlinEndpoint{"V", "ns", `["p","v"]`}); err != nil {
		t.Fatal(err)
	}
	if err := index.add(10, gremlinEndpoint{"V", "ns", `["q","v"]`}); err == nil {
		t.Fatal("duplicate graph ID")
	}
	for _, tc := range []struct {
		id        int64
		label, ns string
	}{{11, "V", "ns"}, {10, "wrong", "ns"}, {10, "V", "wrong"}} {
		if _, err := index.lookup(tc.id, tc.label, tc.ns); err == nil {
			t.Fatal("invalid endpoint accepted")
		}
	}
	if _, _, err := canonicalGremlinTargetEdge("E", `{"source_key":1,"relationship_id":"e"}`, `["wrong","e"]`, `["p","v"]`, `["q","w"]`); err == nil {
		t.Fatal("edge partition must match physical source")
	}
	if err := newGremlinEndpointIndex(1).add(1, gremlinEndpoint{"V", "ns", strings.Repeat("x", 1025)}); err == nil {
		t.Fatal("unbounded endpoint")
	}
	if _, err := CompareGremlinTarget(Manifest{CanonicalVersion: GremlinCanonicalVersion, Source: "fixture"}, Manifest{CanonicalVersion: GremlinCanonicalVersion, Source: "cosmos-gremlin-offline"}); err == nil {
		t.Fatal("offline result accepted as target")
	}
}

type gremlinSQLTrace struct{ queries []string }

func (t *gremlinSQLTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, d pgx.TraceQueryStartData) context.Context {
	t.queries = append(t.queries, d.SQL)
	return ctx
}
func (*gremlinSQLTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

// A dedicated local SQL contract. AF_GREMLIN_SQL_TEST_AGE=1 uses real AGE
// catalogs/graphid/agtype storage; otherwise AGE-shaped text tables are used.
// Neither mode is a migration/Azure/GUI qualification. Each run keeps a new DB.
func TestGremlinTargetSQLContract(t *testing.T) {
	dsn := os.Getenv("AF_GREMLIN_SQL_TEST_ADMIN_DSN")
	if dsn == "" {
		t.Skip("requires isolated local PostgreSQL admin DSN")
	}
	cfg, err := pgx.ParseConfig(dsn)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Host != "127.0.0.1" && !strings.HasPrefix(cfg.Host, "192.168.64.") && !strings.HasPrefix(cfg.Host, "192.168.65.") {
		t.Fatal("local contract server required")
	}
	ctx := t.Context()
	admin, err := pgx.ConnectConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer admin.Close(context.Background())
	database := fmt.Sprintf("af_gremlin_test_%d", time.Now().UnixNano())
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+pgx.Identifier{database}.Sanitize()); err != nil {
		t.Fatal(err)
	}
	cfg.Database = database
	trace := &gremlinSQLTrace{}
	cfg.Tracer = trace
	conn, err := pgx.ConnectConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close(context.Background())
	t.Logf("retained SQL-contract database: %s", database)
	const job = "11111111-1111-4111-8111-111111111111"
	root := filepath.Join(t.TempDir(), "fixture")
	f, err := fixture.Generate(ctx, fixture.GenerateConfig{Phase: fixture.PhaseTiny, Output: root, Seed: 20260829, Shards: 4, Workers: 2})
	if err != nil {
		t.Fatal(err)
	}
	expected, err := GremlinFixtureManifest(ctx, filepath.Join(root, "manifest.json"), 100000)
	if err != nil {
		t.Fatal(err)
	}
	exec := func(sql string, args ...any) {
		t.Helper()
		if _, err := conn.Exec(ctx, sql, args...); err != nil {
			t.Fatal(err)
		}
	}
	realAGE := os.Getenv("AF_GREMLIN_SQL_TEST_AGE") == "1"
	gid := func(expr string) string {
		if realAGE {
			return "(" + expr + ")::bigint::text::ag_catalog.graphid"
		}
		return "(" + expr + ")::bigint"
	}
	prop := func(expr string) string {
		if realAGE {
			return "(" + expr + ")::ag_catalog.agtype"
		}
		return "(" + expr + ")::text"
	}
	if realAGE {
		exec(`CREATE EXTENSION IF NOT EXISTS age; LOAD 'age'; SET search_path=ag_catalog,public; SELECT ag_catalog.create_graph('gremlin_contract')`)
		var version, server string
		if err := conn.QueryRow(ctx, `SELECT extversion,current_setting('server_version') FROM pg_extension WHERE extname='age'`).Scan(&version, &server); err != nil {
			t.Fatal(err)
		}
		t.Logf("REAL AGE storage contract: AGE %s / PostgreSQL %s", version, server)
	} else {
		exec(`CREATE SCHEMA ag_catalog; CREATE SCHEMA gremlin_contract;
 CREATE TABLE ag_catalog.ag_graph(graphid bigint,name text);
 CREATE TABLE ag_catalog.ag_label(graph bigint,name text,kind text);
 CREATE TABLE gremlin_contract._ag_label_vertex(id bigint,properties text); CREATE TABLE gremlin_contract._ag_label_edge(id bigint,properties text);
 INSERT INTO ag_catalog.ag_graph VALUES(1,'gremlin_contract');
 INSERT INTO ag_catalog.ag_label VALUES(1,'_ag_label_vertex','v'),(1,'_ag_label_edge','e')`)
	}
	exec(`CREATE SCHEMA agefreighter_meta;
 CREATE TABLE agefreighter_meta.graph_generation(graph_generation_id bigint,graph_name text);
 CREATE TABLE agefreighter_meta.load_job(job_id uuid,graph_generation_id bigint,status text);
 CREATE TABLE agefreighter_meta.label_generation(graph_generation_id bigint,label_generation_id bigint,label_name text,kind text,mapping_generation bigint);
 CREATE TABLE agefreighter_meta.vertex_identity(graph_generation_id bigint,label_generation_id bigint,graph_id bigint,external_id text,source_namespace text);
 CREATE TABLE agefreighter_meta.edge_identity(graph_generation_id bigint,label_generation_id bigint,graph_id bigint,external_id text,source_namespace text,start_graph_id bigint,end_graph_id bigint);
 INSERT INTO agefreighter_meta.graph_generation VALUES(1,'gremlin_contract')`)
	exec(`INSERT INTO agefreighter_meta.load_job VALUES($1,1,'committed')`, job)
	ids := map[string]int64{}
	vertices := map[string]fixture.VertexSpec{}
	for n, v := range f.Plan.VertexSpecs {
		vertices[v.Label] = v
		table := pgx.Identifier{"gremlin_contract", v.Label}.Sanitize()
		if realAGE {
			exec(`SELECT ag_catalog.create_vlabel('gremlin_contract',$1)`, v.Label)
		} else {
			exec("CREATE TABLE " + table + "(id bigint,properties text)")
			exec(`INSERT INTO ag_catalog.ag_label VALUES(1,$1,'v')`, v.Label)
		}
		exec(`INSERT INTO agefreighter_meta.label_generation VALUES(1,$1,$2,'v',1)`, n+1, v.Label)
		b, _ := newRangeBuilder(100000)
		b.begin("v", v.Label)
		_, err := digestFixtureFiles(ctx, root, fixturePaths(f, "node", v.Label), func(row []string) (int64, []byte, error) {
			key, line, err := gremlinFixtureVertex(v, row)
			if err != nil {
				return 0, nil, err
			}
			fields := bytes.SplitN(line, []byte{0}, 5)
			id := string(fields[3])
			graphID := int64(n+3)<<48 | key*2
			ids[v.Label+"\x00"+id] = graphID
			exec("INSERT INTO "+table+"(id,properties) VALUES("+gid("$1")+","+prop("$2")+")", graphID, string(bytes.TrimSuffix(fields[4], []byte{'\n'})))
			exec(`INSERT INTO agefreighter_meta.vertex_identity VALUES(1,$1,$2,$3,'ns')`, n+1, graphID, id)
			return key, line, nil
		}, b)
		if err != nil {
			t.Fatal(err)
		}
	}
	for n, e := range f.Plan.EdgeSpecs {
		table := pgx.Identifier{"gremlin_contract", e.Type}.Sanitize()
		if realAGE {
			exec(`SELECT ag_catalog.create_elabel('gremlin_contract',$1)`, e.Type)
		} else {
			exec("CREATE TABLE " + table + "(id bigint,properties text,start_id bigint,end_id bigint)")
			exec(`INSERT INTO ag_catalog.ag_label VALUES(1,$1,'e')`, e.Type)
		}
		exec(`INSERT INTO agefreighter_meta.label_generation VALUES(1,$1,$2,'e',1)`, 100+n, e.Type)
		b, _ := newRangeBuilder(100000)
		b.begin("e", e.Type)
		_, err := digestFixtureFiles(ctx, root, fixturePaths(f, "edge", e.Type), func(row []string) (int64, []byte, error) {
			key, line, err := gremlinFixtureEdge(e, vertices, row)
			if err != nil {
				return 0, nil, err
			}
			fields := bytes.SplitN(line, []byte{0}, 7)
			graphID := int64(n+12)<<48 | key*2
			start, end := ids[e.Start+"\x00"+string(fields[4])], ids[e.End+"\x00"+string(fields[5])]
			exec("INSERT INTO "+table+"(id,properties,start_id,end_id) VALUES("+gid("$1")+","+prop("$2")+","+gid("$3")+","+gid("$4")+")", graphID, string(bytes.TrimSuffix(fields[6], []byte{'\n'})), start, end)
			exec(`INSERT INTO agefreighter_meta.edge_identity VALUES(1,$1,$2,$3,'ns',$4,$5)`, 100+n, graphID, string(fields[3]), start, end)
			return key, line, nil
		}, b)
		if err != nil {
			t.Fatal(err)
		}
	}
	// The verifier uses a fresh session, without borrowing the setup connection's
	// LOAD/search_path state. Mutations remain on the separate setup connection.
	reader, err := pgx.ConnectConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close(context.Background())
	trace.queries = nil
	actual, err := gremlinTargetSnapshot(ctx, reader, f, job, 100000)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CompareGremlinTarget(expected, actual); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(trace.queries[0], "repeatable read read only") {
		t.Fatalf("missing read-only snapshot: %s", trace.queries[0])
	}
	for _, query := range trace.queries {
		lower := strings.ToLower(strings.TrimSpace(query))
		if !strings.HasPrefix(lower, "select") && !strings.HasPrefix(lower, "begin") && !strings.HasPrefix(lower, "set local") && lower != "commit" {
			t.Fatalf("unexpected verifier statement: %s", query)
		}
	}
	tests := []struct{ name, apply, undo string }{
		{"orphan-identity", `INSERT INTO agefreighter_meta.vertex_identity VALUES(1,9999,999999,'bad','ns')`, `DELETE FROM agefreighter_meta.vertex_identity WHERE label_generation_id=9999`},
		{"physical-endpoint", `UPDATE gremlin_contract."SUPPLIES" SET end_id=` + gid("end_id::text::bigint+2"), `UPDATE gremlin_contract."SUPPLIES" SET end_id=` + gid("end_id::text::bigint-2")},
		{"metadata-endpoint", `UPDATE agefreighter_meta.edge_identity SET end_graph_id=end_graph_id+2`, `UPDATE agefreighter_meta.edge_identity SET end_graph_id=end_graph_id-2`},
		{"namespace", `UPDATE agefreighter_meta.edge_identity SET source_namespace='other'`, `UPDATE agefreighter_meta.edge_identity SET source_namespace='ns'`},
		{"wrong-partition", `UPDATE agefreighter_meta.vertex_identity SET external_id=replace(external_id,'Supplier-','changed-')`, `UPDATE agefreighter_meta.vertex_identity SET external_id=replace(external_id,'changed-','Supplier-')`},
		{"missing-metadata", `UPDATE agefreighter_meta.vertex_identity SET graph_generation_id=2 WHERE label_generation_id=1`, `UPDATE agefreighter_meta.vertex_identity SET graph_generation_id=1`},
		{"physical-orphan", `INSERT INTO gremlin_contract."Supplier"(id,properties) VALUES(` + gid("999999") + `,'{}')`, `DELETE FROM gremlin_contract."Supplier" WHERE id::text::bigint=999999`},
		{"unlabeled", `INSERT INTO gremlin_contract._ag_label_vertex(id,properties) VALUES(` + gid("999999") + `,'{}')`, `DELETE FROM gremlin_contract._ag_label_vertex WHERE id::text::bigint=999999`},
		{"uncommitted", `UPDATE agefreighter_meta.load_job SET status='running'`, `UPDATE agefreighter_meta.load_job SET status='committed'`},
	}
	if realAGE {
		tests = append(tests, struct{ name, apply, undo string }{"extra-label", `SELECT ag_catalog.create_vlabel('gremlin_contract','extra')`, `SELECT ag_catalog.drop_label('gremlin_contract','extra',false)`})
	} else {
		tests = append(tests, struct{ name, apply, undo string }{"extra-label", `INSERT INTO ag_catalog.ag_label VALUES(1,'extra','v')`, `DELETE FROM ag_catalog.ag_label WHERE name='extra'`})
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := conn.Exec(ctx, tc.apply); err != nil {
				t.Fatal(err)
			}
			defer func() {
				if _, err := conn.Exec(ctx, tc.undo); err != nil {
					t.Error(err)
				}
			}()
			a, err := gremlinTargetSnapshot(ctx, reader, f, job, 100000)
			if err == nil {
				_, err = CompareGremlinTarget(expected, a)
			}
			if err == nil {
				t.Fatal("corrupted target accepted")
			}
		})
	}
	// Mutate a property without changing any metadata, then restore exact bytes.
	var props string
	var graphID int64
	if err := conn.QueryRow(ctx, `SELECT id::text::bigint,properties::text FROM gremlin_contract."Supplier" LIMIT 1`).Scan(&graphID, &props); err != nil {
		t.Fatal(err)
	}
	var p map[string]any
	json.Unmarshal([]byte(props), &p)
	p["name"] = "changed"
	bad, _ := json.Marshal(p)
	exec(`UPDATE gremlin_contract."Supplier" SET properties=`+prop("$1")+` WHERE id::text::bigint=$2`, string(bad), graphID)
	a, err := gremlinTargetSnapshot(ctx, reader, f, job, 100000)
	if err == nil {
		_, err = CompareGremlinTarget(expected, a)
	}
	if err == nil {
		t.Fatal("property mutation accepted")
	}
	exec(`UPDATE gremlin_contract."Supplier" SET properties=`+prop("$1")+` WHERE id::text::bigint=$2`, props, graphID)
	// A consistent metadata+physical edit to a different valid endpoint must
	// still fail the canonical comparison; it cannot be detected by counts alone.
	var edgeID, oldEnd, newEnd int64
	if err := conn.QueryRow(ctx, `SELECT id::text::bigint,end_id::text::bigint FROM gremlin_contract."SUPPLIES" LIMIT 1`).Scan(&edgeID, &oldEnd); err != nil {
		t.Fatal(err)
	}
	if err := conn.QueryRow(ctx, `SELECT id::text::bigint FROM gremlin_contract."Product" WHERE id::text::bigint<>$1 LIMIT 1`, oldEnd).Scan(&newEnd); err != nil {
		t.Fatal(err)
	}
	exec(`UPDATE gremlin_contract."SUPPLIES" SET end_id=`+gid("$1")+` WHERE id::text::bigint=$2`, newEnd, edgeID)
	exec(`UPDATE agefreighter_meta.edge_identity SET end_graph_id=$1 WHERE graph_id=$2`, newEnd, edgeID)
	a, err = gremlinTargetSnapshot(ctx, reader, f, job, 100000)
	if err != nil {
		t.Fatal("valid substituted endpoint should reach digest comparison", err)
	}
	if _, err := CompareGremlinTarget(expected, a); err == nil {
		t.Fatal("consistent wrong endpoint accepted")
	}
	exec(`UPDATE gremlin_contract."SUPPLIES" SET end_id=`+gid("$1")+` WHERE id::text::bigint=$2`, oldEnd, edgeID)
	exec(`UPDATE agefreighter_meta.edge_identity SET end_graph_id=$1 WHERE graph_id=$2`, oldEnd, edgeID)
	t.Log("SQL contract PASS: " + strconv.FormatInt(actual.RecordCount, 10) + " records; full target root " + actual.RootSHA256)
}
