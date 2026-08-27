package app

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
)

type sourceModeJobFactory func(mode config.LoadMode, dataset string) config.LoadJob

type sourceModeTypedExpectations struct {
	predicates []string
	quoted     bool
}

type sourceModeIdentityState struct {
	p1GraphID    int64
	e1GraphID    int64
	e1StartGraph int64
	e1EndGraph   int64
}

func TestCSVSourceModeMatrixIntegration(t *testing.T) {
	targetDSN := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if targetDSN == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN")
	}
	suffix := time.Now().UnixNano()
	graph := fmt.Sprintf("csv_mode_matrix_%d", suffix)
	dir := t.TempDir()
	files := map[string][2]string{
		"create": {
			"id,name,active,score,tags,profile\n" +
				"p1,Ada Create,true,1.5,math,London\n" +
				"p2,Grace Create,false,2,compiler,New York\n",
			"id,start,end,weight\ne1,p1,p2,1\n",
		},
		"replace": {
			"id,name,active,score,tags,profile\n" +
				"p1,Ada Replace,true,2.5,math,London\n" +
				"p2,Grace Replace,false,3,compiler,New York\n",
			"id,start,end,weight\ne1,p1,p2,2\n",
		},
		"append": {
			"id,name,active,score,tags,profile\n" +
				"p3,Katherine Append,true,4.5,space,Washington\n",
			"id,start,end,weight\ne2,p2,p3,3\n",
		},
		"upsert": {
			"id,name,active,score,tags,profile\n" +
				"p1,Ada Upsert,true,9.5,math,London\n" +
				"p4,Dorothy Upsert,true,5.5,navy,Arlington\n",
			"id,start,end,weight\n" +
				"e1,p1,p4,9\n" +
				"e3,p3,p4,4\n",
		},
	}
	for dataset, contents := range files {
		if err := os.WriteFile(
			filepath.Join(dir, dataset+"-vertices.csv"),
			[]byte(contents[0]),
			0o600,
		); err != nil {
			t.Fatalf("write %s vertex fixture: %v", dataset, err)
		}
		if err := os.WriteFile(
			filepath.Join(dir, dataset+"-edges.csv"),
			[]byte(contents[1]),
			0o600,
		); err != nil {
			t.Fatalf("write %s edge fixture: %v", dataset, err)
		}
	}

	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", targetDSN)
	runSourceModeMatrix(
		t,
		targetDSN,
		dir,
		graph,
		"csv",
		func(mode config.LoadMode, dataset string) config.LoadJob {
			job := testLoadJob(
				graph,
				filepath.Join(dir, dataset+"-vertices.csv"),
				filepath.Join(dir, dataset+"-edges.csv"),
			)
			job.Metadata.Name = "csv-matrix-" + dataset
			job.Target.Mode = mode
			job.Source.CSV.Vertices[0].Properties = map[string]string{
				"name": "name", "active": "active", "score": "score",
				"tags": "tags", "profile": "profile",
			}
			job.Source.CSV.Edges[0].ExternalIDColumn = "id"
			job.Source.CSV.Edges[0].Properties = map[string]string{"weight": "weight"}
			return job
		},
		sourceModeTypedExpectations{
			predicates: []string{
				"n.active = 'true'",
				"n.score = '9.5'",
				"n.tags = 'math'",
				"n.profile = 'London'",
			},
			quoted: true,
		},
	)
}

func runSourceModeMatrix(
	t *testing.T,
	targetDSN string,
	dir string,
	graph string,
	connector string,
	jobFactory sourceModeJobFactory,
	typed sourceModeTypedExpectations,
) {
	t.Helper()
	connection, err := pgx.Connect(t.Context(), targetDSN)
	if err != nil {
		t.Fatalf("connect AGE target: %v", err)
	}
	t.Cleanup(func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = connection.Close(closeCtx)
	})
	if _, err := connection.Exec(t.Context(), "LOAD 'age'"); err != nil {
		t.Fatalf("load AGE extension: %v", err)
	}
	if _, err := connection.Exec(
		t.Context(),
		`SET search_path = ag_catalog, "$user", public`,
	); err != nil {
		t.Fatalf("set AGE search path: %v", err)
	}

	phases := []struct {
		mode       config.LoadMode
		dataset    string
		records    uint64
		vertices   int64
		edges      int64
		adaName    string
		edgeFrom   string
		edgeTo     string
		edgeWeight string
	}{
		{
			mode: config.LoadCreate, dataset: "create", records: 3,
			vertices: 2, edges: 1, adaName: "Ada Create",
			edgeFrom: "Ada Create", edgeTo: "Grace Create", edgeWeight: "1",
		},
		{
			mode: config.LoadReplace, dataset: "replace", records: 3,
			vertices: 2, edges: 1, adaName: "Ada Replace",
			edgeFrom: "Ada Replace", edgeTo: "Grace Replace", edgeWeight: "2",
		},
		{
			mode: config.LoadAppend, dataset: "append", records: 2,
			vertices: 3, edges: 2, adaName: "Ada Replace",
			edgeFrom: "Grace Replace", edgeTo: "Katherine Append", edgeWeight: "3",
		},
		{
			mode: config.LoadUpsert, dataset: "upsert", records: 4,
			vertices: 4, edges: 3, adaName: "Ada Upsert",
			edgeFrom: "Ada Upsert", edgeTo: "Dorothy Upsert", edgeWeight: "9",
		},
	}

	matrixT := t
	var replacedIdentities sourceModeIdentityState
	for _, phase := range phases {
		t.Run(string(phase.mode), func(t *testing.T) {
			job := jobFactory(phase.mode, phase.dataset)
			path := writeLoadJob(
				t,
				dir,
				fmt.Sprintf("%s-%s.yaml", connector, phase.mode),
				job,
			)
			result, loadErr := Load(t.Context(), path)
			if result.JobID != "" {
				registerCleanup(matrixT, targetDSN, graph, result.JobID)
				if phase.mode == config.LoadReplace {
					registerReplacementArtifacts(
						matrixT,
						targetDSN,
						graph,
						result.JobID,
					)
				}
			}
			if loadErr != nil {
				t.Fatalf("Load(%s): %v", phase.mode, loadErr)
			}
			if result.Status != meta.JobCommitted ||
				result.Metrics.RecordsCommitted != phase.records {
				t.Fatalf("Load(%s) = %#v", phase.mode, result)
			}
			if _, err := Verify(t.Context(), path, result.JobID); err != nil {
				t.Fatalf("Verify(%s): %v", phase.mode, err)
			}
			assertCypherCount(
				t,
				connection,
				graph,
				"MATCH (n:Person) RETURN count(n)",
				phase.vertices,
			)
			assertCypherCount(
				t,
				connection,
				graph,
				"MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r)",
				phase.edges,
			)
			weight := phase.edgeWeight
			if typed.quoted {
				weight = "'" + weight + "'"
			}
			assertCypherCount(
				t,
				connection,
				graph,
				fmt.Sprintf(
					"MATCH (a:Person)-[r:KNOWS]->(b:Person) "+
						"WHERE a.name = '%s' AND b.name = '%s' "+
						"AND r.weight = %s RETURN count(r)",
					phase.edgeFrom,
					phase.edgeTo,
					weight,
				),
				1,
			)
			assertCypherCount(
				t,
				connection,
				graph,
				fmt.Sprintf(
					"MATCH (n:Person) WHERE n.name = '%s' RETURN count(n)",
					phase.adaName,
				),
				1,
			)
			if phase.mode == config.LoadReplace {
				replacedIdentities = readSourceModeIdentityState(
					t,
					connection,
					graph,
				)
				replacedPhysical := readSourceModePhysicalState(
					t,
					connection,
					graph,
					"Ada Replace",
					"Grace Replace",
					"2",
					typed.quoted,
				)
				assertSourceModeIdentityMatchesPhysical(
					t,
					replacedIdentities,
					replacedPhysical,
				)
				if _, err := Cleanup(t.Context(), path, result.JobID); err != nil {
					t.Fatalf("Cleanup(replace backup): %v", err)
				}
			}
			if phase.mode == config.LoadUpsert {
				assertSourceModeTypedProperties(t, connection, graph, typed)
				upsertedIdentities := readSourceModeIdentityState(
					t,
					connection,
					graph,
				)
				p4GraphID := readSourceModeVertexGraphID(
					t,
					connection,
					graph,
					"p4",
				)
				if upsertedIdentities.p1GraphID != replacedIdentities.p1GraphID {
					t.Fatalf(
						"p1 graph ID changed from %d to %d",
						replacedIdentities.p1GraphID,
						upsertedIdentities.p1GraphID,
					)
				}
				if upsertedIdentities.e1GraphID != replacedIdentities.e1GraphID {
					t.Fatalf(
						"e1 graph ID changed from %d to %d",
						replacedIdentities.e1GraphID,
						upsertedIdentities.e1GraphID,
					)
				}
				if upsertedIdentities.e1StartGraph != upsertedIdentities.p1GraphID ||
					upsertedIdentities.e1EndGraph != p4GraphID {
					t.Fatalf(
						"e1 endpoints = (%d, %d), want (%d, %d)",
						upsertedIdentities.e1StartGraph,
						upsertedIdentities.e1EndGraph,
						upsertedIdentities.p1GraphID,
						p4GraphID,
					)
				}
				upsertedPhysical := readSourceModePhysicalState(
					t,
					connection,
					graph,
					"Ada Upsert",
					"Dorothy Upsert",
					"9",
					typed.quoted,
				)
				assertSourceModeIdentityMatchesPhysical(
					t,
					upsertedIdentities,
					upsertedPhysical,
				)
				if upsertedPhysical.e1EndGraph != p4GraphID {
					t.Fatalf(
						"physical p4 graph ID = %d, metadata = %d",
						upsertedPhysical.e1EndGraph,
						p4GraphID,
					)
				}
				newEdgeWeight := "4"
				if typed.quoted {
					newEdgeWeight = "'4'"
				}
				assertCypherCount(
					t,
					connection,
					graph,
					"MATCH (a:Person)-[r:KNOWS]->(b:Person) "+
						"WHERE a.name = 'Katherine Append' "+
						"AND b.name = 'Dorothy Upsert' "+
						"AND r.weight = "+newEdgeWeight+" RETURN count(r)",
					1,
				)
			}
		})
	}
}

func registerReplacementArtifacts(
	t *testing.T,
	targetDSN string,
	graph string,
	jobID string,
) {
	t.Helper()
	backup, err := age.DeriveGraphName(graph, age.BackupName, jobID)
	if err != nil {
		t.Fatalf("derive replacement backup: %v", err)
	}
	shadow, err := age.DeriveGraphName(graph, age.ShadowName, jobID)
	if err != nil {
		t.Fatalf("derive replacement shadow: %v", err)
	}
	registerReplaceCleanup(t, targetDSN, jobID, graph, shadow, backup)
}

func assertSourceModeTypedProperties(
	t *testing.T,
	connection *pgx.Conn,
	graph string,
	want sourceModeTypedExpectations,
) {
	t.Helper()
	for _, predicate := range want.predicates {
		assertCypherCount(
			t,
			connection,
			graph,
			fmt.Sprintf(
				"MATCH (n:Person) WHERE n.name = 'Ada Upsert' AND %s RETURN count(n)",
				predicate,
			),
			1,
		)
	}
}

func readSourceModeIdentityState(
	t *testing.T,
	connection *pgx.Conn,
	graph string,
) sourceModeIdentityState {
	t.Helper()
	var state sourceModeIdentityState
	state.p1GraphID = readSourceModeVertexGraphID(t, connection, graph, "p1")
	if err := connection.QueryRow(
		t.Context(),
		`SELECT edge.graph_id, edge.start_graph_id, edge.end_graph_id
		 FROM agefreighter_meta.edge_identity edge
		 JOIN agefreighter_meta.graph_generation generation
		   USING (graph_generation_id)
		 WHERE generation.graph_name = $1
		   AND generation.state = 'active'
		   AND edge.source_namespace = 'crm'
		   AND edge.external_id = 'e1'`,
		graph,
	).Scan(&state.e1GraphID, &state.e1StartGraph, &state.e1EndGraph); err != nil {
		t.Fatalf("read e1 identity: %v", err)
	}
	return state
}

func readSourceModeVertexGraphID(
	t *testing.T,
	connection *pgx.Conn,
	graph string,
	externalID string,
) int64 {
	t.Helper()
	var graphID int64
	if err := connection.QueryRow(
		t.Context(),
		`SELECT vertex.graph_id
		 FROM agefreighter_meta.vertex_identity vertex
		 JOIN agefreighter_meta.graph_generation generation
		   USING (graph_generation_id)
		 WHERE generation.graph_name = $1
		   AND generation.state = 'active'
		   AND vertex.source_namespace = 'crm'
		   AND vertex.external_id = $2`,
		graph,
		externalID,
	).Scan(&graphID); err != nil {
		t.Fatalf("read %s identity: %v", externalID, err)
	}
	return graphID
}

func readSourceModePhysicalState(
	t *testing.T,
	connection *pgx.Conn,
	graph string,
	startName string,
	endName string,
	weight string,
	quoted bool,
) sourceModeIdentityState {
	t.Helper()
	weightLiteral := weight
	if quoted {
		weightLiteral = "'" + weight + "'"
	}
	var state sourceModeIdentityState
	state.p1GraphID = readSourceModeCypherID(
		t,
		connection,
		graph,
		fmt.Sprintf(
			"MATCH (n:Person) WHERE n.name = '%s' RETURN id(n)",
			startName,
		),
	)
	statement := fmt.Sprintf(
		`SELECT edge_id::text, start_id::text, end_id::text
		 FROM ag_catalog.cypher(
		   '%s',
		   $$MATCH (a:Person)-[r:KNOWS]->(b:Person)
		     WHERE a.name = '%s' AND b.name = '%s' AND r.weight = %s
		     RETURN id(r), id(a), id(b)$$
		 ) AS result(
		   edge_id ag_catalog.agtype,
		   start_id ag_catalog.agtype,
		   end_id ag_catalog.agtype
		 )`,
		graph,
		startName,
		endName,
		weightLiteral,
	)
	var edgeID, startID, endID string
	if err := connection.QueryRow(t.Context(), statement).Scan(
		&edgeID,
		&startID,
		&endID,
	); err != nil {
		t.Fatalf("read physical e1 identity: %v", err)
	}
	state.e1GraphID = parseSourceModeGraphID(t, edgeID)
	state.e1StartGraph = parseSourceModeGraphID(t, startID)
	state.e1EndGraph = parseSourceModeGraphID(t, endID)
	return state
}

func readSourceModeCypherID(
	t *testing.T,
	connection *pgx.Conn,
	graph string,
	query string,
) int64 {
	t.Helper()
	statement := fmt.Sprintf(
		`SELECT value::text
		 FROM ag_catalog.cypher('%s', $$%s$$)
		 AS result(value ag_catalog.agtype)`,
		graph,
		query,
	)
	var value string
	if err := connection.QueryRow(t.Context(), statement).Scan(&value); err != nil {
		t.Fatalf("Cypher %q physical identity: %v", query, err)
	}
	return parseSourceModeGraphID(t, value)
}

func parseSourceModeGraphID(t *testing.T, value string) int64 {
	t.Helper()
	graphID, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		t.Fatalf("parse graph ID %q: %v", value, err)
	}
	return graphID
}

func assertSourceModeIdentityMatchesPhysical(
	t *testing.T,
	identity sourceModeIdentityState,
	physical sourceModeIdentityState,
) {
	t.Helper()
	if identity != physical {
		t.Fatalf("metadata identities = %#v, physical graph IDs = %#v", identity, physical)
	}
}
