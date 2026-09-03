package app

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/pggraph"
)

func TestPropertyGraphDefinitionSourceMatrix(t *testing.T) {
	base := testLoadJob("property_graph", "vertices.csv", "edges.csv")
	base.Target.Type = config.TargetPostgreSQLPropertyGraph
	base.Target.Schema = "public"
	base.Target.AppendDuplicate = ""
	postgresVertices := []config.VertexQuery{{Label: "Person"}}
	postgresEdges := []config.EdgeQuery{{
		Label: "KNOWS", Start: config.EndpointMapping{Label: "Person"},
		End: config.EndpointMapping{Label: "Person"},
	}}
	tests := map[string]config.Source{
		"csv": base.Source,
		"postgresql": {
			Type: config.SourcePostgreSQL,
			PostgreSQL: &config.PostgreSQLSource{
				Vertices: postgresVertices, Edges: postgresEdges,
			},
		},
		"neo4j": {
			Type: config.SourceNeo4j,
			Neo4j: &config.Neo4jSource{
				Vertices: postgresVertices, Edges: postgresEdges,
			},
		},
		"cosmos": {
			Type: config.SourceCosmos,
			Cosmos: &config.CosmosSource{
				Vertices: []config.CosmosVertexQuery{{Label: "Person"}},
				Edges: []config.CosmosEdgeQuery{{
					Label: "KNOWS", Start: config.EndpointMapping{Label: "Person"},
					End: config.EndpointMapping{Label: "Person"},
				}},
			},
		},
	}
	var fingerprint string
	for name, source := range tests {
		job := base
		job.Source = source
		definition, err := propertyGraphDefinition(job)
		if err != nil {
			t.Fatalf("propertyGraphDefinition(%s): %v", name, err)
		}
		got, err := definition.Fingerprint()
		if err != nil {
			t.Fatalf("Fingerprint(%s): %v", name, err)
		}
		if fingerprint == "" {
			fingerprint = got
		} else if got != fingerprint {
			t.Fatalf("source-neutral definition fingerprint %s = %s, want %s", name, got, fingerprint)
		}
	}
}

func TestPostgreSQLPropertyGraphCreateAndResumeIntegration(t *testing.T) {
	dsn := strings.TrimSpace(os.Getenv("AGEFREIGHTER_PGGRAPH_TEST_DSN"))
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_PGGRAPH_TEST_DSN to run property graph app integration tests")
	}
	t.Setenv("AGEFREIGHTER_PGGRAPH_APP_TEST_DSN", dsn)
	t.Run("clean", func(t *testing.T) {
		job, cleanup := propertyGraphCSVJob(t, dsn, "clean")
		defer cleanup()
		jobID, err := newJobID()
		if err != nil {
			t.Fatal(err)
		}
		defer cleanupPropertyGraphJob(t, dsn, jobID)
		result, err := execute(t.Context(), job, jobID, false)
		if err != nil {
			t.Fatalf("execute clean property graph load: %v", err)
		}
		if result.Status != meta.JobCommitted || result.Metrics.RecordsCommitted != 5 {
			t.Fatalf("clean property graph result = %#v", result)
		}
		assertPropertyGraphLoad(t, dsn, job, jobID, 3, 2)
	})

	t.Run("missing endpoint rolls back edge batch", func(t *testing.T) {
		job, cleanup := propertyGraphCSVJob(t, dsn, "missing-endpoint")
		defer cleanup()
		if err := os.WriteFile(job.Source.CSV.Edges[0].Path,
			[]byte("id,start,end\ne1,p1,p2\ne2,p2,missing\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		job.Runtime.BatchRows = 3
		jobID, err := newJobID()
		if err != nil {
			t.Fatal(err)
		}
		defer cleanupPropertyGraphJob(t, dsn, jobID)

		_, err = execute(t.Context(), job, jobID, false)
		if err == nil || !strings.Contains(err.Error(), "resolved 1 of 2 endpoints") {
			t.Fatalf("execute missing-endpoint load error = %v", err)
		}
		definition, err := propertyGraphDefinition(job)
		if err != nil {
			t.Fatal(err)
		}
		pool, err := pgxpool.New(t.Context(), dsn)
		if err != nil {
			t.Fatal(err)
		}
		defer pool.Close()
		var vertexCount, edgeCount int64
		if err := pool.QueryRow(t.Context(), "SELECT count(*) FROM "+
			pggraph.QuoteIdentifier(job.Target.Schema)+"."+
			pggraph.QuoteIdentifier(definition.Vertices[0].Table)).Scan(&vertexCount); err != nil {
			t.Fatal(err)
		}
		if err := pool.QueryRow(t.Context(), "SELECT count(*) FROM "+
			pggraph.QuoteIdentifier(job.Target.Schema)+"."+
			pggraph.QuoteIdentifier(definition.Edges[0].Table)).Scan(&edgeCount); err != nil {
			t.Fatal(err)
		}
		if vertexCount != 3 || edgeCount != 0 {
			t.Fatalf("partially committed counts = vertices %d, edges %d", vertexCount, edgeCount)
		}
		var status meta.JobStatus
		if err := pool.QueryRow(t.Context(), `SELECT status
			FROM agefreighter_meta.load_job WHERE job_id = $1::uuid`, jobID).Scan(&status); err != nil {
			t.Fatal(err)
		}
		if status != meta.JobFailed {
			t.Fatalf("missing-endpoint job status = %q", status)
		}
	})

	t.Run("cancelled batch resumes same target", func(t *testing.T) {
		job, cleanup := propertyGraphCSVJob(t, dsn, "resume")
		defer cleanup()
		var vertices, edges strings.Builder
		vertices.WriteString("id,name\n")
		edges.WriteString("id,start,end\n")
		for index := 1; index <= 200; index++ {
			fmt.Fprintf(&vertices, "p%d,Person %d\n", index, index)
			if index > 1 {
				fmt.Fprintf(&edges, "e%d,p%d,p%d\n", index-1, index-1, index)
			}
		}
		if err := os.WriteFile(job.Source.CSV.Vertices[0].Path,
			[]byte(vertices.String()), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(job.Source.CSV.Edges[0].Path,
			[]byte(edges.String()), 0o600); err != nil {
			t.Fatal(err)
		}
		job.Runtime.BatchRows = 1
		jobID, err := newJobID()
		if err != nil {
			t.Fatal(err)
		}
		defer cleanupPropertyGraphJob(t, dsn, jobID)
		runCtx, cancel := context.WithCancel(t.Context())
		defer cancel()
		resultChannel := make(chan error, 1)
		go func() {
			_, runErr := execute(runCtx, job, jobID, false)
			resultChannel <- runErr
		}()
		pool, err := pgxpool.New(t.Context(), dsn)
		if err != nil {
			t.Fatal(err)
		}
		deadline := time.Now().Add(10 * time.Second)
		for time.Now().Before(deadline) {
			var committed int64
			err := pool.QueryRow(t.Context(), `SELECT committed_rows
				FROM agefreighter_meta.load_job WHERE job_id = $1::uuid`, jobID).Scan(&committed)
			if err == nil && committed >= 5 {
				cancel()
				break
			}
			time.Sleep(time.Millisecond)
		}
		pool.Close()
		if runErr := <-resultChannel; runErr == nil {
			t.Fatal("cancelled property graph load succeeded")
		}
		result, err := execute(t.Context(), job, jobID, true)
		if err != nil {
			t.Fatalf("resume property graph load: %v", err)
		}
		if result.Status != meta.JobCommitted {
			t.Fatalf("resumed property graph result = %#v", result)
		}
		assertPropertyGraphLoad(t, dsn, job, jobID, 200, 199)
	})
}

func cleanupPropertyGraphJob(t *testing.T, dsn, jobID string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return
	}
	defer pool.Close()
	_, _ = pool.Exec(ctx, `DELETE FROM agefreighter_meta.load_job WHERE job_id = $1::uuid`, jobID)
}

func propertyGraphCSVJob(
	t *testing.T,
	dsn string,
	suffix string,
) (config.LoadJob, func()) {
	t.Helper()
	directory := t.TempDir()
	vertices := filepath.Join(directory, "vertices.csv")
	edges := filepath.Join(directory, "edges.csv")
	if err := os.WriteFile(vertices,
		[]byte("id,name\np1,Alice\np2,Bob\np3,Carol\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(edges,
		[]byte("id,start,end\ne1,p1,p2\ne2,p2,p3\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	unique := fmt.Sprintf("af_pgq_%s_%d", suffix, time.Now().UnixNano())
	job := testLoadJob(unique, vertices, edges)
	job.Target.Type = config.TargetPostgreSQLPropertyGraph
	job.Target.Schema = unique
	job.Target.Connection = config.SecretRef{Env: "AGEFREIGHTER_PGGRAPH_APP_TEST_DSN"}
	job.Target.AppendDuplicate = ""
	job.Runtime.BatchRows = 3
	cleanup := func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		pool, err := pgxpool.New(ctx, dsn)
		if err != nil {
			return
		}
		defer pool.Close()
		_, _ = pool.Exec(ctx, "DROP SCHEMA IF EXISTS "+pggraph.QuoteIdentifier(unique)+" CASCADE")
	}
	return job, cleanup
}

func assertPropertyGraphLoad(
	t *testing.T,
	dsn string,
	job config.LoadJob,
	jobID string,
	wantVertices int64,
	wantEdges int64,
) {
	t.Helper()
	definition, err := propertyGraphDefinition(job)
	if err != nil {
		t.Fatal(err)
	}
	pool, err := pgxpool.New(t.Context(), dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	var state meta.PropertyGraphState
	if err := pool.QueryRow(t.Context(), `
		SELECT state FROM agefreighter_meta.property_graph_generation
		WHERE job_id = $1::uuid`, jobID).Scan(&state); err != nil {
		t.Fatal(err)
	}
	if state != meta.PropertyGraphActive {
		t.Fatalf("property graph state = %q", state)
	}
	var vertices, edges int64
	if err := pool.QueryRow(t.Context(), "SELECT count(*) FROM "+
		pggraph.QuoteIdentifier(job.Target.Schema)+"."+
		pggraph.QuoteIdentifier(definition.Vertices[0].Table)).Scan(&vertices); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(t.Context(), "SELECT count(*) FROM "+
		pggraph.QuoteIdentifier(job.Target.Schema)+"."+
		pggraph.QuoteIdentifier(definition.Edges[0].Table)).Scan(&edges); err != nil {
		t.Fatal(err)
	}
	if vertices != wantVertices || edges != wantEdges {
		t.Fatalf("property graph table counts = %d, %d", vertices, edges)
	}
	connection, err := pool.Acquire(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Release()
	if _, err := connection.Exec(t.Context(), "SET search_path TO "+
		pggraph.QuoteIdentifier(job.Target.Schema)+", pg_catalog"); err != nil {
		t.Fatal(err)
	}
	var matches int64
	query := fmt.Sprintf(`SELECT count(*) FROM GRAPH_TABLE (
		%s MATCH (a IS %s)-[e IS %s]->(b IS %s)
		COLUMNS (a.external_id AS source, b.external_id AS target)
	)`, pggraph.QuoteIdentifier(job.Target.Graph),
		pggraph.QuoteIdentifier(definition.Vertices[0].Label),
		pggraph.QuoteIdentifier(definition.Edges[0].Label),
		pggraph.QuoteIdentifier(definition.Vertices[0].Label))
	if err := connection.QueryRow(t.Context(), query).Scan(&matches); err != nil {
		t.Fatal(err)
	}
	if matches != wantEdges {
		t.Fatalf("GRAPH_TABLE matches = %d, want %d", matches, wantEdges)
	}
}
