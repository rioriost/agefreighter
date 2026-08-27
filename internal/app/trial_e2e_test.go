package app

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"go.yaml.in/yaml/v3"
)

func TestCSVTrialMigrationIntegration(t *testing.T) {
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run app integration tests")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	dir := t.TempDir()
	vertices := filepath.Join(dir, "people.csv")
	edges := filepath.Join(dir, "knows.csv")
	if err := os.WriteFile(
		vertices,
		[]byte("id,name\np1,Ada\np2,Grace\np3,Linus\np4,Barbara\n"),
		0o600,
	); err != nil {
		t.Fatalf("write vertices: %v", err)
	}
	if err := os.WriteFile(
		edges,
		[]byte("id,start,end\ne1,p1,p2\ne2,p2,p3\ne3,p2,p1\n"),
		0o600,
	); err != nil {
		t.Fatalf("write edges: %v", err)
	}
	graph := "af_trial_" + strings.ToLower(time.Now().Format("150405000000"))
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)
	job := testLoadJob(graph, vertices, edges)
	job.Trial = &config.TrialOptions{
		Enabled:             true,
		MaxVerticesPerLabel: 2,
		MaxVertices:         2,
		MaxEdges:            1,
		MaxBytes:            1 << 20,
	}
	jobPath := filepath.Join(dir, "job.yaml")
	encoded, err := yaml.Marshal(job)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if err := os.WriteFile(jobPath, encoded, 0o600); err != nil {
		t.Fatalf("write job: %v", err)
	}

	result, err := Load(ctx, jobPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	registerCleanup(t, dsn, graph, result.JobID)

	if result.Status != meta.JobCommitted ||
		result.Metrics.RecordsCommitted != 3 ||
		result.Trial == nil ||
		result.Trial.TotalVertices != 2 ||
		result.Trial.TotalEdges != 1 ||
		result.Trial.SkippedVertices != 2 ||
		result.Trial.SkippedEdges != 2 ||
		!reflect.DeepEqual(
			result.Trial.LimitsReached,
			[]string{"maxEdges", "maxVertices", "maxVerticesPerLabel"},
		) {
		t.Fatalf("Load() = %#v", result)
	}
	if result.Trial.VerticesPerLabel["Person"] != 2 {
		t.Fatalf("trial labels = %#v", result.Trial.VerticesPerLabel)
	}
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("open Cypher pool: %v", err)
	}
	defer pool.Close()
	connection, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("acquire Cypher connection: %v", err)
	}
	defer connection.Release()
	if _, err := connection.Exec(ctx, "LOAD 'age'"); err != nil {
		t.Fatalf("load AGE extension: %v", err)
	}
	if _, err := connection.Exec(
		ctx,
		`SET search_path = ag_catalog, "$user", public`,
	); err != nil {
		t.Fatalf("set AGE search path: %v", err)
	}
	assertCypherCount(
		t,
		connection.Conn(),
		graph,
		"MATCH (n:Person) RETURN count(n)",
		2,
	)
	assertCypherCount(
		t,
		connection.Conn(),
		graph,
		"MATCH ()-[r:KNOWS]->() RETURN count(r)",
		1,
	)
	if _, err := Verify(ctx, jobPath, result.JobID); err != nil {
		t.Fatalf("Verify() error = %v", err)
	}
	if _, err := Resume(ctx, jobPath, result.JobID); err == nil ||
		!strings.Contains(err.Error(), "trial jobs cannot be resumed") {
		t.Fatalf("Resume() error = %v", err)
	}
}

func TestTrialResumeRejectedBeforeTargetConnection(t *testing.T) {
	job := testLoadJob("trial_resume", "vertices.csv", "edges.csv")
	job.Trial = &config.TrialOptions{
		Enabled:             true,
		MaxVerticesPerLabel: 1,
		MaxVertices:         1,
		MaxEdges:            1,
		MaxBytes:            1,
	}
	job.Target.Connection = config.SecretRef{
		Env: "AGEFREIGHTER_UNSET_TRIAL_TEST_DSN",
	}
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	path := filepath.Join(t.TempDir(), "trial.yaml")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write trial job: %v", err)
	}

	_, err = Resume(
		t.Context(),
		path,
		"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
	)
	if err == nil || !strings.Contains(err.Error(), "trial jobs cannot be resumed") {
		t.Fatalf("Resume() error = %v", err)
	}
}
