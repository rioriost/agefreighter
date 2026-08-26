package app

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"go.yaml.in/yaml/v3"
)

func TestPostgreSQLCreateIntegration(t *testing.T) {
	sourceDSN := os.Getenv("AGEFREIGHTER_POSTGRES_TEST_DSN")
	targetDSN := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if sourceDSN == "" || targetDSN == "" {
		t.Skip("set PostgreSQL source and AGE target test DSNs")
	}
	ctx := t.Context()
	sourcePool, err := pgxpool.New(ctx, sourceDSN)
	if err != nil {
		t.Fatalf("open PostgreSQL source: %v", err)
	}
	t.Cleanup(sourcePool.Close)

	suffix := time.Now().UnixNano()
	people := fmt.Sprintf("agefreighter_people_%d", suffix)
	knows := fmt.Sprintf("agefreighter_knows_%d", suffix)
	peopleTable := pgx.Identifier{people}.Sanitize()
	knowsTable := pgx.Identifier{knows}.Sanitize()
	if _, err := sourcePool.Exec(
		ctx,
		fmt.Sprintf(
			`CREATE TABLE %s (
				person_id text PRIMARY KEY,
				full_name text NOT NULL,
				score numeric NOT NULL,
				active boolean NOT NULL,
				tags text[] NOT NULL,
				profile jsonb NOT NULL
			);
			CREATE TABLE %s (
				relationship_id text PRIMARY KEY,
				from_id text NOT NULL REFERENCES %s(person_id),
				to_id text NOT NULL REFERENCES %s(person_id),
				weight bigint NOT NULL
			);
			INSERT INTO %s VALUES
				('p1', 'Ada', 1.5, true, ARRAY['math'], '{"city":"London"}'),
				('p2', 'Grace', 2, false, ARRAY['compiler'], '{"city":"New York"}');
			INSERT INTO %s VALUES ('e1', 'p1', 'p2', 7)`,
			peopleTable,
			knowsTable,
			peopleTable,
			peopleTable,
			peopleTable,
			knowsTable,
		),
	); err != nil {
		t.Fatalf("create PostgreSQL fixture: %v", err)
	}
	t.Cleanup(func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = sourcePool.Exec(
			cleanupCtx,
			fmt.Sprintf("DROP TABLE IF EXISTS %s, %s", knowsTable, peopleTable),
		)
	})

	graph := fmt.Sprintf("postgres_e2e_%d", suffix)
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", targetDSN)
	t.Setenv("AGEFREIGHTER_POSTGRES_APP_TEST_DSN", sourceDSN)
	job := testLoadJob(graph, "unused-vertices", "unused-edges")
	job.Metadata.Name = "postgres-create"
	job.Source = config.Source{
		Type: config.SourcePostgreSQL, Namespace: "crm",
		PostgreSQL: &config.PostgreSQLSource{
			Connection: config.SecretRef{
				Env: "AGEFREIGHTER_POSTGRES_APP_TEST_DSN",
			},
			ReadMode:  config.PostgreSQLReadCopy,
			FetchRows: 2,
			Vertices: []config.VertexQuery{{
				Label:   "Person",
				Query:   fmt.Sprintf("SELECT * FROM %s ORDER BY person_id", peopleTable),
				IDField: "person_id",
				Properties: map[string]string{
					"name": "full_name", "score": "score", "active": "active",
					"tags": "tags", "profile": "profile",
				},
			}},
			Edges: []config.EdgeQuery{{
				Label: "KNOWS",
				Query: fmt.Sprintf(
					"SELECT * FROM %s ORDER BY relationship_id",
					knowsTable,
				),
				ExternalIDField: "relationship_id",
				Start: config.EndpointMapping{
					Label: "Person", Field: "from_id",
				},
				End: config.EndpointMapping{
					Label: "Person", Field: "to_id",
				},
				Properties: map[string]string{"weight": "weight"},
			}},
		},
	}
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatalf("marshal PostgreSQL job: %v", err)
	}
	path := filepath.Join(t.TempDir(), "postgres.yaml")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write PostgreSQL job: %v", err)
	}

	result, err := Load(ctx, path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	registerCleanup(t, targetDSN, graph, result.JobID)
	if result.Status != meta.JobCommitted ||
		result.Metrics.RecordsCommitted != 3 {
		t.Fatalf("Load() = %#v", result)
	}
	targetPool, err := pgxpool.New(ctx, targetDSN)
	if err != nil {
		t.Fatalf("open AGE target: %v", err)
	}
	defer targetPool.Close()
	connection, err := targetPool.Acquire(ctx)
	if err != nil {
		t.Fatalf("acquire AGE target: %v", err)
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
		"MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r)",
		1,
	)
}
