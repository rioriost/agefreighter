package app

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/config"
)

func TestPostgreSQLSourceModeMatrixIntegration(t *testing.T) {
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
				dataset text NOT NULL,
				person_id text NOT NULL,
				full_name text NOT NULL,
				score numeric NOT NULL,
				active boolean NOT NULL,
				tags text[] NOT NULL,
				profile jsonb NOT NULL,
				PRIMARY KEY (dataset, person_id)
			);
			CREATE TABLE %s (
				dataset text NOT NULL,
				relationship_id text NOT NULL,
				from_id text NOT NULL,
				to_id text NOT NULL,
				weight bigint NOT NULL,
				PRIMARY KEY (dataset, relationship_id)
			);
			INSERT INTO %s VALUES
				('create', 'p1', 'Ada Create', 1.5, true, ARRAY['math'], '{"city":"London"}'),
				('create', 'p2', 'Grace Create', 2, false, ARRAY['compiler'], '{"city":"New York"}'),
				('replace', 'p1', 'Ada Replace', 2.5, true, ARRAY['math'], '{"city":"London"}'),
				('replace', 'p2', 'Grace Replace', 3, false, ARRAY['compiler'], '{"city":"New York"}'),
				('append', 'p3', 'Katherine Append', 4.5, true, ARRAY['space'], '{"city":"Washington"}'),
				('upsert', 'p1', 'Ada Upsert', 9.5, true, ARRAY['math'], '{"city":"London"}'),
				('upsert', 'p4', 'Dorothy Upsert', 5.5, true, ARRAY['navy'], '{"city":"Arlington"}');
			INSERT INTO %s VALUES
				('create', 'e1', 'p1', 'p2', 1),
				('replace', 'e1', 'p1', 'p2', 2),
				('append', 'e2', 'p2', 'p3', 3),
				('upsert', 'e1', 'p1', 'p4', 9),
				('upsert', 'e3', 'p3', 'p4', 4)`,
			peopleTable,
			knowsTable,
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
	runSourceModeMatrix(
		t,
		targetDSN,
		t.TempDir(),
		graph,
		"postgres",
		func(mode config.LoadMode, dataset string) config.LoadJob {
			job := testLoadJob(graph, "unused-vertices", "unused-edges")
			job.Metadata.Name = "postgres-matrix-" + dataset
			job.Target.Mode = mode
			job.Source = config.Source{
				Type: config.SourcePostgreSQL, Namespace: "crm",
				PostgreSQL: &config.PostgreSQLSource{
					Connection: config.SecretRef{
						Env: "AGEFREIGHTER_POSTGRES_APP_TEST_DSN",
					},
					ReadMode: config.PostgreSQLReadCopy, FetchRows: 2,
					Vertices: []config.VertexQuery{{
						Label: "Person",
						Query: fmt.Sprintf(
							"SELECT * FROM %s WHERE dataset = '%s' ORDER BY person_id",
							peopleTable,
							dataset,
						),
						IDField: "person_id",
						Properties: map[string]string{
							"name": "full_name", "score": "score", "active": "active",
							"tags": "tags", "profile": "profile",
						},
					}},
					Edges: []config.EdgeQuery{{
						Label: "KNOWS",
						Query: fmt.Sprintf(
							"SELECT * FROM %s WHERE dataset = '%s' ORDER BY relationship_id",
							knowsTable,
							dataset,
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
			return job
		},
		sourceModeTypedExpectations{
			predicates: []string{
				"n.active = true",
				"n.score = 9.5",
				"n.tags[0] = 'math'",
				"n.profile.city = 'London'",
			},
		},
	)
}
