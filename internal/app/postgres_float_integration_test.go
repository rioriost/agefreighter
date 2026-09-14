package app

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/config"
)

// Verify exact AGE property serialization, not numeric equality (74 = 74.0).
func TestPostgreSQLFloatPreservationIntegration(t *testing.T) {
	sourceDSN, targetDSN := os.Getenv("AGEFREIGHTER_POSTGRES_TEST_DSN"), os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if sourceDSN == "" || targetDSN == "" {
		t.Skip("set PostgreSQL source and AGE target test DSNs")
	}
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", targetDSN)
	t.Setenv("AGEFREIGHTER_POSTGRES_APP_TEST_DSN", sourceDSN)
	for _, mode := range []config.PostgreSQLReadMode{config.PostgreSQLReadCopy, config.PostgreSQLReadCursor, config.PostgreSQLReadKeyset} {
		t.Run(string(mode), func(t *testing.T) {
			graph := fmt.Sprintf("postgres_float_%d", time.Now().UnixNano())
			vertexQuery := "SELECT id::bigint AS id, 74::double precision AS score, 9223372036854775807::bigint AS quantity FROM generate_series(1,2) id"
			edgeQuery := "SELECT 1::bigint AS id, 1::bigint AS start_id, 2::bigint AS end_id, 38::real AS distance_km"
			key := ""
			if mode == config.PostgreSQLReadKeyset {
				vertexQuery += " WHERE ($1::bigint IS NULL OR id > $1) ORDER BY id LIMIT $2"
				edgeQuery += " WHERE ($1::bigint IS NULL OR 1 > $1) ORDER BY id LIMIT $2"
				key = "id"
			} else {
				vertexQuery += " ORDER BY id"
				edgeQuery += " ORDER BY id"
			}
			job := testLoadJob(graph, "unused", "unused")
			job.Source = config.Source{Type: config.SourcePostgreSQL, Namespace: "crm", PostgreSQL: &config.PostgreSQLSource{
				Connection: config.SecretRef{Env: "AGEFREIGHTER_POSTGRES_APP_TEST_DSN"}, ReadMode: mode, FetchRows: 1,
				Vertices: []config.VertexQuery{{Label: "Person", Query: vertexQuery, IDField: "id", KeyField: key, Properties: map[string]string{"score": "score", "quantity": "quantity"}}},
				Edges: []config.EdgeQuery{{Label: "KNOWS", Query: edgeQuery, ExternalIDField: "id", KeyField: key,
					Start: config.EndpointMapping{Label: "Person", Field: "start_id"}, End: config.EndpointMapping{Label: "Person", Field: "end_id"}, Properties: map[string]string{"distance_km": "distance_km"}}},
			}}
			path := writeLoadJob(t, t.TempDir(), "float.yaml", job)
			result, err := Load(t.Context(), path)
			if result.JobID != "" {
				registerCleanup(t, targetDSN, graph, result.JobID)
			}
			if err != nil {
				t.Fatal(err)
			}
			if result.Metrics.RecordsCommitted != 3 {
				t.Fatalf("records: %d", result.Metrics.RecordsCommitted)
			}
			if _, err := Verify(t.Context(), path, result.JobID); err != nil {
				t.Fatal(err)
			}
			conn, err := pgx.Connect(t.Context(), targetDSN)
			if err != nil {
				t.Fatal(err)
			}
			defer conn.Close(context.Background())
			if _, err := conn.Exec(t.Context(), "LOAD 'age'; SET search_path = ag_catalog, public"); err != nil {
				t.Fatal(err)
			}
			for _, item := range []struct {
				label, property, expected string
				count                     int
			}{
				{"Person", "score", "74.0", 2}, {"Person", "quantity", "9223372036854775807", 2}, {"KNOWS", "distance_km", "38.0", 1},
			} {
				var count int
				// AGE scalar extraction normalizes integral floats, so inspect the
				// serialized whole object through PostgreSQL json (not jsonb).
				query := "SELECT count(*) FROM " + pgx.Identifier{graph, item.label}.Sanitize() + " WHERE properties::text::json ->> $1 = $2"
				if err := conn.QueryRow(t.Context(), query, item.property, item.expected).Scan(&count); err != nil {
					t.Fatal(err)
				}
				if count != item.count {
					t.Fatalf("%s.%s: %d exact %s values, want %d", item.label, item.property, count, item.expected, item.count)
				}
			}
		})
	}
}
