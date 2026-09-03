package app

import (
	"context"
	"errors"
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
	"github.com/rioriost/agefreighter/internal/report"
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

func TestPropertyGraphDefinitionValidation(t *testing.T) {
	base := testLoadJob("property_graph", "vertices.csv", "edges.csv")
	base.Target.Type = config.TargetPostgreSQLPropertyGraph
	base.Target.Schema = "public"
	base.Target.AppendDuplicate = ""
	for _, test := range []struct {
		name string
		edit func(*config.LoadJob)
		want string
	}{
		{"missing CSV", func(job *config.LoadJob) { job.Source.CSV = nil }, "CSV source"},
		{"missing PostgreSQL", func(job *config.LoadJob) {
			job.Source = config.Source{Type: config.SourcePostgreSQL}
		}, "PostgreSQL source"},
		{"missing Neo4j", func(job *config.LoadJob) {
			job.Source = config.Source{Type: config.SourceNeo4j}
		}, "Neo4j source"},
		{"missing Cosmos", func(job *config.LoadJob) {
			job.Source = config.Source{Type: config.SourceCosmos}
		}, "Cosmos source"},
		{"unsupported", func(job *config.LoadJob) {
			job.Source = config.Source{Type: "future"}
		}, "not implemented"},
		{"conflicting edge endpoints", func(job *config.LoadJob) {
			job.Source.CSV.Vertices = append(job.Source.CSV.Vertices,
				config.CSVVertex{Label: "Other"})
			job.Source.CSV.Edges = append(job.Source.CSV.Edges,
				config.CSVEdge{Label: "KNOWS",
					Start: config.EndpointMapping{Label: "Other"},
					End:   config.EndpointMapping{Label: "Person"}})
		}, "conflicting endpoints"},
		{"unmapped endpoint", func(job *config.LoadJob) {
			job.Source.CSV.Edges[0].Start.Label = "Missing"
		}, "unmapped endpoint"},
	} {
		t.Run(test.name, func(t *testing.T) {
			job := base
			csv := *base.Source.CSV
			csv.Vertices = append([]config.CSVVertex(nil), base.Source.CSV.Vertices...)
			csv.Edges = append([]config.CSVEdge(nil), base.Source.CSV.Edges...)
			job.Source.CSV = &csv
			test.edit(&job)
			if _, err := propertyGraphDefinition(job); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("propertyGraphDefinition() error = %v, want %q", err, test.want)
			}
		})
	}
	if _, err := propertyGraphJobVerification("job", "fingerprint",
		pggraph.Definition{}); err == nil {
		t.Fatal("propertyGraphJobVerification accepted an invalid definition")
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
		path := writeLoadJob(t, t.TempDir(), "property-graph.yaml", job)
		if _, err := Verify(t.Context(), path, jobID); err != nil {
			t.Fatalf("Verify() property graph: %v", err)
		}
		verification, err := VerificationReport(t.Context(), path, jobID,
			VerifyOptions{Counts: true, Integrity: true})
		if err != nil || verification.Outcome != report.OutcomePass {
			t.Fatalf("VerificationReport() = %#v, %v", verification, err)
		}
		migration, err := MigrationReport(t.Context(), path, jobID,
			ReportOptions{IncludeCounts: true, LimitBatches: 10})
		if err != nil || migration.Outcome != report.OutcomePass {
			t.Fatalf("MigrationReport() = %#v, %v", migration, err)
		}
		migration, err = MigrationReport(t.Context(), path, jobID, ReportOptions{})
		if err != nil || migration.Outcome != report.OutcomePass {
			t.Fatalf("MigrationReport(without counts) = %#v, %v", migration, err)
		}
		doctor, err := Doctor(t.Context(), path, DoctorOptions{Persist: true})
		if err != nil || doctor.Outcome != report.OutcomePass {
			t.Fatalf("Doctor() = %#v, %v", doctor, err)
		}
		history, err := DoctorHistory(t.Context(), path, 10, time.Time{})
		if err != nil || history.Outcome != report.OutcomePass {
			t.Fatalf("DoctorHistory() = %#v, %v", history, err)
		}
		missingJobID := "11111111-2222-4333-8444-555555555555"
		if _, err := VerificationReport(t.Context(), path, missingJobID, VerifyOptions{}); err == nil {
			t.Fatal("VerificationReport() accepted an unknown property graph job")
		}
		if _, err := MigrationReport(t.Context(), path, missingJobID, ReportOptions{}); err == nil {
			t.Fatal("MigrationReport() accepted an unknown property graph job")
		}
		wrongTarget := job
		wrongTarget.Target.Graph += "_changed"
		wrongPath := writeLoadJob(t, t.TempDir(), "wrong-target.yaml", wrongTarget)
		_, err = VerificationReport(t.Context(), wrongPath, jobID,
			VerifyOptions{Counts: true})
		if err == nil || !strings.Contains(err.Error(), "target identity changed") {
			t.Fatalf("VerificationReport(changed target) error = %v", err)
		}
		cancelled, cancel := context.WithCancel(t.Context())
		cancel()
		for name, run := range map[string]func() error{
			"verify": func() error { _, err := Verify(cancelled, path, jobID); return err },
			"verification report": func() error {
				_, err := VerificationReport(cancelled, path, jobID, VerifyOptions{Counts: true})
				return err
			},
			"migration report": func() error {
				_, err := MigrationReport(cancelled, path, jobID, ReportOptions{})
				return err
			},
			"doctor": func() error { _, err := Doctor(cancelled, path, DoctorOptions{}); return err },
			"doctor history": func() error {
				_, err := DoctorHistory(cancelled, path, 1, time.Time{})
				return err
			},
		} {
			if err := run(); err == nil {
				t.Fatalf("%s ignored context cancellation", name)
			}
		}
	})

	t.Run("empty create and resume guards", func(t *testing.T) {
		job, cleanup := propertyGraphCSVJob(t, dsn, "empty")
		defer cleanup()
		if err := os.WriteFile(job.Source.CSV.Vertices[0].Path, []byte("id,name\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(job.Source.CSV.Edges[0].Path, []byte("id,start,end\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		jobID, err := newJobID()
		if err != nil {
			t.Fatal(err)
		}
		defer cleanupPropertyGraphJob(t, dsn, jobID)
		result, err := execute(t.Context(), job, jobID, false)
		if err != nil || result.Status != meta.JobCommitted || result.Metrics.RecordsCommitted != 0 {
			t.Fatalf("empty create = %#v, %v", result, err)
		}
		if _, err := execute(t.Context(), job, jobID, true); err == nil ||
			!strings.Contains(err.Error(), "not resumable") {
			t.Fatalf("committed resume error = %v", err)
		}
		changed := job
		changed.Runtime.BatchRows++
		if _, err := execute(t.Context(), changed, jobID, true); err == nil ||
			!strings.Contains(err.Error(), "fingerprint changed") {
			t.Fatalf("changed resume error = %v", err)
		}
		wrongTarget := job
		wrongTarget.Target.Graph = "other_graph"
		if _, err := execute(t.Context(), wrongTarget, jobID, true); err == nil ||
			!strings.Contains(err.Error(), "target identity") {
			t.Fatalf("changed target resume error = %v", err)
		}
	})

	t.Run("trial and quarantine", func(t *testing.T) {
		job, cleanup := propertyGraphCSVJob(t, dsn, "trial-quarantine")
		defer cleanup()
		if err := os.WriteFile(job.Source.CSV.Vertices[0].Path,
			[]byte("id,name\np1,Alice\nbroken\np2,Bob\np3,Carol\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		job.Trial = &config.TrialOptions{
			Enabled: true, MaxVerticesPerLabel: 3, MaxVertices: 3,
			MaxEdges: 2, MaxBytes: 1 << 20,
		}
		job.Errors.MalformedRecord = config.MalformedQuarantine
		job.Errors.RejectLimit = 1
		job.Errors.QuarantinePath = filepath.Join(t.TempDir(), "rejects.jsonl")
		jobID, err := newJobID()
		if err != nil {
			t.Fatal(err)
		}
		defer cleanupPropertyGraphJob(t, dsn, jobID)
		result, err := execute(t.Context(), job, jobID, false)
		if err != nil || result.Status != meta.JobCommitted || result.Trial == nil {
			t.Fatalf("trial create = %#v, %v", result, err)
		}
		stored, err := Status(t.Context(), writeLoadJob(t, t.TempDir(), "trial.yaml", job), jobID)
		if err != nil || stored.SourceRejectedRows != 1 {
			t.Fatalf("trial rejected records = %#v, %v", stored, err)
		}
	})

	t.Run("doctor before create", func(t *testing.T) {
		job, cleanup := propertyGraphCSVJob(t, dsn, "doctor-before-create")
		defer cleanup()
		path := writeLoadJob(t, t.TempDir(), "doctor.yaml", job)
		document, err := Doctor(t.Context(), path, DoctorOptions{})
		if err != nil || document.Outcome != report.OutcomePass {
			t.Fatalf("Doctor(before create) = %#v, %v", document, err)
		}
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
		path := writeLoadJob(t, t.TempDir(), "failed.yaml", job)
		if _, err := Verify(t.Context(), path, jobID); err == nil {
			t.Fatal("Verify() accepted a failed property graph job")
		}
		if _, err := VerificationReport(t.Context(), path, jobID, VerifyOptions{}); err == nil {
			t.Fatal("VerificationReport() accepted a failed property graph job")
		}
		migration, err := MigrationReport(t.Context(), path, jobID,
			ReportOptions{IncludeCounts: true})
		if err != nil || migration.Outcome != report.OutcomeFail {
			t.Fatalf("MigrationReport(failed job) = %#v, %v", migration, err)
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

		reference := job
		reference.Target.Schema = fmt.Sprintf("af_pgq_reference_%d", time.Now().UnixNano())
		reference.Target.Graph = reference.Target.Schema
		reference.Metadata.Name = reference.Target.Schema
		referenceID, err := newJobID()
		if err != nil {
			t.Fatal(err)
		}
		defer cleanupPropertyGraphJob(t, dsn, referenceID)
		defer dropPropertyGraphSchema(t, dsn, reference.Target.Schema)
		if _, err := execute(t.Context(), reference, referenceID, false); err != nil {
			t.Fatalf("execute clean digest reference: %v", err)
		}
		resumedRoot := propertyGraphDigestRoot(t, dsn, jobID)
		cleanRoot := propertyGraphDigestRoot(t, dsn, referenceID)
		if resumedRoot != cleanRoot {
			t.Fatalf("recovered digest root %s differs from clean root %s",
				resumedRoot, cleanRoot)
		}
	})
}

func TestPostgreSQLPropertyGraphCorruptionDetectionIntegration(t *testing.T) {
	dsn := strings.TrimSpace(os.Getenv("AGEFREIGHTER_PGGRAPH_TEST_DSN"))
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_PGGRAPH_TEST_DSN to run property graph corruption tests")
	}
	t.Setenv("AGEFREIGHTER_PGGRAPH_APP_TEST_DSN", dsn)
	mutations := map[string]func(*testing.T, *pgxpool.Pool, config.LoadJob, pggraph.Definition, string){
		"changed properties": func(t *testing.T, pool *pgxpool.Pool, job config.LoadJob, definition pggraph.Definition, _ string) {
			_, err := pool.Exec(t.Context(), "UPDATE "+propertyGraphTable(job, definition.Vertices[0].Table)+
				` SET properties = '{"changed":true}'::jsonb WHERE external_id = 'p1'`)
			if err != nil {
				t.Fatal(err)
			}
		},
		"changed identity": func(t *testing.T, pool *pgxpool.Pool, job config.LoadJob, definition pggraph.Definition, _ string) {
			_, err := pool.Exec(t.Context(), "UPDATE "+propertyGraphTable(job, definition.Vertices[0].Table)+
				` SET external_id = 'changed-p1' WHERE external_id = 'p1'`)
			if err != nil {
				t.Fatal(err)
			}
		},
		"changed endpoint binding": func(t *testing.T, pool *pgxpool.Pool, job config.LoadJob, definition pggraph.Definition, _ string) {
			_, err := pool.Exec(t.Context(), "UPDATE "+propertyGraphTable(job, definition.Edges[0].Table)+
				" SET start_id = (SELECT id FROM "+propertyGraphTable(job, definition.Vertices[0].Table)+
				` WHERE external_id = 'p3') WHERE external_id = 'e1'`)
			if err != nil {
				t.Fatal(err)
			}
		},
		"deleted record": func(t *testing.T, pool *pgxpool.Pool, job config.LoadJob, definition pggraph.Definition, _ string) {
			_, err := pool.Exec(t.Context(), "DELETE FROM "+propertyGraphTable(job, definition.Edges[0].Table)+
				` WHERE external_id = 'e1'`)
			if err != nil {
				t.Fatal(err)
			}
		},
		"added record": func(t *testing.T, pool *pgxpool.Pool, job config.LoadJob, definition pggraph.Definition, _ string) {
			_, err := pool.Exec(t.Context(), "INSERT INTO "+propertyGraphTable(job, definition.Vertices[0].Table)+
				` (source_namespace, external_id, properties, digest_range, source_digest)
				 SELECT source_namespace, 'added', properties, digest_range, source_digest
				 FROM `+propertyGraphTable(job, definition.Vertices[0].Table)+` WHERE external_id = 'p1'`)
			if err != nil {
				t.Fatal(err)
			}
		},
		"missing digest baseline": func(t *testing.T, pool *pgxpool.Pool, _ config.LoadJob, _ pggraph.Definition, jobID string) {
			_, err := pool.Exec(t.Context(), `DELETE FROM agefreighter_meta.property_graph_digest_range
				WHERE job_id = $1::uuid`, jobID)
			if err != nil {
				t.Fatal(err)
			}
		},
		"changed digest baseline": func(t *testing.T, pool *pgxpool.Pool, _ config.LoadJob, _ pggraph.Definition, jobID string) {
			_, err := pool.Exec(t.Context(), `UPDATE agefreighter_meta.property_graph_digest_range
				SET digest = repeat('0', 64)
				WHERE ctid = (SELECT ctid FROM agefreighter_meta.property_graph_digest_range
					WHERE job_id = $1::uuid ORDER BY label_name, range_id LIMIT 1)`, jobID)
			if err != nil {
				t.Fatal(err)
			}
		},
		"missing digest root": func(t *testing.T, pool *pgxpool.Pool, _ config.LoadJob, _ pggraph.Definition, jobID string) {
			_, err := pool.Exec(t.Context(), `UPDATE agefreighter_meta.property_graph_generation
				SET digest_root = NULL WHERE job_id = $1::uuid`, jobID)
			if err != nil {
				t.Fatal(err)
			}
		},
		"changed digest range count": func(t *testing.T, pool *pgxpool.Pool, _ config.LoadJob, _ pggraph.Definition, jobID string) {
			_, err := pool.Exec(t.Context(), `UPDATE agefreighter_meta.property_graph_generation
				SET digest_range_count = 255 WHERE job_id = $1::uuid`, jobID)
			if err != nil {
				t.Fatal(err)
			}
		},
		"changed committed row count": func(t *testing.T, pool *pgxpool.Pool, _ config.LoadJob, _ pggraph.Definition, jobID string) {
			_, err := pool.Exec(t.Context(), `UPDATE agefreighter_meta.load_job
				SET committed_rows = committed_rows + 1 WHERE job_id = $1::uuid`, jobID)
			if err != nil {
				t.Fatal(err)
			}
		},
		"unexpected reject record": func(t *testing.T, pool *pgxpool.Pool, _ config.LoadJob, _ pggraph.Definition, jobID string) {
			_, err := pool.Exec(t.Context(), `INSERT INTO agefreighter_meta.reject_record (
				job_id, batch_id, attempt, resume_token, error_class, error_message
			)
			SELECT job_id, batch_id, attempt, 'injected-reject', 'test', 'injected'
			FROM agefreighter_meta.load_batch
			WHERE job_id = $1::uuid AND status = 'committed'
			ORDER BY batch_id LIMIT 1`, jobID)
			if err != nil {
				t.Fatal(err)
			}
		},
		"changed stored digest range": func(t *testing.T, pool *pgxpool.Pool, job config.LoadJob, definition pggraph.Definition, _ string) {
			_, err := pool.Exec(t.Context(), "UPDATE "+propertyGraphTable(job, definition.Vertices[0].Table)+
				` SET digest_range = (digest_range + 1) % 256 WHERE external_id = 'p1'`)
			if err != nil {
				t.Fatal(err)
			}
		},
		"inactive mapping": func(t *testing.T, pool *pgxpool.Pool, _ config.LoadJob, _ pggraph.Definition, jobID string) {
			_, err := pool.Exec(t.Context(), `UPDATE agefreighter_meta.property_graph_generation
				SET state = 'loading' WHERE job_id = $1::uuid`, jobID)
			if err != nil {
				t.Fatal(err)
			}
		},
		"changed mapping fingerprint": func(t *testing.T, pool *pgxpool.Pool, _ config.LoadJob, _ pggraph.Definition, jobID string) {
			_, err := pool.Exec(t.Context(), `UPDATE agefreighter_meta.property_graph_generation
				SET definition_fingerprint = repeat('0', 64) WHERE job_id = $1::uuid`, jobID)
			if err != nil {
				t.Fatal(err)
			}
		},
		"missing identity constraint": func(t *testing.T, pool *pgxpool.Pool, job config.LoadJob, definition pggraph.Definition, _ string) {
			constraint := definition.Vertices[0].Table + "_source_namespace_external_id_key"
			_, err := pool.Exec(t.Context(), "ALTER TABLE "+propertyGraphTable(job, definition.Vertices[0].Table)+
				" DROP CONSTRAINT "+pggraph.QuoteIdentifier(constraint))
			if err != nil {
				t.Fatal(err)
			}
		},
		"missing endpoint constraint": func(t *testing.T, pool *pgxpool.Pool, job config.LoadJob, definition pggraph.Definition, _ string) {
			constraint := definition.Edges[0].Table + "_start_id_fkey"
			_, err := pool.Exec(t.Context(), "ALTER TABLE "+propertyGraphTable(job, definition.Edges[0].Table)+
				" DROP CONSTRAINT "+pggraph.QuoteIdentifier(constraint))
			if err != nil {
				t.Fatal(err)
			}
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			job, cleanup := propertyGraphCSVJob(t, dsn, strings.ReplaceAll(name, " ", "-"))
			defer cleanup()
			jobID, err := newJobID()
			if err != nil {
				t.Fatal(err)
			}
			defer cleanupPropertyGraphJob(t, dsn, jobID)
			if _, err := execute(t.Context(), job, jobID, false); err != nil {
				t.Fatalf("execute corruption fixture: %v", err)
			}
			definition, err := propertyGraphDefinition(job)
			if err != nil {
				t.Fatal(err)
			}
			pool, err := pgxpool.New(t.Context(), dsn)
			if err != nil {
				t.Fatal(err)
			}
			mutate(t, pool, job, definition, jobID)
			pool.Close()
			path := writeLoadJob(t, t.TempDir(), "corrupt.yaml", job)
			if _, err := Verify(t.Context(), path, jobID); err == nil {
				t.Fatal("Verify() accepted corrupted property graph")
			}
			document, err := VerificationReport(t.Context(), path, jobID,
				VerifyOptions{Counts: true, Integrity: true})
			if err != nil {
				t.Fatalf("VerificationReport() returned operational error: %v", err)
			}
			if document.Outcome != report.OutcomeFail ||
				!hasStatus(document, report.CheckFail) {
				t.Fatalf("corruption report = %#v", document)
			}
			migration, err := MigrationReport(t.Context(), path, jobID,
				ReportOptions{IncludeCounts: true, LimitBatches: 1})
			if name == "changed mapping fingerprint" {
				if !errors.Is(err, meta.ErrGenerationMismatch) {
					t.Fatalf("MigrationReport(mapping mismatch) error = %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("MigrationReport() returned operational error: %v", err)
			}
			if migration.Outcome != report.OutcomePass && migration.Outcome != report.OutcomeFail {
				t.Fatalf("migration corruption report = %#v", migration)
			}
		})
	}
}

func propertyGraphTable(job config.LoadJob, table string) string {
	return pggraph.QuoteIdentifier(job.Target.Schema) + "." + pggraph.QuoteIdentifier(table)
}

func propertyGraphDigestRoot(t *testing.T, dsn, jobID string) string {
	t.Helper()
	pool, err := pgxpool.New(t.Context(), dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	var root string
	if err := pool.QueryRow(t.Context(), `SELECT digest_root::text
		FROM agefreighter_meta.property_graph_generation
		WHERE job_id = $1::uuid`, jobID).Scan(&root); err != nil {
		t.Fatal(err)
	}
	return root
}

func dropPropertyGraphSchema(t *testing.T, dsn, schema string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return
	}
	defer pool.Close()
	_, _ = pool.Exec(ctx, "DROP SCHEMA IF EXISTS "+pggraph.QuoteIdentifier(schema)+" CASCADE")
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
	var digestRoot string
	var digestRows int64
	var digestRangeCount int
	if err := pool.QueryRow(t.Context(), `
		SELECT state, digest_root::text, digest_rows, digest_range_count
		FROM agefreighter_meta.property_graph_generation
		WHERE job_id = $1::uuid`, jobID).Scan(
		&state, &digestRoot, &digestRows, &digestRangeCount); err != nil {
		t.Fatal(err)
	}
	if state != meta.PropertyGraphActive {
		t.Fatalf("property graph state = %q", state)
	}
	if len(digestRoot) != 64 || digestRows != wantVertices+wantEdges ||
		digestRangeCount != pggraph.DigestRangeCount {
		t.Fatalf("property graph digest = root %q, rows %d, ranges %d",
			digestRoot, digestRows, digestRangeCount)
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
