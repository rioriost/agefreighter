package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/report"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	sourcecsv "github.com/rioriost/agefreighter/internal/source/csv"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.yaml.in/yaml/v3"
)

type limitedIterator struct {
	sourcecontract.Iterator
	remaining int
}

func (iterator *limitedIterator) Next(
	ctx context.Context,
) (sourcecontract.Item, error) {
	if iterator.remaining == 0 {
		return sourcecontract.Item{}, io.EOF
	}
	item, err := iterator.Iterator.Next(ctx)
	if err == nil {
		iterator.remaining--
	}
	return item, err
}

func TestLoadCSVIntegration(t *testing.T) {
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run app integration tests")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	dir := t.TempDir()
	vertices := filepath.Join(dir, "people.csv")
	edges := filepath.Join(dir, "knows.csv")
	if err := os.WriteFile(vertices, []byte("id,name\np1,Ada\nbroken\np2,Grace\n"), 0o600); err != nil {
		t.Fatalf("write vertices: %v", err)
	}
	if err := os.WriteFile(edges, []byte("id,start,end\ne1,p1,p2\n"), 0o600); err != nil {
		t.Fatalf("write edges: %v", err)
	}
	graphName := "af_app_" + strings.ToLower(time.Now().Format("150405000000"))
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)
	job := testLoadJob(graphName, vertices, edges)
	job.Errors.MalformedRecord = config.MalformedQuarantine
	job.Errors.RejectLimit = 1
	job.Errors.QuarantinePath = filepath.Join(dir, "rejects.jsonl")
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

	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		if adapter, openErr := age.Open(cleanupCtx, dsn, age.PoolOptions{
			MinConnections: 1, MaxConnections: 2,
			ConnectTimeout: time.Second, OperationTimeout: 5 * time.Second,
		}); openErr == nil {
			_ = adapter.InTransaction(cleanupCtx, func(tx *age.Transaction) error {
				return tx.DropGraph(cleanupCtx, graphName, true)
			})
			adapter.Close()
		}
		pool, poolErr := pgxpool.New(cleanupCtx, dsn)
		if poolErr == nil {
			_ = deleteAppTestJob(cleanupCtx, pool, result.JobID)
			pool.Close()
		}
	})
	if result.Status != meta.JobCommitted ||
		result.Metrics.RecordsCommitted != 3 {
		t.Fatalf("Load() = %#v", result)
	}
	status, err := Status(ctx, jobPath, result.JobID)
	if err != nil || status.Status != meta.JobCommitted ||
		status.CommittedRows != 3 || status.RejectedRows != 1 {
		t.Fatalf("Status() = %#v, %v", status, err)
	}
	if _, err := Verify(ctx, jobPath, result.JobID); err != nil {
		t.Fatalf("Verify() error = %v", err)
	}
	statsPool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("open optimizer statistics check: %v", err)
	}
	defer statsPool.Close()
	manualAnalyzeCount := func() int {
		var count int
		if err := statsPool.QueryRow(ctx, `
			SELECT COUNT(*)::integer
			FROM pg_catalog.pg_stat_user_tables
			WHERE schemaname = $1
			  AND last_analyze IS NOT NULL`,
			graphName,
		).Scan(&count); err != nil {
			t.Fatalf("read optimizer statistics check: %v", err)
		}
		return count
	}
	beforeRecommendationOnly := manualAnalyzeCount()
	optimizer, err := OptimizationReport(
		ctx,
		jobPath,
		OptimizeOptions{},
	)
	if err != nil || optimizer.Command != "optimize" {
		t.Fatalf("OptimizationReport() = %#v, %v", optimizer, err)
	}
	var recommendations string
	for _, section := range optimizer.Sections {
		if section.Title != "Recommendations" {
			continue
		}
		for _, field := range section.Fields {
			recommendations += field.Value + "\n"
		}
	}
	if strings.Contains(recommendations, "property-index") ||
		strings.Contains(recommendations, "expression-index") {
		t.Fatalf("property recommendation emitted without evidence: %q", recommendations)
	}
	propertyUnavailable := false
	for _, check := range optimizer.Checks {
		if check.ID == "property-statistics" &&
			check.Status == report.CheckUnavailable &&
			check.Detail == propertyEvidenceUnavailable {
			propertyUnavailable = true
		}
	}
	if !propertyUnavailable {
		t.Fatalf("property unavailable evidence = %#v", optimizer.Checks)
	}
	if after := manualAnalyzeCount(); after != beforeRecommendationOnly {
		t.Fatalf(
			"recommendation-only optimizer changed analyze statistics: before=%d after=%d",
			beforeRecommendationOnly,
			after,
		)
	}
	optimizer, err = OptimizationReport(
		ctx,
		jobPath,
		OptimizeOptions{Analyze: true},
	)
	if err != nil || optimizer.Command != "optimize" {
		t.Fatalf("OptimizationReport(--apply-analyze) = %#v, %v", optimizer, err)
	}
	if after := manualAnalyzeCount(); after <= beforeRecommendationOnly {
		t.Fatalf(
			"explicit optimizer ANALYZE did not update graph label statistics: before=%d after=%d",
			beforeRecommendationOnly,
			after,
		)
	}
	if _, err := Verify(
		ctx, jobPath, "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
	); !errors.Is(err, meta.ErrNotFound) {
		t.Fatalf("Verify(missing job) error = %v", err)
	}
	if _, err := Resume(ctx, jobPath, result.JobID); err == nil ||
		!strings.Contains(err.Error(), "not failed") {
		t.Fatalf("Resume(committed) error = %v", err)
	}
	changed := job
	changed.Metadata.Name = "changed"
	changedPath := filepath.Join(dir, "changed.yaml")
	changedYAML, err := yaml.Marshal(changed)
	if err != nil {
		t.Fatalf("Marshal(changed) error = %v", err)
	}
	if err := os.WriteFile(changedPath, changedYAML, 0o600); err != nil {
		t.Fatalf("write changed job: %v", err)
	}
	if _, err := Verify(ctx, changedPath, result.JobID); err == nil ||
		!strings.Contains(err.Error(), "fingerprint changed") {
		t.Fatalf("Verify(changed) error = %v", err)
	}
	for index, status := range []meta.JobStatus{meta.JobPending, meta.JobRunning} {
		resumedJob := job
		resumedJob.Target.Graph = fmt.Sprintf("app_resume_%d_%d", time.Now().UnixNano(), index)
		resumedPath := filepath.Join(dir, fmt.Sprintf("resume-%d.yaml", index))
		resumedYAML, marshalErr := yaml.Marshal(resumedJob)
		if marshalErr != nil {
			t.Fatalf("Marshal(resume %s) error = %v", status, marshalErr)
		}
		if writeErr := os.WriteFile(resumedPath, resumedYAML, 0o600); writeErr != nil {
			t.Fatalf("write resume %s job: %v", status, writeErr)
		}
		resumeID, idErr := newJobID()
		if idErr != nil {
			t.Fatalf("newJobID(resume %s) error = %v", status, idErr)
		}
		adapter, store, openErr := openTarget(ctx, resumedJob)
		if openErr != nil {
			t.Fatalf("openTarget(resume %s) error = %v", status, openErr)
		}
		fingerprint, fingerprintErr := jobFingerprint(resumedJob)
		if fingerprintErr != nil {
			adapter.Close()
			t.Fatalf("jobFingerprint(resume %s) error = %v", status, fingerprintErr)
		}
		if createErr := store.CreateJob(ctx, meta.Job{
			ID: resumeID, Name: resumedJob.Metadata.Name,
			SourceType: string(resumedJob.Source.Type), LoadMode: string(resumedJob.Target.Mode),
			TargetGraph: resumedJob.Target.Graph, ConfigFingerprint: fingerprint,
		}); createErr != nil {
			adapter.Close()
			t.Fatalf("CreateJob(resume %s) error = %v", status, createErr)
		}
		if status == meta.JobRunning {
			if startErr := store.StartJob(ctx, resumeID); startErr != nil {
				adapter.Close()
				t.Fatalf("StartJob(resume running) error = %v", startErr)
			}
		}
		adapter.Close()
		registerCleanup(t, dsn, resumedJob.Target.Graph, resumeID)
		if status == meta.JobPending {
			if _, verifyErr := Verify(ctx, resumedPath, resumeID); verifyErr == nil ||
				!strings.Contains(verifyErr.Error(), "not committed") {
				t.Fatalf("Verify(pending) error = %v", verifyErr)
			}
		}
		resumed, resumeErr := Resume(ctx, resumedPath, resumeID)
		if resumeErr != nil || resumed.Status != meta.JobCommitted {
			t.Fatalf("Resume(%s) = %#v, %v", status, resumed, resumeErr)
		}
		pool, poolErr := pgxpool.New(ctx, dsn)
		if poolErr != nil {
			t.Fatalf("open mutation pool: %v", poolErr)
		}
		if index == 0 {
			if _, mutationErr := pool.Exec(ctx, `
				UPDATE agefreighter_meta.graph_generation
				SET graph_oid = (graph_oid::bigint + 1)::oid,
				    namespace_oid = (namespace_oid::bigint + 1)::oid
				WHERE job_id = $1::uuid`, resumeID); mutationErr != nil {
				pool.Close()
				t.Fatalf("mutate graph generation: %v", mutationErr)
			}
			if _, verifyErr := Verify(ctx, resumedPath, resumeID); verifyErr == nil ||
				!strings.Contains(verifyErr.Error(), "graph generation") {
				pool.Close()
				t.Fatalf("Verify(graph mismatch) error = %v", verifyErr)
			}
			if _, restoreErr := pool.Exec(ctx, `
				UPDATE agefreighter_meta.graph_generation
				SET graph_oid = (graph_oid::bigint - 1)::oid,
				    namespace_oid = (namespace_oid::bigint - 1)::oid
				WHERE job_id = $1::uuid`, resumeID); restoreErr != nil {
				pool.Close()
				t.Fatalf("restore graph generation: %v", restoreErr)
			}
			mutationAdapter, mutationErr := age.Open(ctx, dsn, age.PoolOptions{
				MinConnections: 1, MaxConnections: 2,
				ConnectTimeout: time.Second, OperationTimeout: 5 * time.Second,
			})
			if mutationErr != nil {
				pool.Close()
				t.Fatalf("open label mutation adapter: %v", mutationErr)
			}
			mutationErr = mutationAdapter.InTransaction(ctx, func(tx *age.Transaction) error {
				if dropErr := tx.DropLabel(ctx, resumedJob.Target.Graph, "KNOWS", false); dropErr != nil {
					return dropErr
				}
				return tx.CreateLabel(ctx, resumedJob.Target.Graph, "KNOWS", age.EdgeLabel)
			})
			mutationAdapter.Close()
			if mutationErr != nil {
				pool.Close()
				t.Fatalf("recreate AGE label: %v", mutationErr)
			}
			if _, verifyErr := Verify(ctx, resumedPath, resumeID); verifyErr == nil ||
				!strings.Contains(verifyErr.Error(), "label generation") {
				pool.Close()
				t.Fatalf("Verify(label mismatch) error = %v", verifyErr)
			}
		} else {
			mutationAdapter, mutationErr := age.Open(ctx, dsn, age.PoolOptions{
				MinConnections: 1, MaxConnections: 2,
				ConnectTimeout: time.Second, OperationTimeout: 5 * time.Second,
			})
			if mutationErr != nil {
				pool.Close()
				t.Fatalf("open catalog removal adapter: %v", mutationErr)
			}
			mutationErr = mutationAdapter.InTransaction(ctx, func(tx *age.Transaction) error {
				return tx.DropLabel(ctx, resumedJob.Target.Graph, "KNOWS", false)
			})
			if mutationErr != nil {
				mutationAdapter.Close()
				pool.Close()
				t.Fatalf("drop AGE label: %v", mutationErr)
			}
			if _, verifyErr := Verify(ctx, resumedPath, resumeID); verifyErr == nil ||
				!strings.Contains(verifyErr.Error(), "label catalog") {
				mutationAdapter.Close()
				pool.Close()
				t.Fatalf("Verify(missing label) error = %v", verifyErr)
			}
			if mutationErr = mutationAdapter.InTransaction(ctx, func(tx *age.Transaction) error {
				return tx.DropGraph(ctx, resumedJob.Target.Graph, true)
			}); mutationErr != nil {
				mutationAdapter.Close()
				pool.Close()
				t.Fatalf("drop AGE graph: %v", mutationErr)
			}
			mutationAdapter.Close()
			if _, verifyErr := Verify(ctx, resumedPath, resumeID); verifyErr == nil ||
				!strings.Contains(verifyErr.Error(), "graph catalog") {
				pool.Close()
				t.Fatalf("Verify(missing graph) error = %v", verifyErr)
			}
		}
		pool.Close()
	}
	duplicate, err := Load(ctx, jobPath)
	if err == nil || duplicate.JobID == "" {
		t.Fatalf("Load(duplicate graph) = %#v, %v", duplicate, err)
	}
	registerCleanup(t, dsn, "missing_duplicate_graph", duplicate.JobID)
}

func TestResumeAfterPreBatchFailureIntegration(t *testing.T) {
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run app integration tests")
	}

	t.Run("target and catalog failures", func(t *testing.T) {
		dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
		if dsn == "" {
			t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run app integration tests")
		}
		dir := t.TempDir()
		t.Setenv("AGEFREIGHTER_BAD_APP_DSN", "not-a-postgresql-dsn")
		badTarget := testLoadJob("bad_target_graph", "vertices.csv", "edges.csv")
		badTarget.Target.Connection.Env = "AGEFREIGHTER_BAD_APP_DSN"
		data, err := yaml.Marshal(badTarget)
		if err != nil {
			t.Fatalf("Marshal(bad target) error = %v", err)
		}
		path := filepath.Join(dir, "bad-target.yaml")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatalf("write bad target job: %v", err)
		}
		for name, run := range map[string]func() error{
			"load": func() error {
				_, runErr := Load(t.Context(), path)
				return runErr
			},
			"resume": func() error {
				_, runErr := Resume(
					t.Context(), path, "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
				)
				return runErr
			},
			"status": func() error {
				_, runErr := Status(
					t.Context(), path, "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
				)
				return runErr
			},
			"verify": func() error {
				_, runErr := Verify(
					t.Context(), path, "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
				)
				return runErr
			},
		} {
			if runErr := run(); runErr == nil {
				t.Fatalf("%s accepted an invalid target DSN", name)
			}
		}

		adapter, err := age.Open(t.Context(), dsn, age.PoolOptions{
			MinConnections: 1, MaxConnections: 2,
			ConnectTimeout: time.Second, OperationTimeout: 5 * time.Second,
		})
		if err != nil {
			t.Fatalf("open catalog test adapter: %v", err)
		}
		defer adapter.Close()
		invalidCatalog := testLoadJob("invalid_catalog", "vertices.csv", "edges.csv")
		invalidCatalog.Source.CSV.Edges[0].Label = "Person"
		if _, _, err := createCatalog(
			t.Context(), adapter, invalidCatalog, "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
		); err == nil {
			t.Fatal("createCatalog() accepted conflicting labels")
		}
		cancelled, cancel := context.WithCancel(t.Context())
		cancel()
		if _, _, err := createCatalog(
			cancelled,
			adapter,
			testLoadJob("cancelled_catalog", "vertices.csv", "edges.csv"),
			"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
		); !errors.Is(err, context.Canceled) {
			t.Fatalf("createCatalog(cancelled) error = %v", err)
		}
	})
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	dir := t.TempDir()
	vertices := filepath.Join(dir, "people.csv")
	edges := filepath.Join(dir, "knows.csv")
	if err := os.WriteFile(vertices, []byte("id,name\np1,Ada\np2,Grace\n"), 0o600); err != nil {
		t.Fatalf("write vertices: %v", err)
	}
	if err := os.WriteFile(edges, []byte("id,start,end\ne1,p1,p2\n"), 0o600); err != nil {
		t.Fatalf("write edges: %v", err)
	}
	graphName := "af_resume_" + strings.ToLower(time.Now().Format("150405000000"))
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)
	job := testLoadJob(graphName, vertices, edges)
	job.Errors.MalformedRecord = config.MalformedQuarantine
	job.Errors.QuarantinePath = filepath.Join(dir, "missing", "rejects.jsonl")
	jobPath := filepath.Join(dir, "resume.yaml")
	encoded, err := yaml.Marshal(job)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if err := os.WriteFile(jobPath, encoded, 0o600); err != nil {
		t.Fatalf("write job: %v", err)
	}
	failed, err := Load(ctx, jobPath)
	if err == nil || failed.JobID == "" {
		t.Fatalf("Load() = %#v, %v; want resumable failure", failed, err)
	}
	registerCleanup(t, dsn, graphName, failed.JobID)
	if err := os.Mkdir(filepath.Dir(job.Errors.QuarantinePath), 0o700); err != nil {
		t.Fatalf("Mkdir() error = %v", err)
	}
	resumed, err := Resume(ctx, jobPath, failed.JobID)
	if err != nil {
		t.Fatalf("Resume() error = %v", err)
	}
	if resumed.JobID != failed.JobID || resumed.Status != meta.JobCommitted ||
		resumed.Metrics.RecordsCommitted != 3 {
		t.Fatalf("Resume() = %#v", resumed)
	}
}

func TestRunningAndFailedBatchResumeIntegration(t *testing.T) {
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run app integration tests")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	dir := t.TempDir()
	vertices := filepath.Join(dir, "people.csv")
	edges := filepath.Join(dir, "knows.csv")
	if err := os.WriteFile(vertices, []byte("id,name\np1,Ada\np2,Grace\n"), 0o600); err != nil {
		t.Fatalf("write vertices: %v", err)
	}
	if err := os.WriteFile(edges, []byte("id,start,end\ne1,p1,p2\n"), 0o600); err != nil {
		t.Fatalf("write edges: %v", err)
	}
	graphName := fmt.Sprintf("app_attempt_%d", time.Now().UnixNano())
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)
	job := testLoadJob(graphName, vertices, edges)
	jobPath := filepath.Join(dir, "job.yaml")
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if err := os.WriteFile(jobPath, data, 0o600); err != nil {
		t.Fatalf("write job: %v", err)
	}
	jobID, err := newJobID()
	if err != nil {
		t.Fatalf("newJobID() error = %v", err)
	}
	registerCleanup(t, dsn, graphName, jobID)
	adapter, store, err := openTarget(ctx, job)
	if err != nil {
		t.Fatalf("openTarget() error = %v", err)
	}
	fingerprint, err := jobFingerprint(job)
	if err != nil {
		adapter.Close()
		t.Fatalf("jobFingerprint() error = %v", err)
	}
	if err := store.CreateJob(ctx, meta.Job{
		ID: jobID, Name: job.Metadata.Name,
		SourceType: string(job.Source.Type), LoadMode: string(job.Target.Mode),
		TargetGraph: graphName, ConfigFingerprint: fingerprint,
	}); err != nil {
		adapter.Close()
		t.Fatalf("CreateJob() error = %v", err)
	}
	if err := store.StartJob(ctx, jobID); err != nil {
		adapter.Close()
		t.Fatalf("StartJob() error = %v", err)
	}
	if _, _, err := createCatalog(ctx, adapter, job, jobID); err != nil {
		adapter.Close()
		t.Fatalf("createCatalog() error = %v", err)
	}
	attempt := meta.BatchAttempt{
		JobID: jobID, BatchID: 1, Attempt: 1, Rows: 1, Bytes: 1,
		First: meta.Position{Resource: vertices, Line: 2, Token: "different-input"},
	}
	if _, err := store.StartBatch(ctx, attempt); err != nil {
		adapter.Close()
		t.Fatalf("StartBatch() error = %v", err)
	}
	adapter.Close()

	if _, err := Resume(ctx, jobPath, jobID); !errors.Is(err, meta.ErrConflict) {
		t.Fatalf("Resume(running conflict) error = %v", err)
	}
	status, err := Status(ctx, jobPath, jobID)
	if err != nil || status.Status != meta.JobRunning {
		t.Fatalf("Status(after conflict) = %#v, %v", status, err)
	}
	adapter, store, err = openTarget(ctx, job)
	if err != nil {
		t.Fatalf("reopen target: %v", err)
	}
	if err := store.RecordFailedBatch(ctx, attempt, "simulated hard crash"); err != nil {
		adapter.Close()
		t.Fatalf("RecordFailedBatch() error = %v", err)
	}
	adapter.Close()
	resumed, err := Resume(ctx, jobPath, jobID)
	if err != nil || resumed.Status != meta.JobCommitted {
		t.Fatalf("Resume(failed attempt) = %#v, %v", resumed, err)
	}
}

func TestResumeAfterCommittedBatchIntegration(t *testing.T) {
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
		[]byte("id,name\np1,Ada\np2,Grace\np3,Linus\n"),
		0o600,
	); err != nil {
		t.Fatalf("write vertices: %v", err)
	}
	if err := os.WriteFile(
		edges,
		[]byte("id,start,end\ne1,p1,p3\n"),
		0o600,
	); err != nil {
		t.Fatalf("write edges: %v", err)
	}
	graphName := fmt.Sprintf("app_boundary_%d", time.Now().UnixNano())
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)
	job := testLoadJob(graphName, vertices, edges)
	job.Runtime.BatchRows = 2
	jobPath := filepath.Join(dir, "job.yaml")
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if err := os.WriteFile(jobPath, data, 0o600); err != nil {
		t.Fatalf("write job: %v", err)
	}
	jobID, err := newJobID()
	if err != nil {
		t.Fatalf("newJobID() error = %v", err)
	}
	registerCleanup(t, dsn, graphName, jobID)
	adapter, store, err := openTarget(ctx, job)
	if err != nil {
		t.Fatalf("openTarget() error = %v", err)
	}
	fingerprint, err := jobFingerprint(job)
	if err != nil {
		adapter.Close()
		t.Fatalf("jobFingerprint() error = %v", err)
	}
	if err := store.CreateJob(ctx, meta.Job{
		ID: jobID, Name: job.Metadata.Name,
		SourceType: string(job.Source.Type), LoadMode: string(job.Target.Mode),
		TargetGraph: graphName, ConfigFingerprint: fingerprint,
	}); err != nil {
		adapter.Close()
		t.Fatalf("CreateJob() error = %v", err)
	}
	if err := store.StartJob(ctx, jobID); err != nil {
		adapter.Close()
		t.Fatalf("StartJob() error = %v", err)
	}
	graph, labels, err := createCatalog(ctx, adapter, job, jobID)
	if err != nil {
		adapter.Close()
		t.Fatalf("createCatalog() error = %v", err)
	}
	iterator, err := sourcecsv.NewIterator(ctx, sourcecsv.IteratorOptions{
		Namespace:           job.Source.Namespace,
		Source:              *job.Source.CSV,
		PreencodeProperties: true,
		OptimizeRFC4180:     true,
	})
	if err != nil {
		adapter.Close()
		t.Fatalf("NewIterator() error = %v", err)
	}
	runner, err := newPipelineRunner(job, 1, 1)
	if err != nil {
		_ = iterator.Close()
		adapter.Close()
		t.Fatalf("newPipelineRunner() error = %v", err)
	}
	target, err := age.NewLoadSink(ctx, adapter, age.LoadSinkOptions{
		JobID: jobID, Graph: graph, Labels: labels,
		MissingEndpoint: job.Errors.MissingEndpoint,
	})
	if err != nil {
		_ = iterator.Close()
		adapter.Close()
		t.Fatalf("NewLoadSink() error = %v", err)
	}
	if err := runner.Run(ctx, &limitedIterator{
		Iterator:  iterator,
		remaining: 2,
	}, target); err != nil {
		adapter.Close()
		t.Fatalf("first batch Run() error = %v", err)
	}
	stored, err := store.GetJob(ctx, jobID)
	if err != nil || stored.NextBatchID != 2 || stored.CommittedRows != 2 ||
		stored.Status != meta.JobRunning || stored.ResumeToken == "" {
		adapter.Close()
		t.Fatalf("job after first batch = %#v, %v", stored, err)
	}
	adapter.Close()

	resumed, err := Resume(ctx, jobPath, jobID)
	if err != nil || resumed.Status != meta.JobCommitted ||
		resumed.Metrics.RecordsCommitted != 2 {
		t.Fatalf("Resume() = %#v, %v", resumed, err)
	}
	committed, err := Status(ctx, jobPath, jobID)
	if err != nil || committed.CommittedRows != 4 ||
		committed.NextBatchID != 3 {
		t.Fatalf("committed job = %#v, %v", committed, err)
	}
}

func TestAppHelpers(t *testing.T) {
	t.Setenv("APP_SECRET", "postgres://example")
	if value, err := resolveSecret(config.SecretRef{Env: "APP_SECRET"}); err != nil ||
		value != "postgres://example" {
		t.Fatalf("resolveSecret(env) = %q, %v", value, err)
	}

	if _, err := resolveSecret(config.SecretRef{Env: "MISSING_APP_SECRET"}); err == nil {
		t.Fatal("resolveSecret() accepted missing environment variable")
	}
	path := filepath.Join(t.TempDir(), "secret")
	if err := os.WriteFile(path, []byte("value\r\n"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	if value, err := resolveSecret(config.SecretRef{File: path}); err != nil ||
		value != "value" {
		t.Fatalf("resolveSecret(file) = %q, %v", value, err)
	}
	if _, err := resolveSecret(config.SecretRef{File: "missing"}); err == nil {
		t.Fatal("resolveSecret() accepted missing file")
	}
	empty := filepath.Join(t.TempDir(), "empty")
	if err := os.WriteFile(empty, nil, 0o600); err != nil {
		t.Fatalf("WriteFile(empty) error = %v", err)
	}
	if _, err := resolveSecret(config.SecretRef{File: empty}); err == nil {
		t.Fatal("resolveSecret() accepted empty file")
	}
	large := filepath.Join(t.TempDir(), "large")
	if err := os.WriteFile(large, make([]byte, maxSecretBytes+1), 0o600); err != nil {
		t.Fatalf("WriteFile(large) error = %v", err)
	}
	if _, err := resolveSecret(config.SecretRef{File: large}); err == nil {
		t.Fatal("resolveSecret() accepted oversized file")
	}
	first, err := newJobID()
	if err != nil {
		t.Fatalf("newJobID() error = %v", err)
	}
	if len(first) != 36 || first != strings.ToLower(first) {
		t.Fatalf("newJobID() = %q", first)
	}
	job := testLoadJob("graph", "vertices", "edges")
	left, err := jobFingerprint(job)
	if err != nil {
		t.Fatalf("jobFingerprint() error = %v", err)
	}
	right, _ := jobFingerprint(job)
	if left != right || len(left) != 64 {
		t.Fatalf("job fingerprints = %q, %q", left, right)
	}
	if _, err := configuredLabels(job); err != nil {
		t.Fatalf("configuredLabels() error = %v", err)
	}
	job.Source.CSV.Edges[0].Label = "Person"
	if _, err := configuredLabels(job); err == nil {
		t.Fatal("configuredLabels() accepted vertex/edge collision")
	}
	job = testLoadJob("graph", "vertices", "edges")
	job.Source.CSV.Edges[0].Start.Label = "Missing"
	if _, err := configuredLabels(job); err == nil {
		t.Fatal("configuredLabels() accepted missing start label")
	}
	job = testLoadJob("graph", "vertices", "edges")
	job.Source.CSV.Edges[0].End.Label = "Missing"
	if _, err := configuredLabels(job); err == nil {
		t.Fatal("configuredLabels() accepted missing end label")
	}
	job = testLoadJob("graph", "vertices", "edges")
	job.Source.CSV.Edges[0].Label = "Person"
	if _, _, err := admitIncrementalCatalog(
		t.Context(),
		nil,
		job,
		first,
	); err == nil {
		t.Fatal("admitIncrementalCatalog() accepted conflicting labels")
	}
	if _, err := Load(t.Context(), "missing.yaml"); err == nil {
		t.Fatal("Load() accepted missing configuration")
	}
	if _, err := Resume(t.Context(), "missing.yaml", "bad"); err == nil {
		t.Fatal("Resume() accepted missing configuration")
	}
	if _, err := Status(t.Context(), "missing.yaml", "bad"); err == nil {
		t.Fatal("Status() accepted missing configuration")
	}
	if _, err := Verify(t.Context(), "missing.yaml", "bad"); err == nil {
		t.Fatal("Verify() accepted missing configuration")
	}
	if _, err := Cleanup(t.Context(), "missing.yaml", "bad"); err == nil {
		t.Fatal("Cleanup() accepted missing configuration")
	}
	missingTarget := testLoadJob("graph", "vertices", "edges")
	missingTarget.Target.Connection.Env = "MISSING_TARGET_DSN"
	if _, _, err := openTarget(t.Context(), missingTarget); err == nil {
		t.Fatal("openTarget() accepted missing target secret")
	}
	job.Source.Type = config.SourceType("unsupported")
	if _, err := execute(t.Context(), job, first, false); err == nil {
		t.Fatal("execute() accepted unsupported source")
	}
	job = testLoadJob("graph", "vertices", "edges")
	job.Target.Mode = config.LoadAppend
	if _, err := execute(t.Context(), job, first, false); err == nil {
		t.Fatal("execute() accepted unsupported mode")
	}
	if name, err := loadGraphName(job, first); err != nil || name != "graph" {
		t.Fatalf("loadGraphName(append) = %q, %v", name, err)
	}
	job.Target.Mode = config.LoadUpsert
	if name, err := loadGraphName(job, first); err != nil || name != "graph" {
		t.Fatalf("loadGraphName(upsert) = %q, %v", name, err)
	}
	job.Target.Mode = config.LoadMode("unsupported")
	if _, err := loadGraphName(job, first); err == nil {
		t.Fatal("loadGraphName() accepted unsupported mode")
	}
	job.Target.Mode = config.LoadReplace
	job.Target.Graph = "x"
	if err := promoteReplace(
		t.Context(),
		nil,
		job,
		first,
		meta.GraphGeneration{},
	); err == nil {
		t.Fatal("promoteReplace() accepted an invalid target name")
	}
	job = testLoadJob("graph", "vertices", "edges")
	job.Errors.MissingEndpoint = config.MissingEndpointDefer
	if _, err := execute(t.Context(), job, first, false); err == nil {
		t.Fatal("execute() accepted deferred missing endpoints")
	}
	job = testLoadJob("graph", "vertices", "edges")
	job.Runtime.MemoryLimit = 128
	job.Runtime.BatchBytes = 64
	job.Runtime.BatchRows = 2
	if _, err := execute(t.Context(), job, first, false); err == nil {
		t.Fatal("execute() accepted an overcommitted pipeline")
	}
}

func TestExecuteEmitsSafeOpenTelemetrySpan(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(
		sdktrace.WithSyncer(exporter),
	)
	t.Cleanup(func() {
		if err := provider.Shutdown(context.Background()); err != nil {
			t.Fatalf("shutdown trace provider: %v", err)
		}
	})
	ctx, span := provider.Tracer("test").Start(t.Context(), "parent")
	job := testLoadJob("sensitive_graph_name", "vertices.csv", "edges.csv")
	job.Source.Type = "unsupported"

	if _, err := execute(ctx, job, "secret-job-id", false); err == nil {
		t.Fatal("execute() accepted unsupported source")
	}
	span.End()

	spans := exporter.GetSpans()
	if len(spans) != 2 || spans[0].Name != "load.execute" {
		t.Fatalf("spans = %#v", spans)
	}
	attributes := make(map[string]string)
	for _, item := range spans[0].Attributes {
		attributes[string(item.Key)] = item.Value.Emit()
	}
	if attributes["source.type"] != "unsupported" ||
		attributes["target.type"] != "apache-age" ||
		attributes["load.mode"] != "create" {
		t.Fatalf("span attributes = %#v", attributes)
	}
	for key, value := range attributes {
		if strings.Contains(key, "job") ||
			strings.Contains(value, "sensitive_graph_name") ||
			strings.Contains(value, "secret-job-id") {
			t.Fatalf("span exposed sensitive identifier: %s=%q", key, value)
		}
	}
}

func testLoadJob(graph, vertices, edges string) config.LoadJob {
	header := true
	nullValue := ""
	return config.LoadJob{
		APIVersion: config.APIVersion, Kind: config.KindLoadJob,
		Metadata: config.Metadata{Name: "app-test"},
		Source: config.Source{
			Type: config.SourceCSV, Namespace: "crm",
			CSV: &config.CSVSource{
				Defaults: config.DelimitedOptions{
					Delimiter: ",", Quote: `"`, Escape: `"`,
					Header: &header, Encoding: "utf-8", NullValue: &nullValue,
				},
				Vertices: []config.CSVVertex{{
					Label: "Person", Path: vertices, IDColumn: "id",
					Properties: map[string]string{"name": "name"},
				}},
				Edges: []config.CSVEdge{{
					Label: "KNOWS", Path: edges, ExternalIDColumn: "id",
					Start: config.EndpointMapping{Label: "Person", Field: "start"},
					End:   config.EndpointMapping{Label: "Person", Field: "end"},
				}},
			},
		},
		Target: config.Target{
			Type: config.TargetApacheAGE, Graph: graph, Mode: config.LoadCreate,
			Connection:      config.SecretRef{Env: "AGEFREIGHTER_APP_TEST_DSN"},
			PropertyMode:    config.PropertiesReplace,
			AppendDuplicate: config.AppendDuplicateError,
		},
		Runtime: config.Runtime{
			MemoryLimit: 16 << 20, BatchRows: 2, BatchBytes: 1 << 20,
			MaxSourceConcurrency: 1, MaxTransformConcurrency: 1,
			MaxTargetConnections: 2, OperationTimeout: config.Duration(10 * time.Second),
		},
		Errors: config.ErrorPolicies{
			MalformedRecord:  config.MalformedFail,
			MissingEndpoint:  config.MissingEndpointError,
			MaxDeferredEdges: 100_000,
		},
	}
}

func registerCleanup(t *testing.T, dsn, graphName, jobID string) {
	t.Helper()
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		if adapter, openErr := age.Open(cleanupCtx, dsn, age.PoolOptions{
			MinConnections: 1, MaxConnections: 2,
			ConnectTimeout: time.Second, OperationTimeout: 5 * time.Second,
		}); openErr == nil {
			_ = adapter.InTransaction(cleanupCtx, func(tx *age.Transaction) error {
				return tx.DropGraph(cleanupCtx, graphName, true)
			})
			adapter.Close()
		}
		pool, poolErr := pgxpool.New(cleanupCtx, dsn)
		if poolErr == nil {
			_ = deleteAppTestJob(cleanupCtx, pool, jobID)
			pool.Close()
		}
	})
}

func deleteAppTestJob(ctx context.Context, pool *pgxpool.Pool, jobID string) error {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, `
		UPDATE agefreighter_meta.load_job
		SET graph_generation_id = NULL
		WHERE job_id = $1::uuid`, jobID); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, `
		DELETE FROM agefreighter_meta.graph_generation
		WHERE job_id = $1::uuid`, jobID); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, `
		DELETE FROM agefreighter_meta.load_job
		WHERE job_id = $1::uuid`, jobID); err != nil {
		return err
	}
	return tx.Commit(ctx)
}
