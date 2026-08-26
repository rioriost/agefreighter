package app

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"go.yaml.in/yaml/v3"
)

func TestReplaceFailureResumeAndCleanupIntegration(t *testing.T) {
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run app integration tests")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	dir := t.TempDir()
	vertices := filepath.Join(dir, "people.csv")
	edges := filepath.Join(dir, "knows.csv")
	if err := os.WriteFile(
		vertices,
		[]byte("id,name\np1,Ada\np2,Grace\n"),
		0o600,
	); err != nil {
		t.Fatalf("write vertices: %v", err)
	}
	if err := os.WriteFile(
		edges,
		[]byte("id,start,end\ne1,p1,p2\n"),
		0o600,
	); err != nil {
		t.Fatalf("write edges: %v", err)
	}

	graphName := fmt.Sprintf("app_replace_%d", time.Now().UnixNano())
	jobID, err := newJobID()
	if err != nil {
		t.Fatalf("newJobID() error = %v", err)
	}
	shadowName, err := age.DeriveGraphName(graphName, age.ShadowName, jobID)
	if err != nil {
		t.Fatalf("derive shadow name: %v", err)
	}
	backupName, err := age.DeriveGraphName(graphName, age.BackupName, jobID)
	if err != nil {
		t.Fatalf("derive backup name: %v", err)
	}
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)
	job := testLoadJob(graphName, vertices, edges)
	job.Target.Mode = config.LoadReplace
	jobPath := filepath.Join(dir, "replace.yaml")
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatalf("marshal replace job: %v", err)
	}
	if err := os.WriteFile(jobPath, data, 0o600); err != nil {
		t.Fatalf("write replace job: %v", err)
	}
	registerReplaceCleanup(
		t,
		dsn,
		jobID,
		graphName,
		shadowName,
		backupName,
	)

	adapter, _, err := openTarget(ctx, job)
	if err != nil {
		t.Fatalf("open replacement target: %v", err)
	}
	if err := adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		if err := transaction.CreateGraph(ctx, graphName); err != nil {
			return err
		}
		if err := transaction.CreateLabel(
			ctx,
			graphName,
			"Person",
			age.VertexLabel,
		); err != nil {
			return err
		}
		return transaction.CreateGraph(ctx, backupName)
	}); err != nil {
		adapter.Close()
		t.Fatalf("create replacement fixtures: %v", err)
	}
	oldCatalog, err := adapter.LookupGraph(ctx, graphName)
	adapter.Close()
	if err != nil {
		t.Fatalf("lookup old target: %v", err)
	}
	if err := createLegacyPerson(ctx, dsn, graphName); err != nil {
		t.Fatalf("create legacy person: %v", err)
	}

	result, err := execute(ctx, job, jobID, false)
	if err == nil || result.JobID != jobID {
		t.Fatalf("execute(with backup collision) = %#v, %v", result, err)
	}
	adapter, store, err := openTarget(ctx, job)
	if err != nil {
		t.Fatalf("reopen failed replacement: %v", err)
	}
	currentTarget, err := adapter.LookupGraph(ctx, graphName)
	if err != nil || currentTarget.GraphOID != oldCatalog.GraphOID {
		adapter.Close()
		t.Fatalf("target after failed promotion = %#v, %v", currentTarget, err)
	}
	if _, err := adapter.LookupGraph(ctx, shadowName); err != nil {
		adapter.Close()
		t.Fatalf("shadow after failed promotion: %v", err)
	}
	failedJob, err := store.GetJob(ctx, jobID)
	if err != nil || failedJob.Status != meta.JobFailed {
		adapter.Close()
		t.Fatalf("failed replacement job = %#v, %v", failedJob, err)
	}
	if err := adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		return transaction.DropGraph(ctx, backupName, true)
	}); err != nil {
		adapter.Close()
		t.Fatalf("remove backup collision: %v", err)
	}
	adapter.Close()

	constraintName := "af_replace_block_" + jobID[:8]
	if err := setPromotionBlock(
		ctx,
		dsn,
		constraintName,
		jobID,
		true,
	); err != nil {
		t.Fatalf("install promotion failure: %v", err)
	}
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(
			context.Background(),
			5*time.Second,
		)
		defer cleanupCancel()
		_ = setPromotionBlock(
			cleanupCtx,
			dsn,
			constraintName,
			jobID,
			false,
		)
	})
	if _, err := Resume(ctx, jobPath, jobID); err == nil {
		t.Fatal("Resume() succeeded with injected metadata activation failure")
	}
	adapter, store, err = openTarget(ctx, job)
	if err != nil {
		t.Fatalf("open rolled-back replacement: %v", err)
	}
	rolledBackTarget, err := adapter.LookupGraph(ctx, graphName)
	if err != nil || rolledBackTarget.GraphOID != oldCatalog.GraphOID {
		adapter.Close()
		t.Fatalf("target after rename rollback = %#v, %v", rolledBackTarget, err)
	}
	if _, err := adapter.LookupGraph(ctx, shadowName); err != nil {
		adapter.Close()
		t.Fatalf("shadow after rename rollback: %v", err)
	}
	if _, err := adapter.LookupGraph(ctx, backupName); !errors.Is(
		err,
		age.ErrCatalogEntryNotFound,
	) {
		adapter.Close()
		t.Fatalf("backup after rename rollback error = %v", err)
	}
	rolledBackJob, err := store.GetJob(ctx, jobID)
	adapter.Close()
	if err != nil || rolledBackJob.Status != meta.JobFailed {
		t.Fatalf("job after rename rollback = %#v, %v", rolledBackJob, err)
	}
	if err := setPromotionBlock(
		ctx,
		dsn,
		constraintName,
		jobID,
		false,
	); err != nil {
		t.Fatalf("remove promotion failure: %v", err)
	}

	resumed, err := Resume(ctx, jobPath, jobID)
	if err != nil || resumed.Status != meta.JobCommitted {
		t.Fatalf("Resume(replace) = %#v, %v", resumed, err)
	}
	committed, err := Status(ctx, jobPath, jobID)
	if err != nil ||
		committed.BackupGraphName != backupName ||
		committed.BackupCleanedAt != nil {
		t.Fatalf("committed replacement = %#v, %v", committed, err)
	}
	adapter, store, err = openTarget(ctx, job)
	if err != nil {
		t.Fatalf("open committed replacement: %v", err)
	}
	activeCatalog, err := adapter.LookupGraph(ctx, graphName)
	if err != nil || activeCatalog.GraphOID == oldCatalog.GraphOID {
		adapter.Close()
		t.Fatalf("promoted target = %#v, %v", activeCatalog, err)
	}
	retainedCatalog, err := adapter.LookupGraph(ctx, backupName)
	if err != nil || retainedCatalog.GraphOID != oldCatalog.GraphOID {
		adapter.Close()
		t.Fatalf("retained backup = %#v, %v", retainedCatalog, err)
	}
	generation, err := store.GraphGenerationForJob(ctx, jobID)
	adapter.Close()
	if err != nil ||
		generation.State != meta.GenerationActive ||
		generation.GraphName != graphName ||
		generation.GraphOID != activeCatalog.GraphOID ||
		generation.ReplacesGraphOID != oldCatalog.GraphOID {
		t.Fatalf("active replacement generation = %#v, %v", generation, err)
	}
	if count, err := countGraphNodes(ctx, dsn, graphName); err != nil || count != 2 {
		t.Fatalf("promoted graph node count = %d, %v", count, err)
	}
	if count, err := countGraphNodes(ctx, dsn, backupName); err != nil || count != 1 {
		t.Fatalf("backup graph node count = %d, %v", count, err)
	}
	if _, err := Verify(ctx, jobPath, jobID); err != nil {
		t.Fatalf("Verify(replace) error = %v", err)
	}

	cleaned, err := Cleanup(ctx, jobPath, jobID)
	if err != nil || cleaned.BackupCleanedAt == nil {
		t.Fatalf("Cleanup() = %#v, %v", cleaned, err)
	}
	repeated, err := Cleanup(ctx, jobPath, jobID)
	if err != nil || repeated.BackupCleanedAt == nil {
		t.Fatalf("idempotent Cleanup() = %#v, %v", repeated, err)
	}
	adapter, _, err = openTarget(ctx, job)
	if err != nil {
		t.Fatalf("open cleaned replacement: %v", err)
	}
	defer adapter.Close()
	if _, err := adapter.LookupGraph(ctx, graphName); err != nil {
		t.Fatalf("cleanup removed active graph: %v", err)
	}
	if _, err := adapter.LookupGraph(ctx, backupName); !errors.Is(
		err,
		age.ErrCatalogEntryNotFound,
	) {
		t.Fatalf("backup after cleanup error = %v", err)
	}
}

func TestReplaceManagedGenerationIntegration(t *testing.T) {
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run app integration tests")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	dir := t.TempDir()
	vertices := filepath.Join(dir, "people.csv")
	edges := filepath.Join(dir, "knows.csv")
	if err := os.WriteFile(
		vertices,
		[]byte("id,name\np1,Ada\np2,Grace\n"),
		0o600,
	); err != nil {
		t.Fatalf("write vertices: %v", err)
	}
	if err := os.WriteFile(
		edges,
		[]byte("id,start,end\ne1,p1,p2\n"),
		0o600,
	); err != nil {
		t.Fatalf("write edges: %v", err)
	}
	graphName := fmt.Sprintf("app_managed_replace_%d", time.Now().UnixNano())
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)

	createJob := testLoadJob(graphName, vertices, edges)
	createPath := writeLoadJob(t, dir, "create.yaml", createJob)
	created, err := Load(ctx, createPath)
	if err != nil {
		t.Fatalf("Load(create) error = %v", err)
	}
	registerCleanup(t, dsn, graphName, created.JobID)
	adapter, store, err := openTarget(ctx, createJob)
	if err != nil {
		t.Fatalf("open created graph: %v", err)
	}
	createdGeneration, err := store.GraphGenerationForJob(ctx, created.JobID)
	adapter.Close()
	if err != nil {
		t.Fatalf("created generation: %v", err)
	}

	replaceJob := createJob
	replaceJob.Target.Mode = config.LoadReplace
	replacePath := writeLoadJob(t, dir, "replace-managed.yaml", replaceJob)
	replaced, err := Load(ctx, replacePath)
	if err != nil {
		t.Fatalf("Load(replace) error = %v", err)
	}
	backupName, err := age.DeriveGraphName(
		graphName,
		age.BackupName,
		replaced.JobID,
	)
	if err != nil {
		t.Fatalf("derive managed backup: %v", err)
	}
	shadowName, err := age.DeriveGraphName(
		graphName,
		age.ShadowName,
		replaced.JobID,
	)
	if err != nil {
		t.Fatalf("derive managed shadow: %v", err)
	}
	registerReplaceCleanup(
		t,
		dsn,
		replaced.JobID,
		graphName,
		shadowName,
		backupName,
	)

	adapter, store, err = openTarget(ctx, replaceJob)
	if err != nil {
		t.Fatalf("open replaced graph: %v", err)
	}
	active, err := store.GraphGenerationForJob(ctx, replaced.JobID)
	if err != nil {
		adapter.Close()
		t.Fatalf("active managed generation: %v", err)
	}
	retired, err := store.GraphGenerationForJob(ctx, created.JobID)
	if err != nil {
		adapter.Close()
		t.Fatalf("retired managed generation: %v", err)
	}
	backupCatalog, err := adapter.LookupGraph(ctx, backupName)
	adapter.Close()
	if err != nil {
		t.Fatalf("lookup managed backup: %v", err)
	}
	if active.State != meta.GenerationActive ||
		active.GraphName != graphName ||
		active.Generation != createdGeneration.Generation+1 ||
		active.ReplacesGraphOID != createdGeneration.GraphOID {
		t.Fatalf("active managed generation = %#v", active)
	}
	if retired.State != meta.GenerationRetired ||
		retired.GraphName != backupName ||
		retired.GraphOID != createdGeneration.GraphOID ||
		backupCatalog.GraphOID != retired.GraphOID {
		t.Fatalf(
			"retired managed generation = %#v, backup = %#v",
			retired,
			backupCatalog,
		)
	}
}

func writeLoadJob(
	t *testing.T,
	dir string,
	name string,
	job config.LoadJob,
) string {
	t.Helper()
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatalf("marshal %s: %v", name, err)
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	return path
}

func setPromotionBlock(
	ctx context.Context,
	dsn string,
	constraintName string,
	jobID string,
	enabled bool,
) error {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return err
	}
	defer pool.Close()
	constraint := pgx.Identifier{constraintName}.Sanitize()
	if !enabled {
		_, err = pool.Exec(
			ctx,
			`ALTER TABLE agefreighter_meta.graph_generation
			 DROP CONSTRAINT IF EXISTS `+constraint,
		)
		return err
	}
	_, err = pool.Exec(
		ctx,
		fmt.Sprintf(
			`ALTER TABLE agefreighter_meta.graph_generation
			 ADD CONSTRAINT %s CHECK (
				job_id <> '%s'::uuid OR state <> 'active'
			 ) NOT VALID`,
			constraint,
			jobID,
		),
	)
	return err
}

func createLegacyPerson(ctx context.Context, dsn, graphName string) error {
	connection, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return err
	}
	defer connection.Close()
	acquired, err := connection.Acquire(ctx)
	if err != nil {
		return err
	}
	defer acquired.Release()
	if _, err := acquired.Exec(ctx, "SELECT ag_catalog.age_pi()"); err != nil {
		return err
	}
	if _, err := acquired.Exec(
		ctx,
		`SET search_path = ag_catalog, "$user", public`,
	); err != nil {
		return err
	}
	query := fmt.Sprintf(
		`SELECT value::text
		 FROM ag_catalog.cypher(
			'%s',
			$$CREATE (person:Person {name: "Legacy"}) RETURN id(person)$$
		 ) AS result(value ag_catalog.agtype)`,
		graphName,
	)
	var created string
	return acquired.QueryRow(ctx, query).Scan(&created)
}

func countGraphNodes(ctx context.Context, dsn, graphName string) (int64, error) {
	connection, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return 0, err
	}
	defer connection.Close()
	acquired, err := connection.Acquire(ctx)
	if err != nil {
		return 0, err
	}
	defer acquired.Release()
	if _, err := acquired.Exec(ctx, "SELECT ag_catalog.age_pi()"); err != nil {
		return 0, err
	}
	if _, err := acquired.Exec(
		ctx,
		`SET search_path = ag_catalog, "$user", public`,
	); err != nil {
		return 0, err
	}
	query := fmt.Sprintf(
		`SELECT count(*)
		 FROM ag_catalog.cypher(
			'%s',
			$$MATCH (node) RETURN node$$
		 ) AS result(node ag_catalog.agtype)`,
		graphName,
	)
	var count int64
	if err := acquired.QueryRow(ctx, query).Scan(&count); err != nil {
		return 0, err
	}
	return count, nil
}

func registerReplaceCleanup(
	t *testing.T,
	dsn string,
	jobID string,
	graphNames ...string,
) {
	t.Helper()
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if adapter, err := age.Open(ctx, dsn, age.PoolOptions{
			MinConnections: 1, MaxConnections: 2,
			ConnectTimeout: time.Second, OperationTimeout: 5 * time.Second,
		}); err == nil {
			for _, name := range graphNames {
				_ = adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
					if _, lookupErr := transaction.LookupGraph(ctx, name); errors.Is(
						lookupErr,
						age.ErrCatalogEntryNotFound,
					) {
						return nil
					} else if lookupErr != nil {
						return lookupErr
					}
					return transaction.DropGraph(ctx, name, true)
				})
			}
			adapter.Close()
		}
		if pool, err := pgxpool.New(ctx, dsn); err == nil {
			_ = deleteAppTestJob(ctx, pool, jobID)
			pool.Close()
		}
	})
}
