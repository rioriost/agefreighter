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
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
)

func TestCSVIncrementalLoadIntegration(t *testing.T) {
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run app integration tests")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	graphName := fmt.Sprintf("incremental_e2e_%d", time.Now().UnixNano())
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)
	jobIDs := make([]string, 0, 4)
	registerIncrementalCleanup(t, dsn, graphName, &jobIDs)

	_, createPath := incrementalCSVJob(
		t,
		graphName,
		config.LoadCreate,
		"id,name,city\np1,Ada,London\np2,Grace,New York\n",
		"id,start,end,weight\ne1,p1,p2,1\n",
	)
	created, err := Load(ctx, createPath)
	jobIDs = append(jobIDs, created.JobID)
	if err != nil {
		t.Fatalf("create Load() error = %v", err)
	}
	if created.Status != meta.JobCommitted {
		t.Fatalf("create Load() = %#v", created)
	}
	assertIncrementalFinalizeLock(t, ctx, dsn, graphName, &jobIDs)
	for _, policy := range []config.AppendDuplicatePolicy{
		config.AppendDuplicateError,
		config.AppendDuplicateIgnoreIdentical,
	} {
		liveDuplicateJob, liveDuplicatePath := incrementalCSVJob(
			t,
			graphName,
			config.LoadAppend,
			"id,name,city\n",
			"id,start,end,weight\ne1,p1,missing-live,1\n",
		)
		liveDuplicateJob.Target.AppendDuplicate = policy
		liveDuplicateJob.Errors.MissingEndpoint = config.MissingEndpointDefer
		liveDuplicateJob.Errors.MaxDeferredEdges = 10
		writeLoadJob(
			t,
			filepath.Dir(liveDuplicatePath),
			filepath.Base(liveDuplicatePath),
			liveDuplicateJob,
		)
		liveDuplicate, err := Load(ctx, liveDuplicatePath)
		jobIDs = append(jobIDs, liveDuplicate.JobID)
		if err == nil || !strings.Contains(err.Error(), "existing edge") &&
			!strings.Contains(err.Error(), "duplicate edge identity") {
			t.Fatalf(
				"live duplicate deferred append (%s) error = %v",
				policy,
				err,
			)
		}
	}

	mutationPool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("open catalog mutation pool: %v", err)
	}
	_, graphMismatchPath := incrementalCSVJob(
		t,
		graphName,
		config.LoadAppend,
		"id,name,city\n",
		"id,start,end,weight\n",
	)
	if _, err := mutationPool.Exec(
		ctx,
		`UPDATE agefreighter_meta.graph_generation
		 SET graph_oid = (graph_oid::bigint + 100)::oid,
		     namespace_oid = (namespace_oid::bigint + 100)::oid
		 WHERE graph_name = $1 AND state = 'active'`,
		graphName,
	); err != nil {
		mutationPool.Close()
		t.Fatalf("mutate active graph generation: %v", err)
	}
	graphMismatch, err := Load(ctx, graphMismatchPath)
	jobIDs = append(jobIDs, graphMismatch.JobID)
	if !errors.Is(err, meta.ErrGenerationMismatch) {
		mutationPool.Close()
		t.Fatalf("graph mismatch Load() error = %v", err)
	}
	if _, err := mutationPool.Exec(
		ctx,
		`UPDATE agefreighter_meta.graph_generation
		 SET graph_oid = (graph_oid::bigint - 100)::oid,
		     namespace_oid = (namespace_oid::bigint - 100)::oid
		 WHERE graph_name = $1 AND state = 'active'`,
		graphName,
	); err != nil {
		mutationPool.Close()
		t.Fatalf("restore active graph generation: %v", err)
	}
	_, labelMismatchPath := incrementalCSVJob(
		t,
		graphName,
		config.LoadAppend,
		"id,name,city\n",
		"id,start,end,weight\n",
	)
	if _, err := mutationPool.Exec(
		ctx,
		`UPDATE agefreighter_meta.label_generation
		 SET relation_oid = (relation_oid::bigint + 100)::oid
		 WHERE graph_generation_id = (
		   SELECT graph_generation_id
		   FROM agefreighter_meta.graph_generation
		   WHERE graph_name = $1 AND state = 'active'
		 )
		   AND label_name = 'Person'`,
		graphName,
	); err != nil {
		mutationPool.Close()
		t.Fatalf("mutate active label generation: %v", err)
	}
	labelMismatch, err := Load(ctx, labelMismatchPath)
	jobIDs = append(jobIDs, labelMismatch.JobID)
	if !errors.Is(err, meta.ErrGenerationMismatch) {
		mutationPool.Close()
		t.Fatalf("label mismatch Load() error = %v", err)
	}
	if _, err := mutationPool.Exec(
		ctx,
		`UPDATE agefreighter_meta.label_generation
		 SET relation_oid = (relation_oid::bigint - 100)::oid
		 WHERE graph_generation_id = (
		   SELECT graph_generation_id
		   FROM agefreighter_meta.graph_generation
		   WHERE graph_name = $1 AND state = 'active'
		 )
		   AND label_name = 'Person'`,
		graphName,
	); err != nil {
		mutationPool.Close()
		t.Fatalf("restore active label generation: %v", err)
	}
	mutationPool.Close()

	appendJob, appendPath := incrementalCSVJob(
		t,
		graphName,
		config.LoadAppend,
		"id,name,city\np1,Ada,London\np3,Linus,Helsinki\n",
		"id,start,end,weight\ne2,p3,p1,2\ne3,p3,p4,3\n",
	)
	appendJob.Target.AppendDuplicate = config.AppendDuplicateIgnoreIdentical
	appendJob.Errors.MissingEndpoint = config.MissingEndpointDefer
	appendJob.Errors.MaxDeferredEdges = 10
	writeLoadJob(
		t,
		filepath.Dir(appendPath),
		filepath.Base(appendPath),
		appendJob,
	)
	appended, err := Load(ctx, appendPath)
	jobIDs = append(jobIDs, appended.JobID)
	if err != nil {
		t.Fatalf("append Load() error = %v", err)
	}
	if appended.Status != meta.JobCommitted {
		t.Fatalf("append Load() = %#v", appended)
	}

	_, lockPath := incrementalCSVJob(
		t,
		graphName,
		config.LoadAppend,
		"id,name,city\np-lock,Locked,Tokyo\n",
		"id,start,end,weight\n",
	)
	lockPool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("open lock pool: %v", err)
	}
	lockConnection, err := lockPool.Acquire(ctx)
	if err != nil {
		lockPool.Close()
		t.Fatalf("acquire lock connection: %v", err)
	}
	lockHeld := true
	t.Cleanup(func() {
		if !lockHeld {
			return
		}
		cleanupCtx, cleanupCancel := context.WithTimeout(
			context.Background(),
			5*time.Second,
		)
		defer cleanupCancel()
		_, _ = lockConnection.Exec(
			cleanupCtx,
			`SELECT pg_catalog.pg_advisory_unlock(
				pg_catalog.hashtextextended(
					'agefreighter:graph-lifecycle:' || $1,
					$2
				)
			)`,
			graphName,
			int64(0x6167656672),
		)
		lockConnection.Release()
		lockPool.Close()
	})
	if _, err := lockConnection.Exec(
		ctx,
		`SELECT pg_catalog.pg_advisory_lock(
			pg_catalog.hashtextextended(
				'agefreighter:graph-lifecycle:' || $1,
				$2
			)
		)`,
		graphName,
		int64(0x6167656672),
	); err != nil {
		lockConnection.Release()
		lockPool.Close()
		t.Fatalf("hold incremental graph lock: %v", err)
	}
	locked, err := Load(ctx, lockPath)
	jobIDs = append(jobIDs, locked.JobID)
	if !errors.Is(err, meta.ErrIncrementalConflict) {
		t.Fatalf("concurrent append Load() error = %v", err)
	}
	if _, err := lockConnection.Exec(
		ctx,
		`SELECT pg_catalog.pg_advisory_unlock(
			pg_catalog.hashtextextended(
				'agefreighter:graph-lifecycle:' || $1,
				$2
			)
		)`,
		graphName,
		int64(0x6167656672),
	); err != nil {
		lockConnection.Release()
		lockPool.Close()
		t.Fatalf("release incremental graph lock: %v", err)
	}
	lockHeld = false
	lockConnection.Release()
	lockPool.Close()
	resumed, err := Resume(ctx, lockPath, locked.JobID)
	if err != nil || resumed.Status != meta.JobCommitted {
		t.Fatalf("Resume(concurrent append) = %#v, %v", resumed, err)
	}
	conflictJob, conflictPath := incrementalCSVJob(
		t,
		graphName,
		config.LoadAppend,
		"id,name,city\np1,Wrong,London\n",
		"id,start,end,weight\n",
	)
	conflictJob.Target.AppendDuplicate = config.AppendDuplicateIgnoreIdentical
	writeLoadJob(
		t,
		filepath.Dir(conflictPath),
		filepath.Base(conflictPath),
		conflictJob,
	)
	conflicted, err := Load(ctx, conflictPath)
	jobIDs = append(jobIDs, conflicted.JobID)
	if err == nil {
		t.Fatal("conflicting append Load() succeeded")
	}

	pendingAppendJob, pendingAppendPath := incrementalCSVJob(
		t,
		graphName,
		config.LoadAppend,
		"id,name,city\n",
		"id,start,end,weight\ne3,p3,p1,20\n",
	)
	pendingAppendJob.Target.AppendDuplicate =
		config.AppendDuplicateIgnoreIdentical
	writeLoadJob(
		t,
		filepath.Dir(pendingAppendPath),
		filepath.Base(pendingAppendPath),
		pendingAppendJob,
	)
	pendingAppend, err := Load(ctx, pendingAppendPath)
	jobIDs = append(jobIDs, pendingAppend.JobID)
	if err == nil || !strings.Contains(err.Error(), "older pending edge") {
		t.Fatalf("append ahead of pending edge Load() error = %v", err)
	}

	deferredErrorJob, deferredErrorPath := incrementalCSVJob(
		t,
		graphName,
		config.LoadAppend,
		"id,name,city\n",
		"id,start,end,weight\ne3,p3,p4,3\n",
	)
	deferredErrorJob.Errors.MissingEndpoint = config.MissingEndpointDefer
	deferredErrorJob.Errors.MaxDeferredEdges = 10
	writeLoadJob(
		t,
		filepath.Dir(deferredErrorPath),
		filepath.Base(deferredErrorPath),
		deferredErrorJob,
	)
	deferredError, err := Load(ctx, deferredErrorPath)
	jobIDs = append(jobIDs, deferredError.JobID)
	if err == nil || !strings.Contains(err.Error(), "duplicate deferred edge") {
		t.Fatalf("error-policy deferred replay Load() error = %v", err)
	}
	deferredConflictJob, deferredConflictPath := incrementalCSVJob(
		t,
		graphName,
		config.LoadAppend,
		"id,name,city\n",
		"id,start,end,weight\ne3,p3,p4,99\n",
	)
	deferredConflictJob.Target.AppendDuplicate =
		config.AppendDuplicateIgnoreIdentical
	deferredConflictJob.Errors.MissingEndpoint = config.MissingEndpointDefer
	deferredConflictJob.Errors.MaxDeferredEdges = 10
	writeLoadJob(
		t,
		filepath.Dir(deferredConflictPath),
		filepath.Base(deferredConflictPath),
		deferredConflictJob,
	)
	deferredConflict, err := Load(ctx, deferredConflictPath)
	jobIDs = append(jobIDs, deferredConflict.JobID)
	if err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("conflicting deferred replay Load() error = %v", err)
	}
	deferredReplayJob, deferredReplayPath := incrementalCSVJob(
		t,
		graphName,
		config.LoadAppend,
		"id,name,city\n",
		"id,start,end,weight\ne3,p3,p4,3\n",
	)
	deferredReplayJob.Target.AppendDuplicate =
		config.AppendDuplicateIgnoreIdentical
	deferredReplayJob.Errors.MissingEndpoint = config.MissingEndpointDefer
	deferredReplayJob.Errors.MaxDeferredEdges = 10
	writeLoadJob(
		t,
		filepath.Dir(deferredReplayPath),
		filepath.Base(deferredReplayPath),
		deferredReplayJob,
	)
	deferredReplay, err := Load(ctx, deferredReplayPath)
	jobIDs = append(jobIDs, deferredReplay.JobID)
	if err != nil || deferredReplay.Status != meta.JobCommitted {
		t.Fatalf("deferred replay Load() = %#v, %v", deferredReplay, err)
	}
	deferredPool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("open deferred verification pool: %v", err)
	}
	var pendingDeferred int64
	if err := deferredPool.QueryRow(
		ctx,
		`SELECT COUNT(*)
		 FROM agefreighter_meta.deferred_edge d
		 JOIN agefreighter_meta.graph_generation g
		   ON g.graph_generation_id = d.graph_generation_id
		 WHERE g.graph_name = $1`,
		graphName,
	).Scan(&pendingDeferred); err != nil {
		deferredPool.Close()
		t.Fatalf("count pending deferred edges: %v", err)
	}
	deferredPool.Close()
	if pendingDeferred != 1 {
		t.Fatalf("pending deferred edges = %d, want 1", pendingDeferred)
	}
	queuedUpsertJob, queuedUpsertPath := incrementalCSVJob(
		t,
		graphName,
		config.LoadUpsert,
		"id,name,city\n",
		"id,start,end,weight\ne3,p3,p1,20\n",
	)
	queuedUpsertJob.Target.PropertyMode = config.PropertiesReplace
	queuedUpsertJob.Errors.MaxDeferredEdges = 0
	writeLoadJob(
		t,
		filepath.Dir(queuedUpsertPath),
		filepath.Base(queuedUpsertPath),
		queuedUpsertJob,
	)
	queuedUpsert, err := Load(ctx, queuedUpsertPath)
	jobIDs = append(jobIDs, queuedUpsert.JobID)
	if err != nil || queuedUpsert.Status != meta.JobCommitted {
		t.Fatalf("queued upsert Load() = %#v, %v", queuedUpsert, err)
	}
	deferredPool, err = pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("reopen deferred verification pool: %v", err)
	}
	if err := deferredPool.QueryRow(
		ctx,
		`SELECT COUNT(*)
		 FROM agefreighter_meta.deferred_edge d
		 JOIN agefreighter_meta.graph_generation g
		   ON g.graph_generation_id = d.graph_generation_id
		 WHERE g.graph_name = $1`,
		graphName,
	).Scan(&pendingDeferred); err != nil {
		deferredPool.Close()
		t.Fatalf("count queued upserts: %v", err)
	}
	deferredPool.Close()
	if pendingDeferred != 2 {
		t.Fatalf("queued deferred edges = %d, want 2", pendingDeferred)
	}

	upsertJob, upsertPath := incrementalCSVJob(
		t,
		graphName,
		config.LoadUpsert,
		"id,name,city\np1,Ada Lovelace,\np4,Margaret,London\n",
		"id,start,end,weight\ne1,p1,p4,9\ne3,p3,p4,30\n",
	)
	upsertJob.Target.PropertyMode = config.PropertiesMergeDeleteNull
	upsertJob.Runtime.BatchRows = 10
	writeLoadJob(
		t,
		filepath.Dir(upsertPath),
		filepath.Base(upsertPath),
		upsertJob,
	)
	upserted, err := Load(ctx, upsertPath)
	jobIDs = append(jobIDs, upserted.JobID)
	if err != nil {
		t.Fatalf("upsert Load() error = %v", err)
	}
	if upserted.Status != meta.JobCommitted {
		t.Fatalf("upsert Load() = %#v", upserted)
	}
	if _, err := Verify(ctx, upsertPath, upserted.JobID); err != nil {
		t.Fatalf("Verify(upsert) error = %v", err)
	}
	capacityJob, capacityPath := incrementalCSVJob(
		t,
		graphName,
		config.LoadAppend,
		"id,name,city\n",
		"id,start,end,weight\ne4,p1,missing-a,4\ne5,p1,missing-b,5\n",
	)
	capacityJob.Errors.MissingEndpoint = config.MissingEndpointDefer
	capacityJob.Errors.MaxDeferredEdges = 1
	writeLoadJob(
		t,
		filepath.Dir(capacityPath),
		filepath.Base(capacityPath),
		capacityJob,
	)
	capacityResult, err := Load(ctx, capacityPath)
	jobIDs = append(jobIDs, capacityResult.JobID)
	if err == nil || !strings.Contains(err.Error(), "capacity exceeded") {
		t.Fatalf("capacity-limited append Load() error = %v", err)
	}

	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("open verification pool: %v", err)
	}
	defer pool.Close()
	var deferred int64
	var edgeIdentities string
	if err := pool.QueryRow(
		ctx,
		`SELECT
			(SELECT COUNT(*)
			 FROM agefreighter_meta.deferred_edge d
			 JOIN agefreighter_meta.graph_generation g
			   ON g.graph_generation_id = d.graph_generation_id
			 WHERE g.graph_name = $1),
			(SELECT COALESCE(string_agg(i.external_id, ',' ORDER BY i.external_id), '')
			 FROM agefreighter_meta.edge_identity i
			 JOIN agefreighter_meta.graph_generation g
			   ON g.graph_generation_id = i.graph_generation_id
			 WHERE g.graph_name = $1)`,
		graphName,
	).Scan(&deferred, &edgeIdentities); err != nil {
		t.Fatalf("inspect incremental edge state: %v", err)
	}
	if deferred != 0 {
		t.Fatalf("deferred edge count = %d, want 0", deferred)
	}
	if edgeIdentities != "e1,e2,e3" {
		t.Fatalf("edge identities = %q, want e1,e2,e3", edgeIdentities)
	}
	connection, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("acquire verification connection: %v", err)
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
		graphName,
		"MATCH (n:Person) RETURN count(n)",
		5,
	)
	assertCypherCount(
		t,
		connection.Conn(),
		graphName,
		"MATCH ()-[r:KNOWS]->() RETURN count(r)",
		3,
	)
	assertCypherCount(
		t,
		connection.Conn(),
		graphName,
		`MATCH (n:Person)
		 WHERE n.name = 'Ada Lovelace' AND n.city IS NULL
		 RETURN count(n)`,
		1,
	)
	assertCypherCount(
		t,
		connection.Conn(),
		graphName,
		`MATCH (:Person {name: 'Ada Lovelace'})
		       -[r:KNOWS]->(:Person {name: 'Margaret'})
		 WHERE r.weight = '9'
		 RETURN count(r)`,
		1,
	)
	assertCypherCount(
		t,
		connection.Conn(),
		graphName,
		`MATCH (:Person {name: 'Linus'})
		       -[r:KNOWS]->(:Person {name: 'Margaret'})
		 WHERE r.weight = '30'
		 RETURN count(r)`,
		1,
	)
}

func assertIncrementalFinalizeLock(
	t *testing.T,
	ctx context.Context,
	dsn string,
	graphName string,
	jobIDs *[]string,
) {
	t.Helper()
	job, _ := incrementalCSVJob(
		t,
		graphName,
		config.LoadAppend,
		"id,name,city\n",
		"id,start,end,weight\n",
	)
	jobID, err := newJobID()
	if err != nil {
		t.Fatalf("new finalization job ID: %v", err)
	}
	*jobIDs = append(*jobIDs, jobID)
	adapter, store, err := openTarget(ctx, job)
	if err != nil {
		t.Fatalf("open finalization target: %v", err)
	}
	defer adapter.Close()
	fingerprint, err := jobFingerprint(job)
	if err != nil {
		t.Fatalf("fingerprint finalization job: %v", err)
	}
	if err := store.CreateJob(ctx, meta.Job{
		ID: jobID, Name: job.Metadata.Name,
		SourceType: string(job.Source.Type), LoadMode: string(job.Target.Mode),
		TargetGraph: graphName, ConfigFingerprint: fingerprint,
	}); err != nil {
		t.Fatalf("create finalization job: %v", err)
	}
	if err := store.StartJob(ctx, jobID); err != nil {
		t.Fatalf("start finalization job: %v", err)
	}
	graph, _, err := admitIncrementalCatalog(ctx, adapter, job, jobID)
	if err != nil {
		t.Fatalf("admit finalization job: %v", err)
	}
	lockPool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("open finalization lock pool: %v", err)
	}
	lockConnection, err := lockPool.Acquire(ctx)
	if err != nil {
		lockPool.Close()
		t.Fatalf("acquire finalization lock connection: %v", err)
	}
	if _, err := lockConnection.Exec(
		ctx,
		`SELECT pg_catalog.pg_advisory_lock(
			pg_catalog.hashtextextended(
				'agefreighter:graph-lifecycle:' || $1,
				$2
			)
		)`,
		graphName,
		int64(0x6167656672),
	); err != nil {
		lockConnection.Release()
		lockPool.Close()
		t.Fatalf("hold finalization lifecycle lock: %v", err)
	}
	if err := completeIncremental(
		ctx,
		adapter,
		jobID,
		graph,
	); !errors.Is(err, meta.ErrIncrementalConflict) {
		t.Fatalf("completeIncremental(locked) error = %v", err)
	}
	if _, err := lockConnection.Exec(
		ctx,
		`SELECT pg_catalog.pg_advisory_unlock(
			pg_catalog.hashtextextended(
				'agefreighter:graph-lifecycle:' || $1,
				$2
			)
		)`,
		graphName,
		int64(0x6167656672),
	); err != nil {
		lockConnection.Release()
		lockPool.Close()
		t.Fatalf("release finalization lifecycle lock: %v", err)
	}
	lockConnection.Release()
	lockPool.Close()
	missingGraph := graph
	missingGraph.GraphName += "_missing"
	if err := completeIncremental(ctx, adapter, jobID, missingGraph); err == nil {
		t.Fatal("completeIncremental() accepted a missing graph")
	}
	if err := completeIncremental(
		ctx,
		adapter,
		"10101010-2020-4030-8040-505050505050",
		graph,
	); !errors.Is(err, meta.ErrNotFound) {
		t.Fatalf("completeIncremental(missing job) error = %v", err)
	}
	mismatchedGraph := graph
	mismatchedGraph.ID++
	if err := completeIncremental(
		ctx,
		adapter,
		jobID,
		mismatchedGraph,
	); !errors.Is(err, meta.ErrGenerationMismatch) {
		t.Fatalf("completeIncremental(generation mismatch) error = %v", err)
	}
	if err := completeIncremental(ctx, adapter, jobID, graph); err != nil {
		t.Fatalf("completeIncremental() error = %v", err)
	}
	stored, err := store.GetJob(ctx, jobID)
	if err != nil || stored.Status != meta.JobCommitted {
		t.Fatalf("finalization job = %#v, %v", stored, err)
	}
}

func TestIncrementalAdmissionRejectsUnmanagedGraphIntegration(t *testing.T) {
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run app integration tests")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	graphName := fmt.Sprintf("unmanaged_incremental_%d", time.Now().UnixNano())
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)
	job := testLoadJob(graphName, "vertices.csv", "edges.csv")
	job.Target.Mode = config.LoadAppend
	job.Target.AppendDuplicate = config.AppendDuplicateError
	jobID, err := newJobID()
	if err != nil {
		t.Fatalf("newJobID() error = %v", err)
	}
	registerCleanup(t, dsn, graphName, jobID)
	adapter, store, err := openTarget(ctx, job)
	if err != nil {
		t.Fatalf("openTarget() error = %v", err)
	}
	defer adapter.Close()
	if err := adapter.InTransaction(ctx, func(tx *age.Transaction) error {
		if err := tx.CreateGraph(ctx, graphName); err != nil {
			return err
		}
		if err := tx.CreateLabel(ctx, graphName, "Person", age.VertexLabel); err != nil {
			return err
		}
		return tx.CreateLabel(ctx, graphName, "KNOWS", age.EdgeLabel)
	}); err != nil {
		t.Fatalf("create unmanaged graph: %v", err)
	}
	fingerprint, err := jobFingerprint(job)
	if err != nil {
		t.Fatalf("jobFingerprint() error = %v", err)
	}
	if err := store.CreateJob(ctx, meta.Job{
		ID: jobID, Name: job.Metadata.Name,
		SourceType: string(job.Source.Type), LoadMode: string(job.Target.Mode),
		TargetGraph: graphName, ConfigFingerprint: fingerprint,
	}); err != nil {
		t.Fatalf("CreateJob() error = %v", err)
	}
	if err := store.StartJob(ctx, jobID); err != nil {
		t.Fatalf("StartJob() error = %v", err)
	}
	if _, _, err := admitIncrementalCatalog(
		ctx,
		adapter,
		job,
		jobID,
	); !errors.Is(err, meta.ErrNotFound) {
		t.Fatalf("admitIncrementalCatalog(unmanaged) error = %v", err)
	}
	stored, err := store.GetJob(ctx, jobID)
	if err != nil || stored.GraphGenerationID != 0 {
		t.Fatalf("unmanaged admission job = %#v, %v", stored, err)
	}
}

func incrementalCSVJob(
	t *testing.T,
	graphName string,
	mode config.LoadMode,
	vertices string,
	edges string,
) (config.LoadJob, string) {
	t.Helper()
	dir := t.TempDir()
	vertexPath := filepath.Join(dir, "vertices.csv")
	edgePath := filepath.Join(dir, "edges.csv")
	if err := os.WriteFile(vertexPath, []byte(vertices), 0o600); err != nil {
		t.Fatalf("write incremental vertices: %v", err)
	}
	if err := os.WriteFile(edgePath, []byte(edges), 0o600); err != nil {
		t.Fatalf("write incremental edges: %v", err)
	}
	job := testLoadJob(graphName, vertexPath, edgePath)
	job.Target.Mode = mode
	job.Target.AppendDuplicate = config.AppendDuplicateError
	job.Source.CSV.Vertices[0].Properties["city"] = "city"
	job.Source.CSV.Edges[0].Properties = map[string]string{"weight": "weight"}
	path := writeLoadJob(t, dir, "job.yaml", job)
	return job, path
}

func registerIncrementalCleanup(
	t *testing.T,
	dsn string,
	graphName string,
	jobIDs *[]string,
) {
	t.Helper()
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		pool, err := pgxpool.New(ctx, dsn)
		if err != nil {
			return
		}
		defer pool.Close()
		_, _ = pool.Exec(
			ctx,
			`SELECT ag_catalog.drop_graph($1, true)`,
			graphName,
		)
		tx, err := pool.Begin(ctx)
		if err != nil {
			return
		}
		defer tx.Rollback(ctx)
		for _, jobID := range *jobIDs {
			if jobID == "" {
				continue
			}
			if _, err := tx.Exec(
				ctx,
				`UPDATE agefreighter_meta.load_job
				 SET graph_generation_id = NULL
				 WHERE job_id = $1::uuid`,
				jobID,
			); err != nil {
				return
			}
		}
		for _, jobID := range *jobIDs {
			if jobID == "" {
				continue
			}
			if _, err := tx.Exec(
				ctx,
				`DELETE FROM agefreighter_meta.graph_generation
				 WHERE job_id = $1::uuid`,
				jobID,
			); err != nil {
				return
			}
		}
		for _, jobID := range *jobIDs {
			if jobID == "" {
				continue
			}
			if _, err := tx.Exec(
				ctx,
				`DELETE FROM agefreighter_meta.load_job
				 WHERE job_id = $1::uuid`,
				jobID,
			); err != nil {
				return
			}
		}
		_ = tx.Commit(ctx)
	})
}
