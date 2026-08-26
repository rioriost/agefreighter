package age

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/internal/checkpoint"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/reject"
	"github.com/rioriost/agefreighter/internal/sink"
	"github.com/rioriost/agefreighter/pkg/model"
)

func TestLoadSinkIntegration(t *testing.T) {
	dsn := integrationDSN(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	adapter := openIntegrationAdapter(t, ctx, dsn, 2)
	t.Cleanup(adapter.Close)
	store, err := adapter.Metadata()
	if err != nil {
		t.Fatalf("Metadata() error = %v", err)
	}
	if err := store.Migrate(ctx); err != nil {
		t.Fatalf("Migrate() error = %v", err)
	}
	if err := adapter.acquireLoadSlot(ctx); err != nil {
		t.Fatalf("reserve invalid ownership test slot: %v", err)
	}
	unlockedOwner, err := adapter.pool.Acquire(ctx)
	if err != nil {
		adapter.releaseLoadSlot()
		t.Fatalf("acquire invalid ownership test connection: %v", err)
	}
	unlockedTarget := &LoadSink{
		adapter: adapter,
		options: LoadSinkOptions{
			JobID: "55555555-6666-4777-8888-999999999999",
		},
	}
	if err := unlockedTarget.releaseBatchOwner(
		unlockedOwner,
		"not-held",
		false,
	); err == nil || !strings.Contains(err.Error(), "was not held") {
		t.Fatalf("releaseBatchOwner(unheld) error = %v", err)
	}
	if err := adapter.acquireLoadSlot(ctx); err != nil {
		t.Fatalf("reserve invalid graph ownership test slot: %v", err)
	}
	unlockedGraphOwner, err := adapter.pool.Acquire(ctx)
	if err != nil {
		adapter.releaseLoadSlot()
		t.Fatalf("acquire invalid graph ownership test connection: %v", err)
	}
	if err := unlockedTarget.releaseBatchOwner(
		unlockedGraphOwner,
		"graph-not-held",
		true,
	); err == nil || !strings.Contains(err.Error(), "graph lock was not held") {
		t.Fatalf("releaseBatchOwner(unheld graph) error = %v", err)
	}
	if err := adapter.acquireLoadSlot(ctx); err != nil {
		t.Fatalf("reserve closed ownership test slot: %v", err)
	}
	closedOwner, err := adapter.pool.Acquire(ctx)
	if err != nil {
		adapter.releaseLoadSlot()
		t.Fatalf("acquire closed ownership test connection: %v", err)
	}
	if err := closedOwner.Conn().Close(ctx); err != nil {
		closedOwner.Release()
		adapter.releaseLoadSlot()
		t.Fatalf("close ownership test connection: %v", err)
	}
	if err := unlockedTarget.releaseBatchOwner(
		closedOwner,
		"closed",
		false,
	); err == nil || !strings.Contains(err.Error(), "unlock AGE load batch") {
		t.Fatalf("releaseBatchOwner(closed) error = %v", err)
	}
	blocker, err := adapter.pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("acquire ownership blocker: %v", err)
	}
	const blockedLockKey = "99/1"
	if _, err := blocker.Exec(
		ctx,
		`SELECT pg_catalog.pg_advisory_lock(
			pg_catalog.hashtext($1), pg_catalog.hashtext($2)
		)`,
		unlockedTarget.options.JobID,
		blockedLockKey,
	); err != nil {
		blocker.Release()
		t.Fatalf("lock ownership blocker: %v", err)
	}
	blockedCtx, blockedCancel := context.WithTimeout(ctx, 50*time.Millisecond)
	if _, err := unlockedTarget.Begin(blockedCtx, sink.BatchMetadata{
		ID: 99, Attempt: 1, Rows: 1, Bytes: 1,
		FirstPosition: loadPosition(1, "blocked"),
		LastPosition:  loadPosition(1, "blocked"),
	}); err == nil || !strings.Contains(err.Error(), "lock AGE load batch") {
		blockedCancel()
		blocker.Release()
		t.Fatalf("Begin(blocked ownership) error = %v", err)
	}
	blockedCancel()
	if _, err := blocker.Exec(
		ctx,
		`SELECT pg_catalog.pg_advisory_unlock(
			pg_catalog.hashtext($1), pg_catalog.hashtext($2)
		)`,
		unlockedTarget.options.JobID,
		blockedLockKey,
	); err != nil {
		blocker.Release()
		t.Fatalf("unlock ownership blocker: %v", err)
	}
	blocker.Release()

	const (
		graphName = "af_it_load_sink"
		jobID     = "55555555-6666-4777-8888-999999999999"
	)
	dropGraphIfPresent(t, ctx, adapter, graphName)
	deleteLoadSinkJob(t, ctx, adapter, jobID)
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		dropGraphIfPresent(t, cleanupCtx, adapter, graphName)
		deleteLoadSinkJob(t, cleanupCtx, adapter, jobID)
	})

	if err := store.CreateJob(ctx, meta.Job{
		ID:                jobID,
		Name:              "load-sink-integration",
		SourceType:        "csv",
		LoadMode:          "create",
		TargetGraph:       graphName,
		ConfigFingerprint: strings.Repeat("a", 64),
	}); err != nil {
		t.Fatalf("CreateJob() error = %v", err)
	}
	if err := store.StartJob(ctx, jobID); err != nil {
		t.Fatalf("StartJob() error = %v", err)
	}

	var graph GraphCatalog
	var person, knows LabelCatalog
	if err := adapter.InTransaction(ctx, func(transaction *Transaction) error {
		if err := transaction.CreateGraph(ctx, graphName); err != nil {
			return err
		}
		if err := transaction.CreateLabel(ctx, graphName, "Person", VertexLabel); err != nil {
			return err
		}
		if err := transaction.CreateLabel(ctx, graphName, "KNOWS", EdgeLabel); err != nil {
			return err
		}
		var lookupErr error
		graph, lookupErr = transaction.LookupGraph(ctx, graphName)
		if lookupErr != nil {
			return lookupErr
		}
		person, lookupErr = transaction.LookupLabel(ctx, graphName, "Person")
		if lookupErr != nil {
			return lookupErr
		}
		knows, lookupErr = transaction.LookupLabel(ctx, graphName, "KNOWS")
		return lookupErr
	}); err != nil {
		t.Fatalf("create sink graph: %v", err)
	}
	graphGeneration, err := store.RegisterGraphGeneration(ctx, meta.GraphGeneration{
		JobID:        jobID,
		GraphName:    graphName,
		GraphOID:     graph.GraphOID,
		NamespaceOID: graph.NamespaceOID,
		Generation:   1,
		State:        meta.GenerationLoading,
	})
	if err != nil {
		t.Fatalf("RegisterGraphGeneration() error = %v", err)
	}
	personGeneration := registerLoadLabel(
		t,
		ctx,
		store,
		graphGeneration,
		person,
	)
	knowsGeneration := registerLoadLabel(
		t,
		ctx,
		store,
		graphGeneration,
		knows,
	)
	if _, err := NewLoadSink(ctx, adapter, LoadSinkOptions{
		JobID: jobID, Graph: graphGeneration,
		MissingEndpoint: config.MissingEndpointDefer,
	}); err == nil || !strings.Contains(err.Error(), "deferred endpoints") {
		t.Fatalf("deferred NewLoadSink() error = %v", err)
	}
	if _, err := NewLoadSink(ctx, adapter, LoadSinkOptions{
		JobID: jobID, Graph: graphGeneration,
		MissingEndpoint: config.MissingEndpointError,
	}); err == nil || !strings.Contains(err.Error(), "at least one label") {
		t.Fatalf("label-less NewLoadSink() error = %v", err)
	}
	if _, err := NewLoadSink(ctx, adapter, LoadSinkOptions{
		JobID: jobID, Graph: graphGeneration,
		Labels: []LoadLabel{
			{Catalog: person, Generation: personGeneration},
			{Catalog: person, Generation: personGeneration},
		},
		MissingEndpoint: config.MissingEndpointError,
	}); err == nil || !strings.Contains(err.Error(), "duplicate load label") {
		t.Fatalf("duplicate-label NewLoadSink() error = %v", err)
	}
	mismatchedLabel := personGeneration
	mismatchedLabel.RelationOID++
	if _, err := NewLoadSink(ctx, adapter, LoadSinkOptions{
		JobID: jobID, Graph: graphGeneration,
		Labels: []LoadLabel{
			{Catalog: person, Generation: mismatchedLabel},
		},
		MissingEndpoint: config.MissingEndpointError,
	}); err == nil || !strings.Contains(err.Error(), "does not match AGE catalog") {
		t.Fatalf("mismatched-label NewLoadSink() error = %v", err)
	}
	changedMapping := personGeneration
	changedMapping.MappingGeneration++
	if _, err := NewLoadSink(ctx, adapter, LoadSinkOptions{
		JobID: jobID, Graph: graphGeneration,
		Labels: []LoadLabel{
			{Catalog: person, Generation: changedMapping},
		},
		MissingEndpoint: config.MissingEndpointError,
	}); err == nil || !strings.Contains(err.Error(), "admit load label") {
		t.Fatalf("changed-mapping NewLoadSink() error = %v", err)
	}
	changedGraph := graphGeneration
	changedGraph.GraphOID++
	changedGraph.NamespaceOID++
	if _, err := NewLoadSink(ctx, adapter, LoadSinkOptions{
		JobID: jobID, Graph: changedGraph,
		Labels: []LoadLabel{
			{Catalog: person, Generation: personGeneration},
		},
		MissingEndpoint: config.MissingEndpointError,
	}); err == nil || !strings.Contains(err.Error(), "admit load graph generation") {
		t.Fatalf("changed-graph NewLoadSink() error = %v", err)
	}
	suppliedPersonGeneration := personGeneration
	suppliedPersonGeneration.ID += 100_000
	suppliedKnowsGeneration := knowsGeneration
	suppliedKnowsGeneration.ID += 100_000
	suppliedGraphGeneration := graphGeneration
	suppliedGraphGeneration.ID += 100_000
	quarantinePath := filepath.Join(t.TempDir(), "rejects.jsonl")
	quarantine, err := reject.NewJSONLWriter(quarantinePath)
	if err != nil {
		t.Fatalf("NewJSONLWriter() error = %v", err)
	}
	t.Cleanup(func() { _ = quarantine.Close() })
	target, err := NewLoadSink(ctx, adapter, LoadSinkOptions{
		JobID: jobID,
		Graph: suppliedGraphGeneration,
		Labels: []LoadLabel{
			{Catalog: person, Generation: suppliedPersonGeneration},
			{Catalog: knows, Generation: suppliedKnowsGeneration},
		},
		MissingEndpoint: config.MissingEndpointQuarantine,
		Quarantine:      quarantine,
	})
	if err != nil {
		t.Fatalf("NewLoadSink() error = %v", err)
	}
	if target.labels["Person"].Generation.ID != personGeneration.ID ||
		target.labels["KNOWS"].Generation.ID != knowsGeneration.ID ||
		target.options.Graph.ID != graphGeneration.ID {
		t.Fatalf("NewLoadSink() retained non-canonical generation IDs")
	}
	concurrentTarget, err := NewLoadSink(ctx, adapter, LoadSinkOptions{
		JobID: jobID,
		Graph: graphGeneration,
		Labels: []LoadLabel{
			{Catalog: person, Generation: personGeneration},
			{Catalog: knows, Generation: knowsGeneration},
		},
		MissingEndpoint: config.MissingEndpointQuarantine,
	})
	if err != nil {
		t.Fatalf("NewLoadSink(concurrent target) error = %v", err)
	}

	vertexRecords := []model.Record{
		vertexLoadRecord("p1", "Ada", 2, "v1"),
		vertexLoadRecord("p2", "Grace", 3, "v2"),
		vertexLoadRecord("p3", "Isolated", 4, "v3"),
	}
	runLoadBatch(t, ctx, target, 1, vertexRecords, 300)

	edgeRecords := []model.Record{
		edgeLoadRecord("e1", "p1", "p2", 5, "e1"),
		edgeLoadRecord("e2", "p1", "p1", 6, "e2"),
		edgeLoadRecord("e3", "p1", "p2", 7, "e3"),
		edgeLoadRecord("missing", "p1", "absent", 8, "e4"),
	}
	runLoadBatch(t, ctx, target, 2, edgeRecords, 400)
	runLoadBatch(t, ctx, target, 2, edgeRecords, 400)
	mismatchedReplay := &committedReplayTransaction{
		sink: target,
		metadata: sink.BatchMetadata{
			ID: 2, Attempt: 1, Rows: len(edgeRecords), Bytes: 400,
			LastPosition: loadPosition(99, "wrong-checkpoint"),
		},
		wrote: true,
	}
	if err := mismatchedReplay.Commit(ctx, checkpoint.State{
		BatchID: 2, Attempt: 1, Phase: checkpoint.PhaseCommitted,
		Position: loadPosition(99, "wrong-checkpoint"),
	}); err == nil || !strings.Contains(err.Error(), "stored checkpoint") {
		t.Fatalf("mismatched replay Commit() error = %v", err)
	}
	missingReplay := &committedReplayTransaction{
		sink: target,
		metadata: sink.BatchMetadata{
			ID: 999, Attempt: 1, Rows: 1, Bytes: 1,
			LastPosition: loadPosition(99, "missing-checkpoint"),
		},
		wrote: true,
	}
	if err := missingReplay.Commit(ctx, checkpoint.State{
		BatchID: 999, Attempt: 1, Phase: checkpoint.PhaseCommitted,
		Position: loadPosition(99, "missing-checkpoint"),
	}); err == nil || !strings.Contains(err.Error(), "read committed replay") {
		t.Fatalf("missing replay Commit() error = %v", err)
	}

	errorTarget, err := NewLoadSink(ctx, adapter, LoadSinkOptions{
		JobID: jobID,
		Graph: graphGeneration,
		Labels: []LoadLabel{
			{Catalog: person, Generation: personGeneration},
			{Catalog: knows, Generation: knowsGeneration},
		},
		MissingEndpoint: config.MissingEndpointError,
	})
	if err != nil {
		t.Fatalf("NewLoadSink(error policy) error = %v", err)
	}
	missingOnly := []model.Record{
		edgeLoadRecord("missing-error", "p1", "absent", 9, "e5"),
	}
	first, _ := missingOnly[0].SourcePosition()
	failedBatch := sink.BatchMetadata{
		ID: 3, Attempt: 1, Rows: 1, Bytes: 100,
		FirstPosition: first, LastPosition: first,
	}
	failedTransaction, err := errorTarget.Begin(ctx, failedBatch)
	if err != nil {
		t.Fatalf("Begin(missing error) error = %v", err)
	}
	if err := failedTransaction.Write(ctx, missingOnly); err == nil ||
		!strings.Contains(err.Error(), "missing endpoints") {
		t.Fatalf("Write(missing error) error = %v", err)
	}
	if err := failedTransaction.(*loadTransaction).tx.Rollback(ctx); err != nil {
		t.Fatalf("close missing endpoint transaction: %v", err)
	}
	if err := failedTransaction.Rollback(ctx); err != nil {
		t.Fatalf("Rollback(missing error) error = %v", err)
	}
	if err := failedTransaction.Rollback(ctx); err != nil {
		t.Fatalf("repeated Rollback(missing error) error = %v", err)
	}
	if err := store.StartJob(ctx, jobID); err != nil {
		t.Fatalf("resume after missing endpoint: %v", err)
	}
	finalEdges := []model.Record{
		edgeLoadRecord("e4", "p2", "p1", 10, "e6"),
	}
	runLoadBatchAttempt(t, ctx, target, 3, 2, finalEdges, 100)
	runLoadBatchAttempt(t, ctx, target, 3, 2, finalEdges, 100)
	runLoadBatch(t, ctx, target, 4, []model.Record{
		edgeLoadRecord("missing-only", "absent-a", "absent-b", 11, "e7"),
	}, 100)
	runConcurrentReplayBatch(
		t,
		ctx,
		target,
		concurrentTarget,
		5,
		[]model.Record{edgeLoadRecord("", "p3", "p1", 12, "e8")},
		100,
	)

	duplicateEdge := []model.Record{
		edgeLoadRecord("e1", "p2", "p3", 13, "e9"),
	}
	failLoadBatch(
		t,
		ctx,
		target,
		6,
		1,
		duplicateEdge,
		"COPY edge identities",
	)
	if err := store.StartJob(ctx, jobID); err != nil {
		t.Fatalf("resume after duplicate edge: %v", err)
	}
	runLoadBatchAttempt(t, ctx, target, 6, 2, []model.Record{
		edgeLoadRecord("e5", "p2", "p3", 14, "e10"),
	}, 100)

	unknownEndpoint := edgeLoadRecord("bad-label", "p1", "p2", 15, "e11")
	unknownEndpoint.Edge.Start.Label = "Unknown"
	failLoadBatch(
		t,
		ctx,
		target,
		7,
		1,
		[]model.Record{unknownEndpoint},
		"not a registered vertex label",
	)
	if err := store.StartJob(ctx, jobID); err != nil {
		t.Fatalf("resume after unknown endpoint label: %v", err)
	}
	runLoadBatchAttempt(t, ctx, target, 7, 2, []model.Record{
		edgeLoadRecord("e6", "p2", "p3", 16, "e12"),
	}, 100)
	quarantineData, err := os.ReadFile(quarantinePath)
	if err != nil {
		t.Fatalf("read quarantine output: %v", err)
	}
	if lines := strings.Count(strings.TrimSpace(string(quarantineData)), "\n") + 1; lines != 2 {
		t.Fatalf("quarantine output lines = %d, want 2", lines)
	}

	assertLoadSinkRows(t, ctx, adapter, graphName, jobID)
	if err := store.SetGraphGenerationState(
		ctx,
		graphGeneration.ID,
		meta.GenerationLoading,
		meta.GenerationActive,
	); err != nil {
		t.Fatalf("activate graph generation: %v", err)
	}
	if err := store.CompleteJob(ctx, jobID); err != nil {
		t.Fatalf("CompleteJob() error = %v", err)
	}
	activeGeneration := graphGeneration
	activeGeneration.State = meta.GenerationActive
	const (
		appendJobID     = "12121212-3434-4567-8890-121212121212"
		competitorJobID = "23232323-4545-4678-8901-232323232323"
		upsertJobID     = "34343434-5656-4789-9012-343434343434"
		staleJobID      = "56565656-7878-4012-8345-565656565656"
	)
	for _, incrementalJobID := range []string{
		appendJobID,
		competitorJobID,
		upsertJobID,
		staleJobID,
	} {
		deleteLoadSinkJob(t, ctx, adapter, incrementalJobID)
		incrementalJobID := incrementalJobID
		t.Cleanup(func() {
			cleanupCtx, cleanupCancel := context.WithTimeout(
				context.Background(),
				10*time.Second,
			)
			defer cleanupCancel()
			deleteLoadSinkJob(
				t,
				cleanupCtx,
				adapter,
				incrementalJobID,
			)
		})
	}
	appendTarget := createIncrementalLoadTarget(
		t,
		ctx,
		adapter,
		store,
		appendJobID,
		activeGeneration,
		person,
		personGeneration,
		knows,
		knowsGeneration,
		LoadSinkOptions{
			Mode:             config.LoadAppend,
			AppendDuplicate:  config.AppendDuplicateIgnoreIdentical,
			PropertyMode:     config.PropertiesReplace,
			MissingEndpoint:  config.MissingEndpointDefer,
			MaxDeferredEdges: 10,
		},
	)
	competitorAdapter := openIntegrationAdapter(t, ctx, dsn, 2)
	competitorStore, err := competitorAdapter.Metadata()
	if err != nil {
		competitorAdapter.Close()
		t.Fatalf("open competitor metadata: %v", err)
	}
	competitorTarget := createIncrementalLoadTarget(
		t,
		ctx,
		competitorAdapter,
		competitorStore,
		competitorJobID,
		activeGeneration,
		person,
		personGeneration,
		knows,
		knowsGeneration,
		LoadSinkOptions{
			Mode:            config.LoadAppend,
			AppendDuplicate: config.AppendDuplicateError,
			PropertyMode:    config.PropertiesReplace,
			MissingEndpoint: config.MissingEndpointError,
		},
	)
	appendVertices := []model.Record{
		vertexLoadRecord("p1", "Ada", 20, "append-v1"),
		vertexLoadRecord("p-new", "New", 21, "append-v2"),
	}
	firstPosition, _ := appendVertices[0].SourcePosition()
	lastPosition, _ := appendVertices[len(appendVertices)-1].SourcePosition()
	held, err := appendTarget.Begin(ctx, sink.BatchMetadata{
		ID: 1, Attempt: 1, Rows: len(appendVertices), Bytes: 200,
		FirstPosition: firstPosition, LastPosition: lastPosition,
	})
	if err != nil {
		t.Fatalf("hold append batch: %v", err)
	}
	if _, err := competitorTarget.Begin(ctx, sink.BatchMetadata{
		ID: 1, Attempt: 1, Rows: 1, Bytes: 100,
		FirstPosition: loadPosition(22, "competitor"),
		LastPosition:  loadPosition(22, "competitor"),
	}); !errors.Is(err, meta.ErrIncrementalConflict) {
		_ = held.Rollback(ctx)
		competitorAdapter.Close()
		t.Fatalf("concurrent incremental Begin() error = %v", err)
	}
	if err := held.Rollback(ctx); err != nil {
		t.Fatalf("rollback held append batch: %v", err)
	}
	if err := store.StartJob(ctx, appendJobID); err != nil {
		t.Fatalf("restart append job: %v", err)
	}
	runLoadBatchAttempt(t, ctx, appendTarget, 1, 2, appendVertices, 200)
	runLoadBatch(t, ctx, appendTarget, 2, []model.Record{
		edgeLoadRecord("e1", "p1", "p2", 22, "append-e1"),
		edgeLoadRecord("e-new", "p-new", "p1", 23, "append-e2"),
		edgeLoadRecord("e-deferred", "p-new", "p-later", 24, "append-e3"),
	}, 300)
	runLoadBatch(t, ctx, appendTarget, 3, []model.Record{
		vertexLoadRecord("p1", "Ada", 25, "append-identical-vertex"),
	}, 100)
	runLoadBatch(t, ctx, appendTarget, 4, []model.Record{
		edgeLoadRecord("e1", "p1", "p2", 26, "append-identical-edge"),
	}, 100)
	runLoadBatch(t, ctx, appendTarget, 5, []model.Record{
		edgeLoadRecord("", "p-new", "p-anonymous-later", 27, "append-anonymous-edge"),
	}, 100)
	runLoadBatch(t, ctx, appendTarget, 6, []model.Record{
		vertexLoadRecord("p-anonymous-later", "Anonymous endpoint", 28, "append-anonymous-endpoint"),
	}, 100)
	failLoadBatch(t, ctx, appendTarget, 7, 1, []model.Record{
		edgeLoadRecord("e1", "p2", "p1", 27, "append-conflicting-edge"),
	}, "conflicts with existing endpoints")
	if err := store.StartJob(ctx, appendJobID); err != nil {
		t.Fatalf("restart conflicting append job: %v", err)
	}
	runLoadBatchAttempt(t, ctx, appendTarget, 7, 2, []model.Record{
		edgeLoadRecord("e1", "p1", "p2", 28, "append-recovered-edge"),
	}, 100)
	if err := store.CompleteJob(ctx, appendJobID); err != nil {
		t.Fatalf("complete append job: %v", err)
	}
	failLoadBatch(t, ctx, competitorTarget, 1, 1, []model.Record{
		vertexLoadRecord("p1", "Ada", 29, "append-error-duplicate"),
	}, "append duplicate identity")
	if err := competitorStore.StartJob(ctx, competitorJobID); err != nil {
		competitorAdapter.Close()
		t.Fatalf("restart competitor append job: %v", err)
	}
	competitorTarget.options.MissingEndpoint = config.MissingEndpointDefer
	competitorTarget.options.MaxDeferredEdges = 10
	failLoadBatch(t, ctx, competitorTarget, 1, 2, []model.Record{
		edgeLoadRecord("same-batch-deferred", "missing-a", "missing-b", 30, "deferred-a"),
		edgeLoadRecord("same-batch-deferred", "missing-a", "missing-b", 31, "deferred-b"),
	}, "duplicates external identity in batch")
	competitorAdapter.Close()

	upsertTarget := createIncrementalLoadTarget(
		t,
		ctx,
		adapter,
		store,
		upsertJobID,
		activeGeneration,
		person,
		personGeneration,
		knows,
		knowsGeneration,
		LoadSinkOptions{
			Mode:            config.LoadUpsert,
			AppendDuplicate: config.AppendDuplicateError,
			PropertyMode:    config.PropertiesMerge,
			MissingEndpoint: config.MissingEndpointError,
		},
	)
	runLoadBatch(t, ctx, upsertTarget, 1, []model.Record{
		vertexLoadRecord("p1", "Ada Updated", 25, "upsert-v1"),
		vertexLoadRecord("p-later", "Later", 26, "upsert-v2"),
	}, 200)
	runLoadBatch(t, ctx, upsertTarget, 2, []model.Record{
		edgeLoadRecord("e1", "p1", "p-new", 27, "upsert-e1"),
	}, 100)
	failLoadBatch(t, ctx, upsertTarget, 3, 1, []model.Record{
		edgeLoadRecord("", "p1", "p-new", 30, "upsert-missing-identity"),
	}, "external identity is required")
	if err := store.StartJob(ctx, upsertJobID); err != nil {
		t.Fatalf("restart missing-identity upsert job: %v", err)
	}
	runLoadBatchAttempt(t, ctx, upsertTarget, 3, 2, []model.Record{
		edgeLoadRecord("e1", "p1", "p-new", 31, "upsert-recovered-edge"),
	}, 100)
	if err := store.CompleteJob(ctx, upsertJobID); err != nil {
		t.Fatalf("complete upsert job: %v", err)
	}
	var deferred, vertices, edges int64
	if err := adapter.pool.QueryRow(
		ctx,
		fmt.Sprintf(
			`SELECT
				(SELECT COUNT(*)
				 FROM agefreighter_meta.deferred_edge
				 WHERE graph_generation_id = $1),
				(SELECT COUNT(*) FROM %s),
				(SELECT COUNT(*) FROM %s)`,
			pgxIdentifier(graphName, "Person"),
			pgxIdentifier(graphName, "KNOWS"),
		),
		activeGeneration.ID,
	).Scan(&deferred, &vertices, &edges); err != nil {
		t.Fatalf("inspect incremental sink rows: %v", err)
	}
	if deferred != 0 || vertices != 6 || edges != 10 {
		t.Fatalf(
			"incremental rows = deferred %d vertices %d edges %d",
			deferred,
			vertices,
			edges,
		)
	}
	validationConnection, err := adapter.pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("acquire validation connection: %v", err)
	}
	originalGraph := appendTarget.options.Graph
	appendTarget.options.Graph.GraphName = graphName + "_missing"
	if err := appendTarget.validateIncrementalGeneration(
		ctx,
		validationConnection.Conn(),
	); err == nil {
		validationConnection.Release()
		t.Fatal("validateIncrementalGeneration() accepted a missing graph")
	}
	appendTarget.options.Graph = originalGraph
	appendTarget.options.Graph.ID++
	if err := appendTarget.validateIncrementalGeneration(
		ctx,
		validationConnection.Conn(),
	); !errors.Is(err, meta.ErrGenerationMismatch) {
		validationConnection.Release()
		t.Fatalf("validateIncrementalGeneration(graph ID) error = %v", err)
	}
	appendTarget.options.Graph = originalGraph
	personBinding := appendTarget.labels[model.Label("Person")]
	personBinding.Catalog.LabelName = "MissingLabel"
	appendTarget.labels[model.Label("Person")] = personBinding
	if err := appendTarget.validateIncrementalGeneration(
		ctx,
		validationConnection.Conn(),
	); err == nil {
		validationConnection.Release()
		t.Fatal("validateIncrementalGeneration() accepted a missing label")
	}
	personBinding.Catalog.LabelName = "Person"
	personBinding.Generation.RelationOID++
	appendTarget.labels[model.Label("Person")] = personBinding
	if err := appendTarget.validateIncrementalGeneration(
		ctx,
		validationConnection.Conn(),
	); !errors.Is(err, meta.ErrGenerationMismatch) {
		validationConnection.Release()
		t.Fatalf("validateIncrementalGeneration(label catalog) error = %v", err)
	}
	personBinding.Generation.RelationOID--
	appendTarget.labels[model.Label("Person")] = personBinding
	validationConnection.Release()
	staleTarget := createIncrementalLoadTarget(
		t,
		ctx,
		adapter,
		store,
		staleJobID,
		activeGeneration,
		person,
		personGeneration,
		knows,
		knowsGeneration,
		LoadSinkOptions{
			Mode:            config.LoadAppend,
			AppendDuplicate: config.AppendDuplicateError,
			PropertyMode:    config.PropertiesReplace,
			MissingEndpoint: config.MissingEndpointError,
		},
	)
	if err := store.SetGraphGenerationState(
		ctx,
		activeGeneration.ID,
		meta.GenerationActive,
		meta.GenerationRetired,
	); err != nil {
		t.Fatalf("retire stale incremental generation: %v", err)
	}
	if _, err := staleTarget.Begin(ctx, sink.BatchMetadata{
		ID: 1, Attempt: 1, Rows: 1, Bytes: 100,
		FirstPosition: loadPosition(32, "stale-generation"),
		LastPosition:  loadPosition(32, "stale-generation"),
	}); !errors.Is(err, meta.ErrGenerationMismatch) {
		t.Fatalf("Begin(stale generation) error = %v", err)
	}
	if _, err := target.Begin(ctx, sink.BatchMetadata{
		ID: 8, Attempt: 1, Rows: 1, Bytes: 1,
		FirstPosition: loadPosition(17, "after-complete"),
		LastPosition:  loadPosition(17, "after-complete"),
	}); err == nil || !strings.Contains(err.Error(), "start AGE load batch") {
		t.Fatalf("Begin(after completion) error = %v", err)
	}
}

func TestIncrementalSinkClosedTransactionIntegration(t *testing.T) {
	dsn := integrationDSN(t)
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	adapter := openIntegrationAdapter(t, ctx, dsn, 2)
	defer adapter.Close()
	tx, err := adapter.pool.Begin(ctx)
	if err != nil {
		t.Fatalf("begin closed incremental transaction: %v", err)
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatalf("close incremental transaction: %v", err)
	}
	graph := meta.GraphGeneration{
		ID: 1, GraphName: "closed_graph", GraphOID: 1, NamespaceOID: 1,
		State: meta.GenerationActive,
	}
	vertexBinding := LoadLabel{
		Catalog: LabelCatalog{
			GraphName: "closed_graph", LabelName: "Person",
			GraphOID: 1, NamespaceOID: 1, LabelID: 1,
			RelationOID: 2, SequenceOID: 3, Kind: VertexLabel,
		},
		Generation: meta.LabelGeneration{
			ID: 1, GraphGenerationID: 1, LabelName: "Person",
			Kind: meta.VertexLabel, GraphNamespaceOID: 1, LabelID: 1,
			RelationOID: 2, SequenceOID: 3, MappingGeneration: 1,
		},
	}
	edgeBinding := vertexBinding
	edgeBinding.Catalog.LabelName = "KNOWS"
	edgeBinding.Catalog.LabelID = 2
	edgeBinding.Catalog.Kind = EdgeLabel
	edgeBinding.Generation.ID = 2
	edgeBinding.Generation.LabelName = "KNOWS"
	edgeBinding.Generation.LabelID = 2
	edgeBinding.Generation.Kind = meta.EdgeLabel
	target := &LoadSink{
		adapter: adapter,
		options: LoadSinkOptions{
			JobID:            "45454545-6767-4890-8123-454545454545",
			Graph:            graph,
			Mode:             config.LoadAppend,
			AppendDuplicate:  config.AppendDuplicateError,
			PropertyMode:     config.PropertiesReplace,
			MissingEndpoint:  config.MissingEndpointDefer,
			MaxDeferredEdges: 1,
		},
		labels: map[model.Label]LoadLabel{
			"Person": vertexBinding,
			"KNOWS":  edgeBinding,
		},
	}
	transaction := &loadTransaction{
		sink: target,
		tx:   tx,
		metadata: sink.BatchMetadata{
			ID: 1, Attempt: 1, Rows: 1, Bytes: 1,
		},
	}
	vertex := model.Vertex{
		Label: "Person", Namespace: "crm", ExternalID: "p1",
		Position: loadPosition(1, "closed-vertex"),
	}
	edge := model.Edge{
		Label: "KNOWS", Namespace: "crm", ExternalID: "e1",
		Start: model.Endpoint{
			Label: "Person", Namespace: "crm", ExternalID: "p1",
		},
		End: model.Endpoint{
			Label: "Person", Namespace: "crm", ExternalID: "p2",
		},
		Position: loadPosition(2, "closed-edge"),
	}
	staged := stagedEdge{
		record: &edge, label: edgeBinding,
		startLabelID: 1, endLabelID: 1, properties: []byte(`{}`),
	}
	resolved := resolvedEdge{
		stagedEdge: staged,
		startID:    GraphID(1<<48 | 1),
		endID:      GraphID(1<<48 | 2),
	}
	checks := []struct {
		name string
		run  func() error
	}{
		{
			name: "defer",
			run: func() error {
				return transaction.deferMissingEdges(ctx, []stagedEdge{staged})
			},
		},
		{
			name: "drain",
			run:  func() error { return transaction.drainDeferredEdges(ctx) },
		},
		{
			name: "write vertices",
			run: func() error {
				return transaction.writeVerticesIncremental(
					ctx,
					vertexBinding,
					[]*model.Vertex{&vertex},
					[][]byte{[]byte(`{}`)},
				)
			},
		},
		{
			name: "apply vertices",
			run: func() error {
				return transaction.applyIncrementalVertices(
					ctx,
					vertexBinding,
					[]incrementalVertexDecision{{
						graphID: GraphID(1<<48 | 1),
					}},
				)
			},
		},
		{
			name: "assign vertices",
			run: func() error {
				return assignVertexDecisionIDs(
					ctx,
					tx,
					vertexBinding,
					[]incrementalVertexDecision{{isNew: true}},
				)
			},
		},
		{
			name: "write edges",
			run: func() error {
				return transaction.writeEdgesIncremental(
					ctx,
					edgeBinding,
					[]resolvedEdge{resolved},
				)
			},
		},
		{
			name: "apply edges",
			run: func() error {
				return transaction.applyIncrementalEdges(
					ctx,
					edgeBinding,
					[]incrementalEdgeDecision{{
						graphID: GraphID(2<<48 | 1),
						startID: resolved.startID,
						endID:   resolved.endID,
					}},
				)
			},
		},
		{
			name: "assign edges",
			run: func() error {
				return assignEdgeDecisionIDs(
					ctx,
					tx,
					edgeBinding,
					[]incrementalEdgeDecision{{isNew: true}},
				)
			},
		},
		{
			name: "write drain",
			run: func() error {
				drainTransaction := &loadTransaction{
					sink: target,
					tx:   tx,
					metadata: sink.BatchMetadata{
						Rows: 0,
					},
				}
				return drainTransaction.Write(ctx, nil)
			},
		},
		{
			name: "direct vertices",
			run: func() error {
				directTarget := &LoadSink{
					adapter: target.adapter,
					options: target.options,
					labels:  target.labels,
				}
				directTarget.options.Mode = config.LoadCreate
				directTransaction := &loadTransaction{
					sink: directTarget,
					tx:   tx,
				}
				return directTransaction.writeVertices(
					ctx,
					"Person",
					[]*model.Vertex{&vertex},
				)
			},
		},
		{
			name: "insert vertex identities",
			run: func() error {
				return transaction.insertVertexIdentities(
					ctx,
					[]vertexIdentityRow{{
						label:      vertexBinding,
						namespace:  "crm",
						externalID: "p1",
						graphID:    GraphID(1<<48 | 1),
					}},
				)
			},
		},
		{
			name: "resolve edges",
			run: func() error {
				_, _, err := transaction.resolveEdges(
					ctx,
					[]stagedEdge{staged},
				)
				return err
			},
		},
		{
			name: "insert edge identities",
			run: func() error {
				return transaction.insertEdgeIdentities(
					ctx,
					[]resolvedEdge{resolved},
					[]EdgeRow{{
						ID:      GraphID(2<<48 | 1),
						StartID: resolved.startID,
						EndID:   resolved.endID,
					}},
				)
			},
		},
	}
	for _, check := range checks {
		t.Run(check.name, func(t *testing.T) {
			if err := check.run(); err == nil {
				t.Fatalf("%s succeeded with a closed transaction", check.name)
			}
		})
	}
}

func integrationDSN(t *testing.T) string {
	t.Helper()
	dsn := strings.TrimSpace(os.Getenv(integrationDSNEnvironment))
	if dsn == "" {
		t.Skip("set " + integrationDSNEnvironment + " to run Apache AGE integration tests")
	}
	return dsn
}

func registerLoadLabel(
	t *testing.T,
	ctx context.Context,
	store *meta.Store,
	graph meta.GraphGeneration,
	catalog LabelCatalog,
) meta.LabelGeneration {
	t.Helper()
	kind := meta.VertexLabel
	if catalog.Kind == EdgeLabel {
		kind = meta.EdgeLabel
	}
	generation, err := store.RegisterLabelGeneration(ctx, meta.LabelGeneration{
		GraphGenerationID: graph.ID,
		LabelName:         catalog.LabelName,
		Kind:              kind,
		GraphNamespaceOID: catalog.NamespaceOID,
		LabelID:           catalog.LabelID,
		RelationOID:       catalog.RelationOID,
		SequenceOID:       catalog.SequenceOID,
		MappingGeneration: 1,
	})
	if err != nil {
		t.Fatalf("register label %q: %v", catalog.LabelName, err)
	}
	return generation
}

func runLoadBatch(
	t *testing.T,
	ctx context.Context,
	target *LoadSink,
	id uint64,
	records []model.Record,
	bytes int64,
) {
	runLoadBatchAttempt(t, ctx, target, id, 1, records, bytes)
}

func runLoadBatchAttempt(
	t *testing.T,
	ctx context.Context,
	target *LoadSink,
	id uint64,
	attempt uint32,
	records []model.Record,
	bytes int64,
) {
	t.Helper()
	first, _ := records[0].SourcePosition()
	last, _ := records[len(records)-1].SourcePosition()
	batch := sink.BatchMetadata{
		ID:            id,
		Attempt:       attempt,
		Rows:          len(records),
		Bytes:         bytes,
		FirstPosition: first,
		LastPosition:  last,
	}

	transaction, err := target.Begin(ctx, batch)
	if err != nil {
		t.Fatalf("Begin(batch %d) error = %v", id, err)
	}
	if err := transaction.Write(ctx, records); err != nil {
		_ = transaction.Rollback(ctx)
		t.Fatalf("Write(batch %d) error = %v", id, err)
	}
	state := checkpoint.State{
		BatchID: id, Attempt: attempt,
		Phase: checkpoint.PhaseCommitted, Position: last,
	}
	if err := transaction.Commit(ctx, state); err != nil {
		t.Fatalf("Commit(batch %d) error = %v", id, err)
	}
	if err := transaction.Commit(ctx, state); err == nil ||
		!strings.Contains(err.Error(), "finalized") {
		t.Fatalf("repeated Commit(batch %d) error = %v", id, err)
	}
	if err := transaction.Rollback(ctx); err != nil {
		t.Fatalf("Rollback committed batch %d error = %v", id, err)
	}
}

func failLoadBatch(
	t *testing.T,
	ctx context.Context,
	target *LoadSink,
	id uint64,
	attempt uint32,
	records []model.Record,
	want string,
) {
	t.Helper()
	first, _ := records[0].SourcePosition()
	last, _ := records[len(records)-1].SourcePosition()
	transaction, err := target.Begin(ctx, sink.BatchMetadata{
		ID: id, Attempt: attempt, Rows: len(records), Bytes: 100,
		FirstPosition: first, LastPosition: last,
	})
	if err != nil {
		t.Fatalf("Begin(failing batch %d) error = %v", id, err)
	}

	if err := transaction.Write(ctx, records); err == nil ||
		!strings.Contains(err.Error(), want) {
		t.Fatalf("Write(failing batch %d) error = %v, want %q", id, err, want)
	}
	if err := transaction.Rollback(ctx); err != nil {
		t.Fatalf("Rollback(failing batch %d) error = %v", id, err)
	}
}

func runConcurrentReplayBatch(
	t *testing.T,
	ctx context.Context,
	firstTarget *LoadSink,
	secondTarget *LoadSink,
	id uint64,
	records []model.Record,
	bytes int64,
) {
	t.Helper()
	first, _ := records[0].SourcePosition()
	last, _ := records[len(records)-1].SourcePosition()
	batch := sink.BatchMetadata{
		ID: id, Attempt: 1, Rows: len(records), Bytes: bytes,
		FirstPosition: first, LastPosition: last,
	}
	firstTransaction, err := firstTarget.Begin(ctx, batch)
	if err != nil {
		t.Fatalf("first Begin(batch %d) error = %v", id, err)
	}

	type beginResult struct {
		transaction sink.Transaction
		err         error
	}
	secondResult := make(chan beginResult, 1)
	go func() {
		transaction, beginErr := secondTarget.Begin(ctx, batch)
		secondResult <- beginResult{transaction: transaction, err: beginErr}
	}()

	if err := firstTransaction.Write(ctx, records); err != nil {
		_ = firstTransaction.Rollback(ctx)
		t.Fatalf("first Write(batch %d) error = %v", id, err)
	}
	state := checkpoint.State{
		BatchID: id, Attempt: 1,
		Phase: checkpoint.PhaseCommitted, Position: last,
	}
	if err := firstTransaction.Commit(ctx, state); err != nil {
		t.Fatalf("first Commit(batch %d) error = %v", id, err)
	}

	result := <-secondResult
	if result.err != nil {
		t.Fatalf("second Begin(batch %d) error = %v", id, result.err)
	}
	if _, ok := result.transaction.(*committedReplayTransaction); !ok {
		_ = result.transaction.Rollback(ctx)
		t.Fatalf("second Begin(batch %d) did not return a committed replay", id)
	}
	if err := result.transaction.Write(ctx, records); err != nil {
		t.Fatalf("second Write(batch %d) error = %v", id, err)
	}
	if err := result.transaction.Commit(ctx, state); err != nil {
		t.Fatalf("second Commit(batch %d) error = %v", id, err)
	}
}

func vertexLoadRecord(id, name string, line int64, token string) model.Record {
	return model.VertexRecord(model.Vertex{
		Label:      "Person",
		Namespace:  "crm",
		ExternalID: model.ExternalID(id),
		Properties: model.Properties{
			"name": {Kind: model.ValueString, String: name},
		},
		Position: loadPosition(line, token),
	})
}

func edgeLoadRecord(
	id, start, end string,
	line int64,
	token string,
) model.Record {
	return model.EdgeRecord(model.Edge{
		Label:      "KNOWS",
		Namespace:  "crm",
		ExternalID: model.ExternalID(id),
		Start: model.Endpoint{
			Label: "Person", Namespace: "crm", ExternalID: model.ExternalID(start),
		},
		End: model.Endpoint{
			Label: "Person", Namespace: "crm", ExternalID: model.ExternalID(end),
		},
		Properties: model.Properties{
			"source": {Kind: model.ValueString, String: id},
		},
		Position: loadPosition(line, token),
	})
}

func loadPosition(line int64, token string) model.SourcePosition {
	return model.SourcePosition{
		Connector: "csv",
		Resource:  "fixture.csv",
		Line:      line,
		Offset:    line * 10,
		Token:     token,
	}
}

func assertLoadSinkRows(
	t *testing.T,
	ctx context.Context,
	adapter *Adapter,
	graphName string,
	jobID string,
) {
	t.Helper()
	var vertices, edges, vertexIdentities, edgeIdentities, rejects int64
	if err := adapter.pool.QueryRow(
		ctx,
		fmt.Sprintf(
			`SELECT
				(SELECT COUNT(*) FROM %s),
				(SELECT COUNT(*) FROM %s),
				(SELECT COUNT(*) FROM agefreighter_meta.vertex_identity vi
				 JOIN agefreighter_meta.graph_generation g
				   ON g.graph_generation_id = vi.graph_generation_id
				 WHERE g.job_id = $1::uuid),
				(SELECT COUNT(*) FROM agefreighter_meta.edge_identity ei
				 JOIN agefreighter_meta.graph_generation g
				   ON g.graph_generation_id = ei.graph_generation_id
				 WHERE g.job_id = $1::uuid),
				(SELECT COUNT(*) FROM agefreighter_meta.reject_record
				 WHERE job_id = $1::uuid)`,
			pgxIdentifier(graphName, "Person"),
			pgxIdentifier(graphName, "KNOWS"),
		),
		jobID,
	).Scan(
		&vertices,
		&edges,
		&vertexIdentities,
		&edgeIdentities,
		&rejects,
	); err != nil {
		t.Fatalf("query load sink rows: %v", err)
	}
	if vertices != 3 || edges != 7 ||
		vertexIdentities != 3 || edgeIdentities != 6 || rejects != 2 {
		t.Fatalf(
			"load sink counts = vertices %d edges %d vertex IDs %d edge IDs %d rejects %d",
			vertices,
			edges,
			vertexIdentities,
			edgeIdentities,
			rejects,
		)
	}
	var startID, endID int64
	if err := adapter.pool.QueryRow(
		ctx,
		`SELECT start_graph_id, end_graph_id
		 FROM agefreighter_meta.edge_identity
		 WHERE source_namespace = 'crm' AND external_id = 'e2'`,
	).Scan(&startID, &endID); err != nil {
		t.Fatalf("query self-loop identity: %v", err)
	}
	if startID != endID {
		t.Fatalf("self-loop endpoints = %d -> %d", startID, endID)
	}
	job, err := adapter.Metadata()
	if err != nil {
		t.Fatalf("Metadata() error = %v", err)
	}
	stored, err := job.GetJob(ctx, jobID)
	if err != nil {
		t.Fatalf("GetJob() error = %v", err)
	}
	if stored.NextBatchID != 8 ||
		stored.CommittedRows != 12 ||
		stored.RejectedRows != 2 ||
		stored.ResumeToken != "e12" {
		t.Fatalf("load job checkpoint = %#v", stored)
	}
}

func pgxIdentifier(schema, table string) string {
	return `"` + strings.ReplaceAll(schema, `"`, `""`) +
		`"."` + strings.ReplaceAll(table, `"`, `""`) + `"`
}

func TestLoadSinkRollbackDiagnosticIntegration(t *testing.T) {
	dsn := integrationDSN(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	adapter := openIntegrationAdapter(t, ctx, dsn, 3)
	t.Cleanup(adapter.Close)
	store, err := adapter.Metadata()
	if err != nil {
		t.Fatalf("Metadata() error = %v", err)
	}
	if err := store.Migrate(ctx); err != nil {
		t.Fatalf("Migrate() error = %v", err)
	}
	const jobID = "66666666-7777-4888-8999-aaaaaaaaaaaa"
	deleteLoadSinkJob(t, ctx, adapter, jobID)
	t.Cleanup(func() {
		deleteLoadSinkJob(t, context.Background(), adapter, jobID)
	})
	if err := store.CreateJob(ctx, meta.Job{
		ID: jobID, Name: "rollback", SourceType: "csv", LoadMode: "create",
		TargetGraph: "rollback_graph", ConfigFingerprint: strings.Repeat("b", 64),
	}); err != nil {
		t.Fatalf("CreateJob() error = %v", err)
	}
	if err := store.StartJob(ctx, jobID); err != nil {
		t.Fatalf("StartJob() error = %v", err)
	}
	target := &LoadSink{
		adapter:     adapter,
		diagnostics: store,
		options: LoadSinkOptions{
			JobID: jobID,
			Graph: meta.GraphGeneration{ID: 1},
		},
		labels: map[model.Label]LoadLabel{},
	}
	batch := sink.BatchMetadata{
		ID: 1, Attempt: 1, Rows: 1, Bytes: 10,
		FirstPosition: loadPosition(1, "first"),
		LastPosition:  loadPosition(1, "last"),
	}
	transaction, err := target.Begin(ctx, batch)
	if err != nil {
		t.Fatalf("Begin() error = %v", err)
	}
	if _, err := target.Begin(ctx, batch); err == nil ||
		!strings.Contains(err.Error(), "active transaction") {
		t.Fatalf("concurrent Begin() error = %v", err)
	}
	if err := transaction.Rollback(ctx); err != nil {
		t.Fatalf("Rollback() error = %v", err)
	}
	stored, err := store.GetBatch(ctx, jobID, 1, 1)
	if err != nil {
		t.Fatalf("GetBatch() error = %v", err)
	}
	if stored.Status != meta.BatchFailed ||
		!strings.Contains(stored.ErrorMessage, "rolled back") {
		t.Fatalf("rollback diagnostic = %#v", stored)
	}
	if err := transaction.Rollback(ctx); err != nil {
		t.Fatalf("idempotent Rollback() error = %v", err)
	}
	if _, err := target.Begin(ctx, batch); !errors.Is(err, meta.ErrConflict) {
		t.Fatalf("reopen failed batch Begin() error = %v", err)
	}
	if err := store.StartJob(ctx, jobID); err != nil {
		t.Fatalf("resume rollback job: %v", err)
	}
	batch.Attempt = 2
	checkpointFailure, err := target.Begin(ctx, batch)
	if err != nil {
		t.Fatalf("Begin(checkpoint failure) error = %v", err)
	}
	concrete, ok := checkpointFailure.(*loadTransaction)
	if !ok {
		t.Fatalf("checkpoint transaction type = %T", checkpointFailure)
	}
	concrete.wrote = true
	if err := concrete.Commit(ctx, checkpoint.State{}); err == nil ||
		!strings.Contains(err.Error(), "checkpoint does not match") {
		t.Fatalf("Commit(mismatched checkpoint) error = %v", err)
	}
	stored, err = store.GetBatch(ctx, jobID, 1, 2)
	if err != nil || stored.Status != meta.BatchFailed {
		t.Fatalf("checkpoint failure diagnostic = %#v, %v", stored, err)
	}
	if err := store.StartJob(ctx, jobID); err != nil {
		t.Fatalf("resume checkpoint failure job: %v", err)
	}
	batch.Attempt = 3
	metadataFailure, err := target.Begin(ctx, batch)
	if err != nil {
		t.Fatalf("Begin(metadata failure) error = %v", err)
	}
	metadataConcrete, ok := metadataFailure.(*loadTransaction)
	if !ok {
		t.Fatalf("metadata failure transaction type = %T", metadataFailure)
	}
	metadataConcrete.wrote = true
	if err := store.FailJob(ctx, jobID, "injected checkpoint conflict"); err != nil {
		t.Fatalf("inject checkpoint conflict: %v", err)
	}
	if err := metadataConcrete.Commit(ctx, checkpoint.State{
		BatchID: 1, Attempt: 3, Phase: checkpoint.PhaseCommitted,
		Position: batch.LastPosition,
	}); err == nil || !strings.Contains(err.Error(), "commit AGE load checkpoint") {
		t.Fatalf("Commit(metadata failure) error = %v", err)
	}
	stored, err = store.GetBatch(ctx, jobID, 1, 3)
	if err != nil || stored.Status != meta.BatchFailed {
		t.Fatalf("metadata failure diagnostic = %#v, %v", stored, err)
	}
	if err := store.StartJob(ctx, jobID); err != nil {
		t.Fatalf("resume metadata failure job: %v", err)
	}
	batch.Attempt = 4
	unwritten, err := target.Begin(ctx, batch)
	if err != nil {
		t.Fatalf("Begin(unwritten) error = %v", err)
	}
	if err := unwritten.Commit(ctx, checkpoint.State{
		BatchID: 1, Attempt: 4, Phase: checkpoint.PhaseCommitted,
		Position: batch.LastPosition,
	}); err == nil || !strings.Contains(err.Error(), "has not written") {
		t.Fatalf("Commit(unwritten) error = %v", err)
	}
}

func createIncrementalLoadTarget(
	t *testing.T,
	ctx context.Context,
	adapter *Adapter,
	store *meta.Store,
	jobID string,
	graph meta.GraphGeneration,
	person LabelCatalog,
	personGeneration meta.LabelGeneration,
	knows LabelCatalog,
	knowsGeneration meta.LabelGeneration,
	options LoadSinkOptions,
) *LoadSink {
	t.Helper()
	if err := store.CreateJob(ctx, meta.Job{
		ID:                jobID,
		Name:              "incremental-load-sink",
		SourceType:        "csv",
		LoadMode:          string(options.Mode),
		TargetGraph:       graph.GraphName,
		ConfigFingerprint: strings.Repeat("b", 64),
	}); err != nil {
		t.Fatalf("create incremental sink job: %v", err)
	}
	if err := store.StartJob(ctx, jobID); err != nil {
		t.Fatalf("start incremental sink job: %v", err)
	}
	bound, err := store.BindActiveGraphGeneration(ctx, jobID, graph.GraphName)
	if err != nil {
		t.Fatalf("bind incremental sink graph: %v", err)
	}
	options.JobID = jobID
	options.Graph = bound
	options.Labels = []LoadLabel{
		{Catalog: person, Generation: personGeneration},
		{Catalog: knows, Generation: knowsGeneration},
	}
	target, err := NewLoadSink(ctx, adapter, options)
	if err != nil {
		t.Fatalf("NewLoadSink(incremental) error = %v", err)
	}
	return target
}

func deleteLoadSinkJob(
	t *testing.T,
	ctx context.Context,
	adapter *Adapter,
	jobID string,
) {
	t.Helper()
	tx, err := adapter.pool.Begin(ctx)
	if err != nil {
		t.Fatalf("begin load sink cleanup: %v", err)
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(
		ctx,
		`UPDATE agefreighter_meta.load_job
		 SET graph_generation_id = NULL
		 WHERE job_id = $1::uuid`,
		jobID,
	); err != nil {
		t.Fatalf("unbind load sink generation: %v", err)
	}
	if _, err := tx.Exec(
		ctx,
		`DELETE FROM agefreighter_meta.graph_generation
		 WHERE job_id = $1::uuid`,
		jobID,
	); err != nil {
		t.Fatalf("delete load sink generation: %v", err)
	}
	if _, err := tx.Exec(
		ctx,
		`DELETE FROM agefreighter_meta.load_job WHERE job_id = $1::uuid`,
		jobID,
	); err != nil {
		t.Fatalf("delete load sink job: %v", err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatalf("commit load sink cleanup: %v", err)
	}
}
