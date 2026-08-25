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
	); err == nil || !strings.Contains(err.Error(), "was not held") {
		t.Fatalf("releaseBatchOwner(unheld) error = %v", err)
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
	}); err == nil || !strings.Contains(err.Error(), "unsupported missing endpoint") {
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
		"insert edge identities",
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
	if _, err := target.Begin(ctx, sink.BatchMetadata{
		ID: 8, Attempt: 1, Rows: 1, Bytes: 1,
		FirstPosition: loadPosition(17, "after-complete"),
		LastPosition:  loadPosition(17, "after-complete"),
	}); err == nil || !strings.Contains(err.Error(), "start AGE load batch") {
		t.Fatalf("Begin(after completion) error = %v", err)
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
