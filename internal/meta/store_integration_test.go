package meta

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

const metadataTestDSNEnvironment = "AGEFREIGHTER_AGE_TEST_DSN"

func TestStoreIntegration(t *testing.T) {
	dsn := os.Getenv(metadataTestDSNEnvironment)
	if dsn == "" {
		t.Skip("set " + metadataTestDSNEnvironment + " to run metadata integration tests")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("open metadata test pool: %v", err)
	}
	t.Cleanup(pool.Close)
	store, err := New(pool)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := store.Migrate(ctx); err != nil {
		t.Fatalf("Migrate() error = %v", err)
	}
	if err := store.Migrate(ctx); err != nil {
		t.Fatalf("idempotent Migrate() error = %v", err)
	}
	var telemetryTableExists bool
	if err := pool.QueryRow(
		ctx,
		`SELECT pg_catalog.to_regclass(
			'agefreighter_meta.connector_telemetry'
		) IS NOT NULL`,
	).Scan(&telemetryTableExists); err != nil || !telemetryTableExists {
		t.Fatalf("v15 telemetry migration = %v, %v", telemetryTableExists, err)
	}
	jobIDs := []string{
		testJobID,
		"22222222-3333-4444-8555-666666666666",
		"99999999-8888-4777-8666-555555555555",
		"77777777-6666-4555-8444-333333333333",
		"10101010-2222-4333-8444-555555555555",
		"aaaaaaaa-1111-4222-8333-bbbbbbbbbbbb",
	}
	if err := deleteTestJobs(ctx, pool, jobIDs); err != nil {
		t.Fatalf("clean metadata tables: %v", err)
	}
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cleanupCancel()
		_ = deleteTestJobs(cleanupCtx, pool, jobIDs)
	})

	job := Job{
		ID:                testJobID,
		Name:              "people",
		SourceType:        "csv",
		LoadMode:          "create",
		TargetGraph:       "people",
		ConfigFingerprint: strings.Repeat("a", 64),
	}

	if err := store.CreateJob(ctx, job); err != nil {
		t.Fatalf("CreateJob() error = %v", err)
	}
	if err := store.CreateJob(ctx, job); err == nil {
		t.Fatal("duplicate CreateJob() succeeded")
	}
	storedJob, err := store.GetJob(ctx, testJobID)
	if err != nil {
		t.Fatalf("GetJob() error = %v", err)
	}
	if storedJob.Status != JobPending || storedJob.NextBatchID != 1 {
		t.Fatalf("new job = %#v", storedJob)
	}
	telemetry := ConnectorTelemetry{
		JobID: testJobID, Connector: "csv",
	}
	if err := store.PutConnectorTelemetry(ctx, telemetry); err != nil {
		t.Fatalf("PutConnectorTelemetry() error = %v", err)
	}
	if err := store.PutConnectorTelemetry(ctx, telemetry); err != nil {
		t.Fatalf("idempotent PutConnectorTelemetry() error = %v", err)
	}
	storedTelemetry, err := store.GetConnectorTelemetry(ctx, testJobID)
	if err != nil || storedTelemetry.Connector != "csv" ||
		storedTelemetry.RecordedAt.IsZero() {
		t.Fatalf("GetConnectorTelemetry() = %#v, %v", storedTelemetry, err)
	}
	if _, err := store.GetJob(ctx, "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing GetJob() error = %v", err)
	}
	if err := store.CompleteJob(
		ctx,
		"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
	); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing CompleteJob() error = %v", err)
	}
	if _, err := store.LatestBatch(ctx, "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing LatestBatch() error = %v", err)
	}
	if err := store.StartJob(ctx, testJobID); err != nil {
		t.Fatalf("StartJob() error = %v", err)
	}
	if err := store.StartJob(ctx, testJobID); !errors.Is(err, ErrConflict) {
		t.Fatalf("second StartJob() error = %v", err)
	}

	graph, err := store.RegisterGraphGeneration(ctx, GraphGeneration{
		JobID:        testJobID,
		GraphName:    "people",
		GraphOID:     42000,
		NamespaceOID: 42000,
		Generation:   1,
		State:        GenerationLoading,
	})
	if err != nil {
		t.Fatalf("RegisterGraphGeneration() error = %v", err)
	}
	admittedGraph, err := store.AdmitGraphGeneration(ctx, testJobID, graph)
	if err != nil || admittedGraph.ID != graph.ID {
		t.Fatalf("AdmitGraphGeneration() = %#v, %v", admittedGraph, err)
	}
	changedGraph := graph
	changedGraph.GraphOID++
	changedGraph.NamespaceOID++
	if _, err := store.AdmitGraphGeneration(ctx, testJobID, changedGraph); !errors.Is(err, ErrGenerationMismatch) {
		t.Fatalf("changed AdmitGraphGeneration() error = %v", err)
	}
	changedGraph = graph
	changedGraph.Generation++
	if _, err := store.AdmitGraphGeneration(ctx, testJobID, changedGraph); !errors.Is(err, ErrGenerationMismatch) {
		t.Fatalf("changed logical graph generation error = %v", err)
	}
	if _, err := store.RegisterGraphGeneration(ctx, graph); err == nil {
		t.Fatal("duplicate RegisterGraphGeneration() succeeded")
	}
	if _, err := store.AdmitGraphGeneration(
		ctx,
		"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
		graph,
	); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing AdmitGraphGeneration() error = %v", err)
	}

	if _, err := store.RegisterLabelGeneration(ctx, LabelGeneration{
		GraphGenerationID: graph.ID,
		LabelName:         "Wrong",
		Kind:              VertexLabel,
		GraphNamespaceOID: graph.NamespaceOID + 1,
		LabelID:           9,
		RelationOID:       43009,
		SequenceOID:       44009,
		MappingGeneration: 1,
	}); !errors.Is(err, ErrGenerationMismatch) {
		t.Fatalf("namespace mismatch RegisterLabelGeneration() error = %v", err)
	}
	label, err := store.RegisterLabelGeneration(ctx, LabelGeneration{
		GraphGenerationID: graph.ID,
		LabelName:         "Person",
		Kind:              VertexLabel,
		GraphNamespaceOID: graph.NamespaceOID,
		LabelID:           1,
		RelationOID:       43001,
		SequenceOID:       44001,
		MappingGeneration: 1,
	})
	if err != nil {
		t.Fatalf("RegisterLabelGeneration() error = %v", err)
	}
	admittedLabel, err := store.AdmitLabelGeneration(ctx, graph.ID, label)
	if err != nil || admittedLabel.ID != label.ID {
		t.Fatalf("AdmitLabelGeneration() = %#v, %v", admittedLabel, err)
	}
	changedLabel := label
	changedLabel.RelationOID++
	if _, err := store.AdmitLabelGeneration(ctx, graph.ID, changedLabel); !errors.Is(err, ErrGenerationMismatch) {
		t.Fatalf("changed AdmitLabelGeneration() error = %v", err)
	}
	changedLabel = label
	changedLabel.MappingGeneration++
	if _, err := store.AdmitLabelGeneration(ctx, graph.ID, changedLabel); !errors.Is(err, ErrGenerationMismatch) {
		t.Fatalf("changed logical label generation error = %v", err)
	}
	missingLabel := label
	missingLabel.LabelName = "Missing"
	if _, err := store.AdmitLabelGeneration(ctx, graph.ID, missingLabel); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing AdmitLabelGeneration() error = %v", err)
	}
	if _, err := store.RegisterLabelGeneration(ctx, label); err == nil {
		t.Fatal("duplicate RegisterLabelGeneration() succeeded")
	}
	edgeLabel, err := store.RegisterLabelGeneration(ctx, LabelGeneration{
		GraphGenerationID: graph.ID,
		LabelName:         "KNOWS",
		Kind:              EdgeLabel,
		GraphNamespaceOID: graph.NamespaceOID,
		LabelID:           2,
		RelationOID:       43002,
		SequenceOID:       44002,
		MappingGeneration: 1,
	})
	if err != nil {
		t.Fatalf("edge RegisterLabelGeneration() error = %v", err)
	}
	vertexGraphID := int64(1)<<48 | 100
	if _, err := pool.Exec(
		ctx,
		`INSERT INTO agefreighter_meta.vertex_identity (
			graph_generation_id, label_generation_id, label_id,
			source_namespace, external_id, graph_id
		) VALUES ($1, $2, $3, 'crm', 'valid-graph-id', $4)`,
		graph.ID,
		label.ID,
		label.LabelID,
		vertexGraphID,
	); err != nil {
		t.Fatalf("valid vertex identity insert error = %v", err)
	}
	for name, graphID := range map[string]int64{
		"wrong-label-bits": int64(2)<<48 | 101,
		"zero-entry":       int64(1) << 48,
	} {
		if _, err := pool.Exec(
			ctx,
			`INSERT INTO agefreighter_meta.vertex_identity (
				graph_generation_id, label_generation_id, label_id,
				source_namespace, external_id, graph_id
			) VALUES ($1, $2, $3, 'crm', $4, $5)`,
			graph.ID,
			label.ID,
			label.LabelID,
			name,
			graphID,
		); err == nil {
			t.Fatalf("%s vertex graph ID insert succeeded", name)
		}
	}
	for name, ids := range map[string][3]int64{
		"wrong-edge-label-bits": {
			int64(1)<<48 | 201,
			vertexGraphID,
			vertexGraphID,
		},
		"zero-start-label": {
			int64(2)<<48 | 202,
			1,
			vertexGraphID,
		},
		"zero-end-entry": {
			int64(2)<<48 | 203,
			vertexGraphID,
			int64(1) << 48,
		},
	} {
		if _, err := pool.Exec(
			ctx,
			`INSERT INTO agefreighter_meta.edge_identity (
				graph_generation_id, label_generation_id, label_id,
				source_namespace, external_id, graph_id, start_graph_id, end_graph_id
			) VALUES (
				$1, $2, $3, 'crm', $4, $5, $6, $7
			)`,
			graph.ID,
			edgeLabel.ID,
			edgeLabel.LabelID,
			name,
			ids[0],
			ids[1],
			ids[2],
		); err == nil {
			t.Fatalf("%s edge graph ID insert succeeded", name)
		}
	}
	if err := store.SetGraphGenerationState(
		ctx,
		0,
		GenerationLoading,
		GenerationActive,
	); err == nil {
		t.Fatal("zero-ID SetGraphGenerationState() succeeded")
	}
	if err := store.SetGraphGenerationState(
		ctx,
		graph.ID,
		GenerationLoading,
		GenerationLoading,
	); err == nil {
		t.Fatal("no-op SetGraphGenerationState() succeeded")
	}
	if err := store.SetGraphGenerationState(
		ctx,
		graph.ID,
		GenerationLoading,
		GenerationRetired,
	); err == nil {
		t.Fatal("invalid SetGraphGenerationState() succeeded")
	}
	if err := store.SetGraphGenerationState(
		ctx,
		graph.ID,
		GenerationLoading,
		GenerationActive,
	); err != nil {
		t.Fatalf("SetGraphGenerationState() error = %v", err)
	}
	incrementalJob := Job{
		ID:                jobIDs[4],
		Name:              "append-people",
		SourceType:        "csv",
		LoadMode:          "append",
		TargetGraph:       graph.GraphName,
		ConfigFingerprint: strings.Repeat("e", 64),
	}
	if err := store.CreateJob(ctx, incrementalJob); err != nil {
		t.Fatalf("create incremental job: %v", err)
	}
	if err := store.StartJob(ctx, incrementalJob.ID); err != nil {
		t.Fatalf("start incremental job: %v", err)
	}
	bound, err := store.BindActiveGraphGeneration(
		ctx,
		incrementalJob.ID,
		graph.GraphName,
	)
	if err != nil || bound.ID != graph.ID || bound.JobID != graph.JobID {
		t.Fatalf("BindActiveGraphGeneration() = %#v, %v", bound, err)
	}
	boundForJob, err := store.GraphGenerationForJob(ctx, incrementalJob.ID)
	if err != nil || boundForJob.ID != graph.ID {
		t.Fatalf("incremental GraphGenerationForJob() = %#v, %v", boundForJob, err)
	}
	if _, err := store.BindActiveGraphGeneration(
		ctx,
		incrementalJob.ID,
		graph.GraphName,
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("second BindActiveGraphGeneration() error = %v", err)
	}
	createModeJob := incrementalJob
	createModeJob.ID = jobIDs[5]
	createModeJob.LoadMode = "create"
	createModeJob.ConfigFingerprint = strings.Repeat("f", 64)
	if err := store.CreateJob(ctx, createModeJob); err != nil {
		t.Fatalf("create non-incremental binding job: %v", err)
	}
	if err := store.StartJob(ctx, createModeJob.ID); err != nil {
		t.Fatalf("start non-incremental binding job: %v", err)
	}
	if _, err := store.BindActiveGraphGeneration(
		ctx,
		createModeJob.ID,
		graph.GraphName,
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("non-incremental BindActiveGraphGeneration() error = %v", err)
	}
	if _, err := store.BindActiveGraphGeneration(
		ctx,
		incrementalJob.ID,
		"unmanaged",
	); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing BindActiveGraphGeneration() error = %v", err)
	}
	if err := store.SetGraphGenerationState(
		ctx,
		graph.ID,
		GenerationLoading,
		GenerationActive,
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale SetGraphGenerationState() error = %v", err)
	}

	batch := BatchAttempt{
		JobID:   testJobID,
		BatchID: 1,
		Attempt: 1,
		Rows:    2,
		Bytes:   10,
		First: Position{
			Resource:   "people.csv",
			Line:       2,
			ByteOffset: 3,
			Token:      "first",
		},
	}
	futureBatch := batch
	futureBatch.BatchID = 2
	if _, err := store.StartBatch(ctx, futureBatch); !errors.Is(err, ErrConflict) {
		t.Fatalf("future StartBatch() error = %v", err)
	}
	started, err := store.StartBatch(ctx, batch)
	if err != nil || started.Status != BatchRunning {
		t.Fatalf("StartBatch() = %#v, %v", started, err)
	}
	if _, err := store.StartBatch(ctx, batch); err != nil {
		t.Fatalf("idempotent StartBatch() error = %v", err)
	}
	different := batch
	different.Rows++
	if _, err := store.StartBatch(ctx, different); !errors.Is(err, ErrConflict) {
		t.Fatalf("different StartBatch() error = %v", err)
	}
	if _, err := store.GetBatch(ctx, testJobID, 1, 2); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing GetBatch() error = %v", err)
	}
	last := Position{
		Resource:   "people.csv",
		Line:       3,
		ByteOffset: 10,
		Token:      "checkpoint-1",
	}
	if err := store.CommitBatch(ctx, testJobID, 1, 1, Position{}, 0); err == nil {
		t.Fatal("tokenless CommitBatch() succeeded")
	}
	if err := store.CommitBatch(ctx, testJobID, 1, 1, last, -1); err == nil {
		t.Fatal("negative-reject CommitBatch() succeeded")
	}
	if err := store.CommitBatch(ctx, testJobID, 99, 1, last, 0); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing CommitBatch() error = %v", err)
	}
	if err := store.CommitBatch(ctx, testJobID, 1, 1, last, 1); err != nil {
		t.Fatalf("CommitBatch() error = %v", err)
	}
	if err := store.CommitBatch(ctx, testJobID, 1, 1, last, 1); err != nil {
		t.Fatalf("idempotent CommitBatch() error = %v", err)
	}
	committedStart, err := store.StartBatch(ctx, batch)
	if err != nil || committedStart.Status != BatchCommitted {
		t.Fatalf("committed idempotent StartBatch() = %#v, %v", committedStart, err)
	}
	changedLast := last
	changedLast.Line++
	if err := store.CommitBatch(ctx, testJobID, 1, 1, changedLast, 1); !errors.Is(err, ErrConflict) {
		t.Fatalf("changed idempotent CommitBatch() error = %v", err)
	}
	latest, err := store.LatestBatch(ctx, testJobID)
	if err != nil || latest.Status != BatchCommitted || latest.Last != last {
		t.Fatalf("LatestBatch() = %#v, %v", latest, err)
	}
	storedJob, err = store.GetJob(ctx, testJobID)
	if err != nil {
		t.Fatalf("GetJob(after batch) error = %v", err)
	}
	if storedJob.NextBatchID != 2 ||
		storedJob.ResumeToken != last.Token ||
		storedJob.CommittedRows != 2 ||
		storedJob.CommittedBytes != 10 ||
		storedJob.RejectedRows != 1 {
		t.Fatalf("job checkpoint = %#v", storedJob)
	}
	sourcePosition := Position{Resource: "people.csv", Line: 3, Token: "source-reject-2"}
	if err := store.SetSourceRejections(ctx, testJobID, 2, sourcePosition); err != nil {
		t.Fatalf("SetSourceRejections() error = %v", err)
	}
	if err := store.SetSourceRejections(ctx, testJobID, 2, sourcePosition); err != nil {
		t.Fatalf("idempotent SetSourceRejections() error = %v", err)
	}
	if err := store.SetSourceRejections(ctx, testJobID, 1, sourcePosition); !errors.Is(err, ErrConflict) {
		t.Fatalf("decreasing SetSourceRejections() error = %v", err)
	}
	storedJob, err = store.GetJob(ctx, testJobID)
	if err != nil || storedJob.SourceRejectedRows != 2 || storedJob.RejectedRows != 3 {
		t.Fatalf("source rejection checkpoint = %#v, %v", storedJob, err)
	}
	inserted, err := store.PutReject(ctx, RejectRecord{
		JobID:        testJobID,
		BatchID:      1,
		Attempt:      1,
		Position:     Position{Resource: "people.csv", Line: 2, Token: "reject-1"},
		ErrorClass:   "mapping",
		ErrorMessage: "empty ID",
		Record:       []byte(`{"id":""}`),
	})
	if err != nil || !inserted {
		t.Fatalf("PutReject() = %t, %v", inserted, err)
	}
	inserted, err = store.PutReject(ctx, RejectRecord{
		JobID:        testJobID,
		BatchID:      1,
		Attempt:      1,
		Position:     Position{Resource: "people.csv", Line: 2, Token: "reject-1"},
		ErrorClass:   "mapping",
		ErrorMessage: "empty ID",
		Record:       []byte(`{"id":""}`),
	})
	if err != nil || inserted {
		t.Fatalf("idempotent PutReject() = %t, %v", inserted, err)
	}
	if _, err := store.PutReject(ctx, RejectRecord{
		JobID:        testJobID,
		BatchID:      1,
		Attempt:      1,
		Position:     Position{Resource: "people.csv", Line: 2, Token: "reject-1"},
		ErrorClass:   "mapping",
		ErrorMessage: "different error",
		Record:       []byte(`{"id":""}`),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflicting PutReject() error = %v", err)
	}

	failed := BatchAttempt{
		JobID:   testJobID,
		BatchID: 2,
		Attempt: 1,
		Rows:    1,
		Bytes:   5,
		First:   Position{Token: "first-2"},
	}
	if _, err := store.StartBatch(ctx, failed); err != nil {
		t.Fatalf("second StartBatch() error = %v", err)
	}
	prematureFailure := failed
	prematureFailure.Attempt = 2
	if err := store.RecordFailedBatch(
		ctx,
		prematureFailure,
		"premature retry failure",
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("premature retry RecordFailedBatch() error = %v", err)
	}
	if err := store.CompleteJob(ctx, testJobID); !errors.Is(err, ErrConflict) {
		t.Fatalf("CompleteJob(with running batch) error = %v", err)
	}
	if err := store.FailJob(ctx, testJobID, "failure during running batch"); err != nil {
		t.Fatalf("running-batch FailJob() error = %v", err)
	}
	if err := store.CommitBatch(
		ctx,
		testJobID,
		2,
		1,
		Position{Token: "should-not-commit"},
		0,
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("failed-job CommitBatch() error = %v", err)
	}
	if err := store.StartJob(ctx, testJobID); err != nil {
		t.Fatalf("running-batch resume StartJob() error = %v", err)
	}
	failedDiagnostic := failed
	failedDiagnostic.RejectedRows = 2
	if err := store.RecordFailedBatch(ctx, failedDiagnostic, "injected failure"); err != nil {
		t.Fatalf("RecordFailedBatch() error = %v", err)
	}
	storedFailed, err := store.GetBatch(ctx, testJobID, 2, 1)
	if err != nil || storedFailed.RejectedRows != 2 {
		t.Fatalf("failed GetBatch() = %#v, %v", storedFailed, err)
	}
	conflictingFailure := failedDiagnostic
	conflictingFailure.RejectedRows++
	if err := store.RecordFailedBatch(
		ctx,
		conflictingFailure,
		"injected failure",
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflicting RecordFailedBatch() error = %v", err)
	}
	conflictingMetadata := failedDiagnostic
	conflictingMetadata.Rows++
	if err := store.RecordFailedBatch(
		ctx,
		conflictingMetadata,
		"injected failure",
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("metadata-conflicting RecordFailedBatch() error = %v", err)
	}
	skippedFailure := failedDiagnostic
	skippedFailure.Attempt = 3
	if err := store.RecordFailedBatch(
		ctx,
		skippedFailure,
		"skipped retry failure",
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("skipped retry RecordFailedBatch() error = %v", err)
	}
	if err := store.RecordFailedBatch(ctx, failedDiagnostic, ""); err == nil {
		t.Fatal("message-less RecordFailedBatch() succeeded")
	}
	if err := store.RecordFailedBatch(ctx, batch, "cannot fail committed"); !errors.Is(err, ErrConflict) {
		t.Fatalf("committed RecordFailedBatch() error = %v", err)
	}
	storedJob, err = store.GetJob(ctx, testJobID)
	if err != nil || storedJob.Status != JobFailed {
		t.Fatalf("failed GetJob() = %#v, %v", storedJob, err)
	}
	if err := store.StartJob(ctx, testJobID); err != nil {
		t.Fatalf("resume StartJob() error = %v", err)
	}
	if err := store.RecordFailedBatch(
		ctx,
		failedDiagnostic,
		"injected failure",
	); err != nil {
		t.Fatalf("idempotent old RecordFailedBatch() error = %v", err)
	}
	storedJob, err = store.GetJob(ctx, testJobID)
	if err != nil || storedJob.Status != JobRunning {
		t.Fatalf("idempotent failure changed resumed job = %#v, %v", storedJob, err)
	}
	retry := failed
	retry.Attempt = 2
	if _, err := store.StartBatch(ctx, retry); err != nil {
		t.Fatalf("retry StartBatch() error = %v", err)
	}
	concurrentAttempt := retry
	concurrentAttempt.Attempt = 3
	if _, err := store.StartBatch(ctx, concurrentAttempt); !errors.Is(err, ErrConflict) {
		t.Fatalf("concurrent StartBatch() error = %v", err)
	}
	if err := store.RecordFailedBatch(
		ctx,
		failedDiagnostic,
		"injected failure",
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale RecordFailedBatch() error = %v", err)
	}
	retryDiagnostic := retry
	retryDiagnostic.RejectedRows = 1
	if err := store.RecordFailedBatch(
		ctx,
		retryDiagnostic,
		"retry failure",
	); err != nil {
		t.Fatalf("retry RecordFailedBatch() error = %v", err)
	}
	if err := store.StartJob(ctx, testJobID); err != nil {
		t.Fatalf("second resume StartJob() error = %v", err)
	}
	if err := store.CompleteJob(ctx, testJobID); !errors.Is(err, ErrConflict) {
		t.Fatalf("CompleteJob(with unresolved failed batch) error = %v", err)
	}
	if err := store.CompleteJobGeneration(ctx, testJobID, graph.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("CompleteJobGeneration(with unresolved batch) error = %v", err)
	}
	finalRetry := failed
	finalRetry.Attempt = 3
	if _, err := store.StartBatch(ctx, finalRetry); err != nil {
		t.Fatalf("final retry StartBatch() error = %v", err)
	}
	if err := store.CommitBatch(
		ctx,
		testJobID,
		2,
		3,
		Position{Resource: "people.csv", Line: 4, Token: "checkpoint-2"},
		0,
	); err != nil {
		t.Fatalf("final retry CommitBatch() error = %v", err)
	}
	if err := store.FailJob(ctx, testJobID, ""); err == nil {
		t.Fatal("message-less FailJob() succeeded")
	}
	if err := store.FailJob(ctx, testJobID, "manual failure"); err != nil {
		t.Fatalf("FailJob() error = %v", err)
	}
	if err := store.StartJob(ctx, testJobID); err != nil {
		t.Fatalf("third resume StartJob() error = %v", err)
	}
	if err := store.CompleteJob(ctx, testJobID); err != nil {
		t.Fatalf("CompleteJob() error = %v", err)
	}
	if err := store.CompleteJob(ctx, testJobID); !errors.Is(err, ErrConflict) {
		t.Fatalf("second CompleteJob() error = %v", err)
	}
	if err := store.CompleteJobGeneration(
		ctx,
		"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
		graph.ID,
	); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing CompleteJobGeneration() error = %v", err)
	}

	completionJob := job
	completionJob.ID = "77777777-6666-4555-8444-333333333333"
	completionJob.Name = "completion"
	completionJob.TargetGraph = "completion"
	if err := store.CreateJob(ctx, completionJob); err != nil {
		t.Fatalf("completion CreateJob() error = %v", err)
	}
	if err := store.StartJob(ctx, completionJob.ID); err != nil {
		t.Fatalf("completion StartJob() error = %v", err)
	}
	completionGraph, err := store.RegisterGraphGeneration(ctx, GraphGeneration{
		JobID: completionJob.ID, GraphName: completionJob.TargetGraph,
		GraphOID: 62000, NamespaceOID: 62000, Generation: 1, State: GenerationLoading,
	})
	if err != nil {
		t.Fatalf("completion RegisterGraphGeneration() error = %v", err)
	}
	if err := store.CompleteJobGeneration(ctx, completionJob.ID, completionGraph.ID); err != nil {
		t.Fatalf("CompleteJobGeneration() error = %v", err)
	}
	completed, err := store.GetJob(ctx, completionJob.ID)
	if err != nil || completed.Status != JobCommitted {
		t.Fatalf("completed generation job = %#v, %v", completed, err)
	}

	outer, err := pool.Begin(ctx)
	if err != nil {
		t.Fatalf("begin outer transaction: %v", err)
	}
	transactionStore, err := New(outer)
	if err != nil {
		t.Fatalf("New(transaction) error = %v", err)
	}
	rolledBack := job
	rolledBack.ID = "99999999-8888-4777-8666-555555555555"
	if err := transactionStore.CreateJob(ctx, rolledBack); err != nil {
		t.Fatalf("transaction CreateJob() error = %v", err)
	}
	if err := outer.Rollback(ctx); err != nil {
		t.Fatalf("rollback outer transaction: %v", err)
	}
	if _, err := store.GetJob(ctx, rolledBack.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("rolled-back GetJob() error = %v", err)
	}
}

func deleteTestJobs(
	ctx context.Context,
	pool *pgxpool.Pool,
	jobIDs []string,
) error {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(
		ctx,
		`UPDATE agefreighter_meta.load_job
		 SET graph_generation_id = NULL
		 WHERE job_id = ANY($1::uuid[])`,
		jobIDs,
	); err != nil {
		return err
	}
	if _, err := tx.Exec(
		ctx,
		`DELETE FROM agefreighter_meta.graph_generation
		 WHERE job_id = ANY($1::uuid[])`,
		jobIDs,
	); err != nil {
		return err
	}
	if _, err := tx.Exec(
		ctx,
		`DELETE FROM agefreighter_meta.load_job
		 WHERE job_id = ANY($1::uuid[])`,
		jobIDs,
	); err != nil {
		return err
	}
	return tx.Commit(ctx)
}
