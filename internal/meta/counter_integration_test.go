package meta

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

func TestLazyLabelCounterIntegration(t *testing.T) {
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
		t.Fatal(err)
	}
	if err := store.MigrateIfNeeded(ctx); err != nil {
		t.Fatal(err)
	}
	const (
		freshJobID  = "c0000000-0000-4000-8000-000000000001"
		legacyJobID = "c0000000-0000-4000-8000-000000000002"
	)
	jobIDs := []string{freshJobID, legacyJobID}
	if err := deleteTestJobs(ctx, pool, jobIDs); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cleanupCancel()
		_ = deleteTestJobs(cleanupCtx, pool, jobIDs)
	})

	freshLabels := createCounterTestJob(t, ctx, store, freshJobID, "create", 51000)
	batch := BatchAttempt{
		JobID: freshJobID, BatchID: 1, Attempt: 1, Rows: 8, Bytes: 80,
		First: Position{Resource: "fresh.csv", Line: 1, Token: "fresh-first"},
	}
	if _, err := store.StartBatch(ctx, batch); err != nil {
		t.Fatal(err)
	}
	last := Position{Resource: "fresh.csv", Line: 9, Token: "fresh-last"}
	vertexBytes := int64(50)
	edgeBytes := int64(20)
	counters := []BatchLabelCounter{
		{
			LabelGenerationID: freshLabels[0].ID, Kind: freshLabels[0].Kind,
			AcceptedRows: 5, CommittedRows: 5, CommittedBytes: &vertexBytes,
		},
		{
			LabelGenerationID: freshLabels[1].ID, Kind: freshLabels[1].Kind,
			AcceptedRows: 3, CommittedRows: 2, CommittedBytes: &edgeBytes,
			RejectedRows: 1,
		},
	}
	verification := JobVerification{
		JobID: freshJobID, SubmittedConfigFingerprint: strings.Repeat("b", 64),
		ResolvedMappingFingerprint: strings.Repeat("c", 64),
		ResolvedMappingSummary:     []byte(`{"labels":[1,2],"schemaVersion":2}`),
	}
	mismatched := append([]BatchLabelCounter(nil), counters...)
	mismatched[1].Kind = VertexLabel
	if err := store.CommitBatchWithLabelCounters(
		ctx, freshJobID, 1, 1, last, 1, mismatched,
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("mismatched multi-label commit error = %v", err)
	}
	stillRunning, err := store.GetBatch(ctx, freshJobID, 1, 1)
	if err != nil || stillRunning.Status != BatchRunning {
		t.Fatalf("rolled-back batch = %#v, %v", stillRunning, err)
	}
	if values, err := store.ListLabelCounters(ctx, freshJobID, 10); err != nil || len(values) != 0 {
		t.Fatalf("rolled-back job counters = %#v, %v", values, err)
	}

	if err := store.CommitBatchWithLabelCountersAndVerification(
		ctx, freshJobID, 1, 1, last, 1, counters, &verification,
	); err != nil {
		t.Fatal(err)
	}
	assertFreshCounters(t, ctx, store, freshJobID, freshLabels)
	if err := store.CommitBatchWithLabelCountersAndVerification(
		ctx, freshJobID, 1, 1, last, 1, counters, &verification,
	); err != nil {
		t.Fatalf("idempotent counter commit: %v", err)
	}
	assertFreshCounters(t, ctx, store, freshJobID, freshLabels)
	var batchCounterRows int
	if err := pool.QueryRow(ctx, `
		SELECT COUNT(*)::integer
		FROM agefreighter_meta.load_batch_label_counter
		WHERE job_id = $1::uuid`, freshJobID).Scan(&batchCounterRows); err != nil ||
		batchCounterRows != 2 {
		t.Fatalf("batch counter rows = %d, %v", batchCounterRows, err)
	}

	legacyLabels := createCounterTestJob(t, ctx, store, legacyJobID, "create", 52000)
	first := BatchAttempt{
		JobID: legacyJobID, BatchID: 1, Attempt: 1, Rows: 1, Bytes: 10,
		First: Position{Resource: "legacy.csv", Line: 1, Token: "legacy-first"},
	}
	if _, err := store.StartBatch(ctx, first); err != nil {
		t.Fatal(err)
	}
	if err := store.CommitBatch(
		ctx, legacyJobID, 1, 1,
		Position{Resource: "legacy.csv", Line: 2, Token: "legacy-one"}, 0,
	); err != nil {
		t.Fatal(err)
	}
	second := first
	second.BatchID = 2
	second.First = Position{Resource: "legacy.csv", Line: 2, Token: "legacy-two-first"}
	if _, err := store.StartBatch(ctx, second); err != nil {
		t.Fatal(err)
	}
	if err := store.CommitBatchWithLabelCounters(
		ctx, legacyJobID, 2, 1,
		Position{Resource: "legacy.csv", Line: 3, Token: "legacy-two"}, 0,
		[]BatchLabelCounter{{
			LabelGenerationID: legacyLabels[0].ID,
			Kind:              legacyLabels[0].Kind,
			AcceptedRows:      1,
			CommittedRows:     1,
		}},
	); err != nil {
		t.Fatal(err)
	}
	legacy, err := store.ListLabelCounters(ctx, legacyJobID, 10)
	if err != nil || len(legacy) != 1 {
		t.Fatalf("legacy counters = %#v, %v", legacy, err)
	}
	if legacy[0].Completeness != CounterIncomplete ||
		legacy[0].Provenance != CounterProvenanceLegacyResume ||
		legacy[0].AcceptedRows != nil || legacy[0].CommittedRows != nil ||
		legacy[0].RejectedRows != nil {
		t.Fatalf("legacy counter = %#v", legacy[0])
	}
}

func createCounterTestJob(
	t *testing.T,
	ctx context.Context,
	store *Store,
	jobID, mode string,
	oid uint32,
) []LabelGeneration {
	t.Helper()
	graphName := fmt.Sprintf("counter_test_%d", oid)
	if err := store.CreateRunningJob(ctx, Job{
		ID: jobID, Name: "counter-test", SourceType: "csv", LoadMode: mode,
		TargetGraph: graphName, ConfigFingerprint: strings.Repeat("a", 64),
	}); err != nil {
		t.Fatal(err)
	}
	graph, err := store.RegisterGraphGeneration(ctx, GraphGeneration{
		JobID: jobID, GraphName: graphName, GraphOID: oid, NamespaceOID: oid,
		Generation: 1, State: GenerationLoading,
	})
	if err != nil {
		t.Fatal(err)
	}
	specifications := []LabelGeneration{
		{
			GraphGenerationID: graph.ID, LabelName: "Person", Kind: VertexLabel,
			GraphNamespaceOID: oid, LabelID: 1, RelationOID: oid + 1,
			SequenceOID: oid + 2, MappingGeneration: 1,
		},
	}
	if jobID != "c0000000-0000-4000-8000-000000000002" {
		specifications = append(specifications, LabelGeneration{
			GraphGenerationID: graph.ID, LabelName: "KNOWS", Kind: EdgeLabel,
			GraphNamespaceOID: oid, LabelID: 2, RelationOID: oid + 3,
			SequenceOID: oid + 4, MappingGeneration: 1,
		})
	}
	values := make([]LabelGeneration, 0, len(specifications))
	for _, specification := range specifications {
		value, err := store.RegisterLabelGeneration(ctx, specification)
		if err != nil {
			t.Fatal(err)
		}
		values = append(values, value)
	}
	return values
}

func assertFreshCounters(
	t *testing.T,
	ctx context.Context,
	store *Store,
	jobID string,
	labels []LabelGeneration,
) {
	t.Helper()
	values, err := store.ListLabelCounters(ctx, jobID, 10)
	if err != nil || len(values) != 2 {
		t.Fatalf("fresh counters = %#v, %v", values, err)
	}
	for index, expected := range []struct {
		accepted, committed, bytes, rejected int64
	}{{5, 5, 50, 0}, {3, 2, 20, 1}} {
		value := values[index]
		if value.LabelGenerationID != labels[index].ID ||
			value.Completeness != CounterComplete ||
			value.Provenance != CounterProvenanceLifecycle ||
			value.AcceptedRows == nil || *value.AcceptedRows != expected.accepted ||
			value.CommittedRows == nil || *value.CommittedRows != expected.committed ||
			value.CommittedBytes == nil || *value.CommittedBytes != expected.bytes ||
			value.RejectedRows == nil || *value.RejectedRows != expected.rejected {
			t.Fatalf("fresh counter %d = %#v", index, value)
		}
	}
}
