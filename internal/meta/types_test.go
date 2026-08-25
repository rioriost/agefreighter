package meta

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

const testJobID = "11111111-2222-4333-8444-555555555555"

func TestMetadataValidation(t *testing.T) {
	validJob := Job{
		ID:                testJobID,
		Name:              "people",
		SourceType:        "csv",
		LoadMode:          "create",
		TargetGraph:       "people",
		ConfigFingerprint: strings.Repeat("a", 64),
	}
	if err := validateJob(validJob); err != nil {
		t.Fatalf("validateJob() error = %v", err)
	}
	jobTests := []Job{
		{},
		withJob(validJob, func(job *Job) {
			job.ID = "AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE"
		}),
		withJob(validJob, func(job *Job) {
			job.ID = "zzzzzzzz-bbbb-4ccc-8ddd-eeeeeeeeeeee"
		}),
		withJob(validJob, func(job *Job) { job.Name = "" }),
		withJob(validJob, func(job *Job) { job.SourceType = "unknown" }),
		withJob(validJob, func(job *Job) { job.LoadMode = "unknown" }),
		withJob(validJob, func(job *Job) { job.TargetGraph = "" }),
		withJob(validJob, func(job *Job) { job.ConfigFingerprint = "A" + strings.Repeat("a", 63) }),
		withJob(validJob, func(job *Job) { job.ConfigFingerprint = strings.Repeat("z", 64) }),
		withJob(validJob, func(job *Job) { job.Status = JobRunning }),
	}
	for index, job := range jobTests {
		if err := validateJob(job); err == nil {
			t.Fatalf("invalid job %d accepted: %#v", index, job)
		}
	}
	for _, source := range []string{"csv", "postgresql", "neo4j", "cosmos-nosql"} {
		job := validJob
		job.SourceType = source
		if err := validateJob(job); err != nil {
			t.Fatalf("source %q rejected: %v", source, err)
		}
	}
	for _, mode := range []string{"create", "replace", "append", "upsert"} {
		job := validJob
		job.LoadMode = mode
		if err := validateJob(job); err != nil {
			t.Fatalf("mode %q rejected: %v", mode, err)
		}
	}

	validGraph := GraphGeneration{
		JobID:        testJobID,
		GraphName:    "people",
		GraphOID:     42,
		NamespaceOID: 42,
		Generation:   1,
		State:        GenerationLoading,
	}
	if err := validateGraphGeneration(validGraph); err != nil {
		t.Fatalf("validateGraphGeneration() error = %v", err)
	}
	graphTests := []GraphGeneration{
		{},
		withGraph(validGraph, func(graph *GraphGeneration) { graph.GraphName = "" }),
		withGraph(validGraph, func(graph *GraphGeneration) { graph.GraphOID = 0 }),
		withGraph(validGraph, func(graph *GraphGeneration) { graph.NamespaceOID++ }),
		withGraph(validGraph, func(graph *GraphGeneration) { graph.Generation = 0 }),
		withGraph(validGraph, func(graph *GraphGeneration) { graph.Generation = math.MaxInt64 + 1 }),
		withGraph(validGraph, func(graph *GraphGeneration) { graph.State = "bad" }),
	}
	for index, graph := range graphTests {
		if err := validateGraphGeneration(graph); err == nil {
			t.Fatalf("invalid graph %d accepted: %#v", index, graph)
		}
	}

	validLabel := LabelGeneration{
		GraphGenerationID: 1,
		LabelName:         "Person",
		Kind:              VertexLabel,
		GraphNamespaceOID: 42,
		LabelID:           1,
		RelationOID:       43,
		SequenceOID:       44,
		MappingGeneration: 1,
	}
	if err := validateLabelGeneration(validLabel); err != nil {
		t.Fatalf("validateLabelGeneration() error = %v", err)
	}
	labelTests := []LabelGeneration{
		{},
		withLabel(validLabel, func(label *LabelGeneration) { label.LabelName = "" }),
		withLabel(validLabel, func(label *LabelGeneration) { label.Kind = 'x' }),
		withLabel(validLabel, func(label *LabelGeneration) { label.GraphNamespaceOID = 0 }),
		withLabel(validLabel, func(label *LabelGeneration) { label.LabelID = 0 }),
		withLabel(validLabel, func(label *LabelGeneration) { label.RelationOID = 0 }),
		withLabel(validLabel, func(label *LabelGeneration) { label.SequenceOID = 0 }),
		withLabel(validLabel, func(label *LabelGeneration) { label.MappingGeneration = 0 }),
	}
	for index, label := range labelTests {
		if err := validateLabelGeneration(label); err == nil {
			t.Fatalf("invalid label %d accepted: %#v", index, label)
		}
	}
	validLabel.Kind = EdgeLabel
	if err := validateLabelGeneration(validLabel); err != nil {
		t.Fatalf("edge label rejected: %v", err)
	}

	validBatch := BatchAttempt{
		JobID:   testJobID,
		BatchID: 1,
		Attempt: 1,
		Rows:    2,
		Bytes:   10,
		First:   Position{Line: 1, ByteOffset: 1},
	}
	if err := validateBatch(validBatch); err != nil {
		t.Fatalf("validateBatch() error = %v", err)
	}
	batchTests := []BatchAttempt{
		{},
		withBatch(validBatch, func(batch *BatchAttempt) { batch.BatchID = 0 }),
		withBatch(validBatch, func(batch *BatchAttempt) { batch.Attempt = 0 }),
		withBatch(validBatch, func(batch *BatchAttempt) { batch.Attempt = math.MaxInt32 + 1 }),
		withBatch(validBatch, func(batch *BatchAttempt) { batch.Rows = -1 }),
		withBatch(validBatch, func(batch *BatchAttempt) { batch.Bytes = -1 }),
		withBatch(validBatch, func(batch *BatchAttempt) { batch.RejectedRows = -1 }),
		withBatch(validBatch, func(batch *BatchAttempt) { batch.First.Line = -1 }),
	}
	for index, batch := range batchTests {
		if err := validateBatch(batch); err == nil {
			t.Fatalf("invalid batch %d accepted: %#v", index, batch)
		}
	}

	validReject := RejectRecord{
		JobID:        testJobID,
		BatchID:      1,
		Attempt:      1,
		Position:     Position{Token: "token"},
		ErrorClass:   "mapping",
		ErrorMessage: "bad record",
		Record:       json.RawMessage(`{"id":"bad"}`),
	}
	if err := validateReject(validReject); err != nil {
		t.Fatalf("validateReject() error = %v", err)
	}
	rejectTests := []RejectRecord{
		{},
		withReject(validReject, func(record *RejectRecord) { record.BatchID = 0 }),
		withReject(validReject, func(record *RejectRecord) { record.Position.Token = "" }),
		withReject(validReject, func(record *RejectRecord) { record.Position.Line = -1 }),
		withReject(validReject, func(record *RejectRecord) { record.ErrorClass = "" }),
		withReject(validReject, func(record *RejectRecord) { record.ErrorMessage = "" }),
		withReject(validReject, func(record *RejectRecord) { record.Record = []byte("{") }),
	}
	for index, record := range rejectTests {
		if err := validateReject(record); err == nil {
			t.Fatalf("invalid reject %d accepted: %#v", index, record)
		}
	}
}

func TestMetadataHelpers(t *testing.T) {
	if _, err := New(nil); err == nil {
		t.Fatal("New() accepted nil database")
	}
	var store *Store
	if err := store.Migrate(t.Context()); err == nil {
		t.Fatal("nil Store.Migrate() succeeded")
	}
	if err := rowsAffectedOne(pgconn.NewCommandTag("UPDATE 0"), "test"); !errors.Is(err, ErrConflict) {
		t.Fatalf("rowsAffectedOne() error = %v", err)
	}
	if err := rowsAffectedOne(pgconn.NewCommandTag("UPDATE 1"), "test"); err != nil {
		t.Fatalf("rowsAffectedOne() error = %v", err)
	}
	if !validGenerationTransition(GenerationLoading, GenerationActive) ||
		!validGenerationTransition(GenerationActive, GenerationRetired) ||
		validGenerationTransition(GenerationLoading, GenerationRetired) {
		t.Fatal("validGenerationTransition() returned an invalid result")
	}
	if _, err := scanBatch(errorRow{err: pgx.ErrNoRows}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("scanBatch(no rows) error = %v", err)
	}
	injected := errors.New("injected scan failure")
	if _, err := scanBatch(errorRow{err: injected}); !errors.Is(err, injected) {
		t.Fatalf("scanBatch(failure) error = %v", err)
	}
}

func TestStoreRejectsInvalidPublicInputs(t *testing.T) {
	store := &Store{database: panicDatabase{}}
	invalidJob := Job{}
	if err := store.CreateJob(t.Context(), invalidJob); err == nil {
		t.Fatal("CreateJob() accepted invalid job")
	}
	if _, err := store.GetJob(t.Context(), "bad"); err == nil {
		t.Fatal("GetJob() accepted invalid job ID")
	}
	if err := store.StartJob(t.Context(), "bad"); err == nil {
		t.Fatal("StartJob() accepted invalid job ID")
	}
	if err := store.CompleteJob(t.Context(), "bad"); err == nil {
		t.Fatal("CompleteJob() accepted invalid job ID")
	}
	if err := store.FailJob(t.Context(), "bad", "failure"); err == nil {
		t.Fatal("FailJob() accepted invalid job ID")
	}
	if _, err := store.RegisterGraphGeneration(t.Context(), GraphGeneration{}); err == nil {
		t.Fatal("RegisterGraphGeneration() accepted invalid generation")
	}
	if _, err := store.AdmitGraphGeneration(t.Context(), "bad", GraphGeneration{}); err == nil {
		t.Fatal("AdmitGraphGeneration() accepted invalid job ID")
	}
	if _, err := store.AdmitGraphGeneration(
		t.Context(),
		testJobID,
		GraphGeneration{},
	); err == nil {
		t.Fatal("AdmitGraphGeneration() accepted invalid current generation")
	}
	if _, err := store.RegisterLabelGeneration(t.Context(), LabelGeneration{}); err == nil {
		t.Fatal("RegisterLabelGeneration() accepted invalid generation")
	}
	if _, err := store.AdmitLabelGeneration(t.Context(), 0, LabelGeneration{}); err == nil {
		t.Fatal("AdmitLabelGeneration() accepted invalid graph generation ID")
	}
	if _, err := store.AdmitLabelGeneration(
		t.Context(),
		1,
		LabelGeneration{},
	); err == nil {
		t.Fatal("AdmitLabelGeneration() accepted invalid current generation")
	}
	if _, err := store.StartBatch(t.Context(), BatchAttempt{}); err == nil {
		t.Fatal("StartBatch() accepted invalid batch")
	}
	if _, err := store.StartBatch(t.Context(), BatchAttempt{
		JobID:        testJobID,
		BatchID:      1,
		Attempt:      1,
		RejectedRows: 1,
	}); err == nil {
		t.Fatal("StartBatch() accepted initial rejected rows")
	}
	if _, err := store.GetBatch(t.Context(), "bad", 1, 1); err == nil {
		t.Fatal("GetBatch() accepted invalid job ID")
	}
	if _, err := store.GetBatch(t.Context(), testJobID, 0, 1); err == nil {
		t.Fatal("GetBatch() accepted zero batch ID")
	}
	if _, err := store.LatestBatch(t.Context(), "bad"); err == nil {
		t.Fatal("LatestBatch() accepted invalid job ID")
	}
	if err := store.CommitBatch(
		t.Context(),
		"bad",
		1,
		1,
		Position{Token: "token"},
		0,
	); err == nil {
		t.Fatal("CommitBatch() accepted invalid job ID")
	}
	if err := store.CommitBatch(
		t.Context(),
		testJobID,
		math.MaxInt64,
		1,
		Position{Token: "token"},
		0,
	); err == nil {
		t.Fatal("CommitBatch() accepted terminal batch ID")
	}
	if err := store.CommitBatch(
		t.Context(),
		testJobID,
		1,
		1,
		Position{Line: -1, Token: "token"},
		0,
	); err == nil {
		t.Fatal("CommitBatch() accepted invalid last position")
	}
	if err := store.RecordFailedBatch(t.Context(), BatchAttempt{}, "failure"); err == nil {
		t.Fatal("RecordFailedBatch() accepted invalid batch")
	}
	if _, err := store.PutReject(t.Context(), RejectRecord{}); err == nil {
		t.Fatal("PutReject() accepted invalid reject")
	}
}

func TestStorePropagatesDatabaseFailures(t *testing.T) {
	injected := errors.New("injected database failure")
	beginStore := &Store{database: failingDatabase{err: injected}}
	graph := GraphGeneration{
		JobID:        testJobID,
		GraphName:    "people",
		GraphOID:     42,
		NamespaceOID: 42,
		Generation:   1,
		State:        GenerationLoading,
	}
	if err := beginStore.Migrate(t.Context()); !errors.Is(err, injected) {
		t.Fatalf("Migrate() error = %v", err)
	}
	if _, err := beginStore.RegisterGraphGeneration(t.Context(), graph); !errors.Is(err, injected) {
		t.Fatalf("RegisterGraphGeneration() error = %v", err)
	}
	if err := beginStore.CommitBatch(
		t.Context(),
		testJobID,
		1,
		1,
		Position{Token: "token"},
		0,
	); !errors.Is(err, injected) {
		t.Fatalf("CommitBatch() error = %v", err)
	}
	batch := BatchAttempt{JobID: testJobID, BatchID: 1, Attempt: 1}
	if err := beginStore.RecordFailedBatch(
		t.Context(),
		batch,
		"failure",
	); !errors.Is(err, injected) {
		t.Fatalf("RecordFailedBatch() error = %v", err)
	}

	execStore := &Store{database: execFailDatabase{err: injected}}
	job := Job{
		ID:                testJobID,
		Name:              "people",
		SourceType:        "csv",
		LoadMode:          "create",
		TargetGraph:       "people",
		ConfigFingerprint: strings.Repeat("a", 64),
	}
	assertDatabaseFailure(t, injected, execStore.CreateJob(t.Context(), job))
	_, err := execStore.GetJob(t.Context(), testJobID)
	assertDatabaseFailure(t, injected, err)
	assertDatabaseFailure(t, injected, execStore.StartJob(t.Context(), testJobID))
	assertDatabaseFailure(t, injected, execStore.CompleteJob(t.Context(), testJobID))
	assertDatabaseFailure(t, injected, execStore.FailJob(t.Context(), testJobID, "failure"))
	_, err = execStore.StartBatch(t.Context(), batch)
	assertDatabaseFailure(t, injected, err)
	_, err = execStore.GetBatch(t.Context(), testJobID, 1, 1)
	assertDatabaseFailure(t, injected, err)
	_, err = execStore.LatestBatch(t.Context(), testJobID)
	assertDatabaseFailure(t, injected, err)
	label := LabelGeneration{
		GraphGenerationID: 1,
		LabelName:         "Person",
		Kind:              VertexLabel,
		GraphNamespaceOID: 42,
		LabelID:           1,
		RelationOID:       43,
		SequenceOID:       44,
		MappingGeneration: 1,
	}
	_, err = execStore.RegisterLabelGeneration(t.Context(), label)
	assertDatabaseFailure(t, injected, err)
	_, err = execStore.AdmitLabelGeneration(t.Context(), 1, label)
	assertDatabaseFailure(t, injected, err)
	_, err = execStore.AdmitGraphGeneration(t.Context(), testJobID, graph)
	assertDatabaseFailure(t, injected, err)
	assertDatabaseFailure(
		t,
		injected,
		execStore.SetGraphGenerationState(
			t.Context(),
			1,
			GenerationLoading,
			GenerationActive,
		),
	)
	_, err = execStore.PutReject(t.Context(), RejectRecord{
		JobID:        testJobID,
		BatchID:      1,
		Attempt:      1,
		Position:     Position{Token: "token"},
		ErrorClass:   "mapping",
		ErrorMessage: "failure",
	})
	assertDatabaseFailure(t, injected, err)
}

func assertDatabaseFailure(t *testing.T, expected, actual error) {
	t.Helper()
	if !errors.Is(actual, expected) {
		t.Fatalf("database error = %v, want %v", actual, expected)
	}
}

type errorRow struct {
	err error
}

func (row errorRow) Scan(...any) error {
	return row.err
}

type panicDatabase struct{}

func (panicDatabase) Begin(context.Context) (pgx.Tx, error) {
	panic("unexpected database Begin")
}

func (panicDatabase) Exec(
	context.Context,
	string,
	...any,
) (pgconn.CommandTag, error) {
	panic("unexpected database Exec")
}

func (panicDatabase) QueryRow(context.Context, string, ...any) pgx.Row {
	panic("unexpected database QueryRow")
}

type failingDatabase struct {
	err error
}

func (database failingDatabase) Begin(context.Context) (pgx.Tx, error) {
	return nil, database.err
}

func (failingDatabase) Exec(
	context.Context,
	string,
	...any,
) (pgconn.CommandTag, error) {
	panic("unexpected database Exec")
}

func (failingDatabase) QueryRow(context.Context, string, ...any) pgx.Row {
	panic("unexpected database QueryRow")
}

type execFailDatabase struct {
	err error
}

func (database execFailDatabase) Begin(context.Context) (pgx.Tx, error) {
	return nil, database.err
}

func (database execFailDatabase) Exec(
	context.Context,
	string,
	...any,
) (pgconn.CommandTag, error) {
	return pgconn.CommandTag{}, database.err
}

func (database execFailDatabase) QueryRow(context.Context, string, ...any) pgx.Row {
	return errorRow{err: database.err}
}

func withJob(value Job, change func(*Job)) Job {
	change(&value)
	return value
}

func withGraph(value GraphGeneration, change func(*GraphGeneration)) GraphGeneration {
	change(&value)
	return value
}

func withLabel(value LabelGeneration, change func(*LabelGeneration)) LabelGeneration {
	change(&value)
	return value
}

func withBatch(value BatchAttempt, change func(*BatchAttempt)) BatchAttempt {
	change(&value)
	return value
}

func withReject(value RejectRecord, change func(*RejectRecord)) RejectRecord {
	change(&value)
	return value
}
