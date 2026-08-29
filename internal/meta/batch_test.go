package meta

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestStartBatchFastPathAndFallbackStates(t *testing.T) {
	base := BatchAttempt{
		JobID: testJobID, BatchID: 1, Attempt: 1,
		Rows: 3, Bytes: 30, First: Position{Resource: "file", Line: 1, Token: "first"},
	}
	started := base
	started.Status = BatchRunning
	store := &Store{database: &batchFallbackDatabase{fast: batchAttemptRow(started)}}
	got, err := store.StartBatch(t.Context(), base)
	if err != nil || !sameBatchInput(got, base) || got.Status != BatchRunning {
		t.Fatalf("StartBatch() fast path = %#v, %v", got, err)
	}

	for _, test := range []struct {
		name  string
		input BatchAttempt
		tx    *scriptedLifecycleTx
		want  BatchStatus
	}{
		{
			name:  "idempotent running",
			input: base,
			tx: &scriptedLifecycleTx{rows: []scanLifecycleRow{
				jobBatchStateRow(JobRunning, 1),
				batchAttemptRow(started),
			}},
			want: BatchRunning,
		},
		{
			name:  "idempotent committed",
			input: base,
			tx: &scriptedLifecycleTx{rows: []scanLifecycleRow{
				jobBatchStateRow(JobCommitted, 2),
				batchAttemptRow(withBatchStatus(started, BatchCommitted)),
			}},
			want: BatchCommitted,
		},
		{
			name:  "retry failed attempt",
			input: withBatchAttempt(base, 2),
			tx: &scriptedLifecycleTx{
				rows: []scanLifecycleRow{
					jobBatchStateRow(JobRunning, 1),
					batchAttemptRow(withBatchStatus(started, BatchFailed)),
					batchAttemptRow(withBatchStatus(withBatchAttempt(started, 2), BatchRunning)),
				},
				exec: []scriptedLifecycleExec{{tag: pgconn.NewCommandTag("INSERT 0 1")}},
			},
			want: BatchRunning,
		},
		{
			name:  "new first attempt",
			input: base,
			tx: &scriptedLifecycleTx{
				rows: []scanLifecycleRow{
					jobBatchStateRow(JobRunning, 1),
					func(...any) error { return pgx.ErrNoRows },
					batchAttemptRow(started),
				},
				exec: []scriptedLifecycleExec{{tag: pgconn.NewCommandTag("INSERT 0 1")}},
			},
			want: BatchRunning,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			store := &Store{database: &batchFallbackDatabase{
				fast: func(...any) error { return pgx.ErrNoRows },
				tx:   test.tx,
			}}
			got, err := store.StartBatch(t.Context(), test.input)
			if err != nil || got.Status != test.want || got.Attempt != test.input.Attempt {
				t.Fatalf("StartBatch() = %#v, %v", got, err)
			}
		})
	}
}

func TestStartBatchFallbackErrors(t *testing.T) {
	base := BatchAttempt{JobID: testJobID, BatchID: 1, Attempt: 1, Rows: 2}
	injected := errors.New("injected")
	if _, err := (&Store{database: &batchFallbackDatabase{
		fast:     func(...any) error { return pgx.ErrNoRows },
		beginErr: injected,
	}}).StartBatch(t.Context(), base); !errors.Is(err, injected) {
		t.Fatalf("StartBatch begin error = %v", err)
	}
	for _, test := range []struct {
		name string
		tx   *scriptedLifecycleTx
		want error
	}{
		{
			name: "lock",
			tx: &scriptedLifecycleTx{rows: []scanLifecycleRow{
				func(...any) error { return injected },
			}},
			want: injected,
		},
		{
			name: "latest",
			tx: &scriptedLifecycleTx{rows: []scanLifecycleRow{
				jobBatchStateRow(JobRunning, 1),
				func(...any) error { return injected },
			}},
			want: injected,
		},
		{
			name: "different idempotent metadata",
			tx: &scriptedLifecycleTx{rows: []scanLifecycleRow{
				jobBatchStateRow(JobRunning, 1),
				batchAttemptRow(withBatchRows(withBatchStatus(base, BatchRunning), 99)),
			}},
			want: ErrConflict,
		},
		{
			name: "running not ready",
			tx: &scriptedLifecycleTx{rows: []scanLifecycleRow{
				jobBatchStateRow(JobRunning, 2),
				batchAttemptRow(withBatchStatus(base, BatchRunning)),
			}},
			want: ErrConflict,
		},
		{
			name: "committed inconsistent job",
			tx: &scriptedLifecycleTx{rows: []scanLifecycleRow{
				jobBatchStateRow(JobFailed, 2),
				batchAttemptRow(withBatchStatus(base, BatchCommitted)),
			}},
			want: ErrConflict,
		},
		{
			name: "nonretryable latest",
			tx: &scriptedLifecycleTx{rows: []scanLifecycleRow{
				jobBatchStateRow(JobRunning, 1),
				batchAttemptRow(withBatchStatus(base, BatchRunning)),
			}},
			want: ErrConflict,
		},
		{
			name: "max attempt",
			tx: &scriptedLifecycleTx{rows: []scanLifecycleRow{
				jobBatchStateRow(JobRunning, 1),
				batchAttemptRow(withBatchStatus(withBatchAttempt(base, math.MaxUint32), BatchFailed)),
			}},
			want: ErrConflict,
		},
		{
			name: "wrong next attempt",
			tx: &scriptedLifecycleTx{rows: []scanLifecycleRow{
				jobBatchStateRow(JobRunning, 1),
				func(...any) error { return pgx.ErrNoRows },
			}},
			want: ErrConflict,
		},
		{
			name: "insert",
			tx: &scriptedLifecycleTx{
				rows: []scanLifecycleRow{
					jobBatchStateRow(JobRunning, 1),
					func(...any) error { return pgx.ErrNoRows },
				},
				exec: []scriptedLifecycleExec{{err: injected}},
			},
			want: injected,
		},
		{
			name: "affected rows",
			tx: &scriptedLifecycleTx{
				rows: []scanLifecycleRow{
					jobBatchStateRow(JobRunning, 1),
					func(...any) error { return pgx.ErrNoRows },
				},
				exec: []scriptedLifecycleExec{{tag: pgconn.NewCommandTag("INSERT 0 0")}},
			},
			want: ErrConflict,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			input := base
			if test.name == "wrong next attempt" || test.name == "nonretryable latest" {
				input.Attempt = 2
			}
			_, err := (&Store{database: &batchFallbackDatabase{
				fast: func(...any) error { return pgx.ErrNoRows },
				tx:   test.tx,
			}}).StartBatch(t.Context(), input)
			if !errors.Is(err, test.want) {
				t.Fatalf("StartBatch() error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestBatchVerificationAndCounterValidationHelpers(t *testing.T) {
	base := JobVerification{
		JobID:                      testJobID,
		SubmittedConfigFingerprint: strings.Repeat("a", 64),
		ResolvedMappingFingerprint: strings.Repeat("b", 64),
		ResolvedMappingSummary:     []byte(`{"a":1,"b":[2]}`),
	}
	reordered := base
	reordered.ResolvedMappingSummary = []byte(`{"b":[2],"a":1}`)
	if !sameJobVerification(base, reordered) {
		t.Fatal("sameJobVerification rejected equivalent JSON")
	}
	for _, change := range []func(*JobVerification){
		func(value *JobVerification) { value.JobID = "other" },
		func(value *JobVerification) { value.SubmittedConfigFingerprint = "other" },
		func(value *JobVerification) { value.ResolvedMappingFingerprint = "other" },
		func(value *JobVerification) { value.ResolvedMappingSummary = []byte(`{"a":2}`) },
		func(value *JobVerification) { value.ResolvedMappingSummary = []byte(`{`) },
	} {
		value := base
		change(&value)
		if sameJobVerification(base, value) {
			t.Errorf("sameJobVerification accepted %#v", value)
		}
	}

	bytes := int64(10)
	valid := []BatchLabelCounter{
		{LabelGenerationID: 1, Kind: VertexLabel, AcceptedRows: 2, CommittedRows: 2, CommittedBytes: &bytes},
		{LabelGenerationID: 2, Kind: EdgeLabel, RejectedRows: 1},
	}
	if err := validateBatchLabelCounters(valid, 1); err != nil {
		t.Fatalf("validateBatchLabelCounters() error = %v", err)
	}
	for _, counters := range [][]BatchLabelCounter{
		{{LabelGenerationID: 0, Kind: VertexLabel}},
		{{LabelGenerationID: 1, Kind: LabelKind('x')}},
		{{LabelGenerationID: 1, Kind: VertexLabel, AcceptedRows: -1}},
		{{LabelGenerationID: 1, Kind: VertexLabel, CommittedRows: -1}},
		{{LabelGenerationID: 1, Kind: VertexLabel, RejectedRows: -1}},
		{{LabelGenerationID: 1, Kind: VertexLabel, CommittedBytes: int64Pointer(-1)}},
		{{LabelGenerationID: 1, Kind: VertexLabel}, {LabelGenerationID: 1, Kind: EdgeLabel}},
		{{LabelGenerationID: 1, Kind: VertexLabel, RejectedRows: 2}},
	} {
		if err := validateBatchLabelCounters(counters, 1); err == nil {
			t.Errorf("validateBatchLabelCounters(%#v) succeeded", counters)
		}
	}
}

func TestRecordFailedBatchExistingNewAndIdempotent(t *testing.T) {
	base := BatchAttempt{
		JobID: testJobID, BatchID: 1, Attempt: 1,
		Rows: 2, Bytes: 20, RejectedRows: 1, First: Position{Token: "first"},
	}
	for _, test := range []struct {
		name string
		tx   *scriptedLifecycleTx
	}{
		{
			name: "running",
			tx: &scriptedLifecycleTx{
				rows: []scanLifecycleRow{
					jobBatchStateRow(JobRunning, 1),
					batchAttemptRow(withBatchStatus(base, BatchRunning)),
				},
				exec: []scriptedLifecycleExec{
					{tag: pgconn.NewCommandTag("UPDATE 1")},
					{tag: pgconn.NewCommandTag("UPDATE 1")},
				},
			},
		},
		{
			name: "new",
			tx: &scriptedLifecycleTx{
				rows: []scanLifecycleRow{
					jobBatchStateRow(JobPending, 1),
					func(...any) error { return pgx.ErrNoRows },
				},
				exec: []scriptedLifecycleExec{
					{tag: pgconn.NewCommandTag("INSERT 0 1")},
					{tag: pgconn.NewCommandTag("UPDATE 1")},
				},
			},
		},
		{
			name: "idempotent",
			tx: &scriptedLifecycleTx{rows: []scanLifecycleRow{
				jobBatchStateRow(JobFailed, 1),
				batchAttemptRow(withBatchError(withBatchStatus(base, BatchFailed), "failure")),
			}},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := (&Store{database: &batchFallbackDatabase{tx: test.tx}}).
				RecordFailedBatch(t.Context(), base, "failure")
			if err != nil {
				t.Fatalf("RecordFailedBatch() error = %v", err)
			}
		})
	}
}

func TestRecordFailedBatchConflictAndDatabaseErrors(t *testing.T) {
	base := BatchAttempt{JobID: testJobID, BatchID: 1, Attempt: 1, Rows: 2}
	injected := errors.New("injected")
	tests := []struct {
		name string
		tx   *scriptedLifecycleTx
		want error
	}{
		{"job state", &scriptedLifecycleTx{rows: []scanLifecycleRow{
			jobBatchStateRow(JobCommitted, 1),
		}}, ErrConflict},
		{"newer attempt", &scriptedLifecycleTx{rows: []scanLifecycleRow{
			jobBatchStateRow(JobRunning, 1),
			batchAttemptRow(withBatchAttempt(base, 2)),
		}}, ErrConflict},
		{"metadata differs", &scriptedLifecycleTx{rows: []scanLifecycleRow{
			jobBatchStateRow(JobRunning, 1),
			batchAttemptRow(withBatchRows(withBatchStatus(base, BatchRunning), 3)),
		}}, ErrConflict},
		{"committed", &scriptedLifecycleTx{rows: []scanLifecycleRow{
			jobBatchStateRow(JobRunning, 1),
			batchAttemptRow(withBatchStatus(base, BatchCommitted)),
		}}, ErrConflict},
		{"failed differs", &scriptedLifecycleTx{rows: []scanLifecycleRow{
			jobBatchStateRow(JobFailed, 1),
			batchAttemptRow(withBatchError(withBatchStatus(base, BatchFailed), "other")),
		}}, ErrConflict},
		{"unsupported state", &scriptedLifecycleTx{rows: []scanLifecycleRow{
			jobBatchStateRow(JobRunning, 1),
			batchAttemptRow(withBatchStatus(base, BatchStatus("odd"))),
		}}, ErrConflict},
		{"update batch", &scriptedLifecycleTx{
			rows: []scanLifecycleRow{
				jobBatchStateRow(JobRunning, 1),
				batchAttemptRow(withBatchStatus(base, BatchRunning)),
			},
			exec: []scriptedLifecycleExec{{err: injected}},
		}, injected},
		{"update job", &scriptedLifecycleTx{
			rows: []scanLifecycleRow{
				jobBatchStateRow(JobRunning, 1),
				func(...any) error { return pgx.ErrNoRows },
			},
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("INSERT 0 1")},
				{err: injected},
			},
		}, injected},
		{"commit", &scriptedLifecycleTx{
			rows: []scanLifecycleRow{
				jobBatchStateRow(JobRunning, 1),
				func(...any) error { return pgx.ErrNoRows },
			},
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("INSERT 0 1")},
				{tag: pgconn.NewCommandTag("UPDATE 1")},
			},
			commitErr: injected,
		}, injected},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := (&Store{database: &batchFallbackDatabase{tx: test.tx}}).
				RecordFailedBatch(t.Context(), base, "failure")
			if !errors.Is(err, test.want) {
				t.Fatalf("RecordFailedBatch() error = %v, want %v", err, test.want)
			}
		})
	}
}

type batchFallbackDatabase struct {
	fast     scanLifecycleRow
	tx       pgx.Tx
	beginErr error
}

func (database *batchFallbackDatabase) Begin(context.Context) (pgx.Tx, error) {
	return database.tx, database.beginErr
}

func (*batchFallbackDatabase) Exec(context.Context, string, ...any) (pgconn.CommandTag, error) {
	panic("unexpected Exec")
}

func (database *batchFallbackDatabase) QueryRow(context.Context, string, ...any) pgx.Row {
	return database.fast
}

func jobBatchStateRow(status JobStatus, next uint64) scanLifecycleRow {
	return func(dest ...any) error {
		*dest[0].(*JobStatus) = status
		*dest[1].(*uint64) = next
		return nil
	}
}

func batchAttemptRow(value BatchAttempt) scanLifecycleRow {
	return func(dest ...any) error {
		*dest[0].(*string) = value.JobID
		*dest[1].(*uint64) = value.BatchID
		*dest[2].(*uint32) = value.Attempt
		*dest[3].(*BatchStatus) = value.Status
		*dest[4].(*int64) = value.Rows
		*dest[5].(*int64) = value.Bytes
		*dest[6].(*int64) = value.RejectedRows
		*dest[7].(*string) = value.First.Resource
		*dest[8].(*int64) = value.First.Line
		*dest[9].(*int64) = value.First.ByteOffset
		*dest[10].(*string) = value.First.Token
		*dest[11].(*string) = value.Last.Resource
		*dest[12].(*int64) = value.Last.Line
		*dest[13].(*int64) = value.Last.ByteOffset
		*dest[14].(*string) = value.Last.Token
		*dest[15].(*string) = value.ErrorMessage
		*dest[16].(*time.Time) = value.StartedAt
		if value.FinishedAt != nil {
			*dest[17].(**time.Time) = value.FinishedAt
		}
		return nil
	}
}

func withBatchStatus(value BatchAttempt, status BatchStatus) BatchAttempt {
	value.Status = status
	return value
}

func withBatchAttempt(value BatchAttempt, attempt uint32) BatchAttempt {
	value.Attempt = attempt
	return value
}

func withBatchRows(value BatchAttempt, rows int64) BatchAttempt {
	value.Rows = rows
	return value
}

func withBatchError(value BatchAttempt, message string) BatchAttempt {
	value.ErrorMessage = message
	return value
}

func int64Pointer(value int64) *int64 { return &value }
