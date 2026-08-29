package meta

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestEnsureLabelCountersInitializesWithoutIdentityScan(t *testing.T) {
	tests := []struct {
		name       string
		mode       string
		historical bool
		complete   CounterCompleteness
		provenance CounterProvenance
		initial    any
	}{
		{
			name: "new create", mode: "create",
			complete: CounterComplete, provenance: CounterProvenanceLifecycle,
			initial: int64(0),
		},
		{
			name: "legacy resume", mode: "create", historical: true,
			complete: CounterIncomplete, provenance: CounterProvenanceLegacyResume,
		},
		{
			name: "legacy incremental resume", mode: "upsert", historical: true,
			complete: CounterIncomplete, provenance: CounterProvenanceLegacyResume,
		},
		{
			name: "incremental baseline", mode: "append",
			complete: CounterIncomplete, provenance: CounterProvenanceBaselineUnavailable,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tx := &counterTx{
				mode: test.mode, historical: test.historical,
				insertTag: pgconn.NewCommandTag("INSERT 0 1"),
			}
			store := &Store{database: counterDatabase{tx: tx}}
			if err := store.EnsureLabelCounters(
				t.Context(),
				testJobID,
				[]LabelGeneration{{ID: 9, Kind: VertexLabel}},
			); err != nil {
				t.Fatalf("EnsureLabelCounters() error = %v", err)
			}
			if len(tx.execStatements) != 1 ||
				strings.Contains(strings.ToLower(tx.execStatements[0]), "count(") {
				t.Fatalf("counter initialization statements = %#v", tx.execStatements)
			}
			arguments := tx.execArguments[0]
			if arguments[3] != string(test.complete) ||
				arguments[4] != string(test.provenance) ||
				arguments[5] != test.initial {
				t.Fatalf("counter initialization arguments = %#v", arguments)
			}
			if !tx.committed || !tx.rollbackBounded {
				t.Fatalf("committed=%t rollbackBounded=%t", tx.committed, tx.rollbackBounded)
			}
		})
	}
}

func TestEnsureLabelCountersPropagatesCancellationWithBoundedRollback(t *testing.T) {
	tx := &counterTx{execErr: context.Canceled}
	store := &Store{database: counterDatabase{tx: tx}}
	err := store.EnsureLabelCounters(
		t.Context(),
		testJobID,
		[]LabelGeneration{{ID: 9, Kind: VertexLabel}},
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("EnsureLabelCounters() error = %v", err)
	}
	if !tx.rollbackBounded {
		t.Fatal("rollback did not have an independent deadline")
	}
}

type counterDatabase struct {
	tx pgx.Tx
}

func (database counterDatabase) Begin(context.Context) (pgx.Tx, error) {
	return database.tx, nil
}

func (counterDatabase) Exec(context.Context, string, ...any) (pgconn.CommandTag, error) {
	panic("unexpected database Exec")
}

func (counterDatabase) QueryRow(context.Context, string, ...any) pgx.Row {
	panic("unexpected database QueryRow")
}

type counterTx struct {
	pgx.Tx
	mode            string
	historical      bool
	insertTag       pgconn.CommandTag
	execErr         error
	execStatements  []string
	execArguments   [][]any
	committed       bool
	rollbackBounded bool
}

func (tx *counterTx) QueryRow(
	context.Context,
	string,
	...any,
) pgx.Row {
	return counterRow(func(dest ...any) error {
		*dest[0].(*JobStatus) = JobRunning
		*dest[1].(*string) = tx.mode
		if tx.historical {
			*dest[2].(*int64) = 1
			*dest[6].(*bool) = true
		}
		return nil
	})
}

func (tx *counterTx) Exec(
	_ context.Context,
	statement string,
	arguments ...any,
) (pgconn.CommandTag, error) {
	tx.execStatements = append(tx.execStatements, statement)
	tx.execArguments = append(tx.execArguments, arguments)
	if tx.execErr != nil {
		return pgconn.CommandTag{}, tx.execErr
	}
	return tx.insertTag, nil
}

func (tx *counterTx) Commit(context.Context) error {
	tx.committed = true
	return nil
}

func (tx *counterTx) Rollback(ctx context.Context) error {
	_, tx.rollbackBounded = ctx.Deadline()
	return nil
}

type counterRow func(...any) error

func (row counterRow) Scan(dest ...any) error {
	return row(dest...)
}

func TestJobVerificationPutGetAndValidation(t *testing.T) {
	valid := JobVerification{
		JobID:                      testJobID,
		SubmittedConfigFingerprint: strings.Repeat("a", 64),
		ResolvedMappingFingerprint: strings.Repeat("b", 64),
		ResolvedMappingSummary:     []byte(`{"labels":2}`),
	}
	database := &directCounterDatabase{
		execTag: pgconn.NewCommandTag("INSERT 0 1"),
		row: counterRow(func(dest ...any) error {
			*dest[0].(*string) = valid.JobID
			*dest[1].(*string) = valid.SubmittedConfigFingerprint
			*dest[2].(*string) = valid.ResolvedMappingFingerprint
			*dest[3].(*json.RawMessage) = append(json.RawMessage(nil), valid.ResolvedMappingSummary...)
			return nil
		}),
	}
	store := &Store{database: database}
	if err := store.PutJobVerification(t.Context(), valid); err != nil {
		t.Fatalf("PutJobVerification() error = %v", err)
	}
	got, err := store.GetJobVerification(t.Context(), testJobID)
	if err != nil || !sameJobVerification(got, valid) {
		t.Fatalf("GetJobVerification() = %#v, %v", got, err)
	}

	for _, change := range []func(*JobVerification){
		func(value *JobVerification) { value.JobID = "bad" },
		func(value *JobVerification) { value.SubmittedConfigFingerprint = "bad" },
		func(value *JobVerification) { value.ResolvedMappingFingerprint = strings.Repeat("G", 64) },
		func(value *JobVerification) { value.ResolvedMappingSummary = nil },
		func(value *JobVerification) { value.ResolvedMappingSummary = []byte(`[]`) },
		func(value *JobVerification) { value.ResolvedMappingSummary = []byte(`{`) },
	} {
		value := valid
		change(&value)
		if err := validateJobVerification(value); err == nil {
			t.Errorf("validateJobVerification(%#v) succeeded", value)
		}
	}
	injected := errors.New("database failed")
	database.execErr = injected
	if err := store.PutJobVerification(t.Context(), valid); !errors.Is(err, injected) {
		t.Fatalf("PutJobVerification() database error = %v", err)
	}
	database.execErr = nil
	database.execTag = pgconn.NewCommandTag("INSERT 0 0")
	if err := store.PutJobVerification(t.Context(), valid); !errors.Is(err, ErrConflict) {
		t.Fatalf("PutJobVerification() affected rows error = %v", err)
	}
	database.row = counterRow(func(...any) error { return pgx.ErrNoRows })
	if _, err := store.GetJobVerification(t.Context(), testJobID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetJobVerification() missing error = %v", err)
	}
	database.row = counterRow(func(...any) error { return injected })
	if _, err := store.GetJobVerification(t.Context(), testJobID); !errors.Is(err, injected) {
		t.Fatalf("GetJobVerification() database error = %v", err)
	}
}

func TestEnsureLabelCountersConflictAndFailureBranches(t *testing.T) {
	validLabels := []LabelGeneration{
		{ID: 2, Kind: EdgeLabel},
		{ID: 1, Kind: VertexLabel},
	}
	store := &Store{database: panicDatabase{}}
	for _, labels := range [][]LabelGeneration{
		{{ID: 0, Kind: VertexLabel}},
		{{ID: 1, Kind: LabelKind('x')}},
		{{ID: 1, Kind: VertexLabel}, {ID: 1, Kind: EdgeLabel}},
	} {
		if err := store.EnsureLabelCounters(t.Context(), testJobID, labels); err == nil {
			t.Errorf("EnsureLabelCounters(%#v) succeeded", labels)
		}
	}
	if err := store.EnsureLabelCounters(t.Context(), "bad", nil); err == nil {
		t.Fatal("EnsureLabelCounters accepted invalid job ID")
	}
	injected := errors.New("injected")
	if err := (&Store{database: counterBeginDatabase{err: injected}}).
		EnsureLabelCounters(t.Context(), testJobID, validLabels); !errors.Is(err, injected) {
		t.Fatalf("EnsureLabelCounters() begin error = %v", err)
	}

	lockRunning := func(dest ...any) error {
		*dest[0].(*JobStatus) = JobRunning
		*dest[1].(*string) = "create"
		return nil
	}
	for _, test := range []struct {
		name string
		tx   *scriptedCounterTx
		want error
	}{
		{"missing job", &scriptedCounterTx{rows: []counterRow{
			func(...any) error { return pgx.ErrNoRows },
		}}, ErrNotFound},
		{"lock error", &scriptedCounterTx{rows: []counterRow{
			func(...any) error { return injected },
		}}, injected},
		{"not running", &scriptedCounterTx{rows: []counterRow{
			func(dest ...any) error {
				*dest[0].(*JobStatus) = JobFailed
				*dest[1].(*string) = "create"
				return nil
			},
		}}, ErrConflict},
		{"read existing error", &scriptedCounterTx{
			rows: []counterRow{lockRunning, func(...any) error { return injected }},
			tags: []pgconn.CommandTag{pgconn.NewCommandTag("INSERT 0 0")},
		}, injected},
		{"kind changed", &scriptedCounterTx{
			rows: []counterRow{lockRunning, storedCounterRow(EdgeLabel, CounterComplete, CounterProvenanceLifecycle)},
			tags: []pgconn.CommandTag{pgconn.NewCommandTag("INSERT 0 0")},
		}, ErrConflict},
		{"bad completeness", &scriptedCounterTx{
			rows: []counterRow{lockRunning, storedCounterRow(VertexLabel, CounterCompleteness("bad"), CounterProvenanceLifecycle)},
			tags: []pgconn.CommandTag{pgconn.NewCommandTag("INSERT 0 0")},
		}, ErrConflict},
		{"bad provenance", &scriptedCounterTx{
			rows: []counterRow{lockRunning, storedCounterRow(VertexLabel, CounterComplete, CounterProvenance("bad"))},
			tags: []pgconn.CommandTag{pgconn.NewCommandTag("INSERT 0 0")},
		}, ErrConflict},
		{"commit", &scriptedCounterTx{
			rows:      []counterRow{lockRunning},
			tags:      []pgconn.CommandTag{pgconn.NewCommandTag("INSERT 0 1")},
			commitErr: injected,
		}, injected},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := (&Store{database: counterDatabase{tx: test.tx}}).
				EnsureLabelCounters(t.Context(), testJobID, []LabelGeneration{{ID: 1, Kind: VertexLabel}})
			if !errors.Is(err, test.want) {
				t.Fatalf("EnsureLabelCounters() error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestLabelCounterListsAndUnclassifiedRejects(t *testing.T) {
	ctx := t.Context()
	accepted, committed, bytes, rejected := int64(5), int64(4), int64(40), int64(1)
	scan := func(dest ...any) error {
		*dest[0].(*string) = testJobID
		*dest[1].(*int64) = 9
		*dest[2].(*LabelKind) = VertexLabel
		*dest[3].(*CounterCompleteness) = CounterComplete
		*dest[4].(*CounterProvenance) = CounterProvenanceLifecycle
		*dest[5].(**int64) = &accepted
		*dest[6].(**int64) = &committed
		*dest[7].(**int64) = &bytes
		*dest[8].(**int64) = &rejected
		return nil
	}
	for _, run := range []func(*Store) ([]LabelCounter, error){
		func(store *Store) ([]LabelCounter, error) {
			return store.ListLabelCounters(ctx, testJobID, 2)
		},
		func(store *Store) ([]LabelCounter, error) {
			return store.ListLabelCountersByID(ctx, testJobID, []int64{9})
		},
	} {
		rows := &stubRows{rows: []func(...any) error{scan}}
		values, err := run(&Store{database: &targetedReadDatabase{rows: rows}})
		if err != nil || len(values) != 1 || values[0].CommittedBytes == nil || *values[0].CommittedBytes != 40 {
			t.Errorf("label counter list = %#v, %v", values, err)
		}
		rows = &stubRows{rows: []func(...any) error{func(...any) error { return errors.New("scan") }}}
		if _, err := run(&Store{database: &targetedReadDatabase{rows: rows}}); err == nil {
			t.Error("label counter scan error = nil")
		}
		rows = &stubRows{err: errors.New("rows")}
		if _, err := run(&Store{database: &targetedReadDatabase{rows: rows}}); err == nil {
			t.Error("label counter rows error = nil")
		}
	}

	if _, err := (&Store{database: panicDatabase{}}).ListLabelCounters(ctx, testJobID, 0); err == nil {
		t.Fatal("ListLabelCounters accepted zero limit")
	}
	if _, err := (&Store{database: panicDatabase{}}).ListLabelCounters(ctx, testJobID, MaxReadLimit+1); err == nil {
		t.Fatal("ListLabelCounters accepted excessive limit")
	}
	if _, err := (&Store{database: panicDatabase{}}).ListLabelCounters(ctx, testJobID, 1); err == nil {
		t.Fatal("ListLabelCounters accepted unsupported database")
	}
	for _, ids := range [][]int64{{0}, {-1}, {2, 2}} {
		if _, err := (&Store{database: panicDatabase{}}).ListLabelCountersByID(ctx, testJobID, ids); err == nil {
			t.Errorf("ListLabelCountersByID(%v) succeeded", ids)
		}
	}
	if values, err := (&Store{database: panicDatabase{}}).ListLabelCountersByID(ctx, testJobID, nil); err != nil || len(values) != 0 {
		t.Fatalf("empty ListLabelCountersByID = %#v, %v", values, err)
	}
	if _, err := (&Store{database: panicDatabase{}}).ListLabelCountersByID(ctx, testJobID, make([]int64, MaxReadLimit+1)); err == nil {
		t.Fatal("ListLabelCountersByID accepted excessive IDs")
	}

	database := &directCounterDatabase{row: counterRow(func(dest ...any) error {
		*dest[0].(*int64) = 7
		return nil
	})}
	count, err := (&Store{database: database}).GetUnclassifiedRejects(ctx, testJobID)
	if err != nil || count != 7 {
		t.Fatalf("GetUnclassifiedRejects() = %d, %v", count, err)
	}
	database.row = func(...any) error { return pgx.ErrNoRows }
	if _, err := (&Store{database: database}).GetUnclassifiedRejects(ctx, testJobID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing GetUnclassifiedRejects() error = %v", err)
	}
	database.row = func(...any) error { return errors.New("read") }
	if _, err := (&Store{database: database}).GetUnclassifiedRejects(ctx, testJobID); err == nil {
		t.Fatal("GetUnclassifiedRejects database error = nil")
	}
	if !jsonValidObject([]byte(" \n{\"a\":1}\t")) || jsonValidObject([]byte("[]")) || jsonValidObject([]byte("{")) {
		t.Fatal("jsonValidObject returned unexpected result")
	}
}

type directCounterDatabase struct {
	execTag pgconn.CommandTag
	execErr error
	row     counterRow
}

func (*directCounterDatabase) Begin(context.Context) (pgx.Tx, error) {
	panic("unexpected Begin")
}

func (database *directCounterDatabase) Exec(context.Context, string, ...any) (pgconn.CommandTag, error) {
	return database.execTag, database.execErr
}

func (database *directCounterDatabase) QueryRow(context.Context, string, ...any) pgx.Row {
	return database.row
}

type counterBeginDatabase struct{ err error }

func (database counterBeginDatabase) Begin(context.Context) (pgx.Tx, error) {
	return nil, database.err
}

func (counterBeginDatabase) Exec(context.Context, string, ...any) (pgconn.CommandTag, error) {
	panic("unexpected Exec")
}

func (counterBeginDatabase) QueryRow(context.Context, string, ...any) pgx.Row {
	panic("unexpected QueryRow")
}

type scriptedCounterTx struct {
	pgx.Tx
	rows      []counterRow
	tags      []pgconn.CommandTag
	commitErr error
}

func (tx *scriptedCounterTx) QueryRow(context.Context, string, ...any) pgx.Row {
	row := tx.rows[0]
	tx.rows = tx.rows[1:]
	return row
}

func (tx *scriptedCounterTx) Exec(context.Context, string, ...any) (pgconn.CommandTag, error) {
	tag := tx.tags[0]
	tx.tags = tx.tags[1:]
	return tag, nil
}

func (tx *scriptedCounterTx) Commit(context.Context) error { return tx.commitErr }

func (*scriptedCounterTx) Rollback(context.Context) error { return nil }

func storedCounterRow(
	kind LabelKind,
	completeness CounterCompleteness,
	provenance CounterProvenance,
) counterRow {
	return func(dest ...any) error {
		*dest[0].(*LabelKind) = kind
		*dest[1].(*CounterCompleteness) = completeness
		*dest[2].(*CounterProvenance) = provenance
		return nil
	}
}
