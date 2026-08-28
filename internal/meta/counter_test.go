package meta

import (
	"context"
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
