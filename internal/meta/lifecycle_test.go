package meta

import (
	"context"
	"errors"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestLifecycleMetadataValidation(t *testing.T) {
	store := &Store{}
	if err := store.CompleteJobGeneration(t.Context(), "bad", 1); err == nil {
		t.Fatal("CompleteJobGeneration() accepted invalid job ID")
	}
	if err := store.CompleteJobGeneration(t.Context(), testJobID, 0); err == nil {
		t.Fatal("CompleteJobGeneration() accepted invalid generation ID")
	}
	if _, err := store.CountLabelIdentities(t.Context(), 0, 1, VertexLabel); err == nil {
		t.Fatal("CountLabelIdentities() accepted invalid graph generation")
	}
	if _, err := store.CountLabelIdentities(t.Context(), 1, 1, LabelKind('x')); err == nil {
		t.Fatal("CountLabelIdentities() accepted invalid label kind")
	}
	if err := store.SetSourceRejections(t.Context(), "bad", 0, Position{}); err == nil {
		t.Fatal("SetSourceRejections() accepted invalid job ID")
	}
	if err := store.SetSourceRejections(t.Context(), testJobID, -1, Position{}); err == nil {
		t.Fatal("SetSourceRejections() accepted negative count")
	}
	if err := store.SetSourceRejections(t.Context(), testJobID, 1, Position{}); err == nil {
		t.Fatal("SetSourceRejections() accepted missing token")
	}
}

func TestLifecycleMetadataDatabaseErrors(t *testing.T) {
	injected := errors.New("injected database failure")
	store := &Store{database: errorDatabase{err: injected}}
	if err := store.CompleteJobGeneration(t.Context(), testJobID, 1); !errors.Is(err, injected) {
		t.Fatalf("CompleteJobGeneration(begin) error = %v", err)
	}
	if err := store.SetSourceRejections(
		t.Context(), testJobID, 0, Position{},
	); !errors.Is(err, injected) {
		t.Fatalf("SetSourceRejections() error = %v", err)
	}
	if _, err := store.CountLabelIdentities(
		t.Context(), 1, 1, VertexLabel,
	); !errors.Is(err, injected) {
		t.Fatalf("CountLabelIdentities() error = %v", err)
	}
}

func TestCompleteJobGenerationDatabaseFailures(t *testing.T) {
	injected := errors.New("injected database failure")
	jobRow := func(dest ...any) error {
		*dest[0].(*JobStatus) = JobRunning
		*dest[1].(*uint64) = 1
		*dest[2].(*int64) = 1
		return nil
	}
	unresolvedRow := func(dest ...any) error {
		*dest[0].(*int) = 0
		return nil
	}
	tests := []struct {
		name string
		tx   *scriptedLifecycleTx
	}{
		{name: "lock job", tx: &scriptedLifecycleTx{
			rows: []scanLifecycleRow{func(...any) error { return injected }},
		}},
		{name: "check unresolved", tx: &scriptedLifecycleTx{
			rows: []scanLifecycleRow{jobRow, func(...any) error { return injected }},
		}},
		{name: "activate graph", tx: &scriptedLifecycleTx{
			rows: []scanLifecycleRow{jobRow, unresolvedRow},
			exec: []scriptedLifecycleExec{{err: injected}},
		}},
		{name: "complete job", tx: &scriptedLifecycleTx{
			rows: []scanLifecycleRow{jobRow, unresolvedRow},
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("UPDATE 1")},
				{err: injected},
			},
		}},
		{name: "commit", tx: &scriptedLifecycleTx{
			rows: []scanLifecycleRow{jobRow, unresolvedRow},
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("UPDATE 1")},
				{tag: pgconn.NewCommandTag("UPDATE 1")},
			},
			commitErr: injected,
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store := &Store{database: scriptedLifecycleDatabase{tx: test.tx}}
			if err := store.CompleteJobGeneration(
				t.Context(), testJobID, 1,
			); !errors.Is(err, injected) {
				t.Fatalf("CompleteJobGeneration() error = %v", err)
			}
		})
	}
	for name, tx := range map[string]*scriptedLifecycleTx{
		"wrong generation": {
			rows: []scanLifecycleRow{func(dest ...any) error {
				*dest[0].(*JobStatus) = JobRunning
				*dest[1].(*uint64) = 1
				*dest[2].(*int64) = 2
				return nil
			}},
		},
		"missing graph update": {
			rows: []scanLifecycleRow{jobRow, unresolvedRow},
			exec: []scriptedLifecycleExec{{tag: pgconn.NewCommandTag("UPDATE 0")}},
		},
		"missing job update": {
			rows: []scanLifecycleRow{jobRow, unresolvedRow},
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("UPDATE 1")},
				{tag: pgconn.NewCommandTag("UPDATE 0")},
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			store := &Store{database: scriptedLifecycleDatabase{tx: tx}}
			if err := store.CompleteJobGeneration(t.Context(), testJobID, 1); err == nil {
				t.Fatal("CompleteJobGeneration() error = nil")
			}
		})
	}
}

func TestCompleteJobDatabaseFailures(t *testing.T) {
	injected := errors.New("injected database failure")
	jobRow := func(dest ...any) error {
		*dest[0].(*JobStatus) = JobRunning
		*dest[1].(*uint64) = 1
		return nil
	}
	unresolvedRow := func(dest ...any) error {
		*dest[0].(*int) = 0
		return nil
	}
	tests := map[string]*scriptedLifecycleTx{
		"lock job": {
			rows: []scanLifecycleRow{func(...any) error { return injected }},
		},
		"check unresolved": {
			rows: []scanLifecycleRow{jobRow, func(...any) error { return injected }},
		},
		"update job": {
			rows: []scanLifecycleRow{jobRow, unresolvedRow},
			exec: []scriptedLifecycleExec{{err: injected}},
		},
		"missing update": {
			rows: []scanLifecycleRow{jobRow, unresolvedRow},
			exec: []scriptedLifecycleExec{{tag: pgconn.NewCommandTag("UPDATE 0")}},
		},
		"commit": {
			rows:      []scanLifecycleRow{jobRow, unresolvedRow},
			exec:      []scriptedLifecycleExec{{tag: pgconn.NewCommandTag("UPDATE 1")}},
			commitErr: injected,
		},
	}
	for name, tx := range tests {
		t.Run(name, func(t *testing.T) {
			store := &Store{database: scriptedLifecycleDatabase{tx: tx}}
			if err := store.CompleteJob(t.Context(), testJobID); err == nil {
				t.Fatal("CompleteJob() error = nil")
			}
		})
	}
}

type errorDatabase struct {
	err error
}

func (database errorDatabase) Begin(context.Context) (pgx.Tx, error) {
	return nil, database.err
}

func (database errorDatabase) Exec(
	context.Context,
	string,
	...any,
) (pgconn.CommandTag, error) {
	return pgconn.CommandTag{}, database.err
}

func (database errorDatabase) QueryRow(context.Context, string, ...any) pgx.Row {
	return lifecycleErrorRow{err: database.err}
}

type lifecycleErrorRow struct {
	err error
}

func (row lifecycleErrorRow) Scan(...any) error {
	return row.err
}

type scanLifecycleRow func(...any) error

func (row scanLifecycleRow) Scan(dest ...any) error {
	return row(dest...)
}

type scriptedLifecycleExec struct {
	tag pgconn.CommandTag
	err error
}

type scriptedLifecycleDatabase struct {
	tx pgx.Tx
}

func (database scriptedLifecycleDatabase) Begin(context.Context) (pgx.Tx, error) {
	return database.tx, nil
}

func (scriptedLifecycleDatabase) Exec(
	context.Context, string, ...any,
) (pgconn.CommandTag, error) {
	panic("unexpected database Exec")
}

func (scriptedLifecycleDatabase) QueryRow(context.Context, string, ...any) pgx.Row {
	panic("unexpected database QueryRow")
}

type scriptedLifecycleTx struct {
	pgx.Tx
	rows      []scanLifecycleRow
	exec      []scriptedLifecycleExec
	commitErr error
}

func (tx *scriptedLifecycleTx) QueryRow(context.Context, string, ...any) pgx.Row {
	row := tx.rows[0]
	tx.rows = tx.rows[1:]
	return row
}

func (tx *scriptedLifecycleTx) Exec(
	context.Context, string, ...any,
) (pgconn.CommandTag, error) {
	result := tx.exec[0]
	tx.exec = tx.exec[1:]
	return result.tag, result.err
}

func (tx *scriptedLifecycleTx) Commit(context.Context) error {
	return tx.commitErr
}

func (*scriptedLifecycleTx) Rollback(context.Context) error {
	return nil
}
