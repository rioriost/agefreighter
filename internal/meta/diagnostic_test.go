package meta

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestDiagnosticMigrationIsCurrentVersion(t *testing.T) {
	if schemaVersion != 18 || len(migrations) != schemaVersion {
		t.Fatalf("schema version=%d migrations=%d", schemaVersion, len(migrations))
	}
	if len(migrationV16) != 2 ||
		!strings.Contains(migrationV16[0], "diagnostic_history") ||
		!strings.Contains(migrationV16[1], "diagnostic_history_recent_idx") {
		t.Fatalf("migrationV16 = %#v", migrationV16)
	}
	if len(migrationV17) == 0 ||
		!slices.ContainsFunc(migrationV17, func(statement string) bool {
			return strings.Contains(statement, "job_verification")
		}) {
		t.Fatalf("migrationV17 = %#v", migrationV17)
	}
	if !slices.ContainsFunc(migrationV17, func(statement string) bool {
		return strings.Contains(statement, "job_label_counter") &&
			strings.Contains(statement, "counter_completeness") &&
			strings.Contains(statement, "counter_provenance")
	}) {
		t.Fatalf("migrationV17 counter provenance = %#v", migrationV17)
	}
	for _, statement := range migrationV17 {
		normalized := strings.ToUpper(strings.TrimSpace(statement))
		if strings.HasPrefix(normalized, "UPDATE ") ||
			strings.HasPrefix(normalized, "DELETE ") {
			t.Fatalf("migrationV17 destructively rewrites existing rows: %q", statement)
		}
	}
	if len(migrationV18) != 3 ||
		!strings.Contains(migrationV18[0], "target_backend") ||
		!strings.Contains(migrationV18[1], "load_job_target_identity_ck") ||
		!strings.Contains(migrationV18[2], "DROP DEFAULT") {
		t.Fatalf("migrationV18 = %#v", migrationV18)
	}
}

func TestPersistDiagnosticValidation(t *testing.T) {
	store := &Store{}
	if _, err := store.PersistDiagnostic(
		t.Context(),
		DiagnosticRecord{},
	); err == nil {
		t.Fatal("nil metadata store accepted a diagnostic")
	}
	store = &Store{database: errorDatabase{}}
	if _, err := store.PersistDiagnostic(
		t.Context(),
		DiagnosticRecord{
			Outcome:     "pass",
			TargetGraph: "graph",
			Report:      []byte(`{}`),
		},
	); err == nil || !strings.Contains(err.Error(), "requires a deadline") {
		t.Fatalf("missing deadline error = %v", err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	for _, record := range []DiagnosticRecord{
		{Outcome: "bad", TargetGraph: "graph", Report: []byte(`{}`)},
		{Outcome: "pass", TargetGraph: "", Report: []byte(`{}`)},
		{Outcome: "pass", TargetGraph: "graph", Report: []byte(`not-json`)},
	} {
		if _, err := store.PersistDiagnostic(ctx, record); err == nil {
			t.Fatalf("PersistDiagnostic(%#v) succeeded", record)
		}
	}
}

func TestListDiagnosticsRequiresBoundedDeadline(t *testing.T) {
	store := &Store{database: errorDatabase{}}
	if _, err := store.ListDiagnostics(t.Context(), "graph", 20); err == nil ||
		!strings.Contains(err.Error(), "requires a deadline") {
		t.Fatalf("ListDiagnostics() error = %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := store.ListDiagnostics(ctx, "graph", MaxReadLimit+1); err == nil {
		t.Fatal("ListDiagnostics() accepted an unbounded limit")
	}
	if _, err := store.ListDiagnostics(ctx, "", 20); err == nil {
		t.Fatal("ListDiagnostics() accepted an empty target graph")
	}
}

func TestPersistDiagnosticLocksMigrationAndReinspectsWithBoundedRollback(
	t *testing.T,
) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	tx := &diagnosticTx{
		rows: []pgx.Row{
			inspectCatalogRow(true, true),
			inspectVersionRow(1, SupportedSchemaVersion, SupportedSchemaVersion),
			stubInspectRow(func(dest ...any) error {
				*dest[0].(*int64) = 42
				*dest[1].(*time.Time) = time.Now()
				return nil
			}),
		},
		commitCancel: cancel,
	}
	store := &Store{database: diagnosticDatabase{tx: tx}}
	stored, err := store.PersistDiagnostic(ctx, DiagnosticRecord{
		Outcome:               "pass",
		TargetGraph:           "graph",
		MetadataSchemaVersion: SupportedSchemaVersion,
		Report:                []byte(`{}`),
	})
	if err != nil {
		t.Fatalf("PersistDiagnostic() error = %v", err)
	}
	if stored.ID != 42 || len(tx.execStatements) != 2 {
		t.Fatalf("stored=%#v exec=%v", stored, tx.execStatements)
	}
	if !strings.Contains(tx.execStatements[0], "pg_advisory_xact_lock") ||
		len(tx.execArguments[0]) != 1 ||
		tx.execArguments[0][0] != migrationLockID {
		t.Fatalf(
			"persistence lock = %q %#v",
			tx.execStatements[0],
			tx.execArguments[0],
		)
	}
	if tx.queryCount != 3 {
		t.Fatalf(
			"transaction query count = %d, want locked inspection plus insert",
			tx.queryCount,
		)
	}
	if !tx.rollbackCalled {
		t.Fatal("transaction rollback was not attempted")
	}
	if tx.rollbackErrAtCall != nil {
		t.Fatalf("rollback context was already canceled: %v", tx.rollbackErrAtCall)
	}
	if !tx.rollbackBounded {
		t.Fatal("rollback context did not have an explicit deadline")
	}
}

type diagnosticDatabase struct {
	tx pgx.Tx
}

func (database diagnosticDatabase) Begin(context.Context) (pgx.Tx, error) {
	return database.tx, nil
}

func (diagnosticDatabase) Exec(
	context.Context,
	string,
	...any,
) (pgconn.CommandTag, error) {
	panic("unexpected diagnostic database Exec")
}

func (diagnosticDatabase) QueryRow(context.Context, string, ...any) pgx.Row {
	panic("unexpected diagnostic database QueryRow")
}

type diagnosticTx struct {
	pgx.Tx
	rows              []pgx.Row
	execStatements    []string
	execArguments     [][]any
	queryCount        int
	commitCancel      context.CancelFunc
	rollbackCalled    bool
	rollbackBounded   bool
	rollbackErrAtCall error
}

func (tx *diagnosticTx) Exec(
	_ context.Context,
	statement string,
	arguments ...any,
) (pgconn.CommandTag, error) {
	tx.execStatements = append(tx.execStatements, statement)
	tx.execArguments = append(tx.execArguments, arguments)
	return pgconn.NewCommandTag("SELECT 1"), nil
}

func (tx *diagnosticTx) QueryRow(
	_ context.Context,
	_ string,
	_ ...any,
) pgx.Row {
	tx.queryCount++
	row := tx.rows[0]
	tx.rows = tx.rows[1:]
	return row
}

func (tx *diagnosticTx) Commit(context.Context) error {
	tx.commitCancel()
	return nil
}

func (tx *diagnosticTx) Rollback(ctx context.Context) error {
	tx.rollbackCalled = true
	tx.rollbackErrAtCall = ctx.Err()
	_, tx.rollbackBounded = ctx.Deadline()
	return nil
}
