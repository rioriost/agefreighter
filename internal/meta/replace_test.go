package meta

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestReplaceMetadataValidation(t *testing.T) {
	store := &Store{database: panicDatabase{}}
	if _, err := store.NextGraphGeneration(t.Context(), ""); err == nil {
		t.Fatal("NextGraphGeneration() accepted an empty target")
	}
	if _, err := store.PrepareReplacePromotion(t.Context(), "bad", 1); err == nil {
		t.Fatal("PrepareReplacePromotion() accepted an invalid job ID")
	}
	if _, err := store.PrepareReplacePromotion(
		t.Context(),
		testJobID,
		0,
	); err == nil {
		t.Fatal("PrepareReplacePromotion() accepted an invalid generation ID")
	}
	if err := store.CompleteReplacePromotion(
		t.Context(),
		ReplacePromotion{},
		"target",
		"backup",
	); err == nil {
		t.Fatal("CompleteReplacePromotion() accepted empty metadata")
	}
	if _, err := store.PrepareBackupCleanup(t.Context(), "bad"); err == nil {
		t.Fatal("PrepareBackupCleanup() accepted an invalid job ID")
	}
	if err := store.CompleteBackupCleanup(
		t.Context(),
		BackupCleanup{},
	); err == nil {
		t.Fatal("CompleteBackupCleanup() accepted empty metadata")
	}
	if err := store.CompleteBackupCleanup(
		t.Context(),
		BackupCleanup{Job: Job{ID: testJobID}, AlreadyCleaned: true},
	); err == nil {
		t.Fatal("CompleteBackupCleanup() accepted completed cleanup metadata")
	}
}

func TestReplaceMetadataDatabaseFailures(t *testing.T) {
	injected := errors.New("injected replacement database failure")
	store := &Store{database: errorDatabase{err: injected}}
	if _, err := store.NextGraphGeneration(
		t.Context(),
		"target",
	); !errors.Is(err, injected) {
		t.Fatalf("NextGraphGeneration() error = %v", err)
	}
	if _, err := store.PrepareReplacePromotion(
		t.Context(),
		testJobID,
		1,
	); !errors.Is(err, injected) {
		t.Fatalf("PrepareReplacePromotion() error = %v", err)
	}
	if _, err := store.PrepareBackupCleanup(
		t.Context(),
		testJobID,
	); !errors.Is(err, injected) {
		t.Fatalf("PrepareBackupCleanup() error = %v", err)
	}
	promotion := ReplacePromotion{
		Job: Job{ID: testJobID},
		NewGeneration: GraphGeneration{
			ID:               2,
			JobID:            testJobID,
			GraphName:        "target_shadow",
			GraphOID:         2,
			NamespaceOID:     2,
			ReplacesGraphOID: 1,
			Generation:       2,
			State:            GenerationLoading,
		},
	}
	if err := store.CompleteReplacePromotion(
		t.Context(),
		promotion,
		"target",
		"target_backup",
	); !errors.Is(err, injected) {
		t.Fatalf("CompleteReplacePromotion() error = %v", err)
	}
	cleanup := BackupCleanup{
		Job: Job{
			ID:              testJobID,
			BackupGraphName: "target_backup",
		},
		Generation: promotion.NewGeneration,
	}
	if err := store.CompleteBackupCleanup(
		t.Context(),
		cleanup,
	); !errors.Is(err, injected) {
		t.Fatalf("CompleteBackupCleanup() error = %v", err)
	}
	if _, err := scanGraphGeneration(errorRow{
		err: injected,
	}); !errors.Is(err, injected) {
		t.Fatalf("scanGraphGeneration() error = %v", err)
	}
}

func TestReplacementGraphValidation(t *testing.T) {
	graph := GraphGeneration{
		JobID:            testJobID,
		GraphName:        "shadow",
		GraphOID:         42,
		NamespaceOID:     42,
		ReplacesGraphOID: 42,
		Generation:       1,
		State:            GenerationLoading,
	}
	if err := validateGraphGeneration(graph); err == nil ||
		!strings.Contains(err.Error(), "must differ") {
		t.Fatalf("validateGraphGeneration() error = %v", err)
	}
}

func TestPrepareReplacePromotionConflicts(t *testing.T) {
	newGeneration := GraphGeneration{
		ID:               2,
		JobID:            testJobID,
		GraphName:        "target_shadow",
		GraphOID:         20,
		NamespaceOID:     20,
		ReplacesGraphOID: 10,
		Generation:       2,
		State:            GenerationLoading,
	}
	tests := []struct {
		name string
		rows []rowScanner
	}{
		{
			name: "job not ready",
			rows: []rowScanner{replacePromotionJobRow(
				"create",
				JobRunning,
				2,
				"",
			)},
		},
		{
			name: "unresolved batch",
			rows: []rowScanner{
				replacePromotionJobRow("replace", JobRunning, 2, ""),
				countRow(1),
			},
		},
		{
			name: "generation not loading",
			rows: []rowScanner{
				replacePromotionJobRow("replace", JobRunning, 2, ""),
				countRow(0),
				graphRow(withGraph(newGeneration, func(graph *GraphGeneration) {
					graph.ReplacesGraphOID = 0
				})),
			},
		},
		{
			name: "managed predecessor mismatch",
			rows: []rowScanner{
				replacePromotionJobRow("replace", JobRunning, 2, ""),
				countRow(0),
				graphRow(newGeneration),
				graphRow(GraphGeneration{
					ID: 1, JobID: testJobID, GraphName: "target",
					GraphOID: 11, NamespaceOID: 11, Generation: 1,
					State: GenerationActive,
				}),
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store := &Store{database: &replaceScriptDatabase{rows: test.rows}}
			if _, err := store.PrepareReplacePromotion(
				t.Context(),
				testJobID,
				2,
			); err == nil {
				t.Fatal("PrepareReplacePromotion() error = nil")
			}
		})
	}
}

func TestPrepareBackupCleanupConflicts(t *testing.T) {
	cleanedAt := time.Now()
	tests := []struct {
		name           string
		row            rowScanner
		wantIdempotent bool
		extra          []rowScanner
	}{
		{
			name: "not a replacement",
			row:  backupCleanupJobRow("create", JobCommitted, "", nil),
		},
		{
			name:           "already cleaned",
			row:            backupCleanupJobRow("replace", JobCommitted, "target_backup", &cleanedAt),
			wantIdempotent: true,
		},
		{
			name: "generation not active",
			row:  backupCleanupJobRow("replace", JobCommitted, "target_backup", nil),
			extra: []rowScanner{graphRow(GraphGeneration{
				ID: 2, JobID: testJobID, GraphName: "target",
				GraphOID: 20, NamespaceOID: 20, ReplacesGraphOID: 10,
				Generation: 2, State: GenerationLoading,
			})},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			rows := append([]rowScanner{test.row}, test.extra...)
			store := &Store{database: &replaceScriptDatabase{rows: rows}}
			cleanup, err := store.PrepareBackupCleanup(t.Context(), testJobID)
			if test.wantIdempotent {
				if err != nil || !cleanup.AlreadyCleaned {
					t.Fatalf("PrepareBackupCleanup() = %#v, %v", cleanup, err)
				}
				return
			}
			if err == nil {
				t.Fatal("PrepareBackupCleanup() error = nil")
			}
		})
	}
}

func TestCompleteReplacePromotionConflicts(t *testing.T) {
	base := ReplacePromotion{
		Job: Job{ID: testJobID},
		NewGeneration: GraphGeneration{
			ID: 2, JobID: testJobID, GraphName: "target_shadow",
			GraphOID: 20, NamespaceOID: 20, ReplacesGraphOID: 10,
			Generation: 2, State: GenerationLoading,
		},
	}
	withPrevious := base
	withPrevious.PreviousGeneration = &GraphGeneration{
		ID: 1, JobID: testJobID, GraphName: "target",
		GraphOID: 10, NamespaceOID: 10, Generation: 1,
		State: GenerationActive,
	}
	injected := errors.New("injected promotion update failure")
	tests := []struct {
		name      string
		promotion ReplacePromotion
		exec      []scriptedLifecycleExec
	}{
		{
			name:      "retire predecessor error",
			promotion: withPrevious,
			exec:      []scriptedLifecycleExec{{err: injected}},
		},
		{
			name:      "retire predecessor conflict",
			promotion: withPrevious,
			exec: []scriptedLifecycleExec{{
				tag: pgconn.NewCommandTag("UPDATE 0"),
			}},
		},
		{
			name:      "activate conflict",
			promotion: base,
			exec: []scriptedLifecycleExec{{
				tag: pgconn.NewCommandTag("UPDATE 0"),
			}},
		},
		{
			name:      "complete job error",
			promotion: base,
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("UPDATE 1")},
				{err: injected},
			},
		},
		{
			name:      "complete job conflict",
			promotion: base,
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("UPDATE 1")},
				{tag: pgconn.NewCommandTag("UPDATE 0")},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store := &Store{database: &replaceScriptDatabase{exec: test.exec}}
			if err := store.CompleteReplacePromotion(
				t.Context(),
				test.promotion,
				"target",
				"target_backup",
			); err == nil {
				t.Fatal("CompleteReplacePromotion() error = nil")
			}
		})
	}
}

func TestMigrationFailureBoundaries(t *testing.T) {
	injected := errors.New("injected migration failure")
	ok := scriptedLifecycleExec{tag: pgconn.NewCommandTag("SELECT 1")}
	currentRow := func(version int) scanLifecycleRow {
		return func(dest ...any) error {
			*dest[0].(*int) = version
			return nil
		}
	}
	tests := []struct {
		name string
		tx   *scriptedLifecycleTx
	}{
		{
			name: "migration bounds",
			tx: &scriptedLifecycleTx{
				exec: []scriptedLifecycleExec{{err: injected}},
			},
		},
		{
			name: "migration lock",
			tx: &scriptedLifecycleTx{
				exec: []scriptedLifecycleExec{ok, {err: injected}},
			},
		},
		{
			name: "metadata schema",
			tx: &scriptedLifecycleTx{
				exec: []scriptedLifecycleExec{ok, ok, {err: injected}},
			},
		},
		{
			name: "migration table",
			tx: &scriptedLifecycleTx{
				exec: []scriptedLifecycleExec{ok, ok, ok, {err: injected}},
			},
		},
		{
			name: "read current version",
			tx: &scriptedLifecycleTx{
				exec: []scriptedLifecycleExec{ok, ok, ok, ok},
				rows: []scanLifecycleRow{func(...any) error {
					return injected
				}},
			},
		},
		{
			name: "newer schema",
			tx: &scriptedLifecycleTx{
				exec: []scriptedLifecycleExec{ok, ok, ok, ok},
				rows: []scanLifecycleRow{currentRow(schemaVersion + 1)},
			},
		},
		{
			name: "apply version twelve",
			tx: &scriptedLifecycleTx{
				exec: []scriptedLifecycleExec{ok, ok, ok, ok, {err: injected}},
				rows: []scanLifecycleRow{currentRow(11)},
			},
		},
		{
			name: "record version twelve",
			tx: &scriptedLifecycleTx{
				exec: []scriptedLifecycleExec{
					ok, ok, ok, ok,
					ok, ok,
					{err: injected},
				},
				rows: []scanLifecycleRow{currentRow(11)},
			},
		},
		{
			name: "commit",
			tx: &scriptedLifecycleTx{
				exec:      []scriptedLifecycleExec{ok, ok, ok, ok},
				rows:      []scanLifecycleRow{currentRow(schemaVersion)},
				commitErr: injected,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store := &Store{database: scriptedLifecycleDatabase{tx: test.tx}}
			if err := store.Migrate(t.Context()); err == nil {
				t.Fatal("Migrate() error = nil")
			}
		})
	}
}

func TestMigrationBoundsTransactionBeforeLockAndDDL(t *testing.T) {
	ok := scriptedLifecycleExec{tag: pgconn.NewCommandTag("SELECT 1")}
	tx := &scriptedLifecycleTx{
		exec: []scriptedLifecycleExec{ok, ok, ok, ok},
		rows: []scanLifecycleRow{func(dest ...any) error {
			*dest[0].(*int) = schemaVersion
			return nil
		}},
	}
	store := &Store{database: scriptedLifecycleDatabase{tx: tx}}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	if err := store.Migrate(ctx); err != nil {
		t.Fatalf("Migrate() error = %v", err)
	}
	if len(tx.statements) != 4 ||
		!strings.Contains(tx.statements[0], "lock_timeout") ||
		!strings.Contains(tx.statements[0], "statement_timeout") ||
		!strings.Contains(tx.statements[1], "pg_advisory_xact_lock") ||
		!strings.Contains(tx.statements[2], "CREATE SCHEMA") {
		t.Fatalf("migration statement order = %#v", tx.statements)
	}
	if len(tx.arguments[0]) != 1 {
		t.Fatalf("migration timeout arguments = %#v", tx.arguments[0])
	}
	if timeout, ok := tx.arguments[0][0].(string); !ok ||
		!strings.HasSuffix(timeout, "ms") {
		t.Fatalf("migration timeout = %#v", tx.arguments[0][0])
	}
}

func replacePromotionJobRow(
	mode string,
	status JobStatus,
	generationID int64,
	backup string,
) rowScanner {
	return scanLifecycleRow(func(dest ...any) error {
		*dest[0].(*string) = testJobID
		*dest[1].(*string) = mode
		*dest[2].(*string) = "target"
		*dest[3].(*JobStatus) = status
		*dest[4].(*int64) = generationID
		*dest[5].(*uint64) = 1
		*dest[6].(*string) = backup
		return nil
	})
}

func backupCleanupJobRow(
	mode string,
	status JobStatus,
	backup string,
	cleanedAt *time.Time,
) rowScanner {
	return scanLifecycleRow(func(dest ...any) error {
		*dest[0].(*string) = testJobID
		*dest[1].(*string) = mode
		*dest[2].(*string) = "target"
		*dest[3].(*JobStatus) = status
		*dest[4].(*int64) = 2
		*dest[5].(*string) = backup
		*dest[6].(**time.Time) = cleanedAt
		return nil
	})
}

func countRow(count int) rowScanner {
	return scanLifecycleRow(func(dest ...any) error {
		*dest[0].(*int) = count
		return nil
	})
}

func graphRow(graph GraphGeneration) rowScanner {
	return scanLifecycleRow(func(dest ...any) error {
		*dest[0].(*int64) = graph.ID
		*dest[1].(*string) = graph.JobID
		*dest[2].(*string) = graph.GraphName
		*dest[3].(*uint32) = graph.GraphOID
		*dest[4].(*uint32) = graph.NamespaceOID
		*dest[5].(*uint32) = graph.ReplacesGraphOID
		*dest[6].(*uint64) = graph.Generation
		*dest[7].(*GenerationState) = graph.State
		*dest[8].(*time.Time) = time.Now()
		*dest[9].(*time.Time) = time.Now()
		return nil
	})
}

type replaceScriptDatabase struct {
	rows []rowScanner
	exec []scriptedLifecycleExec
}

func (*replaceScriptDatabase) Begin(context.Context) (pgx.Tx, error) {
	panic("unexpected database Begin")
}

func (database *replaceScriptDatabase) Exec(
	context.Context,
	string,
	...any,
) (pgconn.CommandTag, error) {
	result := database.exec[0]
	database.exec = database.exec[1:]
	return result.tag, result.err
}

func (database *replaceScriptDatabase) QueryRow(
	context.Context,
	string,
	...any,
) pgx.Row {
	row := database.rows[0]
	database.rows = database.rows[1:]
	return row
}
