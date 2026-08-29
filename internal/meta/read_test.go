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

func TestBoundedReadsRequireExplicitValidLimits(t *testing.T) {
	store := &Store{database: panicDatabase{}}
	validJobID := "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
	tests := []struct {
		name string
		run  func(int) error
	}{
		{
			name: "jobs",
			run: func(limit int) error {
				_, err := store.ListJobs(t.Context(), limit)
				return err
			},
		},
		{
			name: "active job health",
			run: func(limit int) error {
				_, err := store.ListActiveJobHealth(t.Context(), limit)
				return err
			},
		},
		{
			name: "graphs",
			run: func(limit int) error {
				_, err := store.ListGraphGenerations(t.Context(), limit)
				return err
			},
		},
		{
			name: "current target graphs",
			run: func(limit int) error {
				_, err := store.ListCurrentGraphGenerations(
					t.Context(), "graph", limit,
				)
				return err
			},
		},
		{
			name: "label page",
			run: func(limit int) error {
				_, err := store.ListLabelGenerationPage(t.Context(), 1, limit)
				return err
			},
		},
		{
			name: "labels",
			run: func(limit int) error {
				_, err := store.ListLabelGenerations(t.Context(), 1, limit)
				return err
			},
		},
		{
			name: "batches",
			run: func(limit int) error {
				_, err := store.ListBatches(t.Context(), validJobID, limit)
				return err
			},
		},
		{
			name: "reject summaries",
			run: func(limit int) error {
				_, err := store.ListRejectSummaries(t.Context(), validJobID, limit)
				return err
			},
		},
		{
			name: "backups",
			run: func(limit int) error {
				_, err := store.ListRetainedBackups(t.Context(), limit)
				return err
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, limit := range []int{0, -1, MaxReadLimit + 1} {
				if err := test.run(limit); err == nil {
					t.Fatalf("limit %d succeeded", limit)
				}
			}
		})
	}
}

func TestBoundedReadsValidateIdentifiersBeforeQuery(t *testing.T) {
	store := &Store{database: panicDatabase{}}
	if _, err := store.ListLabelGenerations(t.Context(), 0, 1); err == nil {
		t.Fatal("ListLabelGenerations() accepted zero graph generation ID")
	}
	if _, err := store.ListCurrentGraphGenerations(
		t.Context(), "", 1,
	); err == nil {
		t.Fatal("ListCurrentGraphGenerations() accepted an empty graph")
	}
	if _, err := store.ListLabelGenerationPage(t.Context(), 0, 1); err == nil {
		t.Fatal("ListLabelGenerationPage() accepted zero graph generation ID")
	}
	if _, err := store.ListBatches(t.Context(), "bad", 1); err == nil {
		t.Fatal("ListBatches() accepted invalid job ID")
	}

	if _, err := store.ListRejectSummaries(t.Context(), "bad", 1); err == nil {
		t.Fatal("ListRejectSummaries() accepted invalid job ID")
	}
}

func TestDoctorMetadataQueriesFilterBeforeLimitAndReportCompleteness(t *testing.T) {
	now := time.Now()
	jobRows := &stubRows{rows: []func(...any) error{
		func(dest ...any) error {
			*dest[0].(*string) = testJobID
			*dest[1].(*string) = "graph"
			*dest[2].(*JobStatus) = JobRunning
			*dest[3].(*time.Time) = now
			*dest[4].(*bool) = true
			return nil
		},
		func(...any) error { return nil },
	}}
	database := &targetedReadDatabase{rows: jobRows}
	store := &Store{database: database}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	page, err := store.ListActiveJobHealth(ctx, 1)
	if err != nil {
		t.Fatalf("ListActiveJobHealth() error = %v", err)
	}
	if page.Complete || len(page.Jobs) != 1 || !page.Jobs[0].Conflicting {
		t.Fatalf("active job page = %#v", page)
	}
	assertFilterBeforeLimit(t, database.statement, "WHERE status IN")

	graphRows := &stubRows{}
	database = &targetedReadDatabase{rows: graphRows}
	store = &Store{database: database}
	graphPage, err := store.ListCurrentGraphGenerations(ctx, "graph", 1)
	if err != nil {
		t.Fatalf("ListCurrentGraphGenerations() error = %v", err)
	}
	if !graphPage.Complete {
		t.Fatalf("empty graph page = %#v", graphPage)
	}
	assertFilterBeforeLimit(t, database.statement, "WHERE graph_name = $1")
	if !strings.Contains(database.statement, "state IN ('loading', 'active')") {
		t.Fatalf("graph query does not select current states: %s", database.statement)
	}
}

func assertFilterBeforeLimit(t *testing.T, statement, filter string) {
	t.Helper()
	filterAt := strings.Index(statement, filter)
	orderAt := strings.Index(statement, "ORDER BY")
	limitAt := strings.Index(statement, "LIMIT")
	if filterAt < 0 || orderAt < 0 || limitAt < 0 ||
		filterAt > orderAt || orderAt > limitAt {
		t.Fatalf("query does not filter before ordering/limit: %s", statement)
	}
}

type targetedReadDatabase struct {
	panicDatabase
	statement string
	rows      pgx.Rows
	queryErr  error
}

func (database *targetedReadDatabase) Query(
	_ context.Context,
	statement string,
	_ ...any,
) (pgx.Rows, error) {
	database.statement = statement
	if database.queryErr != nil {
		return nil, database.queryErr
	}
	if database.rows == nil {
		return nil, errors.New("missing stub rows")
	}
	return database.rows, nil
}

type stubRows struct {
	rows   []func(...any) error
	index  int
	closed bool
	err    error
}

func (rows *stubRows) Close() {
	rows.closed = true
}

func (rows *stubRows) Err() error {
	return rows.err
}

func (*stubRows) CommandTag() pgconn.CommandTag {
	return pgconn.CommandTag{}
}

func (*stubRows) FieldDescriptions() []pgconn.FieldDescription {
	return nil
}

func (rows *stubRows) Next() bool {
	if rows.index >= len(rows.rows) {
		rows.closed = true
		return false
	}
	rows.index++
	return true
}

func (rows *stubRows) Scan(dest ...any) error {
	return rows.rows[rows.index-1](dest...)
}

func (*stubRows) Values() ([]any, error) {
	return nil, errors.New("not implemented")
}

func (*stubRows) RawValues() [][]byte {
	return nil
}

func (*stubRows) Conn() *pgx.Conn {
	return nil
}

func TestBoundedReadsRequireContextDeadline(t *testing.T) {
	store := &Store{database: panicDatabase{}}
	_, err := store.ListJobs(context.Background(), 1)
	if err == nil || !strings.Contains(err.Error(), "requires a deadline") {
		t.Fatalf("ListJobs() error = %v", err)
	}
}

func TestReadListsScanValidRowsAndErrors(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	now := time.Now().UTC()

	t.Run("jobs", func(t *testing.T) {
		rows := &stubRows{rows: []func(...any) error{
			func(dest ...any) error { return scanTestJob(dest, now) },
		}}
		values, err := (&Store{database: &targetedReadDatabase{rows: rows}}).ListJobs(ctx, 2)
		if err != nil || len(values) != 1 || values[0].ID != testJobID || !rows.closed {
			t.Fatalf("ListJobs() = %#v, %v, closed=%t", values, err, rows.closed)
		}
	})
	t.Run("graph generations", func(t *testing.T) {
		rows := &stubRows{rows: []func(...any) error{
			func(dest ...any) error { return scanTestGraphGeneration(dest, now) },
		}}
		values, err := (&Store{database: &targetedReadDatabase{rows: rows}}).ListGraphGenerations(ctx, 2)
		if err != nil || len(values) != 1 || values[0].GraphName != "graph" {
			t.Fatalf("ListGraphGenerations() = %#v, %v", values, err)
		}
	})
	t.Run("label generations", func(t *testing.T) {
		rows := &stubRows{rows: []func(...any) error{
			func(dest ...any) error { return scanTestLabelGeneration(dest, now, "v") },
		}}
		values, err := (&Store{database: &targetedReadDatabase{rows: rows}}).ListLabelGenerations(ctx, 1, 2)
		if err != nil || len(values) != 1 || values[0].Kind != VertexLabel {
			t.Fatalf("ListLabelGenerations() = %#v, %v", values, err)
		}
	})
	t.Run("batches", func(t *testing.T) {
		rows := &stubRows{rows: []func(...any) error{
			func(dest ...any) error { return scanTestBatch(dest, now) },
		}}
		values, err := (&Store{database: &targetedReadDatabase{rows: rows}}).ListBatches(ctx, testJobID, 2)
		if err != nil || len(values) != 1 || values[0].Status != BatchRunning {
			t.Fatalf("ListBatches() = %#v, %v", values, err)
		}
	})
	t.Run("retained backups", func(t *testing.T) {
		rows := &stubRows{rows: []func(...any) error{
			func(dest ...any) error {
				*dest[0].(*string) = testJobID
				*dest[1].(*string) = "graph"
				*dest[2].(*string) = "backup"
				*dest[3].(*int64) = 4
				*dest[4].(*uint32) = 5
				return nil
			},
		}}
		values, err := (&Store{database: &targetedReadDatabase{rows: rows}}).ListRetainedBackups(ctx, 2)
		if err != nil || len(values) != 1 || values[0].BackupGraphOID != 5 {
			t.Fatalf("ListRetainedBackups() = %#v, %v", values, err)
		}
	})

	injected := errors.New("scan failed")
	tests := []struct {
		name string
		run  func(*Store) error
	}{
		{"active jobs", func(store *Store) error { _, err := store.ListActiveJobHealth(ctx, 1); return err }},
		{"current graphs", func(store *Store) error { _, err := store.ListCurrentGraphGenerations(ctx, "graph", 1); return err }},
		{"jobs", func(store *Store) error { _, err := store.ListJobs(ctx, 1); return err }},
		{"graphs", func(store *Store) error { _, err := store.ListGraphGenerations(ctx, 1); return err }},
		{"labels", func(store *Store) error { _, err := store.ListLabelGenerations(ctx, 1, 1); return err }},
		{"label page", func(store *Store) error { _, err := store.ListLabelGenerationPage(ctx, 1, 1); return err }},
		{"batches", func(store *Store) error { _, err := store.ListBatches(ctx, testJobID, 1); return err }},
		{"rejects", func(store *Store) error { _, err := store.ListRejectSummaries(ctx, testJobID, 1); return err }},
		{"backups", func(store *Store) error { _, err := store.ListRetainedBackups(ctx, 1); return err }},
	}
	for _, test := range tests {
		t.Run(test.name+" scan error", func(t *testing.T) {
			rows := &stubRows{rows: []func(...any) error{func(...any) error { return injected }}}
			if err := test.run(&Store{database: &targetedReadDatabase{rows: rows}}); !errors.Is(err, injected) {
				t.Fatalf("scan error = %v", err)
			}
		})
		t.Run(test.name+" rows error", func(t *testing.T) {
			rows := &stubRows{err: injected}
			if err := test.run(&Store{database: &targetedReadDatabase{rows: rows}}); !errors.Is(err, injected) {
				t.Fatalf("rows error = %v", err)
			}
		})
		t.Run(test.name+" query error", func(t *testing.T) {
			if err := test.run(&Store{database: &targetedReadDatabase{queryErr: injected}}); !errors.Is(err, injected) {
				t.Fatalf("query error = %v", err)
			}
		})
	}
}

func TestLabelGenerationSelectionsAndPages(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	now := time.Now().UTC()
	store := &Store{database: panicDatabase{}}
	for _, ids := range [][]int64{{0}, {-1}, {2, 2}} {
		if _, err := store.ListLabelGenerationsByID(ctx, 1, ids); err == nil {
			t.Errorf("ListLabelGenerationsByID(%v) succeeded", ids)
		}
	}
	if values, err := store.ListLabelGenerationsByID(ctx, 1, nil); err != nil || len(values) != 0 {
		t.Fatalf("empty ListLabelGenerationsByID() = %#v, %v", values, err)
	}
	if _, err := store.ListLabelGenerationsByID(ctx, 0, []int64{1}); err == nil {
		t.Fatal("ListLabelGenerationsByID accepted zero graph generation")
	}
	if _, err := store.ListLabelGenerationsByID(ctx, 1, make([]int64, MaxReadLimit+1)); err == nil {
		t.Fatal("ListLabelGenerationsByID accepted too many IDs")
	}

	rows := &stubRows{rows: []func(...any) error{
		func(dest ...any) error { return scanTestLabelGeneration(dest, now, "e") },
	}}
	values, err := (&Store{database: &targetedReadDatabase{rows: rows}}).
		ListLabelGenerationsByID(ctx, 1, []int64{9})
	if err != nil || len(values) != 1 || values[0].Kind != EdgeLabel {
		t.Fatalf("ListLabelGenerationsByID() = %#v, %v", values, err)
	}

	pageRows := &stubRows{rows: []func(...any) error{
		func(dest ...any) error { return scanTestLabelGeneration(dest, now, "v") },
		func(dest ...any) error {
			if err := scanTestLabelGeneration(dest, now, "e"); err != nil {
				return err
			}
			*dest[0].(*int64) = 10
			return nil
		},
	}}
	page, err := (&Store{database: &targetedReadDatabase{rows: pageRows}}).
		ListLabelGenerationPage(ctx, 1, 1)
	if err != nil || page.Complete || len(page.Generations) != 1 {
		t.Fatalf("ListLabelGenerationPage() = %#v, %v", page, err)
	}

	for _, run := range []func(*Store) error{
		func(store *Store) error { _, err := store.ListLabelGenerations(ctx, 1, 1); return err },
		func(store *Store) error { _, err := store.ListLabelGenerationsByID(ctx, 1, []int64{9}); return err },
		func(store *Store) error { _, err := store.ListLabelGenerationPage(ctx, 1, 1); return err },
	} {
		rows := &stubRows{rows: []func(...any) error{
			func(dest ...any) error { return scanTestLabelGeneration(dest, now, "bad") },
		}}
		if err := run(&Store{database: &targetedReadDatabase{rows: rows}}); err == nil || !strings.Contains(err.Error(), "kind") {
			t.Errorf("invalid stored kind error = %v", err)
		}
	}
}

func TestRejectSummaryPaginationSortingAndQueryValidation(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	rows := &stubRows{rows: []func(...any) error{
		scanString("zeta"), scanString("alpha"), scanString("zeta"), scanString("ignored"),
	}}
	page, err := (&Store{database: &targetedReadDatabase{rows: rows}}).
		ListRejectSummaries(ctx, testJobID, 3)
	if err != nil || !page.Truncated || page.ScannedRows != 3 ||
		len(page.Summaries) != 2 || page.Summaries[0] != (RejectSummary{ErrorClass: "zeta", Count: 2}) {
		t.Fatalf("ListRejectSummaries() = %#v, %v", page, err)
	}

	if _, err := (*Store)(nil).queryBounded(ctx, "SELECT 1", 1); err == nil {
		t.Fatal("nil store queryBounded succeeded")
	}
	if _, err := (&Store{}).queryBounded(ctx, "SELECT 1", 1); err == nil {
		t.Fatal("nil database queryBounded succeeded")
	}
	if _, err := (&Store{database: panicDatabase{}}).queryBounded(ctx, "SELECT 1", 1); err == nil ||
		!strings.Contains(err.Error(), "does not support") {
		t.Fatalf("unsupported query database error = %v", err)
	}
}

func scanTestJob(dest []any, now time.Time) error {
	*dest[0].(*string) = testJobID
	*dest[1].(*string) = "job"
	*dest[2].(*string) = "csv"
	*dest[3].(*string) = "create"
	*dest[4].(*string) = "graph"
	*dest[5].(*string) = ""
	*dest[6].(*string) = strings.Repeat("a", 64)
	*dest[7].(*JobStatus) = JobRunning
	*dest[8].(*int64) = 1
	*dest[9].(*uint64) = 1
	*dest[10].(*string) = "token"
	*dest[11].(*int64) = 2
	*dest[12].(*int64) = 3
	*dest[13].(*int64) = 4
	*dest[14].(*int64) = 5
	*dest[15].(*string) = ""
	*dest[16].(*time.Time) = now
	*dest[18].(*time.Time) = now
	return nil
}

func scanTestGraphGeneration(dest []any, now time.Time) error {
	*dest[0].(*int64) = 1
	*dest[1].(*string) = testJobID
	*dest[2].(*string) = "graph"
	*dest[3].(*uint32) = 2
	*dest[4].(*uint32) = 2
	*dest[5].(*uint32) = 0
	*dest[6].(*uint64) = 1
	*dest[7].(*GenerationState) = GenerationActive
	*dest[8].(*time.Time) = now
	*dest[9].(*time.Time) = now
	return nil
}

func scanTestLabelGeneration(dest []any, now time.Time, kind string) error {
	*dest[0].(*int64) = 9
	*dest[1].(*int64) = 1
	*dest[2].(*string) = "Person"
	*dest[3].(*string) = kind
	*dest[4].(*uint32) = 2
	*dest[5].(*uint16) = 3
	*dest[6].(*uint32) = 4
	*dest[7].(*uint32) = 5
	*dest[8].(*uint64) = 1
	*dest[9].(*time.Time) = now
	*dest[10].(*time.Time) = now
	return nil
}

func scanTestBatch(dest []any, now time.Time) error {
	*dest[0].(*string) = testJobID
	*dest[1].(*uint64) = 1
	*dest[2].(*uint32) = 1
	*dest[3].(*BatchStatus) = BatchRunning
	*dest[4].(*int64) = 2
	*dest[5].(*int64) = 3
	*dest[6].(*int64) = 0
	*dest[7].(*string) = "file"
	*dest[8].(*int64) = 1
	*dest[9].(*int64) = 0
	*dest[10].(*string) = "first"
	*dest[11].(*string) = ""
	*dest[12].(*int64) = 0
	*dest[13].(*int64) = 0
	*dest[14].(*string) = ""
	*dest[15].(*string) = ""
	*dest[16].(*time.Time) = now
	return nil
}

func scanString(value string) func(...any) error {
	return func(dest ...any) error {
		*dest[0].(*string) = value
		return nil
	}
}
