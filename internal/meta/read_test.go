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
}

func (database *targetedReadDatabase) Query(
	_ context.Context,
	statement string,
	_ ...any,
) (pgx.Rows, error) {
	database.statement = statement
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
