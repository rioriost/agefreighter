package postgres

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
)

type catalogTestRows struct {
	pgx.Rows
	values [][]any
	index  int
	err    error
}

func (r *catalogTestRows) Next() bool { r.index++; return r.index <= len(r.values) }
func (r *catalogTestRows) Close()     {}
func (r *catalogTestRows) Err() error { return r.err }
func (r *catalogTestRows) Scan(dest ...any) error {
	if len(dest) != len(r.values[r.index-1]) {
		return errors.New("bad fixture")
	}
	for i, d := range dest {
		reflect.ValueOf(d).Elem().Set(reflect.ValueOf(r.values[r.index-1][i]))
	}
	return nil
}

type catalogTestQuery struct {
	tables, columns, constraints [][]any
	calls                        int
	err                          error
}

func (q *catalogTestQuery) Query(_ context.Context, sql string, args ...any) (pgx.Rows, error) {
	q.calls++
	if q.err != nil {
		return nil, q.err
	}
	var rows [][]any
	switch sql {
	case catalogTablesSQL:
		if args[1] != CatalogMaxTables+1 {
			panic("unbounded tables")
		}
		rows = q.tables
	case catalogColumnsSQL:
		if args[1] != CatalogMaxColumns+1 {
			panic("unbounded columns")
		}
		rows = q.columns
	case catalogConstraintsSQL:
		if args[1] != CatalogMaxConstraints+1 {
			panic("unbounded constraints")
		}
		rows = q.constraints
	default:
		if !strings.Contains(sql, "has_schema_privilege") {
			panic("unexpected query")
		}
		rows = [][]any{{"public"}}
	}
	return &catalogTestRows{values: rows}, nil
}
func catalogFixture() *catalogTestQuery {
	return &catalogTestQuery{
		tables:      [][]any{{uint32(100), "public", "person", "r", true, false, false}},
		columns:     [][]any{{"id", "pg_catalog", "int8", true, true}},
		constraints: [][]any{{"person_pkey", "p", []string{"id"}, "", "", []string{}, true, true, false, false, true}},
	}
}
func TestCatalogBoundsAndSafeErrors(t *testing.T) {
	for _, schemas := range [][]string{nil, {"public", "public"}, {"pg_catalog"}, {"information_schema"}, {"public; DELETE"}, {strings.Repeat("x", 64)}} {
		if _, err := ReadCatalog(t.Context(), "", schemas); err == nil {
			t.Fatal("accepted invalid schema scope")
		}
	}
	if _, err := ReadCatalog(nil, "dsn", []string{"public"}); err == nil {
		t.Fatal("accepted nil context")
	}
	if _, err := ReadCatalog(t.Context(), "", []string{"public"}); err == nil {
		t.Fatal("accepted empty connection")
	}
	for _, kind := range []string{"tables", "columns", "constraints"} {
		q := catalogFixture()
		switch kind {
		case "tables":
			for len(q.tables) <= CatalogMaxTables {
				q.tables = append(q.tables, q.tables[0])
			}
		case "columns":
			for len(q.columns) <= CatalogMaxColumns {
				q.columns = append(q.columns, q.columns[0])
			}
		case "constraints":
			for len(q.constraints) <= CatalogMaxConstraints {
				q.constraints = append(q.constraints, q.constraints[0])
			}
		}
		doc, err := readCatalog(t.Context(), q, []string{"public"})
		if err == nil || doc.Complete {
			t.Fatalf("accepted oversized %s", kind)
		}
	}
	q := catalogFixture()
	q.err = errors.New("password=private dsn")
	_, err := readCatalog(t.Context(), q, []string{"public"})
	if err == nil || strings.Contains(err.Error(), "private") {
		t.Fatal("unredacted error")
	}
	if _, err := readCatalog(t.Context(), catalogFixture(), []string{"missing"}); err == nil {
		t.Fatal("missing schema accepted")
	}
}
func TestCatalogMetadataOnlyContract(t *testing.T) {
	q := catalogFixture()
	doc, err := readCatalog(t.Context(), q, []string{"public"})
	if err != nil {
		t.Fatal(err)
	}
	if !doc.Complete || doc.Command != "postgres-catalog" || len(doc.Tables) != 1 || doc.Tables[0].Constraints[0].Columns[0] != "id" || q.calls != 4 {
		t.Fatalf("invalid report: %+v", doc)
	}
	q = &catalogTestQuery{}
	doc, err = readCatalog(t.Context(), q, []string{"public"})
	if err != nil || doc.Tables == nil || len(doc.Tables) != 0 || q.calls != 2 {
		t.Fatal("empty catalog contract", err)
	}
	for _, sql := range []string{catalogTablesSQL, catalogColumnsSQL, catalogConstraintsSQL} {
		if !strings.HasPrefix(sql, "SELECT ") || !strings.Contains(sql, "LIMIT $2") {
			t.Fatal("unbounded or non-read query")
		}
	}
}
