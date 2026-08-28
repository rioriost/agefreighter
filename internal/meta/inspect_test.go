package meta

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestInspectSchemaStates(t *testing.T) {
	tests := []struct {
		name string
		rows []pgx.Row
		want SchemaState
	}{
		{
			name: "absent",
			rows: []pgx.Row{inspectCatalogRow(false, false)},
			want: SchemaAbsent,
		},
		{
			name: "missing migration table",
			rows: []pgx.Row{inspectCatalogRow(true, false)},
			want: SchemaInvalid,
		},
		{
			name: "pending",
			rows: []pgx.Row{
				inspectCatalogRow(true, true),
				inspectVersionRow(1, SupportedSchemaVersion-1, SupportedSchemaVersion-1),
			},
			want: SchemaPending,
		},
		{
			name: "empty migration history",
			rows: []pgx.Row{
				inspectCatalogRow(true, true),
				inspectVersionRow(0, 0, 0),
			},
			want: SchemaPending,
		},
		{
			name: "current",
			rows: []pgx.Row{
				inspectCatalogRow(true, true),
				inspectVersionRow(1, SupportedSchemaVersion, SupportedSchemaVersion),
			},
			want: SchemaCurrent,
		},
		{
			name: "newer",
			rows: []pgx.Row{
				inspectCatalogRow(true, true),
				inspectVersionRow(1, SupportedSchemaVersion+1, SupportedSchemaVersion+1),
			},
			want: SchemaNewer,
		},
		{
			name: "gap",
			rows: []pgx.Row{
				inspectCatalogRow(true, true),
				inspectVersionRow(1, SupportedSchemaVersion, SupportedSchemaVersion-1),
			},
			want: SchemaInvalid,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store := &Store{database: &inspectDatabase{rows: test.rows}}
			got, err := store.InspectSchema(t.Context())
			if err != nil {
				t.Fatalf("InspectSchema() error = %v", err)
			}
			if got.State != test.want ||
				got.SupportedVersion != SupportedSchemaVersion {
				t.Fatalf("InspectSchema() = %#v", got)
			}
			if got.State == SchemaPending &&
				got.PendingVersions != SupportedSchemaVersion-got.InstalledVersion {
				t.Fatalf("pending inspection = %#v", got)
			}
		})
	}
}

func TestInspectSchemaClassifiesPermissionAndErrors(t *testing.T) {
	store := &Store{database: &inspectDatabase{rows: []pgx.Row{
		errorRow{err: &pgconn.PgError{Code: "42501"}},
	}}}
	inspection, err := store.InspectSchema(t.Context())
	if err != nil || inspection.State != SchemaUnknown || inspection.Detail == "" {
		t.Fatalf("InspectSchema() = %#v, %v", inspection, err)
	}

	injected := errors.New("injected")
	store = &Store{database: &inspectDatabase{rows: []pgx.Row{
		errorRow{err: injected},
	}}}
	if _, err := store.InspectSchema(t.Context()); !errors.Is(err, injected) {
		t.Fatalf("InspectSchema() error = %v", err)
	}

	var nilStore *Store
	if _, err := nilStore.InspectSchema(t.Context()); err == nil {
		t.Fatal("nil Store.InspectSchema() succeeded")
	}
}

func TestSchemaInspectionRequireCurrent(t *testing.T) {
	current := SchemaInspection{
		State: SchemaCurrent, InstalledVersion: SupportedSchemaVersion,
		SupportedVersion: SupportedSchemaVersion,
	}
	if err := current.RequireCurrent(); err != nil {
		t.Fatalf("RequireCurrent() error = %v", err)
	}
	current.State = SchemaPending
	if err := current.RequireCurrent(); err == nil {
		t.Fatal("pending RequireCurrent() succeeded")
	}
}

func TestSchemaInspectionReadCompatibility(t *testing.T) {
	for _, version := range []int{
		MinimumReadCompatibleSchemaVersion,
		SupportedSchemaVersion,
	} {
		state := SchemaCurrent
		if version < SupportedSchemaVersion {
			state = SchemaPending
		}

		inspection := SchemaInspection{
			State: state, InstalledVersion: version,
			SupportedVersion: SupportedSchemaVersion,
		}
		if err := inspection.RequireReadCompatible(); err != nil {
			t.Fatalf("schema v%d rejected: %v", version, err)
		}
	}
	for _, inspection := range []SchemaInspection{
		{
			State: SchemaPending, InstalledVersion: MinimumReadCompatibleSchemaVersion - 1,
			SupportedVersion: SupportedSchemaVersion,
		},
		{
			State: SchemaNewer, InstalledVersion: SupportedSchemaVersion + 1,
			SupportedVersion: SupportedSchemaVersion,
		},
		{
			State: SchemaInvalid, InstalledVersion: SupportedSchemaVersion,
			SupportedVersion: SupportedSchemaVersion,
		},
	} {
		if err := inspection.RequireReadCompatible(); err == nil {
			t.Fatalf("incompatible schema accepted: %#v", inspection)
		}
	}
}

func TestLockCurrentSchemaRequiresDeadlineAndExactCurrentVersion(t *testing.T) {
	database := &inspectDatabase{}
	store := &Store{database: database}
	if _, err := store.LockCurrentSchema(context.Background()); err == nil {
		t.Fatal("LockCurrentSchema() accepted a context without a deadline")
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	database.rows = []pgx.Row{
		inspectCatalogRow(true, true),
		inspectVersionRow(1, SupportedSchemaVersion, SupportedSchemaVersion),
	}
	inspection, err := store.LockCurrentSchema(ctx)
	if err != nil || inspection.State != SchemaCurrent || database.execCalls != 1 {
		t.Fatalf(
			"LockCurrentSchema() = %#v, %v, execCalls=%d",
			inspection,
			err,
			database.execCalls,
		)
	}
}

func inspectCatalogRow(schema, table bool) pgx.Row {
	return stubInspectRow(func(dest ...any) error {
		*dest[0].(*bool) = schema
		*dest[1].(*bool) = table
		return nil
	})
}

func inspectVersionRow(minimum, maximum, count int) pgx.Row {
	return stubInspectRow(func(dest ...any) error {
		*dest[0].(*int) = minimum
		*dest[1].(*int) = maximum
		*dest[2].(*int) = count
		return nil
	})
}

type inspectDatabase struct {
	rows      []pgx.Row
	execCalls int
	execErr   error
}

func (*inspectDatabase) Begin(context.Context) (pgx.Tx, error) {
	panic("unexpected Begin")
}

func (database *inspectDatabase) Exec(
	context.Context,
	string,
	...any,
) (pgconn.CommandTag, error) {
	database.execCalls++
	return pgconn.CommandTag{}, database.execErr
}

func (database *inspectDatabase) QueryRow(
	context.Context,
	string,
	...any,
) pgx.Row {
	row := database.rows[0]
	database.rows = database.rows[1:]
	return row
}

type stubInspectRow func(...any) error

func (row stubInspectRow) Scan(dest ...any) error {
	return row(dest...)
}
