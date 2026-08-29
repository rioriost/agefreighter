package age

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestDegradedProbeIntegration(t *testing.T) {
	dsn := os.Getenv(integrationDSNEnvironment)
	if dsn == "" {
		t.Skip("set " + integrationDSNEnvironment + " to run Apache AGE integration tests")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	result, err := ProbeDegraded(ctx, dsn, ProbeOptions{
		ConnectTimeout: 5 * time.Second, OperationTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("ProbeDegraded() error = %v", err)
	}
	if result.PostgreSQLStatus != ProbePass ||
		result.AGEPresenceStatus != ProbePass ||
		result.AGEVersionStatus != ProbePass ||
		result.AGELoadabilityStatus != ProbePass {
		t.Fatalf("ProbeDegraded() = %#v", result)
	}
}

func TestDegradedProbeClassifiesSupportedTarget(t *testing.T) {
	database := &probeScript{
		rows: []pgx.Row{
			stubRow(func(dest ...any) error {
				*dest[0].(*string) = "170009"
				*dest[1].(*string) = "17.9"
				return nil
			}),
			stubRow(func(dest ...any) error {
				*dest[0].(*string) = "1.6.0"
				return nil
			}),
			stubRow(func(dest ...any) error {
				*dest[0].(*string) = "age, pg_stat_statements"
				return nil
			}),
		},
	}
	result, err := probeDegradedCapabilities(t.Context(), database)
	if err != nil {
		t.Fatalf("probeDegradedCapabilities() error = %v", err)
	}
	if result.PostgreSQLStatus != ProbePass ||
		result.AGEPresenceStatus != ProbePass ||
		result.AGEVersionStatus != ProbePass ||
		result.AGELoadabilityStatus != ProbePass ||
		result.AGEPreloadStatus != PreloadConfigured {
		t.Fatalf("probe = %#v", result)
	}
}

func TestDegradedProbeReportsUnsupportedVersions(t *testing.T) {
	database := &probeScript{
		rows: []pgx.Row{
			stubRow(func(dest ...any) error {
				*dest[0].(*string) = "160012"
				*dest[1].(*string) = "16.12"
				return nil
			}),
			stubRow(func(dest ...any) error {
				*dest[0].(*string) = "1.5.0"
				return nil
			}),
			stubRow(func(dest ...any) error {
				*dest[0].(*string) = ""
				return nil
			}),
		},
	}
	result, err := probeDegradedCapabilities(t.Context(), database)
	if err != nil {
		t.Fatalf("probeDegradedCapabilities() error = %v", err)
	}
	if result.PostgreSQLStatus != ProbePass ||
		result.AGEVersionStatus != ProbeFail ||
		result.AGELoadabilityStatus != ProbePass {
		t.Fatalf("probe = %#v", result)
	}
}

func TestDegradedProbeClassifiesMissingAndPermissionStates(t *testing.T) {
	t.Run("missing extension", func(t *testing.T) {
		database := &probeScript{rows: []pgx.Row{
			versionProbeRow(),
			stubRow(func(...any) error { return pgx.ErrNoRows }),
		}}
		result, err := probeDegradedCapabilities(t.Context(), database)
		if err != nil {
			t.Fatalf("probeDegradedCapabilities() error = %v", err)
		}
		if result.AGEPresenceStatus != ProbeUnavailable ||
			result.AGEVersionStatus != ProbeUnavailable ||
			result.AGELoadabilityStatus != ProbeUnavailable ||
			database.execCalls != 0 {
			t.Fatalf("probe = %#v, execCalls = %d", result, database.execCalls)
		}
	})

	t.Run("extension catalog permission", func(t *testing.T) {
		database := &probeScript{rows: []pgx.Row{
			versionProbeRow(),
			stubRow(func(...any) error {
				return &pgconn.PgError{Code: "42501"}
			}),
		}}
		result, err := probeDegradedCapabilities(t.Context(), database)
		if err != nil {
			t.Fatalf("probeDegradedCapabilities() error = %v", err)
		}
		if result.AGEPresenceStatus != ProbeUnknown ||
			result.AGEVersionStatus != ProbeUnknown ||
			result.AGELoadabilityStatus != ProbeUnknown {
			t.Fatalf("probe = %#v", result)
		}
	})

	t.Run("load permission", func(t *testing.T) {
		database := &probeScript{
			rows: []pgx.Row{
				versionProbeRow(),
				stubRow(func(dest ...any) error {
					*dest[0].(*string) = "1.6.1"
					return nil
				}),
				stubRow(func(dest ...any) error {
					*dest[0].(*string) = ""
					return nil
				}),
			},
			execErr: &pgconn.PgError{Code: "42501"},
		}
		result, err := probeDegradedCapabilities(t.Context(), database)
		if err != nil {
			t.Fatalf("probeDegradedCapabilities() error = %v", err)
		}
		if result.AGELoadabilityStatus != ProbeUnknown {
			t.Fatalf("probe = %#v", result)
		}
	})
}

func TestDegradedProbeDoesNotHideConnectionErrors(t *testing.T) {
	database := &probeScript{
		rows: []pgx.Row{
			versionProbeRow(),
			stubRow(func(dest ...any) error {
				*dest[0].(*string) = "1.6.0"
				return nil
			}),
			stubRow(func(dest ...any) error {
				*dest[0].(*string) = ""
				return nil
			}),
		},
		execErr: &pgconn.PgError{Code: "08006"},
	}
	if _, err := probeDegradedCapabilities(t.Context(), database); err == nil {
		t.Fatal("probeDegradedCapabilities() hid connection failure")
	}

	database = &probeScript{rows: []pgx.Row{
		stubRow(func(...any) error { return errors.New("injected") }),
	}}
	if _, err := probeDegradedCapabilities(context.Background(), database); err == nil {
		t.Fatal("probeDegradedCapabilities() hid PostgreSQL query failure")
	}
}

func versionProbeRow() pgx.Row {
	return stubRow(func(dest ...any) error {
		*dest[0].(*string) = "170009"
		*dest[1].(*string) = "17.9"
		return nil
	})
}

type probeScript struct {
	rows      []pgx.Row
	execErr   error
	execCalls int
}

func (database *probeScript) QueryRow(
	context.Context,
	string,
	...any,
) pgx.Row {
	row := database.rows[0]
	database.rows = database.rows[1:]
	return row
}

func (database *probeScript) Exec(
	context.Context,
	string,
	...any,
) (pgconn.CommandTag, error) {
	database.execCalls++
	return pgconn.CommandTag{}, database.execErr
}
