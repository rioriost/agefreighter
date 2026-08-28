package meta

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestConnectorTelemetryValidation(t *testing.T) {
	valid := ConnectorTelemetry{
		JobID: testJobID, Connector: "cosmos-nosql",
		Pages: 2, RequestCharge: 3.5, FailedRequestAttempts: 1,
		ThrottledRequests: 1, ContinuationDigest: "abcdef12",
	}
	if err := validateConnectorTelemetry(valid); err != nil {
		t.Fatalf("validateConnectorTelemetry() error = %v", err)
	}
	tests := []ConnectorTelemetry{
		{},
		func() ConnectorTelemetry { value := valid; value.Connector = "bad"; return value }(),
		func() ConnectorTelemetry { value := valid; value.Pages = -1; return value }(),
		func() ConnectorTelemetry { value := valid; value.RequestCharge = math.NaN(); return value }(),
		func() ConnectorTelemetry {
			value := valid
			value.ContinuationDigest = strings.Repeat("x", MaxContinuationDigestBytes+1)
			return value
		}(),
		func() ConnectorTelemetry {
			value := valid
			value.ContinuationDigest = "raw\nsecret"
			return value
		}(),
	}
	for index, value := range tests {
		if err := validateConnectorTelemetry(value); err == nil {
			t.Fatalf("invalid telemetry %d accepted", index)
		}
	}
}

func TestConnectorTelemetryMigrationIsCurrentVersion(t *testing.T) {
	if schemaVersion < 15 || len(migrations) != schemaVersion {
		t.Fatalf("schema version=%d migrations=%d", schemaVersion, len(migrations))
	}
	if len(migrationV15) != 1 ||
		!strings.Contains(migrationV15[0], "connector_telemetry") {
		t.Fatalf("migrationV15 = %#v", migrationV15)
	}
}

func TestConnectorTelemetryReadWrite(t *testing.T) {
	now := time.Date(2026, 8, 28, 6, 0, 0, 0, time.UTC)
	value := ConnectorTelemetry{
		JobID: testJobID, Connector: "neo4j", Pages: 3,
		FailedRequestAttempts: 1,
	}
	database := &telemetryDatabase{
		execTags: []pgconn.CommandTag{pgconn.NewCommandTag("INSERT 0 1")},
		rows: []pgx.Row{stubInspectRow(func(dest ...any) error {
			*dest[0].(*string) = value.JobID
			*dest[1].(*string) = value.Connector
			*dest[2].(*int64) = value.Pages
			*dest[3].(*float64) = value.RequestCharge
			*dest[4].(*int64) = value.FailedRequestAttempts
			*dest[5].(*int64) = value.ThrottledRequests
			*dest[6].(*string) = value.ContinuationDigest
			*dest[7].(*time.Time) = now
			return nil
		})},
	}
	store := &Store{database: database}
	if err := store.PutConnectorTelemetry(t.Context(), value); err != nil {
		t.Fatalf("PutConnectorTelemetry() error = %v", err)
	}
	got, err := store.GetConnectorTelemetry(t.Context(), testJobID)
	if err != nil {
		t.Fatalf("GetConnectorTelemetry() error = %v", err)
	}
	if got.JobID != value.JobID || got.Connector != value.Connector ||
		got.Pages != value.Pages || !got.RecordedAt.Equal(now) {
		t.Fatalf("GetConnectorTelemetry() = %#v", got)
	}
}

func TestConnectorTelemetryReplayAndErrors(t *testing.T) {
	value := ConnectorTelemetry{JobID: testJobID, Connector: "csv"}
	database := &telemetryDatabase{
		execTags: []pgconn.CommandTag{pgconn.NewCommandTag("INSERT 0 0")},
		rows: []pgx.Row{stubInspectRow(func(dest ...any) error {
			*dest[0].(*bool) = true
			return nil
		})},
	}
	if err := (&Store{database: database}).PutConnectorTelemetry(
		t.Context(),
		value,
	); err != nil {
		t.Fatalf("idempotent replay error = %v", err)
	}
	database = &telemetryDatabase{
		execTags: []pgconn.CommandTag{pgconn.NewCommandTag("INSERT 0 0")},
		rows: []pgx.Row{stubInspectRow(func(dest ...any) error {
			*dest[0].(*bool) = false
			return nil
		})},
	}
	if err := (&Store{database: database}).PutConnectorTelemetry(
		t.Context(),
		value,
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflicting replay error = %v", err)
	}
	database = &telemetryDatabase{rows: []pgx.Row{errorRow{err: pgx.ErrNoRows}}}
	if _, err := (&Store{database: database}).GetConnectorTelemetry(
		t.Context(),
		testJobID,
	); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing telemetry error = %v", err)
	}
}

type telemetryDatabase struct {
	execTags []pgconn.CommandTag
	execErr  error
	rows     []pgx.Row
}

func (*telemetryDatabase) Begin(context.Context) (pgx.Tx, error) {
	panic("unexpected Begin")
}

func (database *telemetryDatabase) Exec(
	context.Context,
	string,
	...any,
) (pgconn.CommandTag, error) {
	if database.execErr != nil {
		return pgconn.CommandTag{}, database.execErr
	}
	tag := database.execTags[0]
	database.execTags = database.execTags[1:]
	return tag, nil
}

func (database *telemetryDatabase) QueryRow(
	context.Context,
	string,
	...any,
) pgx.Row {
	row := database.rows[0]
	database.rows = database.rows[1:]
	return row
}
