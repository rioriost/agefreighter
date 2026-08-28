package meta

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

const (
	MaxDiagnosticHistory = 100
	MaxDiagnosticBytes   = 4 << 20
)

type DiagnosticRecord struct {
	ID                      int64           `json:"id"`
	RecordedAt              time.Time       `json:"recordedAt"`
	Outcome                 string          `json:"outcome"`
	TargetGraph             string          `json:"targetGraph"`
	PostgreSQLVersionNumber int             `json:"postgresqlVersionNumber"`
	AGEVersion              string          `json:"ageVersion"`
	MetadataSchemaVersion   int             `json:"metadataSchemaVersion"`
	Report                  json.RawMessage `json:"report"`
}

func (store *Store) PersistDiagnostic(
	ctx context.Context,
	record DiagnosticRecord,
) (DiagnosticRecord, error) {
	if store == nil || store.database == nil {
		return DiagnosticRecord{}, errors.New("metadata store is required")
	}
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		return DiagnosticRecord{}, errors.New("diagnostic write context requires a deadline")
	}
	rollbackTimeout := remainingTimeout(ctx, defaultMigrationTimeout)
	if record.TargetGraph == "" || len(record.TargetGraph) > 63 {
		return DiagnosticRecord{}, errors.New("diagnostic target graph is invalid")
	}
	if record.PostgreSQLVersionNumber < 0 ||
		record.MetadataSchemaVersion < 0 ||
		len(record.AGEVersion) > 64 {
		return DiagnosticRecord{}, errors.New("diagnostic typed fields are invalid")
	}
	switch record.Outcome {
	case "pass", "fail", "incomplete":
	default:
		return DiagnosticRecord{}, fmt.Errorf(
			"diagnostic outcome %q is invalid",
			record.Outcome,
		)
	}
	if len(record.Report) == 0 || len(record.Report) > MaxDiagnosticBytes ||
		!json.Valid(record.Report) {
		return DiagnosticRecord{}, errors.New("diagnostic report is invalid or too large")
	}
	tx, err := store.database.Begin(ctx)
	if err != nil {
		return DiagnosticRecord{}, fmt.Errorf("begin diagnostic persistence: %w", err)
	}
	defer rollbackWithTimeout(ctx, tx, rollbackTimeout)
	if _, err := tx.Exec(
		ctx,
		`SELECT pg_catalog.pg_advisory_xact_lock($1)`,
		migrationLockID,
	); err != nil {
		return DiagnosticRecord{}, fmt.Errorf("lock diagnostic persistence: %w", err)
	}
	inspection, err := (&Store{database: tx}).InspectSchema(ctx)
	if err != nil {
		return DiagnosticRecord{}, fmt.Errorf(
			"inspect locked diagnostic metadata schema: %w",
			err,
		)
	}
	if err := inspection.RequireCurrent(); err != nil ||
		inspection.InstalledVersion != SupportedSchemaVersion ||
		record.MetadataSchemaVersion != SupportedSchemaVersion {
		if err == nil {
			err = fmt.Errorf(
				"diagnostic metadata version is %d; current supported version is %d",
				record.MetadataSchemaVersion,
				SupportedSchemaVersion,
			)
		}
		return DiagnosticRecord{}, fmt.Errorf(
			"persist diagnostic against current metadata schema: %w",
			err,
		)
	}
	err = tx.QueryRow(
		ctx,
		`INSERT INTO agefreighter_meta.diagnostic_history (
			outcome, target_graph, postgresql_version_number,
			age_version, metadata_schema_version, report
		) VALUES ($1, $2, $3, $4, $5, $6::jsonb)
		RETURNING diagnostic_id, recorded_at`,
		record.Outcome,
		record.TargetGraph,
		record.PostgreSQLVersionNumber,
		record.AGEVersion,
		record.MetadataSchemaVersion,
		string(record.Report),
	).Scan(&record.ID, &record.RecordedAt)
	if err != nil {
		return DiagnosticRecord{}, fmt.Errorf("persist diagnostic report: %w", err)
	}
	if _, err := tx.Exec(ctx, `
		DELETE FROM agefreighter_meta.diagnostic_history
		WHERE diagnostic_id NOT IN (
			SELECT diagnostic_id
			FROM agefreighter_meta.diagnostic_history
			ORDER BY recorded_at DESC, diagnostic_id DESC
			LIMIT $1
		)`,
		MaxDiagnosticHistory,
	); err != nil {
		return DiagnosticRecord{}, fmt.Errorf("trim diagnostic history: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return DiagnosticRecord{}, fmt.Errorf("commit diagnostic persistence: %w", err)
	}
	return record, nil
}

func (store *Store) ListDiagnostics(
	ctx context.Context,
	targetGraph string,
	limit int,
) ([]DiagnosticRecord, error) {
	if targetGraph == "" || len(targetGraph) > 63 {
		return nil, errors.New("diagnostic target graph is invalid")
	}
	rows, err := store.queryBounded(
		ctx,
		`SELECT
			diagnostic_id, recorded_at, outcome, target_graph,
			postgresql_version_number, age_version,
			metadata_schema_version
		 FROM agefreighter_meta.diagnostic_history
		 WHERE target_graph = $1
		 ORDER BY recorded_at DESC, diagnostic_id DESC
		 LIMIT $2`,
		limit,
		targetGraph,
	)
	if err != nil {
		return nil, fmt.Errorf("list diagnostic history: %w", err)
	}
	defer rows.Close()
	records := make([]DiagnosticRecord, 0, limit)
	for rows.Next() {
		var record DiagnosticRecord
		if err := rows.Scan(
			&record.ID,
			&record.RecordedAt,
			&record.Outcome,
			&record.TargetGraph,
			&record.PostgreSQLVersionNumber,
			&record.AGEVersion,
			&record.MetadataSchemaVersion,
		); err != nil {
			return nil, fmt.Errorf("read diagnostic history: %w", err)
		}
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list diagnostic history: %w", err)
	}
	return records, nil
}
