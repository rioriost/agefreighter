package meta

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

func (store *Store) PutReject(
	ctx context.Context,
	record RejectRecord,
) (bool, error) {
	if err := validateReject(record); err != nil {
		return false, err
	}
	var payload any
	if len(record.Record) > 0 {
		payload = string(record.Record)
	}
	tag, err := store.database.Exec(
		ctx,
		`INSERT INTO agefreighter_meta.reject_record (
			job_id, batch_id, attempt, resume_token, resource, line,
			byte_offset, error_class, error_message, record
		) VALUES (
			$1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb
		)
		ON CONFLICT (job_id, batch_id, attempt, resume_token) DO NOTHING`,
		record.JobID,
		record.BatchID,
		record.Attempt,
		record.Position.Token,
		record.Position.Resource,
		record.Position.Line,
		record.Position.ByteOffset,
		record.ErrorClass,
		record.ErrorMessage,
		payload,
	)
	if err != nil {
		return false, fmt.Errorf("write rejected record: %w", err)
	}
	if tag.RowsAffected() == 1 {
		return true, nil
	}
	var identical bool
	err = store.database.QueryRow(
		ctx,
		`SELECT
			resource = $5
			AND line = $6
			AND byte_offset = $7
			AND error_class = $8
			AND error_message = $9
			AND record IS NOT DISTINCT FROM $10::jsonb
		 FROM agefreighter_meta.reject_record
		 WHERE job_id = $1::uuid
		   AND batch_id = $2
		   AND attempt = $3
		   AND resume_token = $4`,
		record.JobID,
		record.BatchID,
		record.Attempt,
		record.Position.Token,
		record.Position.Resource,
		record.Position.Line,
		record.Position.ByteOffset,
		record.ErrorClass,
		record.ErrorMessage,
		payload,
	).Scan(&identical)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, fmt.Errorf("%w: rejected record disappeared during replay", ErrConflict)
	}
	if err != nil {
		return false, fmt.Errorf("compare rejected record replay: %w", err)
	}
	if !identical {
		return false, fmt.Errorf(
			"%w: rejected record replay differs for token %q",
			ErrConflict,
			record.Position.Token,
		)
	}
	return false, nil
}
