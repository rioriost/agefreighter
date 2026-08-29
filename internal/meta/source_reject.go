package meta

import (
	"context"
	"errors"
	"fmt"
)

func (store *Store) SetSourceRejections(
	ctx context.Context,
	jobID string,
	count int64,
	position Position,
) error {
	if err := validateJobID(jobID); err != nil {
		return err
	}
	if count < 0 {
		return errors.New("source rejection count cannot be negative")
	}
	if count > 0 && position.Token == "" {
		return errors.New("source rejection checkpoint token is required")
	}
	tag, err := store.database.Exec(
		ctx,
		`WITH updated_job AS (
			UPDATE agefreighter_meta.load_job
			SET rejected_rows = rejected_rows + ($2 - source_rejected_rows),
			    source_rejected_rows = $2,
			    resume_token = CASE
			      WHEN $2 > source_rejected_rows THEN $3
			      ELSE resume_token
			    END,
			    updated_at = clock_timestamp()
			WHERE job_id = $1::uuid
			  AND status = 'running'
			  AND source_rejected_rows <= $2
			  AND NOT EXISTS (
			    SELECT 1
			    FROM agefreighter_meta.job_unclassified_counter counter
			    WHERE counter.job_id = load_job.job_id
			      AND counter.rejected_rows > $2
			  )
			RETURNING job_id
		)
		INSERT INTO agefreighter_meta.job_unclassified_counter (
			job_id, rejected_rows
		)
		SELECT job_id, $2
		FROM updated_job
		ON CONFLICT (job_id) DO UPDATE SET
			rejected_rows = EXCLUDED.rejected_rows,
			updated_at = clock_timestamp()
		WHERE agefreighter_meta.job_unclassified_counter.rejected_rows <=
			EXCLUDED.rejected_rows`,
		jobID,
		count,
		position.Token,
	)
	if err != nil {
		return fmt.Errorf("record source rejections: %w", err)
	}
	return rowsAffectedOne(tag, "set source rejections")
}
