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
		`UPDATE agefreighter_meta.load_job
		 SET rejected_rows = rejected_rows + ($2 - source_rejected_rows),
		     source_rejected_rows = $2,
		     resume_token = CASE WHEN $2 > source_rejected_rows THEN $3 ELSE resume_token END,
		     updated_at = clock_timestamp()
		 WHERE job_id = $1::uuid
		   AND status = 'running'
		   AND source_rejected_rows <= $2`,
		jobID,
		count,
		position.Token,
	)
	if err != nil {
		return fmt.Errorf("record source rejections: %w", err)
	}
	return rowsAffectedOne(tag, "set source rejections")
}
