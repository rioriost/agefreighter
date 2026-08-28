package meta

import (
	"context"
	"errors"
	"fmt"
	"time"
)

const sourceRejectionRollbackTimeout = 30 * time.Second

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
	tx, err := store.database.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin source rejection update: %w", err)
	}
	defer rollbackWithTimeout(ctx, tx, sourceRejectionRollbackTimeout)
	tag, err := tx.Exec(
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
	if err := rowsAffectedOne(tag, "set source rejections"); err != nil {
		return err
	}
	tag, err = tx.Exec(ctx, `
		INSERT INTO agefreighter_meta.job_unclassified_counter (
			job_id, rejected_rows
		) VALUES ($1::uuid, $2)
		ON CONFLICT (job_id) DO UPDATE SET
			rejected_rows = EXCLUDED.rejected_rows,
			updated_at = clock_timestamp()
		WHERE agefreighter_meta.job_unclassified_counter.rejected_rows <=
			EXCLUDED.rejected_rows`,
		jobID, count,
	)
	if err != nil {
		return fmt.Errorf("record unclassified source rejections: %w", err)
	}
	if err := rowsAffectedOne(tag, "set unclassified source rejections"); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit source rejection update: %w", err)
	}
	return nil
}
