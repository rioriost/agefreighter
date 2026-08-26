package meta

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

func (store *Store) CreateJob(ctx context.Context, job Job) error {
	if err := validateJob(job); err != nil {
		return err
	}
	_, err := store.database.Exec(
		ctx,
		`INSERT INTO agefreighter_meta.load_job (
			job_id, name, source_type, load_mode, target_graph,
			config_fingerprint, status
		) VALUES ($1::uuid, $2, $3, $4, $5, $6, 'pending')`,
		job.ID,
		job.Name,
		job.SourceType,
		job.LoadMode,
		job.TargetGraph,
		job.ConfigFingerprint,
	)
	if err != nil {
		return fmt.Errorf("create load job %q: %w", job.ID, err)
	}
	return nil
}

func (store *Store) GetJob(ctx context.Context, jobID string) (Job, error) {
	if err := validateJobID(jobID); err != nil {
		return Job{}, err
	}
	var job Job
	err := store.database.QueryRow(
		ctx,
		`SELECT
			job_id::text, name, source_type, load_mode, target_graph,
			backup_graph_name, config_fingerprint::text, status,
			COALESCE(graph_generation_id, 0),
			next_batch_id, resume_token, committed_rows, committed_bytes,
			rejected_rows, source_rejected_rows, error_message, created_at, started_at, updated_at,
			completed_at, backup_cleaned_at
		 FROM agefreighter_meta.load_job
		 WHERE job_id = $1::uuid`,
		jobID,
	).Scan(
		&job.ID,
		&job.Name,
		&job.SourceType,
		&job.LoadMode,
		&job.TargetGraph,
		&job.BackupGraphName,
		&job.ConfigFingerprint,
		&job.Status,
		&job.GraphGenerationID,
		&job.NextBatchID,
		&job.ResumeToken,
		&job.CommittedRows,
		&job.CommittedBytes,
		&job.RejectedRows,
		&job.SourceRejectedRows,
		&job.ErrorMessage,
		&job.CreatedAt,
		&job.StartedAt,
		&job.UpdatedAt,
		&job.CompletedAt,
		&job.BackupCleanedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return Job{}, fmt.Errorf("%w: load job %q", ErrNotFound, jobID)
	}
	if err != nil {
		return Job{}, fmt.Errorf("read load job %q: %w", jobID, err)
	}
	return job, nil
}

func (store *Store) StartJob(ctx context.Context, jobID string) error {
	if err := validateJobID(jobID); err != nil {
		return err
	}
	tag, err := store.database.Exec(
		ctx,
		`UPDATE agefreighter_meta.load_job
		 SET status = 'running',
		     error_message = '',
		     started_at = COALESCE(started_at, clock_timestamp()),
		     completed_at = NULL,
		     updated_at = clock_timestamp()
		 WHERE job_id = $1::uuid
		   AND status IN ('pending', 'failed')`,
		jobID,
	)
	if err != nil {
		return fmt.Errorf("start load job %q: %w", jobID, err)
	}
	return rowsAffectedOne(tag, "start load job")
}

func (store *Store) CompleteJob(ctx context.Context, jobID string) error {
	if err := validateJobID(jobID); err != nil {
		return err
	}
	tx, err := store.database.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin load job completion: %w", err)
	}
	defer rollback(ctx, tx)
	var status JobStatus
	var nextBatchID uint64
	if err := tx.QueryRow(
		ctx,
		`SELECT status, next_batch_id
		 FROM agefreighter_meta.load_job
		 WHERE job_id = $1::uuid
		 FOR UPDATE`,
		jobID,
	).Scan(&status, &nextBatchID); errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("%w: load job %q", ErrNotFound, jobID)
	} else if err != nil {
		return fmt.Errorf("lock load job %q for completion: %w", jobID, err)
	}
	if status != JobRunning {
		return fmt.Errorf("%w: load job is %s", ErrConflict, status)
	}
	var unresolvedBatches int
	if err := tx.QueryRow(
		ctx,
		`SELECT COUNT(*)
		 FROM agefreighter_meta.load_batch
		 WHERE job_id = $1::uuid
		   AND batch_id = $2
		   AND status IN ('running', 'failed')`,
		jobID,
		nextBatchID,
	).Scan(&unresolvedBatches); err != nil {
		return fmt.Errorf("check unresolved load batches: %w", err)
	}
	if unresolvedBatches != 0 {
		return fmt.Errorf(
			"%w: load job has %d unresolved attempts for batch %d",
			ErrConflict,
			unresolvedBatches,
			nextBatchID,
		)
	}
	tag, err := tx.Exec(
		ctx,
		`UPDATE agefreighter_meta.load_job
		 SET status = 'committed',
		     error_message = '',
		     completed_at = clock_timestamp(),
		     updated_at = clock_timestamp()
		 WHERE job_id = $1::uuid
		   AND status = 'running'`,
		jobID,
	)
	if err != nil {
		return fmt.Errorf("complete load job %q: %w", jobID, err)
	}
	if err := rowsAffectedOne(tag, "complete load job"); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit load job completion: %w", err)
	}
	return nil
}

func (store *Store) CompleteJobGeneration(
	ctx context.Context,
	jobID string,
	graphGenerationID int64,
) error {
	if err := validateJobID(jobID); err != nil {
		return err
	}
	if graphGenerationID <= 0 {
		return errors.New("graph generation ID must be positive")
	}
	tx, err := store.database.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin load generation completion: %w", err)
	}
	defer rollback(ctx, tx)
	var status JobStatus
	var nextBatchID uint64
	var boundGenerationID int64
	if err := tx.QueryRow(
		ctx,
		`SELECT status, next_batch_id, COALESCE(graph_generation_id, 0)
		 FROM agefreighter_meta.load_job
		 WHERE job_id = $1::uuid
		 FOR UPDATE`,
		jobID,
	).Scan(&status, &nextBatchID, &boundGenerationID); errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("%w: load job %q", ErrNotFound, jobID)
	} else if err != nil {
		return fmt.Errorf("lock load job for generation completion: %w", err)
	}
	if status != JobRunning || boundGenerationID != graphGenerationID {
		return fmt.Errorf("%w: load job generation is not running", ErrConflict)
	}
	var unresolved int
	if err := tx.QueryRow(
		ctx,
		`SELECT COUNT(*)
		 FROM agefreighter_meta.load_batch
		 WHERE job_id = $1::uuid
		   AND batch_id = $2
		   AND status IN ('running', 'failed')`,
		jobID,
		nextBatchID,
	).Scan(&unresolved); err != nil {
		return fmt.Errorf("check unresolved load batches: %w", err)
	}
	if unresolved != 0 {
		return fmt.Errorf("%w: load job has unresolved attempts", ErrConflict)
	}
	tag, err := tx.Exec(
		ctx,
		`UPDATE agefreighter_meta.graph_generation
		 SET state = 'active', updated_at = clock_timestamp()
		 WHERE graph_generation_id = $1
		   AND job_id = $2::uuid
		   AND state = 'loading'`,
		graphGenerationID,
		jobID,
	)
	if err != nil {
		return fmt.Errorf("activate completed graph generation: %w", err)
	}
	if err := rowsAffectedOne(tag, "activate completed graph generation"); err != nil {
		return err
	}
	tag, err = tx.Exec(
		ctx,
		`UPDATE agefreighter_meta.load_job
		 SET status = 'committed', error_message = '',
		     completed_at = clock_timestamp(), updated_at = clock_timestamp()
		 WHERE job_id = $1::uuid AND status = 'running'`,
		jobID,
	)
	if err != nil {
		return fmt.Errorf("complete load job generation: %w", err)
	}
	if err := rowsAffectedOne(tag, "complete load job generation"); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit load generation completion: %w", err)
	}
	return nil
}

func (store *Store) FailJob(
	ctx context.Context,
	jobID string,
	message string,
) error {
	if err := validateJobID(jobID); err != nil {
		return err
	}
	if message == "" {
		return errors.New("job failure message is required")
	}
	tag, err := store.database.Exec(
		ctx,
		`UPDATE agefreighter_meta.load_job
		 SET status = 'failed',
		     error_message = $2,
		     completed_at = clock_timestamp(),
		     updated_at = clock_timestamp()
		 WHERE job_id = $1::uuid
		   AND status IN ('pending', 'running', 'failed')`,
		jobID,
		message,
	)
	if err != nil {
		return fmt.Errorf("fail load job %q: %w", jobID, err)
	}
	return rowsAffectedOne(tag, "fail load job")
}
