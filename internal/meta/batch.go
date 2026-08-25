package meta

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/jackc/pgx/v5"
)

func (store *Store) StartBatch(
	ctx context.Context,
	batch BatchAttempt,
) (BatchAttempt, error) {
	if err := validateBatch(batch); err != nil {
		return BatchAttempt{}, err
	}
	if batch.RejectedRows != 0 {
		return BatchAttempt{}, errors.New("new batch rejected rows must be zero")
	}
	tx, err := store.database.Begin(ctx)
	if err != nil {
		return BatchAttempt{}, fmt.Errorf("begin load batch: %w", err)
	}
	defer rollback(ctx, tx)
	jobStatus, nextBatchID, err := lockJobBatchState(ctx, tx, batch.JobID)
	if err != nil {
		return BatchAttempt{}, err
	}
	ready := jobStatus == JobRunning && nextBatchID == batch.BatchID
	latest, latestErr := latestBatchInTransaction(
		ctx,
		tx,
		batch.JobID,
		batch.BatchID,
	)
	if latestErr != nil && !errors.Is(latestErr, ErrNotFound) {
		return BatchAttempt{}, latestErr
	}
	if latestErr == nil && latest.Attempt == batch.Attempt {
		if !sameBatchInput(latest, batch) ||
			(latest.Status == BatchRunning && !ready) ||
			(latest.Status == BatchCommitted &&
				(jobStatus != JobRunning && jobStatus != JobCommitted ||
					nextBatchID != batch.BatchID+1)) ||
			(latest.Status != BatchRunning && latest.Status != BatchCommitted) {
			return BatchAttempt{}, fmt.Errorf(
				"%w: batch attempt already exists with different metadata or state",
				ErrConflict,
			)
		}
		if err := tx.Commit(ctx); err != nil {
			return BatchAttempt{}, fmt.Errorf("commit idempotent load batch start: %w", err)
		}
		return latest, nil
	}
	expectedAttempt := uint32(1)
	if latestErr == nil {
		if latest.Status != BatchFailed || latest.Attempt == math.MaxUint32 {
			return BatchAttempt{}, fmt.Errorf(
				"%w: previous batch attempt is not retryable",
				ErrConflict,
			)
		}
		expectedAttempt = latest.Attempt + 1
	}
	if !ready || batch.Attempt != expectedAttempt {
		return BatchAttempt{}, fmt.Errorf(
			"%w: job is not ready for batch %d attempt %d",
			ErrConflict,
			batch.BatchID,
			batch.Attempt,
		)
	}
	tag, err := tx.Exec(
		ctx,
		`INSERT INTO agefreighter_meta.load_batch (
			job_id, batch_id, attempt, status, rows, bytes,
			first_resource, first_line, first_byte_offset, first_token
		) VALUES (
			$1::uuid, $2, $3, 'running', $4, $5, $6, $7, $8, $9
		)`,
		batch.JobID,
		batch.BatchID,
		batch.Attempt,
		batch.Rows,
		batch.Bytes,
		batch.First.Resource,
		batch.First.Line,
		batch.First.ByteOffset,
		batch.First.Token,
	)
	if err != nil {
		return BatchAttempt{}, fmt.Errorf("start load batch: %w", err)
	}
	if err := rowsAffectedOne(tag, "start load batch"); err != nil {
		return BatchAttempt{}, err
	}
	stored, err := latestBatchInTransaction(ctx, tx, batch.JobID, batch.BatchID)
	if err != nil {
		return BatchAttempt{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return BatchAttempt{}, fmt.Errorf("commit load batch start: %w", err)
	}
	return stored, nil
}

func (store *Store) GetBatch(
	ctx context.Context,
	jobID string,
	batchID uint64,
	attempt uint32,
) (BatchAttempt, error) {
	if err := validateBatchKey(jobID, batchID, attempt); err != nil {
		return BatchAttempt{}, err
	}
	return scanBatch(store.database.QueryRow(
		ctx,
		`SELECT
			job_id::text, batch_id, attempt, status, rows, bytes, rejected_rows,
			first_resource, first_line, first_byte_offset, first_token,
			last_resource, last_line, last_byte_offset, last_token,
			error_message, started_at, finished_at
		 FROM agefreighter_meta.load_batch
		 WHERE job_id = $1::uuid AND batch_id = $2 AND attempt = $3`,
		jobID,
		batchID,
		attempt,
	))
}

func (store *Store) LatestBatch(
	ctx context.Context,
	jobID string,
) (BatchAttempt, error) {
	if err := validateJobID(jobID); err != nil {
		return BatchAttempt{}, err
	}
	return scanBatch(store.database.QueryRow(
		ctx,
		`SELECT
			job_id::text, batch_id, attempt, status, rows, bytes, rejected_rows,
			first_resource, first_line, first_byte_offset, first_token,
			last_resource, last_line, last_byte_offset, last_token,
			error_message, started_at, finished_at
		 FROM agefreighter_meta.load_batch
		 WHERE job_id = $1::uuid
		 ORDER BY batch_id DESC, attempt DESC
		 LIMIT 1`,
		jobID,
	))
}

func scanBatch(row pgx.Row) (BatchAttempt, error) {
	var batch BatchAttempt
	err := row.Scan(
		&batch.JobID,
		&batch.BatchID,
		&batch.Attempt,
		&batch.Status,
		&batch.Rows,
		&batch.Bytes,
		&batch.RejectedRows,
		&batch.First.Resource,
		&batch.First.Line,
		&batch.First.ByteOffset,
		&batch.First.Token,
		&batch.Last.Resource,
		&batch.Last.Line,
		&batch.Last.ByteOffset,
		&batch.Last.Token,
		&batch.ErrorMessage,
		&batch.StartedAt,
		&batch.FinishedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return BatchAttempt{}, ErrNotFound
	}
	if err != nil {
		return BatchAttempt{}, fmt.Errorf("read load batch: %w", err)
	}
	return batch, nil
}

func (store *Store) CommitBatch(
	ctx context.Context,
	jobID string,
	batchID uint64,
	attempt uint32,
	last Position,
	rejectedRows int64,
) error {
	if err := validateBatchKey(jobID, batchID, attempt); err != nil {
		return err
	}
	if batchID == math.MaxInt64 {
		return errors.New("batch ID cannot advance beyond MaxInt64")
	}
	if err := validatePosition(last); err != nil {
		return fmt.Errorf("last position: %w", err)
	}
	if last.Token == "" {
		return errors.New("committed batch resume token is required")
	}
	if rejectedRows < 0 {
		return errors.New("committed rejected rows cannot be negative")
	}

	tx, err := store.database.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin batch commit: %w", err)
	}
	defer rollback(ctx, tx)

	jobStatus, nextBatchID, err := lockJobBatchState(ctx, tx, jobID)
	if err != nil {
		return err
	}
	ready := jobStatus == JobRunning && nextBatchID == batchID
	var rows, bytes int64
	err = tx.QueryRow(
		ctx,
		`UPDATE agefreighter_meta.load_batch
		 SET status = 'committed',
		     last_resource = $4,
		     last_line = $5,
		     last_byte_offset = $6,
		     last_token = $7,
		     rejected_rows = $8,
		     error_message = '',
		     finished_at = clock_timestamp()
		 WHERE job_id = $1::uuid
		   AND batch_id = $2
		   AND attempt = $3
		   AND status = 'running'
		 RETURNING rows, bytes`,
		jobID,
		batchID,
		attempt,
		last.Resource,
		last.Line,
		last.ByteOffset,
		last.Token,
		rejectedRows,
	).Scan(&rows, &bytes)
	if err == nil && !ready {
		return fmt.Errorf("%w: job is not ready to commit batch %d", ErrConflict, batchID)
	}
	if errors.Is(err, pgx.ErrNoRows) {
		var status BatchStatus
		var storedLast Position
		var storedRejected int64
		err = tx.QueryRow(
			ctx,
			`SELECT
				status, last_resource, last_line, last_byte_offset,
				last_token, rejected_rows
			 FROM agefreighter_meta.load_batch
			 WHERE job_id = $1::uuid AND batch_id = $2 AND attempt = $3`,
			jobID,
			batchID,
			attempt,
		).Scan(
			&status,
			&storedLast.Resource,
			&storedLast.Line,
			&storedLast.ByteOffset,
			&storedLast.Token,
			&storedRejected,
		)
		if errors.Is(err, pgx.ErrNoRows) {
			return fmt.Errorf("%w: batch attempt", ErrNotFound)
		}
		if err != nil {
			return fmt.Errorf("read batch commit state: %w", err)
		}
		if status != BatchCommitted ||
			storedLast != last ||
			storedRejected != rejectedRows ||
			nextBatchID != batchID+1 ||
			(jobStatus != JobRunning && jobStatus != JobCommitted) {
			return fmt.Errorf("%w: batch attempt cannot be committed", ErrConflict)
		}
		if err := tx.Commit(ctx); err != nil {
			return fmt.Errorf("commit idempotent batch state: %w", err)
		}
		return nil
	}
	if err != nil {
		return fmt.Errorf("commit load batch: %w", err)
	}

	tag, err := tx.Exec(
		ctx,
		`UPDATE agefreighter_meta.load_job
		 SET next_batch_id = $2 + 1,
		     resume_token = $3,
		     committed_rows = committed_rows + $4,
		     committed_bytes = committed_bytes + $5,
		     rejected_rows = rejected_rows + $6,
		     updated_at = clock_timestamp()
		 WHERE job_id = $1::uuid
		   AND status = 'running'
		   AND next_batch_id = $2`,
		jobID,
		batchID,
		last.Token,
		rows,
		bytes,
		rejectedRows,
	)
	if err != nil {
		return fmt.Errorf("advance load job checkpoint: %w", err)
	}
	if err := rowsAffectedOne(tag, "advance load job checkpoint"); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit batch checkpoint: %w", err)
	}
	return nil
}

func (store *Store) RecordFailedBatch(
	ctx context.Context,
	batch BatchAttempt,
	message string,
) error {
	if err := validateBatch(batch); err != nil {
		return err
	}
	if message == "" {
		return errors.New("batch failure message is required")
	}
	tx, err := store.database.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin failed batch diagnostic: %w", err)
	}
	defer rollback(ctx, tx)

	jobStatus, nextBatchID, err := lockJobBatchState(ctx, tx, batch.JobID)
	if err != nil {
		return err
	}
	if nextBatchID != batch.BatchID ||
		(jobStatus != JobPending &&
			jobStatus != JobRunning &&
			jobStatus != JobFailed) {
		return fmt.Errorf(
			"%w: job is not ready to fail batch %d",
			ErrConflict,
			batch.BatchID,
		)
	}
	latest, latestErr := latestBatchInTransaction(
		ctx,
		tx,
		batch.JobID,
		batch.BatchID,
	)
	if latestErr != nil && !errors.Is(latestErr, ErrNotFound) {
		return latestErr
	}
	if latestErr == nil && latest.Attempt > batch.Attempt {
		return fmt.Errorf(
			"%w: a newer batch attempt already exists",
			ErrConflict,
		)
	}
	if latestErr == nil && latest.Attempt == batch.Attempt {
		if !sameBatchInput(latest, batch) {
			return fmt.Errorf(
				"%w: failed batch metadata differs from the running attempt",
				ErrConflict,
			)
		}
		switch latest.Status {
		case BatchCommitted:
			return fmt.Errorf("%w: committed batch cannot fail", ErrConflict)
		case BatchFailed:
			if latest.RejectedRows != batch.RejectedRows ||
				latest.ErrorMessage != message {
				return fmt.Errorf(
					"%w: failed batch diagnostic differs from the stored attempt",
					ErrConflict,
				)
			}
			if err := tx.Commit(ctx); err != nil {
				return fmt.Errorf("commit idempotent failed batch diagnostic: %w", err)
			}
			return nil
		case BatchRunning:
			tag, updateErr := tx.Exec(
				ctx,
				`UPDATE agefreighter_meta.load_batch
				 SET status = 'failed',
				     rejected_rows = $4,
				     error_message = $5,
				     finished_at = clock_timestamp()
				 WHERE job_id = $1::uuid
				   AND batch_id = $2
				   AND attempt = $3
				   AND status = 'running'`,
				batch.JobID,
				batch.BatchID,
				batch.Attempt,
				batch.RejectedRows,
				message,
			)
			if updateErr != nil {
				return fmt.Errorf("record failed load batch: %w", updateErr)
			}
			if err := rowsAffectedOne(tag, "record failed load batch"); err != nil {
				return err
			}
		default:
			return fmt.Errorf("%w: unsupported batch state %q", ErrConflict, latest.Status)
		}
	} else {
		expectedAttempt := uint32(1)
		if latestErr == nil {
			if latest.Status != BatchFailed || latest.Attempt == math.MaxUint32 {
				return fmt.Errorf(
					"%w: previous batch attempt is not retryable",
					ErrConflict,
				)
			}
			expectedAttempt = latest.Attempt + 1
		}
		if batch.Attempt != expectedAttempt {
			return fmt.Errorf(
				"%w: failed batch attempt is not the next attempt",
				ErrConflict,
			)
		}
		tag, insertErr := tx.Exec(
			ctx,
			`INSERT INTO agefreighter_meta.load_batch (
				job_id, batch_id, attempt, status, rows, bytes, rejected_rows,
				first_resource, first_line, first_byte_offset, first_token,
				error_message, finished_at
			) VALUES (
				$1::uuid, $2, $3, 'failed', $4, $5, $6, $7, $8, $9, $10,
				$11, clock_timestamp()
			)`,
			batch.JobID,
			batch.BatchID,
			batch.Attempt,
			batch.Rows,
			batch.Bytes,
			batch.RejectedRows,
			batch.First.Resource,
			batch.First.Line,
			batch.First.ByteOffset,
			batch.First.Token,
			message,
		)
		if insertErr != nil {
			return fmt.Errorf("record failed load batch: %w", insertErr)
		}
		if err := rowsAffectedOne(tag, "record failed load batch"); err != nil {
			return err
		}
	}
	tag, err := tx.Exec(
		ctx,
		`UPDATE agefreighter_meta.load_job
		 SET status = 'failed',
		     error_message = $2,
		     completed_at = clock_timestamp(),
		     updated_at = clock_timestamp()
		 WHERE job_id = $1::uuid
		   AND status IN ('pending', 'running', 'failed')`,
		batch.JobID,
		message,
	)
	if err != nil {
		return fmt.Errorf("record failed load job: %w", err)
	}
	if err := rowsAffectedOne(tag, "record failed load job"); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit failed batch diagnostic: %w", err)
	}
	return nil
}

func validateBatchKey(jobID string, batchID uint64, attempt uint32) error {
	if err := validateJobID(jobID); err != nil {
		return err
	}
	if batchID == 0 || batchID > math.MaxInt64 || attempt == 0 {
		return errors.New("batch ID and attempt must be positive")
	}
	return nil
}

func lockJobBatchState(
	ctx context.Context,
	tx pgx.Tx,
	jobID string,
) (JobStatus, uint64, error) {
	var status JobStatus
	var nextBatchID uint64
	err := tx.QueryRow(
		ctx,
		`SELECT status, next_batch_id
		 FROM agefreighter_meta.load_job
		 WHERE job_id = $1::uuid
		 FOR UPDATE`,
		jobID,
	).Scan(&status, &nextBatchID)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", 0, fmt.Errorf("%w: load job %q", ErrNotFound, jobID)
	}
	if err != nil {
		return "", 0, fmt.Errorf("lock load job batch state: %w", err)
	}
	return status, nextBatchID, nil
}

func latestBatchInTransaction(
	ctx context.Context,
	tx pgx.Tx,
	jobID string,
	batchID uint64,
) (BatchAttempt, error) {
	return scanBatch(tx.QueryRow(
		ctx,
		`SELECT
			job_id::text, batch_id, attempt, status, rows, bytes, rejected_rows,
			first_resource, first_line, first_byte_offset, first_token,
			last_resource, last_line, last_byte_offset, last_token,
			error_message, started_at, finished_at
		 FROM agefreighter_meta.load_batch
		 WHERE job_id = $1::uuid AND batch_id = $2
		 ORDER BY attempt DESC
		 LIMIT 1
		 FOR UPDATE`,
		jobID,
		batchID,
	))
}

func sameBatchInput(left, right BatchAttempt) bool {
	return left.JobID == right.JobID &&
		left.BatchID == right.BatchID &&
		left.Attempt == right.Attempt &&
		left.Rows == right.Rows &&
		left.Bytes == right.Bytes &&
		left.First == right.First
}
