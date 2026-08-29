package meta

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"

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
	started, fastErr := scanBatch(store.database.QueryRow(
		ctx,
		`WITH
		job_state AS MATERIALIZED (
			SELECT status, next_batch_id
			FROM agefreighter_meta.load_job
			WHERE job_id = $1::uuid
			FOR UPDATE
		),
		latest AS MATERIALIZED (
			SELECT attempt, status
			FROM agefreighter_meta.load_batch
			WHERE job_id = $1::uuid AND batch_id = $2
			ORDER BY attempt DESC
			LIMIT 1
		),
		inserted AS (
			INSERT INTO agefreighter_meta.load_batch (
				job_id, batch_id, attempt, status, rows, bytes,
				first_resource, first_line, first_byte_offset, first_token
			)
			SELECT $1::uuid, $2, $3, 'running', $4, $5, $6, $7, $8, $9
			FROM job_state
			WHERE job_state.status = 'running'
			  AND job_state.next_batch_id = $2
			  AND (
			    (
			      NOT EXISTS (SELECT 1 FROM latest)
			      AND $3::integer = 1
			    )
			    OR EXISTS (
			      SELECT 1
			      FROM latest
			      WHERE latest.status = 'failed'
			        AND latest.attempt < 2147483647
			        AND $3::integer = latest.attempt + 1
			    )
			  )
			RETURNING
				job_id::text, batch_id, attempt, status, rows, bytes,
				rejected_rows, first_resource, first_line, first_byte_offset,
				first_token, last_resource, last_line, last_byte_offset,
				last_token, error_message, started_at, finished_at
		)
		SELECT *
		FROM inserted`,
		pgx.QueryExecModeExec,
		batch.JobID,
		batch.BatchID,
		batch.Attempt,
		batch.Rows,
		batch.Bytes,
		batch.First.Resource,
		batch.First.Line,
		batch.First.ByteOffset,
		batch.First.Token,
	))
	if fastErr == nil {
		return started, nil
	}
	if !errors.Is(fastErr, ErrNotFound) {
		return BatchAttempt{}, fmt.Errorf("start load batch: %w", fastErr)
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
	return store.CommitBatchWithLabelCounters(
		ctx, jobID, batchID, attempt, last, rejectedRows, nil,
	)
}

func (store *Store) CommitBatchWithLabelCounters(
	ctx context.Context,
	jobID string,
	batchID uint64,
	attempt uint32,
	last Position,
	rejectedRows int64,
	counters []BatchLabelCounter,
) error {
	return store.CommitBatchWithLabelCountersAndVerification(
		ctx, jobID, batchID, attempt, last, rejectedRows, counters, nil,
	)
}

func (store *Store) CommitBatchWithLabelCountersAndVerification(
	ctx context.Context,
	jobID string,
	batchID uint64,
	attempt uint32,
	last Position,
	rejectedRows int64,
	counters []BatchLabelCounter,
	verification *JobVerification,
) error {
	counters = slices.Clone(counters)
	slices.SortFunc(counters, func(left, right BatchLabelCounter) int {
		if left.LabelGenerationID < right.LabelGenerationID {
			return -1
		}
		if left.LabelGenerationID > right.LabelGenerationID {
			return 1
		}
		return 0
	})
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
	if err := validateBatchLabelCounters(counters, rejectedRows); err != nil {
		return err
	}
	if verification != nil {
		if verification.JobID != jobID {
			return errors.New("job verification belongs to a different job")
		}
		if err := validateJobVerification(*verification); err != nil {
			return err
		}
	}

	tx, existingTransaction := store.database.(pgx.Tx)
	ownsTransaction := !existingTransaction
	var err error
	if ownsTransaction {
		tx, err = store.database.Begin(ctx)
		if err != nil {
			return fmt.Errorf("begin batch commit: %w", err)
		}
		defer rollback(ctx, tx)
	}

	labelIDs := make([]int64, len(counters))
	labelKinds := make([]string, len(counters))
	acceptedRows := make([]int64, len(counters))
	committedRows := make([]int64, len(counters))
	committedBytes := make([]int64, len(counters))
	committedBytesKnown := make([]bool, len(counters))
	labelRejectedRows := make([]int64, len(counters))
	for index, counter := range counters {
		labelIDs[index] = counter.LabelGenerationID
		labelKinds[index] = string(counter.Kind)
		acceptedRows[index] = counter.AcceptedRows
		committedRows[index] = counter.CommittedRows
		if counter.CommittedBytes != nil {
			committedBytes[index] = *counter.CommittedBytes
			committedBytesKnown[index] = true
		}
		labelRejectedRows[index] = counter.RejectedRows
	}
	var (
		submittedFingerprint, resolvedFingerprint string
		resolvedSummary                           = []byte("{}")
		storeVerification                         bool
	)
	if verification != nil {
		submittedFingerprint = verification.SubmittedConfigFingerprint
		resolvedFingerprint = verification.ResolvedMappingFingerprint
		resolvedSummary = verification.ResolvedMappingSummary
		storeVerification = true
	}

	var (
		jobStatus                        JobStatus
		nextBatchID                      uint64
		rows, bytes                      int64
		batchCommitted                   bool
		batchCounterRows, jobCounterRows int64
		verificationRows                 int64
	)
	err = tx.QueryRow(
		ctx,
		`WITH
		counter_input AS (
			SELECT *
			FROM unnest(
				$9::bigint[], $10::text[], $11::bigint[], $12::bigint[],
				$13::bigint[], $14::boolean[], $15::bigint[]
			) AS counter(
				label_generation_id, kind, accepted_rows, committed_rows,
				committed_bytes, committed_bytes_known, rejected_rows
			)
		),
		job_basis AS MATERIALIZED (
			SELECT job_id, status, next_batch_id, load_mode,
			       committed_rows <> 0 OR committed_bytes <> 0
			       OR rejected_rows <> 0 OR source_rejected_rows <> 0
			       OR EXISTS (
			         SELECT 1
			         FROM agefreighter_meta.load_batch previous
			         WHERE previous.job_id = job.job_id
			           AND previous.status = 'committed'
			       ) AS historical_activity
			FROM agefreighter_meta.load_job job
			WHERE job_id = $1::uuid
			FOR UPDATE
		),
		committed_batch AS (
			UPDATE agefreighter_meta.load_batch batch
			SET status = 'committed',
			    last_resource = $4,
			    last_line = $5,
			    last_byte_offset = $6,
			    last_token = $7,
			    rejected_rows = $8,
			    error_message = '',
			    finished_at = clock_timestamp()
			FROM job_basis
			WHERE batch.job_id = job_basis.job_id
			  AND batch.batch_id = $2
			  AND batch.attempt = $3
			  AND batch.status = 'running'
			  AND job_basis.status = 'running'
			  AND job_basis.next_batch_id = $2
			RETURNING batch.rows, batch.bytes
		),
		advanced_job AS (
			UPDATE agefreighter_meta.load_job job
			SET next_batch_id = $2 + 1,
			    resume_token = $7,
			    committed_rows = job.committed_rows + batch.rows,
			    committed_bytes = job.committed_bytes + batch.bytes,
			    rejected_rows = job.rejected_rows + $8,
			    updated_at = clock_timestamp()
			FROM committed_batch batch
			WHERE job.job_id = $1::uuid
			  AND job.status = 'running'
			  AND job.next_batch_id = $2
			RETURNING job.job_id
		),
		stored_verification AS (
			INSERT INTO agefreighter_meta.job_verification (
				job_id, submitted_config_fingerprint,
				resolved_mapping_fingerprint, resolved_mapping_summary
			)
			SELECT advanced.job_id, $16, $17, $18::jsonb
			FROM advanced_job advanced
			WHERE $19
			ON CONFLICT (job_id) DO UPDATE SET
				submitted_config_fingerprint =
					EXCLUDED.submitted_config_fingerprint,
				resolved_mapping_fingerprint =
					EXCLUDED.resolved_mapping_fingerprint,
				resolved_mapping_summary = EXCLUDED.resolved_mapping_summary
			WHERE agefreighter_meta.job_verification.submitted_config_fingerprint =
					EXCLUDED.submitted_config_fingerprint
			  AND agefreighter_meta.job_verification.resolved_mapping_fingerprint =
					EXCLUDED.resolved_mapping_fingerprint
			  AND agefreighter_meta.job_verification.resolved_mapping_summary =
					EXCLUDED.resolved_mapping_summary
			RETURNING job_id
		),
		stored_unclassified_counter AS (
			INSERT INTO agefreighter_meta.job_unclassified_counter (
				job_id, rejected_rows
			)
			SELECT advanced.job_id, 0
			FROM advanced_job advanced
			ON CONFLICT (job_id) DO NOTHING
			RETURNING job_id
		),
		stored_batch_counters AS (
			INSERT INTO agefreighter_meta.load_batch_label_counter (
				job_id, batch_id, attempt, label_generation_id, kind,
				accepted_rows, committed_rows_delta, committed_bytes, rejected_rows
			)
			SELECT advanced.job_id, $2, $3, counter.label_generation_id,
			       counter.kind, counter.accepted_rows, counter.committed_rows,
			       CASE WHEN counter.committed_bytes_known
			            THEN counter.committed_bytes END,
			       counter.rejected_rows
			FROM advanced_job advanced
			CROSS JOIN counter_input counter
			RETURNING label_generation_id
		),
		updated_job_counters AS (
			UPDATE agefreighter_meta.job_label_counter stored
			SET accepted_rows = CASE
			      WHEN stored.accepted_rows IS NULL THEN NULL
			      ELSE stored.accepted_rows + counter.accepted_rows
			    END,
			    committed_rows = CASE
			      WHEN stored.committed_rows IS NULL THEN NULL
			      ELSE stored.committed_rows + counter.committed_rows
			    END,
			    committed_bytes = CASE
			      WHEN stored.committed_bytes IS NULL
			        OR NOT counter.committed_bytes_known THEN NULL
			      ELSE stored.committed_bytes + counter.committed_bytes
			    END,
			    rejected_rows = CASE
			      WHEN stored.rejected_rows IS NULL THEN NULL
			      ELSE stored.rejected_rows + counter.rejected_rows
			    END,
			    updated_at = clock_timestamp()
			FROM counter_input counter
			CROSS JOIN advanced_job advanced
			WHERE stored.job_id = advanced.job_id
			  AND stored.label_generation_id = counter.label_generation_id
			  AND stored.kind = counter.kind
			RETURNING stored.label_generation_id
		),
		inserted_job_counters AS (
			INSERT INTO agefreighter_meta.job_label_counter (
				job_id, label_generation_id, kind,
				counter_completeness, counter_provenance,
				accepted_rows, committed_rows, committed_bytes, rejected_rows
			)
			SELECT advanced.job_id, counter.label_generation_id, counter.kind,
			       CASE
			         WHEN basis.historical_activity
			           OR basis.load_mode IN ('append', 'upsert')
			         THEN 'incomplete'
			         ELSE 'complete'
			       END,
			       CASE
			         WHEN basis.historical_activity THEN 'legacy-resume'
			         WHEN basis.load_mode IN ('append', 'upsert')
			         THEN 'baseline-unavailable'
			         ELSE 'v17-lifecycle'
			       END,
			       CASE
			         WHEN basis.historical_activity
			           OR basis.load_mode IN ('append', 'upsert')
			         THEN NULL
			         ELSE counter.accepted_rows
			       END,
			       CASE
			         WHEN basis.historical_activity
			           OR basis.load_mode IN ('append', 'upsert')
			         THEN NULL
			         ELSE counter.committed_rows
			       END,
			       CASE
			         WHEN basis.historical_activity
			           OR basis.load_mode IN ('append', 'upsert')
			           OR NOT counter.committed_bytes_known
			         THEN NULL
			         ELSE counter.committed_bytes
			       END,
			       CASE
			         WHEN basis.historical_activity
			           OR basis.load_mode IN ('append', 'upsert')
			         THEN NULL
			         ELSE counter.rejected_rows
			       END
			FROM advanced_job advanced
			JOIN job_basis basis ON basis.job_id = advanced.job_id
			CROSS JOIN counter_input counter
			JOIN agefreighter_meta.label_generation label
			  ON label.label_generation_id = counter.label_generation_id
			 AND label.kind = counter.kind
			WHERE NOT EXISTS (
				SELECT 1
				FROM agefreighter_meta.job_label_counter stored
				WHERE stored.job_id = advanced.job_id
				  AND stored.label_generation_id = counter.label_generation_id
			)
			RETURNING label_generation_id
		)
		SELECT
			job_basis.status,
			job_basis.next_batch_id,
			COALESCE((SELECT rows FROM committed_batch), 0),
			COALESCE((SELECT bytes FROM committed_batch), 0),
			EXISTS (SELECT 1 FROM committed_batch),
			(SELECT count(*) FROM stored_batch_counters),
			(SELECT count(*) FROM updated_job_counters)
			  + (SELECT count(*) FROM inserted_job_counters),
			(SELECT count(*) FROM stored_verification)
		FROM job_basis`,
		pgx.QueryExecModeExec,
		jobID,
		batchID,
		attempt,
		last.Resource,
		last.Line,
		last.ByteOffset,
		last.Token,
		rejectedRows,
		labelIDs,
		labelKinds,
		acceptedRows,
		committedRows,
		committedBytes,
		committedBytesKnown,
		labelRejectedRows,
		submittedFingerprint,
		resolvedFingerprint,
		string(resolvedSummary),
		storeVerification,
	).Scan(
		&jobStatus,
		&nextBatchID,
		&rows,
		&bytes,
		&batchCommitted,
		&batchCounterRows,
		&jobCounterRows,
		&verificationRows,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("%w: load job %q", ErrNotFound, jobID)
	}
	if err != nil {
		return fmt.Errorf("commit batch checkpoint: %w", err)
	}
	if !batchCommitted {
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
		if counters != nil {
			if err := verifyBatchLabelCounters(
				ctx, tx, jobID, batchID, attempt, counters,
			); err != nil {
				return err
			}
		}
		if verification != nil {
			stored, readErr := scanJobVerification(tx.QueryRow(ctx, `
				SELECT job_id::text, submitted_config_fingerprint::text,
				       resolved_mapping_fingerprint::text, resolved_mapping_summary
				FROM agefreighter_meta.job_verification
				WHERE job_id = $1::uuid`, jobID))
			if readErr != nil || !sameJobVerification(stored, *verification) {
				return fmt.Errorf("%w: job verification differs", ErrConflict)
			}
		}
		if ownsTransaction {
			if err := tx.Commit(ctx); err != nil {
				return fmt.Errorf("commit idempotent batch state: %w", err)
			}
		}
		return nil
	}
	expectedCounterRows := int64(len(counters))
	if batchCounterRows != expectedCounterRows ||
		jobCounterRows != expectedCounterRows {
		return fmt.Errorf(
			"%w: stored %d batch and %d job label counters, expected %d",
			ErrConflict,
			batchCounterRows,
			jobCounterRows,
			expectedCounterRows,
		)
	}
	expectedVerificationRows := int64(0)
	if verification != nil {
		expectedVerificationRows = 1
	}
	if verificationRows != expectedVerificationRows {
		return fmt.Errorf("%w: job verification differs", ErrConflict)
	}
	if ownsTransaction {
		if err := tx.Commit(ctx); err != nil {
			return fmt.Errorf("commit batch checkpoint: %w", err)
		}
	}
	return nil
}

func scanJobVerification(row pgx.Row) (JobVerification, error) {
	var value JobVerification
	err := row.Scan(
		&value.JobID,
		&value.SubmittedConfigFingerprint,
		&value.ResolvedMappingFingerprint,
		&value.ResolvedMappingSummary,
	)
	return value, err
}

func sameJobVerification(left, right JobVerification) bool {
	var leftSummary, rightSummary any
	if json.Unmarshal(left.ResolvedMappingSummary, &leftSummary) != nil ||
		json.Unmarshal(right.ResolvedMappingSummary, &rightSummary) != nil {
		return false
	}
	return left.JobID == right.JobID &&
		left.SubmittedConfigFingerprint == right.SubmittedConfigFingerprint &&
		left.ResolvedMappingFingerprint == right.ResolvedMappingFingerprint &&
		reflect.DeepEqual(leftSummary, rightSummary)
}

func validateBatchLabelCounters(
	counters []BatchLabelCounter,
	rejectedRows int64,
) error {
	seen := make(map[int64]struct{}, len(counters))
	var attributedRejected int64
	for _, counter := range counters {
		if counter.LabelGenerationID <= 0 {
			return errors.New("label counter generation ID must be positive")
		}
		if counter.Kind != VertexLabel && counter.Kind != EdgeLabel {
			return fmt.Errorf("unsupported label counter kind %q", counter.Kind)
		}
		if counter.AcceptedRows < 0 || counter.CommittedRows < 0 ||
			counter.RejectedRows < 0 ||
			(counter.CommittedBytes != nil && *counter.CommittedBytes < 0) {
			return errors.New("label counter values cannot be negative")
		}
		if _, exists := seen[counter.LabelGenerationID]; exists {
			return errors.New("duplicate batch label counter")
		}
		seen[counter.LabelGenerationID] = struct{}{}
		attributedRejected += counter.RejectedRows
	}
	if attributedRejected > rejectedRows {
		return errors.New("attributed label rejects exceed batch rejects")
	}
	return nil
}

func verifyBatchLabelCounters(
	ctx context.Context,
	tx pgx.Tx,
	jobID string,
	batchID uint64,
	attempt uint32,
	expected []BatchLabelCounter,
) error {
	rows, err := tx.Query(ctx, `
			SELECT label_generation_id, kind, accepted_rows,
			       committed_rows_delta, committed_bytes, rejected_rows
			FROM agefreighter_meta.load_batch_label_counter
			WHERE job_id = $1::uuid AND batch_id = $2 AND attempt = $3
			ORDER BY label_generation_id`,
		jobID, batchID, attempt,
	)
	if err != nil {
		return fmt.Errorf("read idempotent batch label counters: %w", err)
	}
	defer rows.Close()
	actual := make([]BatchLabelCounter, 0, len(expected))
	for rows.Next() {
		var value BatchLabelCounter
		if err := rows.Scan(
			&value.LabelGenerationID, &value.Kind, &value.AcceptedRows,
			&value.CommittedRows, &value.CommittedBytes, &value.RejectedRows,
		); err != nil {
			return fmt.Errorf("scan idempotent batch label counter: %w", err)
		}
		actual = append(actual, value)
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("read idempotent batch label counters: %w", err)
	}
	if !slices.EqualFunc(actual, expected, func(left, right BatchLabelCounter) bool {
		return left.LabelGenerationID == right.LabelGenerationID &&
			left.Kind == right.Kind &&
			left.AcceptedRows == right.AcceptedRows &&
			left.CommittedRows == right.CommittedRows &&
			left.RejectedRows == right.RejectedRows &&
			(left.CommittedBytes == nil && right.CommittedBytes == nil ||
				left.CommittedBytes != nil && right.CommittedBytes != nil &&
					*left.CommittedBytes == *right.CommittedBytes)
	}) {
		return fmt.Errorf("%w: batch label counters differ", ErrConflict)
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
