package meta

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func (store *Store) CreateJob(ctx context.Context, job Job) error {
	if err := validateJob(job); err != nil {
		return err
	}
	_, err := store.database.Exec(
		ctx,
		`INSERT INTO agefreighter_meta.load_job (
			job_id, name, source_type, load_mode, target_backend,
			target_schema, target_graph,
			config_fingerprint, status
		) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, 'pending')`,
		job.ID,
		job.Name,
		job.SourceType,
		job.LoadMode,
		resolvedTargetBackend(job),
		job.TargetSchema,
		job.TargetGraph,
		job.ConfigFingerprint,
	)
	if err != nil {
		return fmt.Errorf("create load job %q: %w", job.ID, err)
	}
	return nil
}

func (store *Store) CreateRunningJob(ctx context.Context, job Job) error {
	if err := validateJob(job); err != nil {
		return err
	}
	_, err := store.database.Exec(
		ctx,
		`INSERT INTO agefreighter_meta.load_job (
			job_id, name, source_type, load_mode, target_backend,
			target_schema, target_graph,
			config_fingerprint, status, started_at, updated_at
		) VALUES (
			$1::uuid, $2, $3, $4, $5, $6, $7, $8, 'running',
			clock_timestamp(), clock_timestamp()
		)`,
		job.ID,
		job.Name,
		job.SourceType,
		job.LoadMode,
		resolvedTargetBackend(job),
		job.TargetSchema,
		job.TargetGraph,
		job.ConfigFingerprint,
	)
	if err != nil {
		return fmt.Errorf("create running load job %q: %w", job.ID, err)
	}
	return nil
}

func (store *Store) CreateRunningJobIfCurrent(
	ctx context.Context,
	job Job,
) (bool, error) {
	if err := validateJob(job); err != nil {
		return false, err
	}
	tag, err := store.database.Exec(
		ctx,
		`INSERT INTO agefreighter_meta.load_job (
			job_id, name, source_type, load_mode, target_backend,
			target_schema, target_graph,
			config_fingerprint, status, started_at, updated_at
		)
		SELECT
			$1::uuid, $2, $3, $4, $5, $6, $7, $8, 'running',
			clock_timestamp(), clock_timestamp()
		WHERE (
			SELECT COALESCE(MIN(version), 0) = 1
			   AND COALESCE(MAX(version), 0) = $9
			   AND COUNT(*) = $9
			FROM agefreighter_meta.schema_migration
		)`,
		job.ID,
		job.Name,
		job.SourceType,
		job.LoadMode,
		resolvedTargetBackend(job),
		job.TargetSchema,
		job.TargetGraph,
		job.ConfigFingerprint,
		schemaVersion,
	)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) &&
			(pgErr.Code == "42P01" || pgErr.Code == "3F000") {
			return false, nil
		}
		return false, fmt.Errorf("create running load job %q: %w", job.ID, err)
	}
	return tag.RowsAffected() == 1, nil
}

func (store *Store) GetJob(ctx context.Context, jobID string) (Job, error) {
	if err := validateJobID(jobID); err != nil {
		return Job{}, err
	}
	var job Job
	err := store.database.QueryRow(
		ctx,
		`SELECT
			job_id::text, name, source_type, load_mode,
			COALESCE(to_jsonb(job)->>'target_backend', 'apache-age'),
			COALESCE(to_jsonb(job)->>'target_schema', ''), target_graph,
			backup_graph_name, config_fingerprint::text, status,
			COALESCE(graph_generation_id, 0),
			next_batch_id, resume_token, committed_rows, committed_bytes,
			rejected_rows, source_rejected_rows, error_message, created_at, started_at, updated_at,
			completed_at, backup_cleaned_at
		 FROM agefreighter_meta.load_job AS job
		 WHERE job.job_id = $1::uuid`,
		jobID,
	).Scan(
		&job.ID,
		&job.Name,
		&job.SourceType,
		&job.LoadMode,
		&job.TargetBackend,
		&job.TargetSchema,
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
	return store.completeJob(ctx, jobID, nil)
}

func (store *Store) CompleteJobWithTelemetry(
	ctx context.Context,
	jobID string,
	telemetry ConnectorTelemetry,
) error {
	telemetry.JobID = jobID
	if err := validateConnectorTelemetry(telemetry); err != nil {
		return err
	}
	return store.completeJob(ctx, jobID, &telemetry)
}

func (store *Store) completeJob(
	ctx context.Context,
	jobID string,
	telemetry *ConnectorTelemetry,
) error {
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
	if telemetry != nil {
		if err := (&Store{database: tx}).PutConnectorTelemetry(ctx, *telemetry); err != nil {
			return err
		}
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
	return store.completeJobGeneration(ctx, jobID, graphGenerationID, nil)
}

func (store *Store) CompleteJobGenerationWithTelemetry(
	ctx context.Context,
	jobID string,
	graphGenerationID int64,
	telemetry ConnectorTelemetry,
) error {
	telemetry.JobID = jobID
	if err := validateConnectorTelemetry(telemetry); err != nil {
		return err
	}
	return store.completeJobGenerationWithTelemetry(
		ctx, jobID, graphGenerationID, telemetry,
	)
}

func (store *Store) completeJobGenerationWithTelemetry(
	ctx context.Context,
	jobID string,
	graphGenerationID int64,
	telemetry ConnectorTelemetry,
) error {
	if graphGenerationID <= 0 {
		return errors.New("graph generation ID must be positive")
	}
	var (
		status                                       JobStatus
		boundGenerationID                            int64
		unresolved                                   bool
		telemetryRows, generationRows, completedRows int64
	)
	err := store.database.QueryRow(ctx, `
		WITH
		current_state AS MATERIALIZED (
			SELECT job.status, COALESCE(job.graph_generation_id, 0) AS generation_id,
			       EXISTS (
			         SELECT 1
			         FROM agefreighter_meta.load_batch batch
			         WHERE batch.job_id = job.job_id
			           AND batch.batch_id = job.next_batch_id
			           AND batch.status IN ('running', 'failed')
			       ) AS unresolved
			FROM agefreighter_meta.load_job job
			WHERE job.job_id = $1::uuid
			FOR UPDATE
		),
		eligible AS MATERIALIZED (
			SELECT generation.graph_generation_id
			FROM current_state state
			JOIN agefreighter_meta.graph_generation generation
			  ON generation.graph_generation_id = $2
			 AND generation.job_id = $1::uuid
			 AND generation.state = 'loading'
			WHERE state.status = 'running'
			  AND state.generation_id = $2
			  AND NOT state.unresolved
			FOR UPDATE OF generation
		),
		stored_telemetry AS (
			INSERT INTO agefreighter_meta.connector_telemetry (
				job_id, connector, pages, request_charge,
				failed_request_attempts, throttled_requests, continuation_digest
			)
			SELECT $1::uuid, $3, $4, $5, $6, $7, $8
			FROM eligible
			ON CONFLICT (job_id) DO UPDATE SET
				connector = EXCLUDED.connector
			WHERE agefreighter_meta.connector_telemetry.connector = EXCLUDED.connector
			  AND agefreighter_meta.connector_telemetry.pages = EXCLUDED.pages
			  AND agefreighter_meta.connector_telemetry.request_charge =
					EXCLUDED.request_charge
			  AND agefreighter_meta.connector_telemetry.failed_request_attempts =
					EXCLUDED.failed_request_attempts
			  AND agefreighter_meta.connector_telemetry.throttled_requests =
					EXCLUDED.throttled_requests
			  AND agefreighter_meta.connector_telemetry.continuation_digest =
					EXCLUDED.continuation_digest
			RETURNING job_id
		),
		activated AS (
			UPDATE agefreighter_meta.graph_generation generation
			SET state = 'active', updated_at = clock_timestamp()
			FROM eligible, stored_telemetry
			WHERE generation.graph_generation_id = eligible.graph_generation_id
			RETURNING generation.graph_generation_id
		),
		completed AS (
			UPDATE agefreighter_meta.load_job job
			SET status = 'committed', error_message = '',
			    completed_at = clock_timestamp(), updated_at = clock_timestamp()
			FROM activated
			WHERE job.job_id = $1::uuid
			  AND job.status = 'running'
			RETURNING job.job_id
		)
		SELECT state.status, state.generation_id, state.unresolved,
		       (SELECT count(*) FROM stored_telemetry),
		       (SELECT count(*) FROM activated),
		       (SELECT count(*) FROM completed)
		FROM current_state state`,
		pgx.QueryExecModeExec,
		jobID,
		graphGenerationID,
		telemetry.Connector,
		telemetry.Pages,
		telemetry.RequestCharge,
		telemetry.FailedRequestAttempts,
		telemetry.ThrottledRequests,
		telemetry.ContinuationDigest,
	).Scan(
		&status,
		&boundGenerationID,
		&unresolved,
		&telemetryRows,
		&generationRows,
		&completedRows,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("%w: load job %q", ErrNotFound, jobID)
	}
	if err != nil {
		return fmt.Errorf("complete load generation: %w", err)
	}
	if status != JobRunning || boundGenerationID != graphGenerationID ||
		unresolved || telemetryRows != 1 || generationRows != 1 ||
		completedRows != 1 {
		return fmt.Errorf("%w: load job generation is not ready to complete", ErrConflict)
	}
	return nil
}

func (store *Store) completeJobGeneration(
	ctx context.Context,
	jobID string,
	graphGenerationID int64,
	telemetry *ConnectorTelemetry,
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
	if telemetry != nil {
		if err := (&Store{database: tx}).PutConnectorTelemetry(ctx, *telemetry); err != nil {
			return err
		}
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
