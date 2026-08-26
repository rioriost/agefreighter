package meta

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"
)

type ReplacePromotion struct {
	Job                Job
	NewGeneration      GraphGeneration
	PreviousGeneration *GraphGeneration
}

type BackupCleanup struct {
	Job            Job
	Generation     GraphGeneration
	AlreadyCleaned bool
}

func (store *Store) NextGraphGeneration(
	ctx context.Context,
	targetGraph string,
) (uint64, error) {
	if strings.TrimSpace(targetGraph) == "" {
		return 0, errors.New("target graph is required")
	}
	var generation uint64
	if err := store.database.QueryRow(
		ctx,
		`SELECT COALESCE(MAX(generation), 0) + 1
		 FROM agefreighter_meta.graph_generation
		 WHERE graph_name = $1
		   AND state = 'active'`,
		targetGraph,
	).Scan(&generation); err != nil {
		return 0, fmt.Errorf("select next graph generation: %w", err)
	}
	return generation, nil
}

func (store *Store) PrepareReplacePromotion(
	ctx context.Context,
	jobID string,
	graphGenerationID int64,
) (ReplacePromotion, error) {
	if err := validateJobID(jobID); err != nil {
		return ReplacePromotion{}, err
	}
	if graphGenerationID <= 0 {
		return ReplacePromotion{}, errors.New("graph generation ID must be positive")
	}

	var promotion ReplacePromotion
	var nextBatchID uint64
	err := store.database.QueryRow(
		ctx,
		`SELECT
			job_id::text, load_mode, target_graph, status,
			COALESCE(graph_generation_id, 0), next_batch_id,
			backup_graph_name
		 FROM agefreighter_meta.load_job
		 WHERE job_id = $1::uuid
		 FOR UPDATE`,
		jobID,
	).Scan(
		&promotion.Job.ID,
		&promotion.Job.LoadMode,
		&promotion.Job.TargetGraph,
		&promotion.Job.Status,
		&promotion.Job.GraphGenerationID,
		&nextBatchID,
		&promotion.Job.BackupGraphName,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return ReplacePromotion{}, fmt.Errorf("%w: load job %q", ErrNotFound, jobID)
	}
	if err != nil {
		return ReplacePromotion{}, fmt.Errorf("lock replacement job: %w", err)
	}
	if promotion.Job.LoadMode != "replace" ||
		promotion.Job.Status != JobRunning ||
		promotion.Job.GraphGenerationID != graphGenerationID ||
		promotion.Job.BackupGraphName != "" {
		return ReplacePromotion{}, fmt.Errorf(
			"%w: replacement job is not ready for promotion",
			ErrConflict,
		)
	}

	var unresolved int
	if err := store.database.QueryRow(
		ctx,
		`SELECT COUNT(*)
		 FROM agefreighter_meta.load_batch
		 WHERE job_id = $1::uuid
		   AND batch_id = $2
		   AND status IN ('running', 'failed')`,
		jobID,
		nextBatchID,
	).Scan(&unresolved); err != nil {
		return ReplacePromotion{}, fmt.Errorf("check unresolved replacement batches: %w", err)
	}
	if unresolved != 0 {
		return ReplacePromotion{}, fmt.Errorf(
			"%w: replacement job has unresolved attempts",
			ErrConflict,
		)
	}

	promotion.NewGeneration, err = scanGraphGeneration(store.database.QueryRow(
		ctx,
		graphGenerationSelect+`
		 WHERE graph_generation_id = $1
		   AND job_id = $2::uuid
		 FOR UPDATE`,
		graphGenerationID,
		jobID,
	))
	if err != nil {
		return ReplacePromotion{}, fmt.Errorf("lock replacement generation: %w", err)
	}
	if promotion.NewGeneration.State != GenerationLoading ||
		promotion.NewGeneration.ReplacesGraphOID == 0 {
		return ReplacePromotion{}, fmt.Errorf(
			"%w: replacement generation is not loading",
			ErrConflict,
		)
	}

	previous, previousErr := scanGraphGeneration(store.database.QueryRow(
		ctx,
		graphGenerationSelect+`
		 WHERE graph_name = $1
		   AND state = 'active'
		   AND graph_generation_id <> $2
		 FOR UPDATE`,
		promotion.Job.TargetGraph,
		graphGenerationID,
	))
	if previousErr == nil {
		if previous.GraphOID != promotion.NewGeneration.ReplacesGraphOID {
			return ReplacePromotion{}, fmt.Errorf(
				"%w: active generation OID %d does not match replacement OID %d",
				ErrGenerationMismatch,
				previous.GraphOID,
				promotion.NewGeneration.ReplacesGraphOID,
			)
		}
		promotion.PreviousGeneration = &previous
	} else if !errors.Is(previousErr, ErrNotFound) {
		return ReplacePromotion{}, previousErr
	}
	return promotion, nil
}

func (store *Store) CompleteReplacePromotion(
	ctx context.Context,
	promotion ReplacePromotion,
	targetGraph string,
	backupGraph string,
) error {
	if promotion.Job.ID == "" ||
		promotion.NewGeneration.ID <= 0 ||
		strings.TrimSpace(targetGraph) == "" ||
		strings.TrimSpace(backupGraph) == "" ||
		targetGraph == backupGraph {
		return errors.New("valid replacement promotion metadata is required")
	}
	if promotion.PreviousGeneration != nil {
		tag, err := store.database.Exec(
			ctx,
			`UPDATE agefreighter_meta.graph_generation
			 SET graph_name = $2, state = 'retired',
			     updated_at = clock_timestamp()
			 WHERE graph_generation_id = $1
			   AND graph_name = $3
			   AND graph_oid = $4
			   AND state = 'active'`,
			promotion.PreviousGeneration.ID,
			backupGraph,
			targetGraph,
			promotion.NewGeneration.ReplacesGraphOID,
		)
		if err != nil {
			return fmt.Errorf("retire replaced graph generation: %w", err)
		}
		if err := rowsAffectedOne(tag, "retire replaced graph generation"); err != nil {
			return err
		}
	}

	tag, err := store.database.Exec(
		ctx,
		`UPDATE agefreighter_meta.graph_generation
		 SET graph_name = $2, state = 'active',
		     updated_at = clock_timestamp()
		 WHERE graph_generation_id = $1
		   AND job_id = $3::uuid
		   AND graph_name = $4
		   AND graph_oid = $5
		   AND replaces_graph_oid = $6
		   AND state = 'loading'`,
		promotion.NewGeneration.ID,
		targetGraph,
		promotion.Job.ID,
		promotion.NewGeneration.GraphName,
		promotion.NewGeneration.GraphOID,
		promotion.NewGeneration.ReplacesGraphOID,
	)
	if err != nil {
		return fmt.Errorf("activate replacement graph generation: %w", err)
	}
	if err := rowsAffectedOne(tag, "activate replacement graph generation"); err != nil {
		return err
	}

	tag, err = store.database.Exec(
		ctx,
		`UPDATE agefreighter_meta.load_job
		 SET status = 'committed', backup_graph_name = $2,
		     error_message = '', completed_at = clock_timestamp(),
		     updated_at = clock_timestamp()
		 WHERE job_id = $1::uuid
		   AND load_mode = 'replace'
		   AND status = 'running'
		   AND graph_generation_id = $3
		   AND backup_graph_name = ''`,
		promotion.Job.ID,
		backupGraph,
		promotion.NewGeneration.ID,
	)
	if err != nil {
		return fmt.Errorf("complete replacement job: %w", err)
	}
	return rowsAffectedOne(tag, "complete replacement job")
}

func (store *Store) PrepareBackupCleanup(
	ctx context.Context,
	jobID string,
) (BackupCleanup, error) {
	if err := validateJobID(jobID); err != nil {
		return BackupCleanup{}, err
	}
	var cleanup BackupCleanup
	err := store.database.QueryRow(
		ctx,
		`SELECT
			job_id::text, load_mode, target_graph, status,
			COALESCE(graph_generation_id, 0), backup_graph_name,
			backup_cleaned_at
		 FROM agefreighter_meta.load_job
		 WHERE job_id = $1::uuid
		 FOR UPDATE`,
		jobID,
	).Scan(
		&cleanup.Job.ID,
		&cleanup.Job.LoadMode,
		&cleanup.Job.TargetGraph,
		&cleanup.Job.Status,
		&cleanup.Job.GraphGenerationID,
		&cleanup.Job.BackupGraphName,
		&cleanup.Job.BackupCleanedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return BackupCleanup{}, fmt.Errorf("%w: load job %q", ErrNotFound, jobID)
	}
	if err != nil {
		return BackupCleanup{}, fmt.Errorf("lock backup cleanup job: %w", err)
	}
	if cleanup.Job.LoadMode != "replace" ||
		cleanup.Job.Status != JobCommitted ||
		cleanup.Job.BackupGraphName == "" {
		return BackupCleanup{}, fmt.Errorf(
			"%w: job has no committed replacement backup",
			ErrConflict,
		)
	}
	if cleanup.Job.BackupCleanedAt != nil {
		cleanup.AlreadyCleaned = true
		return cleanup, nil
	}

	cleanup.Generation, err = scanGraphGeneration(store.database.QueryRow(
		ctx,
		graphGenerationSelect+`
		 WHERE graph_generation_id = $1
		   AND job_id = $2::uuid
		 FOR UPDATE`,
		cleanup.Job.GraphGenerationID,
		jobID,
	))
	if err != nil {
		return BackupCleanup{}, fmt.Errorf("lock active replacement generation: %w", err)
	}
	if cleanup.Generation.State != GenerationActive ||
		cleanup.Generation.GraphName != cleanup.Job.TargetGraph ||
		cleanup.Generation.ReplacesGraphOID == 0 ||
		cleanup.Job.TargetGraph == cleanup.Job.BackupGraphName {
		return BackupCleanup{}, fmt.Errorf(
			"%w: replacement backup is not safe to clean",
			ErrConflict,
		)
	}
	return cleanup, nil
}

func (store *Store) CompleteBackupCleanup(
	ctx context.Context,
	cleanup BackupCleanup,
) error {
	if cleanup.Job.ID == "" || cleanup.AlreadyCleaned {
		return errors.New("pending backup cleanup metadata is required")
	}
	tag, err := store.database.Exec(
		ctx,
		`UPDATE agefreighter_meta.load_job
		 SET backup_cleaned_at = clock_timestamp(),
		     updated_at = clock_timestamp()
		 WHERE job_id = $1::uuid
		   AND status = 'committed'
		   AND load_mode = 'replace'
		   AND graph_generation_id = $2
		   AND backup_graph_name = $3
		   AND backup_cleaned_at IS NULL`,
		cleanup.Job.ID,
		cleanup.Generation.ID,
		cleanup.Job.BackupGraphName,
	)
	if err != nil {
		return fmt.Errorf("record replacement backup cleanup: %w", err)
	}
	return rowsAffectedOne(tag, "record replacement backup cleanup")
}

const graphGenerationSelect = `SELECT
	graph_generation_id, job_id::text, graph_name, graph_oid, namespace_oid,
	COALESCE(replaces_graph_oid, 0), generation, state, created_at, updated_at
 FROM agefreighter_meta.graph_generation`

type rowScanner interface {
	Scan(...any) error
}

func scanGraphGeneration(row rowScanner) (GraphGeneration, error) {
	var value GraphGeneration
	if err := row.Scan(
		&value.ID,
		&value.JobID,
		&value.GraphName,
		&value.GraphOID,
		&value.NamespaceOID,
		&value.ReplacesGraphOID,
		&value.Generation,
		&value.State,
		&value.CreatedAt,
		&value.UpdatedAt,
	); errors.Is(err, pgx.ErrNoRows) {
		return GraphGeneration{}, ErrNotFound
	} else if err != nil {
		return GraphGeneration{}, fmt.Errorf("read graph generation: %w", err)
	}
	return value, nil
}
