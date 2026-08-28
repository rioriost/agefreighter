package meta

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/jackc/pgx/v5"
)

const MaxReadLimit = 1000

type RejectSummary struct {
	ErrorClass string `json:"errorClass"`
	Count      int64  `json:"count"`
}

type RejectSummaryPage struct {
	Summaries   []RejectSummary `json:"summaries"`
	ScannedRows int             `json:"scannedRows"`
	Truncated   bool            `json:"truncated"`
}

type RetainedBackup struct {
	JobID                   string `json:"jobId"`
	TargetGraph             string `json:"targetGraph"`
	BackupGraphName         string `json:"backupGraphName"`
	ActiveGraphGenerationID int64  `json:"activeGraphGenerationId"`
	BackupGraphOID          uint32 `json:"backupGraphOid"`
}

type rowsQuerier interface {
	Query(context.Context, string, ...any) (pgx.Rows, error)
}

func (store *Store) ListJobs(ctx context.Context, limit int) ([]Job, error) {
	rows, err := store.queryBounded(
		ctx,
		`SELECT
			job_id::text, name, source_type, load_mode, target_graph,
			backup_graph_name, config_fingerprint::text, status,
			COALESCE(graph_generation_id, 0),
			next_batch_id, resume_token, committed_rows, committed_bytes,
			rejected_rows, source_rejected_rows, error_message,
			created_at, started_at, updated_at, completed_at, backup_cleaned_at
		 FROM agefreighter_meta.load_job
		 ORDER BY created_at DESC, job_id
		 LIMIT $1`,
		limit,
	)
	if err != nil {
		return nil, fmt.Errorf("list load jobs: %w", err)
	}
	defer rows.Close()
	values := make([]Job, 0, limit)
	for rows.Next() {
		value, scanErr := scanJob(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		values = append(values, value)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list load jobs: %w", err)
	}
	return values, nil
}

func (store *Store) ListGraphGenerations(
	ctx context.Context,
	limit int,
) ([]GraphGeneration, error) {
	rows, err := store.queryBounded(
		ctx,
		graphGenerationSelect+`
		 ORDER BY graph_generation_id DESC
		 LIMIT $1`,
		limit,
	)
	if err != nil {
		return nil, fmt.Errorf("list graph generations: %w", err)
	}
	defer rows.Close()
	values := make([]GraphGeneration, 0, limit)
	for rows.Next() {
		value, scanErr := scanGraphGeneration(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		values = append(values, value)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list graph generations: %w", err)
	}
	return values, nil
}

func (store *Store) ListLabelGenerations(
	ctx context.Context,
	graphGenerationID int64,
	limit int,
) ([]LabelGeneration, error) {
	if graphGenerationID <= 0 {
		return nil, errors.New("graph generation ID must be positive")
	}
	rows, err := store.queryBounded(
		ctx,
		`SELECT
			label_generation_id, graph_generation_id, label_name, kind,
			graph_namespace_oid, label_id, relation_oid, sequence_oid,
			mapping_generation, created_at, updated_at
		 FROM agefreighter_meta.label_generation
		 WHERE graph_generation_id = $1
		 ORDER BY label_name, mapping_generation DESC, label_generation_id
		 LIMIT $2`,
		limit,
		graphGenerationID,
	)
	if err != nil {
		return nil, fmt.Errorf("list label generations: %w", err)
	}
	defer rows.Close()
	values := make([]LabelGeneration, 0, limit)
	for rows.Next() {
		var value LabelGeneration
		var kind string
		if err := rows.Scan(
			&value.ID,
			&value.GraphGenerationID,
			&value.LabelName,
			&kind,
			&value.GraphNamespaceOID,
			&value.LabelID,
			&value.RelationOID,
			&value.SequenceOID,
			&value.MappingGeneration,
			&value.CreatedAt,
			&value.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("read label generation: %w", err)
		}
		if len(kind) != 1 {
			return nil, fmt.Errorf("stored label kind %q is invalid", kind)
		}
		value.Kind = LabelKind(kind[0])
		values = append(values, value)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list label generations: %w", err)
	}
	return values, nil
}

func (store *Store) ListBatches(
	ctx context.Context,
	jobID string,
	limit int,
) ([]BatchAttempt, error) {
	if err := validateJobID(jobID); err != nil {
		return nil, err
	}
	rows, err := store.queryBounded(
		ctx,
		`SELECT
			job_id::text, batch_id, attempt, status, rows, bytes, rejected_rows,
			first_resource, first_line, first_byte_offset, first_token,
			last_resource, last_line, last_byte_offset, last_token,
			error_message, started_at, finished_at
		 FROM agefreighter_meta.load_batch
		 WHERE job_id = $1::uuid
		 ORDER BY batch_id DESC, attempt DESC
		 LIMIT $2`,
		limit,
		jobID,
	)
	if err != nil {
		return nil, fmt.Errorf("list load batches: %w", err)
	}
	defer rows.Close()
	values := make([]BatchAttempt, 0, limit)
	for rows.Next() {
		value, scanErr := scanBatch(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		values = append(values, value)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list load batches: %w", err)
	}
	return values, nil
}

func (store *Store) ListRejectSummaries(
	ctx context.Context,
	jobID string,
	limit int,
) (RejectSummaryPage, error) {
	if err := validateJobID(jobID); err != nil {
		return RejectSummaryPage{}, err
	}
	rows, err := store.queryBounded(
		ctx,
		`SELECT error_class
		 FROM agefreighter_meta.reject_record
		 WHERE job_id = $1::uuid
		 ORDER BY batch_id, attempt, resume_token
		 LIMIT $2 + 1`,
		limit,
		jobID,
	)
	if err != nil {
		return RejectSummaryPage{}, fmt.Errorf("list reject summaries: %w", err)
	}
	defer rows.Close()
	counts := make(map[string]int64)
	scanned := 0
	truncated := false
	for rows.Next() {
		var errorClass string
		if err := rows.Scan(&errorClass); err != nil {
			return RejectSummaryPage{}, fmt.Errorf("read reject summary: %w", err)
		}
		if scanned == limit {
			truncated = true
			continue
		}
		counts[errorClass]++
		scanned++
	}
	if err := rows.Err(); err != nil {
		return RejectSummaryPage{}, fmt.Errorf("list reject summaries: %w", err)
	}
	values := make([]RejectSummary, 0, len(counts))
	for errorClass, count := range counts {
		values = append(values, RejectSummary{
			ErrorClass: errorClass,
			Count:      count,
		})
	}
	slices.SortFunc(values, func(left, right RejectSummary) int {
		if left.Count != right.Count {
			if left.Count > right.Count {
				return -1
			}
			return 1
		}
		return strings.Compare(left.ErrorClass, right.ErrorClass)
	})
	return RejectSummaryPage{
		Summaries:   values,
		ScannedRows: scanned,
		Truncated:   truncated,
	}, nil
}

func (store *Store) ListRetainedBackups(
	ctx context.Context,
	limit int,
) ([]RetainedBackup, error) {
	rows, err := store.queryBounded(
		ctx,
		`SELECT
			j.job_id::text, j.target_graph, j.backup_graph_name,
			g.graph_generation_id, g.replaces_graph_oid
		 FROM agefreighter_meta.load_job j
		 JOIN agefreighter_meta.graph_generation g
		   ON g.graph_generation_id = j.graph_generation_id
		 WHERE j.load_mode = 'replace'
		   AND j.status = 'committed'
		   AND j.backup_graph_name <> ''
		   AND j.backup_cleaned_at IS NULL
		   AND g.replaces_graph_oid IS NOT NULL
		 ORDER BY j.completed_at DESC, j.job_id
		 LIMIT $1`,
		limit,
	)
	if err != nil {
		return nil, fmt.Errorf("list retained backups: %w", err)
	}
	defer rows.Close()
	values := make([]RetainedBackup, 0, limit)
	for rows.Next() {
		var value RetainedBackup
		if err := rows.Scan(
			&value.JobID,
			&value.TargetGraph,
			&value.BackupGraphName,
			&value.ActiveGraphGenerationID,
			&value.BackupGraphOID,
		); err != nil {
			return nil, fmt.Errorf("read retained backup: %w", err)
		}
		values = append(values, value)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list retained backups: %w", err)
	}
	return values, nil
}

func (store *Store) queryBounded(
	ctx context.Context,
	statement string,
	limit int,
	arguments ...any,
) (pgx.Rows, error) {
	if store == nil || store.database == nil {
		return nil, errors.New("metadata store is required")
	}
	if limit <= 0 || limit > MaxReadLimit {
		return nil, fmt.Errorf("read limit must be within 1..%d", MaxReadLimit)
	}
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		return nil, errors.New("metadata read context requires a deadline")
	}
	database, ok := store.database.(rowsQuerier)
	if !ok {
		return nil, errors.New("metadata database does not support row queries")
	}
	arguments = append(arguments, limit)
	return database.Query(ctx, statement, arguments...)
}

func scanJob(row rowScanner) (Job, error) {
	var job Job
	err := row.Scan(
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
		return Job{}, ErrNotFound
	}
	if err != nil {
		return Job{}, fmt.Errorf("read load job: %w", err)
	}
	return job, nil
}
