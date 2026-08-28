package meta

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/jackc/pgx/v5"
)

func (store *Store) PutJobVerification(ctx context.Context, value JobVerification) error {
	if err := validateJobID(value.JobID); err != nil {
		return err
	}
	if err := validateFingerprint(value.SubmittedConfigFingerprint); err != nil {
		return fmt.Errorf("submitted configuration: %w", err)
	}
	if err := validateFingerprint(value.ResolvedMappingFingerprint); err != nil {
		return fmt.Errorf("resolved mapping: %w", err)
	}
	if len(value.ResolvedMappingSummary) == 0 || !jsonValidObject(value.ResolvedMappingSummary) {
		return errors.New("resolved mapping summary must be a JSON object")
	}
	tag, err := store.database.Exec(ctx, `
		INSERT INTO agefreighter_meta.job_verification (
			job_id, submitted_config_fingerprint,
			resolved_mapping_fingerprint, resolved_mapping_summary
		) VALUES ($1::uuid, $2, $3, $4::jsonb)
		ON CONFLICT (job_id) DO UPDATE SET
			submitted_config_fingerprint = EXCLUDED.submitted_config_fingerprint,
			resolved_mapping_fingerprint = EXCLUDED.resolved_mapping_fingerprint,
			resolved_mapping_summary = EXCLUDED.resolved_mapping_summary
		WHERE agefreighter_meta.job_verification.submitted_config_fingerprint =
				EXCLUDED.submitted_config_fingerprint
		  AND agefreighter_meta.job_verification.resolved_mapping_fingerprint =
				EXCLUDED.resolved_mapping_fingerprint
		  AND agefreighter_meta.job_verification.resolved_mapping_summary =
				EXCLUDED.resolved_mapping_summary`,
		value.JobID,
		value.SubmittedConfigFingerprint,
		value.ResolvedMappingFingerprint,
		value.ResolvedMappingSummary,
	)
	if err != nil {
		return fmt.Errorf("store job verification metadata: %w", err)
	}
	return rowsAffectedOne(tag, "store job verification metadata")
}

func (store *Store) GetJobVerification(
	ctx context.Context,
	jobID string,
) (JobVerification, error) {
	if err := validateJobID(jobID); err != nil {
		return JobVerification{}, err
	}
	var value JobVerification
	err := store.database.QueryRow(ctx, `
		SELECT job_id::text, submitted_config_fingerprint::text,
		       resolved_mapping_fingerprint::text, resolved_mapping_summary
		FROM agefreighter_meta.job_verification
		WHERE job_id = $1::uuid`, jobID).Scan(
		&value.JobID,
		&value.SubmittedConfigFingerprint,
		&value.ResolvedMappingFingerprint,
		&value.ResolvedMappingSummary,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return JobVerification{}, ErrNotFound
	}
	if err != nil {
		return JobVerification{}, fmt.Errorf("read job verification metadata: %w", err)
	}
	return value, nil
}

func (store *Store) EnsureLabelCounters(
	ctx context.Context,
	jobID string,
	labels []LabelGeneration,
) error {
	if err := validateJobID(jobID); err != nil {
		return err
	}
	ordered := slices.Clone(labels)
	slices.SortFunc(ordered, func(left, right LabelGeneration) int {
		if left.ID < right.ID {
			return -1
		}
		if left.ID > right.ID {
			return 1
		}
		return 0
	})
	for index, label := range ordered {
		if label.ID <= 0 || (label.Kind != VertexLabel && label.Kind != EdgeLabel) {
			return errors.New("valid label generation is required")
		}
		if index > 0 && ordered[index-1].ID == label.ID {
			return errors.New("duplicate label generation")
		}
	}
	tx, err := store.database.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin label counter initialization: %w", err)
	}
	defer rollbackWithTimeout(ctx, tx, defaultMigrationTimeout)

	var (
		status                                                     JobStatus
		loadMode                                                   string
		committedRows, committedBytes, rejectedRows, sourceRejects int64
		hasCommittedBatch                                          bool
	)
	err = tx.QueryRow(ctx, `
		SELECT status, load_mode, committed_rows, committed_bytes,
		       rejected_rows, source_rejected_rows,
		       EXISTS (
		         SELECT 1
		         FROM agefreighter_meta.load_batch batch
		         WHERE batch.job_id = job.job_id
		           AND batch.status = 'committed'
		       )
		FROM agefreighter_meta.load_job job
		WHERE job_id = $1::uuid
		FOR UPDATE`, jobID).Scan(
		&status,
		&loadMode,
		&committedRows,
		&committedBytes,
		&rejectedRows,
		&sourceRejects,
		&hasCommittedBatch,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("%w: load job %q", ErrNotFound, jobID)
	}
	if err != nil {
		return fmt.Errorf("lock label counter job: %w", err)
	}
	if status != JobRunning {
		return fmt.Errorf("%w: label counter job is not running", ErrConflict)
	}
	historicalActivity := hasCommittedBatch || committedRows != 0 ||
		committedBytes != 0 || rejectedRows != 0 || sourceRejects != 0
	completeness := CounterIncomplete
	provenance := CounterProvenanceBaselineUnavailable
	if historicalActivity {
		provenance = CounterProvenanceLegacyResume
	} else if loadMode != "append" && loadMode != "upsert" {
		completeness = CounterComplete
		provenance = CounterProvenanceLifecycle
	}
	var initialRows any
	if completeness == CounterComplete {
		initialRows = int64(0)
	}

	for _, label := range ordered {
		tag, err := tx.Exec(ctx, `
			INSERT INTO agefreighter_meta.job_label_counter (
				job_id, label_generation_id, kind,
				counter_completeness, counter_provenance,
				accepted_rows, committed_rows, rejected_rows
			) VALUES ($1::uuid, $2, $3, $4, $5, $6, $6, $6)
			ON CONFLICT (job_id, label_generation_id) DO NOTHING`,
			jobID, label.ID, string(label.Kind),
			string(completeness), string(provenance), initialRows,
		)
		if err != nil {
			return fmt.Errorf("initialize label counter: %w", err)
		}
		if tag.RowsAffected() == 0 {
			var (
				kind               LabelKind
				storedCompleteness CounterCompleteness
				storedProvenance   CounterProvenance
			)
			if err := tx.QueryRow(ctx, `
				SELECT kind, counter_completeness, counter_provenance
				FROM agefreighter_meta.job_label_counter
				WHERE job_id = $1::uuid AND label_generation_id = $2`,
				jobID, label.ID,
			).Scan(&kind, &storedCompleteness, &storedProvenance); err != nil {
				return fmt.Errorf("read initialized label counter: %w", err)
			}
			if kind != label.Kind {
				return fmt.Errorf("%w: label counter kind changed", ErrConflict)
			}
			if storedCompleteness != CounterComplete &&
				storedCompleteness != CounterIncomplete {
				return fmt.Errorf("%w: invalid label counter completeness", ErrConflict)
			}
			switch storedProvenance {
			case CounterProvenanceLifecycle,
				CounterProvenanceLegacyResume,
				CounterProvenanceBaselineUnavailable:
			default:
				return fmt.Errorf("%w: invalid label counter provenance", ErrConflict)
			}
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit label counter initialization: %w", err)
	}
	return nil
}

func (store *Store) ListLabelCounters(
	ctx context.Context,
	jobID string,
	limit int,
) ([]LabelCounter, error) {
	if err := validateJobID(jobID); err != nil {
		return nil, err
	}
	if limit <= 0 || limit > MaxReadLimit {
		return nil, fmt.Errorf("label counter limit must be within 1..%d", MaxReadLimit)
	}
	queryer, ok := store.database.(rowsQuerier)
	if !ok {
		return nil, errors.New("metadata database does not support row queries")
	}
	rows, err := queryer.Query(ctx, `
		SELECT job_id::text, label_generation_id, kind,
		       counter_completeness, counter_provenance,
		       accepted_rows, committed_rows, committed_bytes, rejected_rows
		FROM agefreighter_meta.job_label_counter
		WHERE job_id = $1::uuid
		ORDER BY label_generation_id
		LIMIT $2`, jobID, limit)
	if err != nil {
		return nil, fmt.Errorf("list label counters: %w", err)
	}
	defer rows.Close()
	values := make([]LabelCounter, 0)
	for rows.Next() {
		var value LabelCounter
		if err := rows.Scan(
			&value.JobID, &value.LabelGenerationID, &value.Kind,
			&value.Completeness, &value.Provenance,
			&value.AcceptedRows, &value.CommittedRows,
			&value.CommittedBytes, &value.RejectedRows,
		); err != nil {
			return nil, fmt.Errorf("scan label counter: %w", err)
		}
		values = append(values, value)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list label counters: %w", err)
	}
	return values, nil
}

func (store *Store) ListLabelCountersByID(
	ctx context.Context,
	jobID string,
	labelGenerationIDs []int64,
) ([]LabelCounter, error) {
	if err := validateJobID(jobID); err != nil {
		return nil, err
	}
	ids := slices.Clone(labelGenerationIDs)
	slices.Sort(ids)
	if len(ids) == 0 {
		return []LabelCounter{}, nil
	}
	if len(ids) > MaxReadLimit {
		return nil, fmt.Errorf("label counter limit must be within 1..%d", MaxReadLimit)
	}
	for index, id := range ids {
		if id <= 0 {
			return nil, errors.New("label counter generation IDs must be positive")
		}
		if index > 0 && ids[index-1] == id {
			return nil, errors.New("label counter generation IDs must be unique")
		}
	}
	queryer, ok := store.database.(rowsQuerier)
	if !ok {
		return nil, errors.New("metadata database does not support row queries")
	}
	rows, err := queryer.Query(ctx, `
		SELECT job_id::text, label_generation_id, kind,
		       counter_completeness, counter_provenance,
		       accepted_rows, committed_rows, committed_bytes, rejected_rows
		FROM agefreighter_meta.job_label_counter
		WHERE job_id = $1::uuid
		  AND label_generation_id = ANY($2::bigint[])
		ORDER BY label_generation_id
		LIMIT $3`, jobID, ids, len(ids))
	if err != nil {
		return nil, fmt.Errorf("list expected label counters: %w", err)
	}
	defer rows.Close()
	values := make([]LabelCounter, 0, len(ids))
	for rows.Next() {
		var value LabelCounter
		if err := rows.Scan(
			&value.JobID, &value.LabelGenerationID, &value.Kind,
			&value.Completeness, &value.Provenance,
			&value.AcceptedRows, &value.CommittedRows,
			&value.CommittedBytes, &value.RejectedRows,
		); err != nil {
			return nil, fmt.Errorf("scan expected label counter: %w", err)
		}
		values = append(values, value)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list expected label counters: %w", err)
	}
	return values, nil
}

func jsonValidObject(value []byte) bool {
	trimmed := bytes.TrimSpace(value)
	return json.Valid(trimmed) &&
		len(trimmed) >= 2 && trimmed[0] == '{' && trimmed[len(trimmed)-1] == '}'
}

func (store *Store) GetUnclassifiedRejects(
	ctx context.Context,
	jobID string,
) (int64, error) {
	if err := validateJobID(jobID); err != nil {
		return 0, err
	}
	var count int64
	err := store.database.QueryRow(ctx, `
		SELECT rejected_rows
		FROM agefreighter_meta.job_unclassified_counter
		WHERE job_id = $1::uuid`, jobID).Scan(&count)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, ErrNotFound
	}
	if err != nil {
		return 0, fmt.Errorf("read unclassified reject counter: %w", err)
	}
	return count, nil
}
