package meta

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

func (store *Store) RegisterGraphGeneration(
	ctx context.Context,
	value GraphGeneration,
) (GraphGeneration, error) {
	if err := validateGraphGeneration(value); err != nil {
		return GraphGeneration{}, err
	}
	tx, existingTransaction := store.database.(pgx.Tx)
	ownsTransaction := !existingTransaction
	var err error
	if ownsTransaction {
		tx, err = store.database.Begin(ctx)
		if err != nil {
			return GraphGeneration{}, fmt.Errorf("begin graph generation registration: %w", err)
		}
		defer rollback(ctx, tx)
	}

	err = tx.QueryRow(
		ctx,
		`INSERT INTO agefreighter_meta.graph_generation (
			job_id, graph_name, graph_oid, namespace_oid, replaces_graph_oid,
			generation, state
		) VALUES ($1::uuid, $2, $3, $4, NULLIF($5, 0)::oid, $6, $7)
		RETURNING graph_generation_id, created_at, updated_at`,
		value.JobID,
		value.GraphName,
		value.GraphOID,
		value.NamespaceOID,
		value.ReplacesGraphOID,
		value.Generation,
		value.State,
	).Scan(&value.ID, &value.CreatedAt, &value.UpdatedAt)
	if err != nil {
		return GraphGeneration{}, fmt.Errorf(
			"register graph generation for job %q: %w",
			value.JobID,
			err,
		)
	}
	tag, err := tx.Exec(
		ctx,
		`UPDATE agefreighter_meta.load_job
		 SET graph_generation_id = $2,
		     updated_at = clock_timestamp()
		 WHERE job_id = $1::uuid
		   AND graph_generation_id IS NULL`,
		value.JobID,
		value.ID,
	)
	if err != nil {
		return GraphGeneration{}, fmt.Errorf("bind graph generation to job: %w", err)
	}
	if err := rowsAffectedOne(tag, "bind graph generation to job"); err != nil {
		return GraphGeneration{}, err
	}
	if ownsTransaction {
		if err := tx.Commit(ctx); err != nil {
			return GraphGeneration{}, fmt.Errorf("commit graph generation registration: %w", err)
		}
	}
	return value, nil
}

func (store *Store) AdmitGraphGeneration(
	ctx context.Context,
	jobID string,
	current GraphGeneration,
) (GraphGeneration, error) {
	if err := validateJobID(jobID); err != nil {
		return GraphGeneration{}, err
	}
	if err := validateGraphGeneration(current); err != nil {
		return GraphGeneration{}, err
	}
	stored, err := store.GraphGenerationForJob(ctx, jobID)
	if err != nil {
		return GraphGeneration{}, err
	}
	if stored.GraphName != current.GraphName ||
		stored.GraphOID != current.GraphOID ||
		stored.NamespaceOID != current.NamespaceOID ||
		stored.ReplacesGraphOID != current.ReplacesGraphOID ||
		stored.JobID != current.JobID ||
		stored.Generation != current.Generation ||
		stored.State != current.State ||
		stored.State == GenerationRetired {
		return GraphGeneration{}, fmt.Errorf(
			"%w: stored graph %q oid=%d namespace=%d, current graph %q oid=%d namespace=%d",
			ErrGenerationMismatch,
			stored.GraphName,
			stored.GraphOID,
			stored.NamespaceOID,
			current.GraphName,
			current.GraphOID,
			current.NamespaceOID,
		)
	}
	return stored, nil
}

func (store *Store) GraphGenerationForJob(
	ctx context.Context,
	jobID string,
) (GraphGeneration, error) {
	if err := validateJobID(jobID); err != nil {
		return GraphGeneration{}, err
	}
	var value GraphGeneration
	err := store.database.QueryRow(
		ctx,
		`SELECT
			g.graph_generation_id, g.job_id::text, g.graph_name,
			g.graph_oid, g.namespace_oid, COALESCE(g.replaces_graph_oid, 0),
			g.generation, g.state,
			g.created_at, g.updated_at
		 FROM agefreighter_meta.graph_generation g
		 JOIN agefreighter_meta.load_job j
		   ON j.graph_generation_id = g.graph_generation_id
		 WHERE j.job_id = $1::uuid`,
		jobID,
	).Scan(
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
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return GraphGeneration{}, fmt.Errorf(
			"%w: graph generation for job %q",
			ErrNotFound,
			jobID,
		)
	}
	if err != nil {
		return GraphGeneration{}, fmt.Errorf(
			"read graph generation for job %q: %w",
			jobID,
			err,
		)
	}
	return value, nil
}

func (store *Store) BindActiveGraphGeneration(
	ctx context.Context,
	jobID string,
	graphName string,
) (GraphGeneration, error) {
	if err := validateJobID(jobID); err != nil {
		return GraphGeneration{}, err
	}
	if graphName == "" {
		return GraphGeneration{}, errors.New("target graph name is required")
	}
	tx, err := store.database.Begin(ctx)
	if err != nil {
		return GraphGeneration{}, fmt.Errorf(
			"begin active graph generation binding: %w",
			err,
		)
	}
	defer rollback(ctx, tx)

	var value GraphGeneration
	err = tx.QueryRow(
		ctx,
		`SELECT
			graph_generation_id, job_id::text, graph_name,
			graph_oid, namespace_oid, COALESCE(replaces_graph_oid, 0),
			generation, state, created_at, updated_at
		 FROM agefreighter_meta.graph_generation
		 WHERE graph_name = $1
		   AND state = 'active'
		 FOR KEY SHARE`,
		graphName,
	).Scan(
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
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return GraphGeneration{}, fmt.Errorf(
			"%w: active graph generation %q",
			ErrNotFound,
			graphName,
		)
	}
	if err != nil {
		return GraphGeneration{}, fmt.Errorf(
			"read active graph generation %q: %w",
			graphName,
			err,
		)
	}
	tag, err := tx.Exec(
		ctx,
		`UPDATE agefreighter_meta.load_job
		 SET graph_generation_id = $2,
		     updated_at = clock_timestamp()
		 WHERE job_id = $1::uuid
		   AND target_graph = $3
		   AND load_mode IN ('append', 'upsert')
		   AND status = 'running'
		   AND graph_generation_id IS NULL`,
		jobID,
		value.ID,
		graphName,
	)
	if err != nil {
		return GraphGeneration{}, fmt.Errorf(
			"bind active graph generation to incremental job: %w",
			err,
		)
	}
	if err := rowsAffectedOne(tag, "bind active graph generation to incremental job"); err != nil {
		return GraphGeneration{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return GraphGeneration{}, fmt.Errorf(
			"commit active graph generation binding: %w",
			err,
		)
	}
	return value, nil
}

func (store *Store) SetGraphGenerationState(
	ctx context.Context,
	graphGenerationID int64,
	from GenerationState,
	to GenerationState,
) error {
	if graphGenerationID <= 0 {
		return errors.New("graph generation ID must be positive")
	}
	if from == to {
		return errors.New("graph generation transition must change state")
	}
	if !validGenerationTransition(from, to) {
		return fmt.Errorf("invalid graph generation transition from %q to %q", from, to)
	}
	tag, err := store.database.Exec(
		ctx,
		`UPDATE agefreighter_meta.graph_generation
		 SET state = $3, updated_at = clock_timestamp()
		 WHERE graph_generation_id = $1
		   AND state = $2`,
		graphGenerationID,
		from,
		to,
	)
	if err != nil {
		return fmt.Errorf("transition graph generation: %w", err)
	}
	return rowsAffectedOne(tag, "transition graph generation")
}

func validGenerationTransition(from, to GenerationState) bool {
	return (from == GenerationLoading && to == GenerationActive) ||
		(from == GenerationActive && to == GenerationRetired)
}

func (store *Store) RegisterLabelGeneration(
	ctx context.Context,
	value LabelGeneration,
) (LabelGeneration, error) {
	if err := validateLabelGeneration(value); err != nil {
		return LabelGeneration{}, err
	}
	var kind = string([]byte{byte(value.Kind)})
	err := store.database.QueryRow(
		ctx,
		`INSERT INTO agefreighter_meta.label_generation (
			graph_generation_id, label_name, kind, graph_namespace_oid,
			label_id, relation_oid, sequence_oid, mapping_generation
		)
		SELECT $1, $2, $3, $4, $5, $6, $7, $8
		FROM agefreighter_meta.graph_generation
		WHERE graph_generation_id = $1
		  AND namespace_oid = $4
		RETURNING label_generation_id, created_at, updated_at`,
		value.GraphGenerationID,
		value.LabelName,
		kind,
		value.GraphNamespaceOID,
		value.LabelID,
		value.RelationOID,
		value.SequenceOID,
		value.MappingGeneration,
	).Scan(&value.ID, &value.CreatedAt, &value.UpdatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return LabelGeneration{}, fmt.Errorf(
			"%w: graph namespace does not match label catalog",
			ErrGenerationMismatch,
		)
	}
	if err != nil {
		return LabelGeneration{}, fmt.Errorf(
			"register label generation %q: %w",
			value.LabelName,
			err,
		)
	}
	return value, nil
}

func (store *Store) AdmitLabelGeneration(
	ctx context.Context,
	graphGenerationID int64,
	current LabelGeneration,
) (LabelGeneration, error) {
	if graphGenerationID <= 0 {
		return LabelGeneration{}, errors.New("graph generation ID must be positive")
	}
	if err := validateLabelGeneration(current); err != nil {
		return LabelGeneration{}, err
	}
	var stored LabelGeneration
	var kind string
	err := store.database.QueryRow(
		ctx,
		`SELECT
			label_generation_id, graph_generation_id, label_name, kind,
			graph_namespace_oid, label_id, relation_oid, sequence_oid,
			mapping_generation, created_at, updated_at
		 FROM agefreighter_meta.label_generation
		 WHERE graph_generation_id = $1
		   AND label_name = $2
		 ORDER BY mapping_generation DESC
		 LIMIT 1`,
		graphGenerationID,
		current.LabelName,
	).Scan(
		&stored.ID,
		&stored.GraphGenerationID,
		&stored.LabelName,
		&kind,
		&stored.GraphNamespaceOID,
		&stored.LabelID,
		&stored.RelationOID,
		&stored.SequenceOID,
		&stored.MappingGeneration,
		&stored.CreatedAt,
		&stored.UpdatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return LabelGeneration{}, fmt.Errorf(
			"%w: label generation %q",
			ErrNotFound,
			current.LabelName,
		)
	}
	if err != nil {
		return LabelGeneration{}, fmt.Errorf(
			"read label generation %q: %w",
			current.LabelName,
			err,
		)
	}
	if len(kind) != 1 {
		return LabelGeneration{}, fmt.Errorf("stored label kind %q is invalid", kind)
	}
	stored.Kind = LabelKind(kind[0])
	if stored.Kind != current.Kind ||
		stored.GraphGenerationID != current.GraphGenerationID ||
		stored.GraphNamespaceOID != current.GraphNamespaceOID ||
		stored.LabelID != current.LabelID ||
		stored.RelationOID != current.RelationOID ||
		stored.SequenceOID != current.SequenceOID ||
		stored.MappingGeneration != current.MappingGeneration {
		return LabelGeneration{}, fmt.Errorf(
			"%w: label %q catalog identity changed",
			ErrGenerationMismatch,
			current.LabelName,
		)
	}
	return stored, nil
}
