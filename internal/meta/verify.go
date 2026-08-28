package meta

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

func (store *Store) CountLabelIdentities(
	ctx context.Context,
	graphGenerationID int64,
	labelGenerationID int64,
	kind LabelKind,
) (int64, error) {
	if graphGenerationID <= 0 || labelGenerationID <= 0 {
		return 0, errors.New("graph and label generation IDs must be positive")
	}
	var table string
	switch kind {
	case VertexLabel:
		table = "agefreighter_meta.vertex_identity"
	case EdgeLabel:
		table = "agefreighter_meta.edge_identity"
	default:
		return 0, fmt.Errorf("unsupported label kind %q", kind)
	}
	var count int64
	if err := store.database.QueryRow(
		ctx,
		`SELECT COUNT(*) FROM `+table+`
		 WHERE graph_generation_id = $1
		   AND label_generation_id = $2`,
		graphGenerationID,
		labelGenerationID,
	).Scan(&count); err != nil {
		return 0, fmt.Errorf("count label identities: %w", err)
	}
	return count, nil
}

func (store *Store) CountLabelIdentitiesWithTimeout(
	ctx context.Context,
	graphGenerationID int64,
	labelGenerationID int64,
	kind LabelKind,
	timeout time.Duration,
) (int64, error) {
	if timeout <= 0 {
		return 0, errors.New("identity count timeout must be positive")
	}
	if graphGenerationID <= 0 || labelGenerationID <= 0 {
		return 0, errors.New("graph and label generation IDs must be positive")
	}
	if kind != VertexLabel && kind != EdgeLabel {
		return 0, fmt.Errorf("unsupported label kind %q", kind)
	}
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		return 0, errors.New("identity count context requires a deadline")
	}
	tx, err := store.database.Begin(ctx)
	if err != nil {
		return 0, fmt.Errorf("begin bounded identity count: %w", err)
	}
	defer rollbackWithTimeout(ctx, tx, timeout)
	if _, err := tx.Exec(
		ctx,
		`SELECT pg_catalog.set_config('statement_timeout', $1, true)`,
		fmt.Sprintf("%dms", max(timeout.Milliseconds(), 1)),
	); err != nil {
		return 0, fmt.Errorf("set identity count statement timeout: %w", err)
	}
	count, err := (&Store{database: tx}).CountLabelIdentities(
		ctx,
		graphGenerationID,
		labelGenerationID,
		kind,
	)
	if err != nil {
		return 0, err
	}
	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("commit bounded identity count: %w", err)
	}
	return count, nil
}

func rollbackWithTimeout(ctx context.Context, tx pgx.Tx, timeout time.Duration) {
	rollbackCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), timeout)
	defer cancel()
	_ = tx.Rollback(rollbackCtx)
}
