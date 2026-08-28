package age

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/meta"
)

type IntegrityResult struct {
	IdentityRows            int64
	MissingPhysicalRows     int64
	MissingEndpointRows     int64
	ChangedEndpointRows     int64
	PhysicalRows            int64
	OrphanPhysicalRows      int64
	IdentityTruncated       bool
	PhysicalTruncated       bool
	PhysicalCoverageChecked bool
}

func (transaction *Transaction) SetStatementTimeout(
	ctx context.Context,
	timeout time.Duration,
) error {
	if timeout <= 0 {
		return errors.New("statement timeout must be positive")
	}
	if _, ok := ctx.Deadline(); !ok {
		return errors.New("statement timeout context requires a deadline")
	}
	if _, err := transaction.tx.Exec(
		ctx,
		`SELECT pg_catalog.set_config('statement_timeout', $1, true)`,
		fmt.Sprintf("%dms", max(timeout.Milliseconds(), 1)),
	); err != nil {
		return fmt.Errorf("set verification statement timeout: %w", err)
	}
	return nil
}

func (transaction *Transaction) VerifyBoundedIntegrity(
	ctx context.Context,
	label LabelCatalog,
	graphGenerationID int64,
	labelGenerationID int64,
	kind meta.LabelKind,
	limit int,
) (IntegrityResult, error) {
	return transaction.VerifyBoundedIntegrityForIdentityCoverage(
		ctx, label, graphGenerationID, labelGenerationID, kind, limit, true,
	)
}

func (transaction *Transaction) VerifyBoundedIntegrityForIdentityCoverage(
	ctx context.Context,
	label LabelCatalog,
	graphGenerationID int64,
	labelGenerationID int64,
	kind meta.LabelKind,
	limit int,
	requireFullCoverage bool,
) (IntegrityResult, error) {
	if graphGenerationID <= 0 || labelGenerationID <= 0 {
		return IntegrityResult{}, errors.New("graph and label generation IDs must be positive")
	}
	if limit <= 0 {
		return IntegrityResult{}, errors.New("integrity limit must be positive")
	}
	current, err := transaction.LookupLabel(ctx, label.GraphName, label.LabelName)
	if err != nil {
		return IntegrityResult{}, err
	}
	if current != label {
		return IntegrityResult{}, fmt.Errorf("label %q catalog changed before verification", label.LabelName)
	}
	table := pgx.Identifier{label.GraphName, label.LabelName}.Sanitize()
	var result IntegrityResult
	switch kind {
	case meta.VertexLabel:
		err = transaction.tx.QueryRow(ctx, fmt.Sprintf(`
			WITH bounded AS MATERIALIZED (
				SELECT graph_id
				FROM agefreighter_meta.vertex_identity
				WHERE graph_generation_id = $1
				  AND label_generation_id = $2
				ORDER BY graph_id
				LIMIT $3 + 1
			), sample AS (
				SELECT graph_id FROM bounded ORDER BY graph_id LIMIT $3
			)
			SELECT
				(SELECT COUNT(*) FROM sample),
				(SELECT COUNT(*) FROM bounded) > $3,
				COUNT(*) FILTER (WHERE physical.id IS NULL)
			FROM sample
			LEFT JOIN %s physical
			  ON physical.id = sample.graph_id::text::graphid`,
			table,
		), graphGenerationID, labelGenerationID, limit).Scan(
			&result.IdentityRows,
			&result.IdentityTruncated,
			&result.MissingPhysicalRows,
		)
	case meta.EdgeLabel:
		err = transaction.tx.QueryRow(ctx, fmt.Sprintf(`
			WITH bounded AS MATERIALIZED (
				SELECT graph_id, start_graph_id, end_graph_id
				FROM agefreighter_meta.edge_identity
				WHERE graph_generation_id = $1
				  AND label_generation_id = $2
				ORDER BY graph_id
				LIMIT $3 + 1
			), sample AS (
				SELECT graph_id, start_graph_id, end_graph_id
				FROM bounded ORDER BY graph_id LIMIT $3
			)
			SELECT
				(SELECT COUNT(*) FROM sample),
				(SELECT COUNT(*) FROM bounded) > $3,
				COUNT(*) FILTER (WHERE physical.id IS NULL),
				COUNT(*) FILTER (
					WHERE start_identity.graph_id IS NULL
					   OR end_identity.graph_id IS NULL
				),
				COUNT(*) FILTER (
					WHERE physical.id IS NOT NULL
					  AND (
						physical.start_id <> sample.start_graph_id::text::graphid
						OR physical.end_id <> sample.end_graph_id::text::graphid
					  )
				)
			FROM sample
			LEFT JOIN %s physical
			  ON physical.id = sample.graph_id::text::graphid
			LEFT JOIN agefreighter_meta.vertex_identity start_identity
			  ON start_identity.graph_generation_id = $1
			 AND start_identity.graph_id = sample.start_graph_id
			LEFT JOIN agefreighter_meta.vertex_identity end_identity
			  ON end_identity.graph_generation_id = $1
			 AND end_identity.graph_id = sample.end_graph_id`,
			table,
		), graphGenerationID, labelGenerationID, limit).Scan(
			&result.IdentityRows,
			&result.IdentityTruncated,
			&result.MissingPhysicalRows,
			&result.MissingEndpointRows,
			&result.ChangedEndpointRows,
		)
	default:
		return IntegrityResult{}, fmt.Errorf("unsupported label kind %q", kind)
	}
	if err != nil {
		return IntegrityResult{}, fmt.Errorf("verify bounded identity integrity for %s: %w", table, err)
	}

	if kind == meta.EdgeLabel && !requireFullCoverage {
		return result, nil
	}
	result.PhysicalCoverageChecked = true
	identityTable := "agefreighter_meta.vertex_identity"
	if kind == meta.EdgeLabel {
		identityTable = "agefreighter_meta.edge_identity"
	}
	if err := transaction.tx.QueryRow(ctx, fmt.Sprintf(`
		WITH bounded AS MATERIALIZED (
			SELECT id::text::bigint AS graph_id
			FROM %s
			ORDER BY id
			LIMIT $3 + 1
		), sample AS (
			SELECT graph_id FROM bounded ORDER BY graph_id LIMIT $3
		)
		SELECT
			(SELECT COUNT(*) FROM sample),
			(SELECT COUNT(*) FROM bounded) > $3,
			COUNT(*) FILTER (WHERE identity.graph_id IS NULL)
		FROM sample
		LEFT JOIN %s identity
		  ON identity.graph_generation_id = $1
		 AND identity.label_generation_id = $2
		 AND identity.graph_id = sample.graph_id`,
		table,
		identityTable,
	), graphGenerationID, labelGenerationID, limit).Scan(
		&result.PhysicalRows,
		&result.PhysicalTruncated,
		&result.OrphanPhysicalRows,
	); err != nil {
		return IntegrityResult{}, fmt.Errorf("verify bounded physical integrity for %s: %w", table, err)
	}
	return result, nil
}
