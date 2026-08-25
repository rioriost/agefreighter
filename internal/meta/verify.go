package meta

import (
	"context"
	"errors"
	"fmt"
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
