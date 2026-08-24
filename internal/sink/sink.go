package sink

import (
	"context"

	"github.com/rioriost/agefreighter/internal/checkpoint"
	"github.com/rioriost/agefreighter/pkg/model"
)

type BatchMetadata struct {
	ID            uint64
	Attempt       uint32
	Rows          int
	Bytes         int64
	FirstPosition model.SourcePosition
	LastPosition  model.SourcePosition
}

type Sink interface {
	Begin(ctx context.Context, batch BatchMetadata) (Transaction, error)
}

type Transaction interface {
	Write(ctx context.Context, records []model.Record) error

	// Commit must atomically commit both the records and checkpoint.
	Commit(ctx context.Context, checkpoint checkpoint.State) error
	Rollback(ctx context.Context) error
}
