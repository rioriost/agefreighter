package source

import (
	"context"

	"github.com/rioriost/agefreighter/pkg/model"
)

type Item struct {
	Record model.Record

	// SizeBytes is the retained heap size of the record and any source buffers
	// that remain live until the target commits it. The pipeline adds its own
	// per-record container overhead to this value.
	SizeBytes int64
}

// Iterator must return promptly after ctx is cancelled. io.EOF marks a
// successful end of input; a record returned with a non-nil error is ignored.
type Iterator interface {
	Next(ctx context.Context) (Item, error)
	Close() error
}
