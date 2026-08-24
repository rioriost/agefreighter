package reject

import (
	"context"

	"github.com/rioriost/agefreighter/pkg/model"
)

type Rejection struct {
	Record   model.Record
	Position model.SourcePosition
	Code     string
	Message  string
}

type Writer interface {
	Write(ctx context.Context, rejection Rejection) error
	Close() error
}
