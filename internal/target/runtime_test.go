package target

import (
	"context"
	"errors"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
)

func TestOpenRejectsUnimplementedPropertyGraphTarget(t *testing.T) {
	runtime, err := Open(
		context.Background(),
		config.TargetPostgreSQLPropertyGraph,
		"not-used",
		Options{},
	)
	if runtime != nil || !errors.Is(err, ErrAdapterNotImplemented) {
		t.Fatalf("Open() = %#v, %v", runtime, err)
	}
}

func TestOpenRejectsUnknownTarget(t *testing.T) {
	runtime, err := Open(context.Background(), "unknown", "not-used", Options{})
	if runtime != nil || err == nil || errors.Is(err, ErrAdapterNotImplemented) {
		t.Fatalf("Open() = %#v, %v", runtime, err)
	}
}
