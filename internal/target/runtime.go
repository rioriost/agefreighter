// Package target owns target-backend selection and connection lifecycle.
// Backend-specific graph operations remain in their adapter packages.
package target

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
)

var ErrAdapterNotImplemented = errors.New("target adapter is not implemented")

type Options struct {
	MaxConnections   int32
	ConnectTimeout   time.Duration
	OperationTimeout time.Duration
}

// Runtime is the backend-neutral connection and metadata boundary used by the
// application. Backend-specific capabilities are exposed by narrow extension
// interfaces such as AGERuntime.
type Runtime interface {
	Backend() config.TargetType
	Metadata() *meta.Store
	InspectMetadata(context.Context) (meta.SchemaInspection, error)
	MigrateMetadata(context.Context) error
	Close()
}

type AGERuntime interface {
	Runtime
	AGEAdapter() *age.Adapter
}

func RequireAGE(runtime Runtime) (AGERuntime, error) {
	if runtime == nil {
		return nil, errors.New("target runtime is nil")
	}
	ageRuntime, ok := runtime.(AGERuntime)
	if !ok {
		return nil, fmt.Errorf(
			"target backend %q does not provide Apache AGE operations",
			runtime.Backend(),
		)
	}
	return ageRuntime, nil
}

func Open(
	ctx context.Context,
	backend config.TargetType,
	dsn string,
	options Options,
) (Runtime, error) {
	if err := RequireImplemented(backend); err != nil {
		return nil, err
	}
	switch backend {
	case config.TargetApacheAGE:
		return openAGE(ctx, dsn, options)
	}
	return nil, fmt.Errorf("unsupported target backend %q", backend)
}

func RequireImplemented(backend config.TargetType) error {
	switch backend {
	case config.TargetApacheAGE:
		return nil
	case config.TargetPostgreSQLPropertyGraph:
		return fmt.Errorf("%w: %s", ErrAdapterNotImplemented, backend)
	default:
		return fmt.Errorf("unsupported target backend %q", backend)
	}
}

// ProbeAGE performs the degraded, read-only capability probe for an AGE
// backend. Keeping dispatch here prevents diagnostic commands from treating a
// different PostgreSQL target as AGE merely because both use a PostgreSQL DSN.
func ProbeAGE(
	ctx context.Context,
	backend config.TargetType,
	dsn string,
	options Options,
) (age.DegradedProbe, error) {
	if err := RequireImplemented(backend); err != nil {
		return age.DegradedProbe{}, err
	}
	if backend != config.TargetApacheAGE {
		return age.DegradedProbe{}, fmt.Errorf(
			"target backend %q does not provide Apache AGE diagnostics",
			backend,
		)
	}
	return age.ProbeDegraded(ctx, dsn, age.ProbeOptions{
		ConnectTimeout:   options.ConnectTimeout,
		OperationTimeout: options.OperationTimeout,
	})
}

type ageRuntime struct {
	adapter *age.Adapter
	store   *meta.Store
}

var _ Runtime = (*ageRuntime)(nil)
var _ AGERuntime = (*ageRuntime)(nil)

func openAGE(
	ctx context.Context,
	dsn string,
	options Options,
) (*ageRuntime, error) {
	adapter, err := age.Open(ctx, dsn, age.PoolOptions{
		MinConnections:   1,
		MaxConnections:   options.MaxConnections,
		ConnectTimeout:   options.ConnectTimeout,
		OperationTimeout: options.OperationTimeout,
	})
	if err != nil {
		return nil, err
	}
	store, err := adapter.Metadata()
	if err != nil {
		adapter.Close()
		return nil, err
	}
	return &ageRuntime{adapter: adapter, store: store}, nil
}

func (runtime *ageRuntime) Backend() config.TargetType {
	return config.TargetApacheAGE
}

func (runtime *ageRuntime) Metadata() *meta.Store {
	return runtime.store
}

func (runtime *ageRuntime) InspectMetadata(
	ctx context.Context,
) (meta.SchemaInspection, error) {
	return runtime.store.InspectSchema(ctx)
}

func (runtime *ageRuntime) MigrateMetadata(ctx context.Context) error {
	return runtime.store.MigrateIfNeeded(ctx)
}

func (runtime *ageRuntime) AGEAdapter() *age.Adapter {
	return runtime.adapter
}

func (runtime *ageRuntime) Close() {
	if runtime != nil && runtime.adapter != nil {
		runtime.adapter.Close()
	}
}
