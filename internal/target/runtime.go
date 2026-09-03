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
	"github.com/rioriost/agefreighter/internal/pggraph"
)

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

type PGGraphRuntime interface {
	Runtime
	PGGraphAdapter() *pggraph.Adapter
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

func RequirePGGraph(runtime Runtime) (PGGraphRuntime, error) {
	if runtime == nil {
		return nil, errors.New("target runtime is nil")
	}
	pgRuntime, ok := runtime.(PGGraphRuntime)
	if !ok {
		return nil, fmt.Errorf(
			"target backend %q does not provide PostgreSQL property graph operations",
			runtime.Backend(),
		)
	}
	return pgRuntime, nil
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
		runtime, err := openAGE(ctx, dsn, options)
		if err != nil {
			return nil, err
		}
		return runtime, nil
	case config.TargetPostgreSQLPropertyGraph:
		runtime, err := openPGGraph(ctx, dsn, options)
		if err != nil {
			return nil, err
		}
		return runtime, nil
	}
	return nil, fmt.Errorf("unsupported target backend %q", backend)
}

func RequireImplemented(backend config.TargetType) error {
	switch backend {
	case config.TargetApacheAGE:
		return nil
	case config.TargetPostgreSQLPropertyGraph:
		return nil
	default:
		return fmt.Errorf("unsupported target backend %q", backend)
	}
}

type pgGraphRuntime struct {
	adapter *pggraph.Adapter
	store   *meta.Store
}

var _ Runtime = (*pgGraphRuntime)(nil)
var _ PGGraphRuntime = (*pgGraphRuntime)(nil)

func openPGGraph(
	ctx context.Context,
	dsn string,
	options Options,
) (*pgGraphRuntime, error) {
	adapter, err := pggraph.Open(ctx, dsn, pggraph.PoolOptions{
		MinConnections: 1, MaxConnections: options.MaxConnections,
		ConnectTimeout: options.ConnectTimeout, OperationTimeout: options.OperationTimeout,
	})
	if err != nil {
		return nil, err
	}
	return &pgGraphRuntime{adapter: adapter, store: adapter.Metadata()}, nil
}

func (*pgGraphRuntime) Backend() config.TargetType {
	return config.TargetPostgreSQLPropertyGraph
}

func (runtime *pgGraphRuntime) Metadata() *meta.Store { return runtime.store }

func (runtime *pgGraphRuntime) InspectMetadata(ctx context.Context) (meta.SchemaInspection, error) {
	return runtime.store.InspectSchema(ctx)
}

func (runtime *pgGraphRuntime) MigrateMetadata(ctx context.Context) error {
	return runtime.store.MigrateIfNeeded(ctx)
}

func (runtime *pgGraphRuntime) PGGraphAdapter() *pggraph.Adapter { return runtime.adapter }

func (runtime *pgGraphRuntime) Close() {
	if runtime != nil && runtime.adapter != nil {
		runtime.adapter.Close()
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
