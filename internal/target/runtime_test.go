package target

import (
	"context"
	"errors"
	"testing"

	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
)

type fakeRuntime struct {
	backend config.TargetType
}

func (runtime fakeRuntime) Backend() config.TargetType { return runtime.backend }
func (fakeRuntime) Metadata() *meta.Store              { return nil }
func (fakeRuntime) InspectMetadata(context.Context) (meta.SchemaInspection, error) {
	return meta.SchemaInspection{}, nil
}
func (fakeRuntime) MigrateMetadata(context.Context) error { return nil }
func (fakeRuntime) Close()                                {}

type fakeAGERuntime struct{ fakeRuntime }

func (fakeAGERuntime) AGEAdapter() *age.Adapter { return nil }

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

func TestProbeAGERejectsNonAGEBackendBeforeUsingDSN(t *testing.T) {
	_, err := ProbeAGE(
		context.Background(),
		config.TargetPostgreSQLPropertyGraph,
		"not-used",
		Options{},
	)
	if !errors.Is(err, ErrAdapterNotImplemented) {
		t.Fatalf("ProbeAGE(property graph) error = %v", err)
	}
}

func TestRequireAGECapability(t *testing.T) {
	if _, err := RequireAGE(nil); err == nil {
		t.Fatal("RequireAGE(nil) succeeded")
	}
	postgresRuntime := fakeRuntime{backend: config.TargetPostgreSQLPropertyGraph}
	if _, err := RequireAGE(postgresRuntime); err == nil {
		t.Fatal("RequireAGE(property graph) succeeded")
	}
	ageRuntime := fakeAGERuntime{fakeRuntime{backend: config.TargetApacheAGE}}
	got, err := RequireAGE(ageRuntime)
	if err != nil || got.Backend() != config.TargetApacheAGE {
		t.Fatalf("RequireAGE(AGE) = %#v, %v", got, err)
	}
}

func TestAGERuntimeLifecycleAccessors(t *testing.T) {
	store := &meta.Store{}
	runtime := &ageRuntime{store: store}
	if runtime.Backend() != config.TargetApacheAGE ||
		runtime.Metadata() != store || runtime.AGEAdapter() != nil {
		t.Fatalf("age runtime accessors = %#v", runtime)
	}
	runtime.Close()
	var nilRuntime *ageRuntime
	nilRuntime.Close()
}
