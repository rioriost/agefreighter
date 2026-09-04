package target

import (
	"context"
	"strings"
	"testing"
	"time"

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

func TestOpenPropertyGraphRequiresConnectionAndOptions(t *testing.T) {
	runtime, err := Open(
		context.Background(),
		config.TargetPostgreSQLPropertyGraph,
		"",
		Options{
			MaxConnections: 1, ConnectTimeout: time.Second,
			OperationTimeout: time.Second,
		},
	)
	if runtime != nil || err == nil ||
		!strings.Contains(err.Error(), "connection string is required") {
		t.Fatalf("Open() = %#v, %v", runtime, err)
	}
}

func TestOpenRejectsUnknownTarget(t *testing.T) {
	runtime, err := Open(context.Background(), "unknown", "not-used", Options{})
	if runtime != nil || err == nil {
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
	if err == nil || !strings.Contains(err.Error(), "does not provide Apache AGE diagnostics") {
		t.Fatalf("ProbeAGE(property graph) error = %v", err)
	}
	if _, err := ProbeAGE(context.Background(), "unknown", "not-used", Options{}); err == nil {
		t.Fatal("ProbeAGE(unknown) succeeded")
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
	if _, err := RequirePGGraph(postgresRuntime); err == nil {
		t.Fatal("RequirePGGraph(runtime without capability) succeeded")
	}
	if _, err := RequirePGGraph(nil); err == nil {
		t.Fatal("RequirePGGraph(nil) succeeded")
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

func TestPGGraphRuntimeLifecycleAccessors(t *testing.T) {
	store := &meta.Store{}
	runtime := &pgGraphRuntime{store: store}
	if runtime.Backend() != config.TargetPostgreSQLPropertyGraph ||
		runtime.Metadata() != store || runtime.PGGraphAdapter() != nil {
		t.Fatalf("property graph runtime accessors = %#v", runtime)
	}
	runtime.Close()
	var nilRuntime *pgGraphRuntime
	nilRuntime.Close()
}
