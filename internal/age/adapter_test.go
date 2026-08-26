package age

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestOpenRejectsInvalidOptions(t *testing.T) {
	valid := PoolOptions{
		MaxConnections:   1,
		ConnectTimeout:   time.Second,
		OperationTimeout: time.Second,
	}
	tests := []struct {
		name    string
		dsn     string
		options PoolOptions
	}{
		{name: "empty DSN", options: valid},
		{
			name: "negative minimum", dsn: "postgres://localhost",
			options: PoolOptions{
				MinConnections: -1, MaxConnections: 1,
				ConnectTimeout: time.Second, OperationTimeout: time.Second,
			},
		},
		{
			name: "zero maximum", dsn: "postgres://localhost",
			options: PoolOptions{
				ConnectTimeout: time.Second, OperationTimeout: time.Second,
			},
		},
		{
			name: "minimum exceeds maximum", dsn: "postgres://localhost",
			options: PoolOptions{
				MinConnections: 2, MaxConnections: 1,
				ConnectTimeout: time.Second, OperationTimeout: time.Second,
			},
		},
		{
			name: "zero connect timeout", dsn: "postgres://localhost",
			options: PoolOptions{MaxConnections: 1, OperationTimeout: time.Second},
		},
		{
			name: "zero operation timeout", dsn: "postgres://localhost",
			options: PoolOptions{MaxConnections: 1, ConnectTimeout: time.Second},
		},
		{name: "malformed DSN", dsn: "://bad", options: valid},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if adapter, err := Open(
				context.Background(),
				test.dsn,
				test.options,
			); err == nil {
				adapter.Close()
				t.Fatal("Open() succeeded")
			}
		})
	}
}

func TestOpenReportsConnectionFailure(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	adapter, err := Open(
		ctx,
		"postgres://127.0.0.1:1/agefreighter?sslmode=disable",
		PoolOptions{
			MaxConnections:   1,
			ConnectTimeout:   100 * time.Millisecond,
			OperationTimeout: time.Second,
		},
	)
	if adapter != nil {
		adapter.Close()
		t.Fatal("Open() returned an adapter for an unreachable server")
	}
	if err == nil {
		t.Fatal("Open() did not report an unreachable server")
	}
}

func TestAdapterLoadSlotsReserveMetadataCapacity(t *testing.T) {
	config, err := pgxpool.ParseConfig(
		"postgres://localhost/unused?connect_timeout=1",
	)
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}
	config.MaxConns = 2
	pool, err := pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		t.Fatalf("NewWithConfig() error = %v", err)
	}
	t.Cleanup(pool.Close)
	adapter := &Adapter{pool: pool}

	if err := adapter.acquireLoadSlot(t.Context()); err != nil {
		t.Fatalf("acquireLoadSlot() error = %v", err)
	}
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	if err := adapter.acquireLoadSlot(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatalf("blocked acquireLoadSlot() error = %v", err)
	}
	adapter.releaseLoadSlot()
	if err := adapter.acquireLoadSlot(t.Context()); err != nil {
		t.Fatalf("reacquireLoadSlot() error = %v", err)
	}
	adapter.releaseLoadSlot()
}

func TestAdapterSmallHelpers(t *testing.T) {
	if !listContains("pg_stat_statements, age", "age") {
		t.Fatal("listContains() did not find AGE")
	}
	if listContains("lineage,other", "age") {
		t.Fatal("listContains() found partial AGE name")
	}
	var adapter *Adapter
	adapter.Close()
	if got := (LabelKind('x')).String(); got != "unknown" {
		t.Fatalf("unknown LabelKind.String() = %q", got)
	}
	if VertexLabel.String() != "vertex" || EdgeLabel.String() != "edge" {
		t.Fatal("known label kind strings are incorrect")
	}
}

func TestCapabilityProbeErrors(t *testing.T) {
	queryError := errors.New("injected query failure")
	for _, row := range []pgx.Row{
		stubRow(func(...any) error { return pgx.ErrNoRows }),
		stubRow(func(...any) error { return queryError }),
		stubRow(func(destinations ...any) error {
			setCapabilityDestinations(destinations, "bad", "1.6.0")
			return nil
		}),
		stubRow(func(destinations ...any) error {
			setCapabilityDestinations(destinations, "170007", "bad")
			return nil
		}),
	} {
		if _, err := ProbeCapabilities(
			context.Background(),
			stubQuerier{row: row},
		); err == nil {
			t.Fatal("ProbeCapabilities() succeeded")
		}
	}
}

func TestCatalogLookupErrors(t *testing.T) {
	injected := errors.New("injected catalog failure")
	if _, err := lookupGraph(
		t.Context(),
		stubDatabase{row: stubRow(func(...any) error { return pgx.ErrNoRows })},
		"valid_graph",
	); !errors.Is(err, ErrCatalogEntryNotFound) {
		t.Fatalf("lookupGraph(missing) error = %v", err)
	}
	if _, err := lookupGraph(
		t.Context(),
		stubDatabase{row: stubRow(func(...any) error { return injected })},
		"valid_graph",
	); !errors.Is(err, injected) {
		t.Fatalf("lookupGraph(database) error = %v", err)
	}
	if _, err := lookupGraph(
		t.Context(),
		stubDatabase{row: stubRow(func(dest ...any) error {
			*dest[0].(*string) = "valid_graph"
			*dest[1].(*uint32) = 10
			*dest[2].(*uint32) = 11
			return nil
		})},
		"valid_graph",
	); err == nil {
		t.Fatal("lookupGraph() accepted mismatched catalog OIDs")
	}

	labelCases := []struct {
		name string
		row  stubRow
	}{
		{
			name: "missing",
			row:  func(...any) error { return pgx.ErrNoRows },
		},
		{
			name: "database",
			row:  func(...any) error { return injected },
		},
		{
			name: "invalid ID",
			row: labelCatalogRow(func(dest ...any) {
				*dest[4].(*int32) = 0
			}),
		},
		{
			name: "invalid kind",
			row: labelCatalogRow(func(dest ...any) {
				*dest[5].(*string) = "x"
			}),
		},
		{
			name: "namespace mismatch",
			row: labelCatalogRow(func(dest ...any) {
				*dest[7].(*uint32) = 11
			}),
		},
	}
	for _, test := range labelCases {
		t.Run(test.name, func(t *testing.T) {
			if _, err := lookupLabel(
				t.Context(),
				stubDatabase{row: test.row},
				"valid_graph",
				"Person",
			); err == nil {
				t.Fatal("lookupLabel() error = nil")
			}
		})
	}
}

func labelCatalogRow(mutate func(...any)) stubRow {
	return func(dest ...any) error {
		*dest[0].(*string) = "valid_graph"
		*dest[1].(*string) = "Person"
		*dest[2].(*uint32) = 10
		*dest[3].(*uint32) = 10
		*dest[4].(*int32) = 1
		*dest[5].(*string) = "v"
		*dest[6].(*uint32) = 12
		*dest[7].(*uint32) = 10
		*dest[8].(*uint32) = 13
		*dest[9].(*string) = "Person_id_seq"
		mutate(dest...)
		return nil
	}
}

func TestPreloadStatusProbe(t *testing.T) {
	status, err := probePreloadStatus(
		context.Background(),
		stubQuerier{row: stubRow(func(destinations ...any) error {
			*destinations[0].(*string) = "other, pg_stat_statements"
			return nil
		})},
	)
	if err != nil || status != PreloadNotConfigured {
		t.Fatalf("probePreloadStatus() = %q, %v", status, err)
	}
	status, err = probePreloadStatus(
		context.Background(),
		stubQuerier{row: stubRow(func(...any) error { return pgx.ErrNoRows })},
	)
	if err != nil || status != PreloadUnknown {
		t.Fatalf("unknown probePreloadStatus() = %q, %v", status, err)
	}
	if _, err := probePreloadStatus(
		context.Background(),
		stubQuerier{row: stubRow(func(...any) error {
			return errors.New("injected preload failure")
		})},
	); err == nil {
		t.Fatal("probePreloadStatus() ignored query failure")
	}
}

type stubQuerier struct {
	row pgx.Row
}

func (querier stubQuerier) QueryRow(context.Context, string, ...any) pgx.Row {
	return querier.row
}

type stubDatabase struct {
	row pgx.Row
}

func (database stubDatabase) Exec(
	context.Context,
	string,
	...any,
) (pgconn.CommandTag, error) {
	return pgconn.CommandTag{}, nil
}

func (database stubDatabase) QueryRow(
	context.Context,
	string,
	...any,
) pgx.Row {
	return database.row
}

type stubRow func(...any) error

func (row stubRow) Scan(destinations ...any) error {
	return row(destinations...)
}

func setCapabilityDestinations(destinations []any, serverVersion, ageVersion string) {
	*destinations[0].(*string) = serverVersion
	*destinations[1].(*string) = ageVersion
	*destinations[2].(*string) = "test_user"
	*destinations[3].(*bool) = false
}

func TestTransactionAndLifecycleRejectInvalidInputs(t *testing.T) {
	adapter := &Adapter{}
	if err := adapter.InTransaction(context.Background(), nil); err == nil {
		t.Fatal("InTransaction() accepted nil callback")
	}
	transaction := &Transaction{}
	ctx := context.Background()
	if err := transaction.CreateGraph(ctx, "x"); err == nil {
		t.Fatal("CreateGraph() accepted invalid graph")
	}
	if err := transaction.DropGraph(ctx, "x", true); err == nil {
		t.Fatal("DropGraph() accepted invalid graph")
	}
	if err := transaction.RenameGraph(ctx, "x", "valid_graph"); err == nil {
		t.Fatal("RenameGraph() accepted invalid old graph")
	}
	if err := transaction.RenameGraph(ctx, "valid_graph", "x"); err == nil {
		t.Fatal("RenameGraph() accepted invalid new graph")
	}
	if err := transaction.CreateLabel(ctx, "x", "Person", VertexLabel); err == nil {
		t.Fatal("CreateLabel() accepted invalid graph")
	}
	if err := transaction.CreateLabel(ctx, "valid_graph", "bad-name", VertexLabel); err == nil {
		t.Fatal("CreateLabel() accepted invalid label")
	}
	if err := transaction.CreateLabel(ctx, "valid_graph", "Person", LabelKind('x')); err == nil {
		t.Fatal("CreateLabel() accepted invalid kind")
	}
	if err := transaction.DropLabel(ctx, "x", "Person", true); err == nil {
		t.Fatal("DropLabel() accepted invalid graph")
	}
	if err := transaction.DropLabel(ctx, "valid_graph", "bad-name", true); err == nil {
		t.Fatal("DropLabel() accepted invalid label")
	}
	if err := transaction.DropLabel(ctx, "valid_graph", "Person", true); err == nil {
		t.Fatal("DropLabel() accepted unsupported force option")
	}
	if err := transaction.LockGraphLifecycle(ctx, "x"); err == nil {
		t.Fatal("LockGraphLifecycle() accepted invalid graph")
	}
	if err := transaction.PreflightGraphRename(ctx, GraphCatalog{}); err == nil {
		t.Fatal("PreflightGraphRename() accepted empty catalog")
	}
	if err := transaction.PreflightGraphRename(ctx, GraphCatalog{
		Name: "valid_graph", GraphOID: 1, NamespaceOID: 2,
	}); err == nil {
		t.Fatal("PreflightGraphRename() accepted mismatched OIDs")
	}
	if err := transaction.LockLabel(ctx, 0, 1); err == nil {
		t.Fatal("LockLabel() accepted zero graph OID")
	}
	if err := transaction.LockLabel(ctx, 1, 0); err == nil {
		t.Fatal("LockLabel() accepted zero label ID")
	}
	if _, err := transaction.ReserveIDs(ctx, LabelCatalog{}, 0); err == nil {
		t.Fatal("ReserveIDs() accepted zero count")
	}
	if _, err := transaction.ReserveIDs(ctx, LabelCatalog{}, MaxEntryID+1); err == nil {
		t.Fatal("ReserveIDs() accepted excessive count")
	}
	if err := transaction.VerifyLabelRows(ctx, LabelCatalog{}, -1); err == nil {
		t.Fatal("VerifyLabelRows() accepted negative count")
	}
}
