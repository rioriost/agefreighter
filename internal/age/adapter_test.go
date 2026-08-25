package age

import (
	"context"
	"testing"
	"time"
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
