package neo4j

import (
	"context"
	"errors"
	"io"
	"math"
	"testing"
)

func TestCountInventory(t *testing.T) {
	client := &fakeClient{streams: []RecordStream{
		&fakeStream{records: []Record{record(map[string]any{"count": int64(160_000_000)})}},
		&fakeStream{records: []Record{record(map[string]any{"count": int64(400_000_000)})}},
	}}
	inventory, err := CountInventory(t.Context(), client)
	if err != nil {
		t.Fatal(err)
	}
	if inventory.Vertices != 160_000_000 || inventory.Edges != 400_000_000 ||
		inventory.TotalRows() != 560_000_000 {
		t.Fatalf("inventory = %#v", inventory)
	}
	if len(client.queries) != 2 || client.queries[0] != countVerticesQuery ||
		client.queries[1] != countEdgesQuery {
		t.Fatalf("queries = %#v", client.queries)
	}
}

func TestCountInventoryRequiresContextAndClient(t *testing.T) {
	client := &fakeClient{}
	if _, err := CountInventory(nil, client); err == nil {
		t.Fatal("nil context accepted")
	}
	if _, err := CountInventory(t.Context(), nil); err == nil {
		t.Fatal("nil client accepted")
	}
	if len(client.queries) != 0 {
		t.Fatal("invalid request queried the source")
	}
	queryErr := errors.New("query failed")
	client.queryErr = queryErr
	if _, err := CountInventory(t.Context(), client); !errors.Is(err, queryErr) {
		t.Fatalf("query failure = %v", err)
	}
}

func TestCountInventoryRetainsReadAndCloseFailures(t *testing.T) {
	closeErr := errors.New("close failed")
	for _, phase := range []string{"vertices", "edges"} {
		for _, failure := range []string{"empty", "close", "canceled", "trailing-read"} {
			t.Run(phase+"/"+failure, func(t *testing.T) {
				stream := &fakeStream{closeErr: closeErr}
				want := closeErr
				switch failure {
				case "empty":
					want = io.EOF
				case "close":
					stream.records = []Record{record(map[string]any{"count": int64(4)})}
				case "canceled":
					stream.block = true
					want = context.Canceled
				case "trailing-read":
					stream.records = []Record{record(map[string]any{"count": int64(4)})}
					stream.nextErr = errors.New("trailing read failed")
					want = stream.nextErr
				}
				client := &fakeClient{streams: []RecordStream{stream}}
				if phase == "edges" {
					client.streams = append([]RecordStream{&fakeStream{
						records: []Record{record(map[string]any{"count": int64(2)})},
					}}, client.streams...)
				}
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				if failure == "canceled" {
					cancel()
				}
				inventory, err := CountInventory(ctx, client)
				if !errors.Is(err, want) || !errors.Is(err, closeErr) || inventory != (Inventory{}) {
					t.Fatalf("inventory = %#v, error = %v", inventory, err)
				}
				if stream.closeCalls != 1 {
					t.Fatalf("stream closed %d times, want 1", stream.closeCalls)
				}
			})
		}
	}
}

func TestInventoryTotalRowsSaturates(t *testing.T) {
	inventory := Inventory{Vertices: math.MaxInt64 - 1, Edges: 2}
	if got := inventory.TotalRows(); got != math.MaxInt64 {
		t.Fatalf("overflow total = %d", got)
	}
}

func TestCountInventoryRejectsInvalidResults(t *testing.T) {
	for name, stream := range map[string]RecordStream{
		"wrong type": &fakeStream{records: []Record{record(map[string]any{"count": "1"})}},
		"negative":   &fakeStream{records: []Record{record(map[string]any{"count": int64(-1)})}},
		"extra row": &fakeStream{records: []Record{
			record(map[string]any{"count": int64(1)}),
			record(map[string]any{"count": int64(2)}),
		}},
		"query error": &fakeStream{nextErr: errors.New("read failed")},
	} {
		t.Run(name, func(t *testing.T) {
			client := &fakeClient{streams: []RecordStream{stream}}
			if _, err := CountInventory(context.Background(), client); err == nil {
				t.Fatal("CountInventory() error = nil")
			}
		})
	}
}
