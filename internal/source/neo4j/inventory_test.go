package neo4j

import (
	"context"
	"errors"
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
