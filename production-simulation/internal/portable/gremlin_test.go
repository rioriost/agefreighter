package portable

import "testing"

func TestGremlinPartitionOrdinal(t *testing.T) {
	for _, tc := range []struct{ id, want string }{{"supplier-000000000001", "Supplier-0"}, {"supplier-000000000064", "Supplier-63"}, {"supplier-000000000065", "Supplier-0"}} {
		got, err := gremlinPartition("Supplier", tc.id)
		if err != nil || got != tc.want {
			t.Fatalf("%s: %s %v", tc.id, got, err)
		}
	}
	for _, id := range []string{"supplier-000000000000", "supplier-00000000001", "other-000000000001", "supplier-00000000000x"} {
		if _, err := gremlinPartition("Supplier", id); err == nil {
			t.Fatalf("accepted %q", id)
		}
	}
}
