package age

import (
	"strconv"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/pkg/model"
)

func TestDenseEndpointCacheAppliesOnlyCommittedIdentities(t *testing.T) {
	cache, err := newDenseEndpointCache(denseEndpointChunkBytes)
	if err != nil {
		t.Fatalf("newDenseEndpointCache() error = %v", err)
	}
	graphID, err := MakeGraphID(7, 11)
	if err != nil {
		t.Fatalf("MakeGraphID() error = %v", err)
	}
	pending, err := cache.prepare([]vertexIdentityRow{{
		externalID: "42",
		graphID:    graphID,
	}})
	if err != nil {
		t.Fatalf("prepare() error = %v", err)
	}
	if _, ok := cache.lookup("42", 7); ok {
		t.Fatal("prepared identity became visible before commit")
	}
	cache.apply(pending)
	if got, ok := cache.lookup("42", 7); !ok || got != graphID {
		t.Fatalf("lookup() = %v, %t; want %v, true", got, ok, graphID)
	}
	if _, ok := cache.lookup("42", 8); ok {
		t.Fatal("lookup() accepted the wrong endpoint label")
	}
}

func TestDenseEndpointCacheRejectsInvalidAndOversizedIdentities(t *testing.T) {
	cache, err := newDenseEndpointCache(denseEndpointChunkBytes)
	if err != nil {
		t.Fatalf("newDenseEndpointCache() error = %v", err)
	}
	graphID, err := MakeGraphID(7, 11)
	if err != nil {
		t.Fatalf("MakeGraphID() error = %v", err)
	}
	for _, externalID := range []model.ExternalID{"-1", "not-an-id"} {
		if _, err := cache.prepare([]vertexIdentityRow{{
			externalID: externalID,
			graphID:    graphID,
		}}); err == nil || !strings.Contains(err.Error(), "internal ID") {
			t.Fatalf("prepare(%q) error = %v", externalID, err)
		}
	}
	first, err := cache.prepare([]vertexIdentityRow{{
		externalID: "0",
		graphID:    graphID,
	}})
	if err != nil {
		t.Fatalf("prepare(first chunk) error = %v", err)
	}
	cache.apply(first)
	if _, err := cache.prepare([]vertexIdentityRow{{
		externalID: "262144",
		graphID:    graphID,
	}}); err == nil || !strings.Contains(err.Error(), "exceeded") {
		t.Fatalf("prepare(oversized) error = %v", err)
	}
}

func TestResolveEdgesDenseBypassesDatabaseWhenComplete(t *testing.T) {
	cache, err := newDenseEndpointCache(denseEndpointChunkBytes)
	if err != nil {
		t.Fatalf("newDenseEndpointCache() error = %v", err)
	}
	startID, _ := MakeGraphID(2, 10)
	endID, _ := MakeGraphID(3, 20)
	pending, err := cache.prepare([]vertexIdentityRow{
		{externalID: "100", graphID: startID},
		{externalID: "200", graphID: endID},
	})
	if err != nil {
		t.Fatalf("prepare() error = %v", err)
	}
	cache.apply(pending)
	transaction := &loadTransaction{sink: &LoadSink{endpointCache: cache}}
	edges := []stagedEdge{{
		record: &model.Edge{
			Start: model.Endpoint{ExternalID: "100"},
			End:   model.Endpoint{ExternalID: "200"},
		},
		startLabelID: 2,
		endLabelID:   3,
	}}
	resolved, complete := transaction.resolveEdgesDense(edges)
	if !complete || len(resolved) != 1 ||
		resolved[0].startID != startID || resolved[0].endID != endID {
		t.Fatalf("resolveEdgesDense() = %#v, %t", resolved, complete)
	}
	edges[0].record.End.ExternalID = "201"
	if _, complete := transaction.resolveEdgesDense(edges); complete {
		t.Fatal("resolveEdgesDense() accepted an incomplete cache")
	}
}

func BenchmarkResolveEdgesDense20000(b *testing.B) {
	cache, err := newDenseEndpointCache(denseEndpointChunkBytes)
	if err != nil {
		b.Fatal(err)
	}
	edges := make([]stagedEdge, 20_000)
	identities := make([]vertexIdentityRow, 40_000)
	for index := range edges {
		startID, _ := MakeGraphID(2, uint64(index+1))
		endID, _ := MakeGraphID(3, uint64(index+1))
		startExternal := model.ExternalID(stringID(index))
		endExternal := model.ExternalID(stringID(index + len(edges)))
		identities[index*2] = vertexIdentityRow{
			externalID: startExternal,
			graphID:    startID,
		}
		identities[index*2+1] = vertexIdentityRow{
			externalID: endExternal,
			graphID:    endID,
		}
		edges[index] = stagedEdge{
			record: &model.Edge{
				Start: model.Endpoint{ExternalID: startExternal},
				End:   model.Endpoint{ExternalID: endExternal},
			},
			startLabelID: 2,
			endLabelID:   3,
		}
	}
	pending, err := cache.prepare(identities)
	if err != nil {
		b.Fatal(err)
	}
	cache.apply(pending)
	transaction := &loadTransaction{sink: &LoadSink{endpointCache: cache}}
	b.ReportAllocs()
	b.SetBytes(int64(len(edges) * 2))
	b.ResetTimer()
	for range b.N {
		resolved, complete := transaction.resolveEdgesDense(edges)
		if !complete || len(resolved) != len(edges) {
			b.Fatal("dense resolution incomplete")
		}
	}
}

func stringID(value int) string {
	return strconv.Itoa(value)
}
