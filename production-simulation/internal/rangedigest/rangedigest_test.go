package rangedigest

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	fixturemodel "github.com/rioriost/agefreighter/production-simulation/internal/fixture"
)

func TestFixtureManifestIsIndependentOfShardLayout(t *testing.T) {
	t.Parallel()

	roots := []string{
		filepath.Join(t.TempDir(), "one-shard"),
		filepath.Join(t.TempDir(), "three-shards"),
	}
	shards := []int{1, 3}
	manifests := make([]Manifest, len(roots))
	for index := range roots {
		_, err := fixturemodel.Generate(context.Background(), fixturemodel.GenerateConfig{
			Phase: fixturemodel.PhaseTiny, Output: roots[index], Shards: shards[index],
			Workers: 2, Seed: 20260829,
		})
		if err != nil {
			t.Fatal(err)
		}
		manifests[index], err = FixtureManifest(
			context.Background(), filepath.Join(roots[index], "manifest.json"), 17,
		)
		if err != nil {
			t.Fatal(err)
		}
	}

	if manifests[0].FixtureRoot == manifests[1].FixtureRoot {
		t.Fatal("fixture file roots unexpectedly match across different shard layouts")
	}
	if manifests[0].RootSHA256 != manifests[1].RootSHA256 {
		t.Fatalf("canonical roots differ: %s != %s", manifests[0].RootSHA256, manifests[1].RootSHA256)
	}
	if manifests[0].RecordCount != 560 || len(manifests[0].Leaves) == 0 {
		t.Fatalf("unexpected digest manifest: %#v", manifests[0])
	}
}

func TestCanonicalJSONPropertiesSortsKeysAndPreservesTypes(t *testing.T) {
	t.Parallel()

	properties, encoded, err := canonicalJSONProperties(
		`{"z":[1,"two"],"float":2.0,"integer":2,"a":true}`,
	)
	if err != nil {
		t.Fatal(err)
	}
	const expected = `{"a":true,"float":2.0,"integer":2,"z":[1,"two"]}`
	if string(encoded) != expected {
		t.Fatalf("encoded = %s, want %s", encoded, expected)
	}
	if properties["float"].Kind == properties["integer"].Kind {
		t.Fatal("float and integer canonical types collapsed")
	}
}

func TestFixtureVertexCanonicalizesEmptyStatusAsNull(t *testing.T) {
	t.Parallel()

	_, canonical, err := fixtureVertex("Supplier", []string{
		"1", "supplier-000000000001", "Supplier-1-東京", "JP-13",
		"2020-01-01T00:00:00Z", "", "1.2500", "true",
		"tier-1;segment-2", "1;2;3", "description",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(canonical), `"status":null`) {
		t.Fatalf("canonical vertex does not preserve Neo4j null status: %s", canonical)
	}
}

func TestCompareReportsFirstLeafMismatch(t *testing.T) {
	t.Parallel()

	leaf := Leaf{Kind: "v", Name: "Supplier", RangeIndex: 0, StartKey: 1, EndKey: 2, Rows: 2, SHA256: strings.Repeat("a", 64)}
	expected := Manifest{
		Version: ManifestVersion, CanonicalVersion: CanonicalVersion,
		Source: "fixture", FixtureRoot: strings.Repeat("b", 64), RangeRows: 2,
		RecordCount: 2, Leaves: []Leaf{leaf}, RootSHA256: strings.Repeat("c", 64),
	}
	actual := expected
	actual.Source = "apache-age"
	comparison, err := Compare(expected, actual)
	if err != nil || comparison.Status != "pass" {
		t.Fatalf("matching compare = %#v, %v", comparison, err)
	}

	actual.Leaves = append([]Leaf(nil), actual.Leaves...)
	actual.Leaves[0].SHA256 = strings.Repeat("d", 64)
	comparison, err = Compare(expected, actual)
	if err == nil || comparison.Status != "fail" || !strings.Contains(comparison.Mismatch, "leaf 0") {
		t.Fatalf("mismatch compare = %#v, %v", comparison, err)
	}
}

func TestRangeBuilderRejectsOutOfOrderKeys(t *testing.T) {
	t.Parallel()

	builder, err := newRangeBuilder(10)
	if err != nil {
		t.Fatal(err)
	}
	if err := builder.begin("v", "Supplier"); err != nil {
		t.Fatal(err)
	}
	if err := builder.add(2, []byte("two")); err != nil {
		t.Fatal(err)
	}
	if err := builder.add(1, []byte("one")); err == nil {
		t.Fatal("accepted an out-of-order source key")
	}
}

func TestCanonicalTargetVertexUsesVisibleExternalIdentity(t *testing.T) {
	t.Parallel()

	key, canonical, err := canonicalTargetVertex(
		"Supplier", `{"source_key":1,"external_id":"supplier-000000000001"}`,
	)
	if err != nil {
		t.Fatal(err)
	}
	if key != 1 || !strings.Contains(string(canonical), "supplier-000000000001") {
		t.Fatalf("unexpected target vertex: key=%d canonical=%q", key, canonical)
	}
}

func TestCanonicalTargetEdgeUsesVisibleEndpointIdentities(t *testing.T) {
	t.Parallel()

	key, canonical, err := canonicalTargetEdge(
		"SUPPLIES",
		`{"source_key":1,"relationship_id":"supplies-000000000001"}`,
		"supplier-000000000001",
		"product-000000000001",
	)
	if err != nil {
		t.Fatal(err)
	}
	text := string(canonical)
	for _, expected := range []string{
		"supplies-000000000001",
		"supplier-000000000001",
		"product-000000000001",
	} {
		if !strings.Contains(text, expected) {
			t.Fatalf("target edge %q does not contain %q", text, expected)
		}
	}
	if key != 1 {
		t.Fatalf("target edge key=%d, want 1", key)
	}
}

func TestTargetEndpointIndexUsesFullGraphID(t *testing.T) {
	t.Parallel()

	index := newTargetEndpointIndex()
	graphID := int64(uint64(7)<<48 | 42)
	if err := index.add("Supplier", graphID, 123); err != nil {
		t.Fatal(err)
	}
	key, err := index.lookup("Supplier", graphID)
	if err != nil || key != 123 {
		t.Fatalf("lookup key=%d err=%v", key, err)
	}
	if _, err := index.lookup("Product", graphID); err == nil {
		t.Fatal("accepted endpoint graph ID for the wrong label")
	}
	if err := index.add("Supplier", graphID, 124); err == nil {
		t.Fatal("accepted duplicate endpoint graph ID")
	}
}
