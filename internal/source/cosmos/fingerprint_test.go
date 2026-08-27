package cosmos

import (
	"context"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
)

func testSource() config.CosmosSource {
	return config.CosmosSource{
		Endpoint: "https://example.documents.azure.com:443/",
		Database: "graphdb",
		PageSize: 100,
		Vertices: []config.CosmosVertexQuery{
			{
				Container: "people",
				Label:     "Person",
				Query:     "SELECT * FROM c WHERE c.kind = @kind",
				IDField:   "/id",
				Parameters: []config.CosmosQueryParameter{
					{Name: "@kind", Value: mustParamValue("person")},
				},
				Properties: map[string]string{"name": "/name"},
			},
		},
		Edges: []config.CosmosEdgeQuery{
			{
				Container:       "friendships",
				Label:           "KNOWS",
				Query:           "SELECT * FROM c",
				ExternalIDField: "/id",
				Start:           config.EndpointMapping{Label: "Person", Field: "/fromId"},
				End:             config.EndpointMapping{Label: "Person", Field: "/toId"},
				Properties:      map[string]string{"since": "/since"},
			},
		},
	}
}

func mustParamValue(native any) config.CosmosParamValue {
	value, err := config.NewCosmosParamValue(native)
	if err != nil {
		panic(err)
	}
	return value
}

func compileTestMappings(t *testing.T, source config.CosmosSource) []compiledMapping {
	t.Helper()
	mappings, err := buildMappings(context.Background(), "ns", source, 1024)
	if err != nil {
		t.Fatalf("buildMappings: %v", err)
	}
	return mappings
}

func TestBindFingerprintDeterministic(t *testing.T) {
	source := testSource()
	mappings := compileTestMappings(t, source)
	first, err := bindFingerprint(source.Endpoint, source.Database, "ns", int32(source.PageSize), mappings)
	if err != nil {
		t.Fatalf("bindFingerprint: %v", err)
	}
	second, err := bindFingerprint(source.Endpoint, source.Database, "ns", int32(source.PageSize), mappings)
	if err != nil {
		t.Fatalf("bindFingerprint: %v", err)
	}
	if first != second {
		t.Errorf("bindFingerprint is not deterministic: %q != %q", first, second)
	}
}

func TestBindFingerprintChangesWithMeaningfulEdits(t *testing.T) {
	base := testSource()
	baseMappings := compileTestMappings(t, base)
	baseFingerprint, err := bindFingerprint(base.Endpoint, base.Database, "ns", int32(base.PageSize), baseMappings)
	if err != nil {
		t.Fatalf("bindFingerprint: %v", err)
	}

	mutate := func(mutateFn func(*config.CosmosSource)) string {
		mutated := testSource()
		mutateFn(&mutated)
		mappings := compileTestMappings(t, mutated)
		fingerprint, err := bindFingerprint(mutated.Endpoint, mutated.Database, "ns", int32(mutated.PageSize), mappings)
		if err != nil {
			t.Fatalf("bindFingerprint: %v", err)
		}
		return fingerprint
	}

	cases := map[string]func(*config.CosmosSource){
		"endpoint changed":  func(s *config.CosmosSource) { s.Endpoint = "https://other.documents.azure.com:443/" },
		"database changed":  func(s *config.CosmosSource) { s.Database = "otherdb" },
		"page size changed": func(s *config.CosmosSource) { s.PageSize = 250 },
		"query changed": func(s *config.CosmosSource) {
			s.Vertices[0].Query = "SELECT * FROM c WHERE c.kind = @kind AND c.active = true"
		},
		"parameter changed": func(s *config.CosmosSource) { s.Vertices[0].Parameters[0].Value = mustParamValue("organization") },
		"property changed":  func(s *config.CosmosSource) { s.Vertices[0].Properties["name"] = "/fullName" },
		"idField changed":   func(s *config.CosmosSource) { s.Vertices[0].IDField = "/pk" },
	}
	for name, mutateFn := range cases {
		if got := mutate(mutateFn); got == baseFingerprint {
			t.Errorf("%s: expected fingerprint to change, stayed %q", name, got)
		}
	}
}

func TestBindFingerprintMappingOrderMatters(t *testing.T) {
	source := testSource()
	// Duplicate the single vertex mapping with a different label so
	// swapping order is meaningful.
	source.Vertices = append(source.Vertices, config.CosmosVertexQuery{
		Container: "orgs", Label: "Organization", Query: "SELECT * FROM c", IDField: "/id",
	})
	forward := compileTestMappings(t, source)
	forwardFingerprint, err := bindFingerprint(source.Endpoint, source.Database, "ns", int32(source.PageSize), forward)
	if err != nil {
		t.Fatalf("bindFingerprint: %v", err)
	}

	swapped := source
	swapped.Vertices = []config.CosmosVertexQuery{source.Vertices[1], source.Vertices[0]}
	reverse := compileTestMappings(t, swapped)
	reverseFingerprint, err := bindFingerprint(swapped.Endpoint, swapped.Database, "ns", int32(swapped.PageSize), reverse)
	if err != nil {
		t.Fatalf("bindFingerprint: %v", err)
	}

	if forwardFingerprint == reverseFingerprint {
		t.Error("bindFingerprint: expected mapping order to affect the fingerprint")
	}
}

func TestBindFingerprintIncludesGremlinInterpretation(t *testing.T) {
	options := *gremlinSource().Gremlin
	vertex, err := gremlinVertexQuery(options, "AppPerson")
	if err != nil {
		t.Fatal(err)
	}
	source := gremlinSource()
	source.Gremlin = nil
	source.Vertices = []config.CosmosVertexQuery{vertex}
	base := compileTestMappings(t, source)
	baseFingerprint, err := bindFingerprint(
		source.Endpoint,
		source.Database,
		"ns",
		int32(source.PageSize),
		base,
	)
	if err != nil {
		t.Fatal(err)
	}

	source.Vertices[0].PartitionKeyProperty = "tenant"
	changed := compileTestMappings(t, source)
	changedFingerprint, err := bindFingerprint(
		source.Endpoint,
		source.Database,
		"ns",
		int32(source.PageSize),
		changed,
	)
	if err != nil {
		t.Fatal(err)
	}
	if changedFingerprint == baseFingerprint {
		t.Fatal("partition-key property did not affect fingerprint")
	}
}
