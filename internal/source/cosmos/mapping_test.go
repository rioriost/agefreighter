package cosmos

import (
	"context"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
)

func TestBuildMappingsPreservesVertexBeforeEdgeOrder(t *testing.T) {
	source := config.CosmosSource{
		Vertices: []config.CosmosVertexQuery{
			{Container: "a", Label: "A", Query: "SELECT * FROM c", IDField: "/id"},
			{Container: "b", Label: "B", Query: "SELECT * FROM c", IDField: "/id"},
		},
		Edges: []config.CosmosEdgeQuery{
			{Container: "c", Label: "C", Query: "SELECT * FROM c",
				Start: config.EndpointMapping{Label: "A", Field: "/fromId"},
				End:   config.EndpointMapping{Label: "B", Field: "/toId"}},
			{Container: "d", Label: "D", Query: "SELECT * FROM c",
				Start: config.EndpointMapping{Label: "A", Field: "/fromId"},
				End:   config.EndpointMapping{Label: "B", Field: "/toId"}},
		},
	}
	mappings, err := buildMappings(context.Background(), "ns", source, 1024)
	if err != nil {
		t.Fatalf("buildMappings: %v", err)
	}
	if len(mappings) != 4 {
		t.Fatalf("buildMappings: got %d mappings, want 4", len(mappings))
	}
	wantKinds := []mappingKind{vertexMapping, vertexMapping, edgeMapping, edgeMapping}
	wantLabels := []string{"A", "B", "C", "D"}
	for index, mapping := range mappings {
		if mapping.kind != wantKinds[index] {
			t.Errorf("mapping[%d].kind = %v, want %v", index, mapping.kind, wantKinds[index])
		}
		if string(mapping.label) != wantLabels[index] {
			t.Errorf("mapping[%d].label = %q, want %q", index, mapping.label, wantLabels[index])
		}
	}
}

func TestBuildMappingsRejectsInvalidPointers(t *testing.T) {
	cases := map[string]config.CosmosSource{
		"vertex idField": {
			Vertices: []config.CosmosVertexQuery{
				{Container: "a", Label: "A", Query: "SELECT * FROM c", IDField: "not-a-pointer"},
			},
		},
		"vertex property": {
			Vertices: []config.CosmosVertexQuery{
				{Container: "a", Label: "A", Query: "SELECT * FROM c", IDField: "/id",
					Properties: map[string]string{"name": "bad"}},
			},
		},
		"edge start field": {
			Edges: []config.CosmosEdgeQuery{
				{Container: "c", Label: "C", Query: "SELECT * FROM c",
					Start: config.EndpointMapping{Label: "A", Field: "bad"},
					End:   config.EndpointMapping{Label: "B", Field: "/toId"}},
			},
		},
		"edge end field": {
			Edges: []config.CosmosEdgeQuery{
				{Container: "c", Label: "C", Query: "SELECT * FROM c",
					Start: config.EndpointMapping{Label: "A", Field: "/fromId"},
					End:   config.EndpointMapping{Label: "B", Field: "bad"}},
			},
		},
		"edge externalIdField": {
			Edges: []config.CosmosEdgeQuery{
				{Container: "c", Label: "C", Query: "SELECT * FROM c",
					ExternalIDField: "bad",
					Start:           config.EndpointMapping{Label: "A", Field: "/fromId"},
					End:             config.EndpointMapping{Label: "B", Field: "/toId"}},
			},
		},
	}
	for name, source := range cases {
		if _, err := buildMappings(context.Background(), "ns", source, 1024); err == nil {
			t.Errorf("%s: expected buildMappings to reject an invalid pointer", name)
		}
	}
}

func TestBuildMappingsRejectsTooManyProperties(t *testing.T) {
	source := config.CosmosSource{
		Vertices: []config.CosmosVertexQuery{
			{Container: "a", Label: "A", Query: "SELECT * FROM c", IDField: "/id",
				Properties: map[string]string{"x": "/x", "y": "/y"}},
		},
	}
	if _, err := buildMappings(context.Background(), "ns", source, 1); err == nil {
		t.Error("expected buildMappings to reject too many properties")
	}
}

func TestBuildMappingsRejectsEmptySource(t *testing.T) {
	if _, err := buildMappings(context.Background(), "ns", config.CosmosSource{}, 1024); err == nil {
		t.Error("expected buildMappings to reject a source with no mappings")
	}
}

func TestBuildMappingsRejectsInvalidDocumentFormat(t *testing.T) {
	base := config.CosmosSource{
		Vertices: []config.CosmosVertexQuery{{
			Container: "a",
			Label:     "A",
			Query:     "SELECT * FROM c",
			IDField:   "/id",
		}},
	}
	tests := []struct {
		name string
		edit func(*config.CosmosVertexQuery)
	}{
		{
			name: "unsupported",
			edit: func(mapping *config.CosmosVertexQuery) {
				mapping.DocumentFormat = "future"
			},
		},
		{
			name: "missing partition key",
			edit: func(mapping *config.CosmosVertexQuery) {
				mapping.DocumentFormat = config.CosmosDocumentGremlin
				mapping.MaxProperties = 10
			},
		},
		{
			name: "missing property limit",
			edit: func(mapping *config.CosmosVertexQuery) {
				mapping.DocumentFormat = config.CosmosDocumentGremlin
				mapping.PartitionKeyProperty = "pk"
			},
		},
		{
			name: "explicit properties",
			edit: func(mapping *config.CosmosVertexQuery) {
				mapping.DocumentFormat = config.CosmosDocumentGremlin
				mapping.PartitionKeyProperty = "pk"
				mapping.MaxProperties = 10
				mapping.Properties = map[string]string{"name": "/name"}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := base
			source.Vertices = append(
				[]config.CosmosVertexQuery(nil),
				base.Vertices...,
			)
			test.edit(&source.Vertices[0])
			if _, err := buildMappings(
				context.Background(),
				"ns",
				source,
				1_024,
			); err == nil {
				t.Fatal("buildMappings() accepted invalid document format")
			}
		})
	}
}

func TestCompileParametersRejectsMissingAtPrefix(t *testing.T) {
	value, err := config.NewCosmosParamValue("x")
	if err != nil {
		t.Fatalf("NewCosmosParamValue: %v", err)
	}
	_, err = compileParameters("Person", []config.CosmosQueryParameter{
		{Name: "kind", Value: value},
	})
	if err == nil {
		t.Error("expected compileParameters to reject a parameter name without an @ prefix")
	}
}

func TestCompileParametersPreservesNativeValues(t *testing.T) {
	value, err := config.NewCosmosParamValue(int64(42))
	if err != nil {
		t.Fatalf("NewCosmosParamValue: %v", err)
	}
	parameters, err := compileParameters("Person", []config.CosmosQueryParameter{
		{Name: "@age", Value: value},
	})
	if err != nil {
		t.Fatalf("compileParameters: %v", err)
	}
	if parameters[0].Name != "@age" || parameters[0].Value != int64(42) {
		t.Errorf("compileParameters = %+v, want {@age 42}", parameters[0])
	}
}
