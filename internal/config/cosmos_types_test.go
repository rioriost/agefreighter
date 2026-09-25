package config

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/santhosh-tekuri/jsonschema/v6"
	"go.yaml.in/yaml/v3"
)

func TestCosmosTypeConfiguration(t *testing.T) {
	job := validCSVJob(t)
	job.Source = Source{Type: SourceCosmos, Namespace: "p1", Cosmos: &CosmosSource{Endpoint: "https://example.documents.azure.com:443/", Database: "p1", Credential: "default-azure", PageSize: 100, Vertices: []CosmosVertexQuery{{Container: "graph", Label: "Person", Query: "SELECT * FROM c", IDField: "/id", Properties: map[string]string{"score": "/score"}, PropertyTypes: map[string]string{"score": "float64"}}}}}
	schema, err := jsonschema.NewCompiler().Compile(filepath.Join(moduleRoot(t), "docs/reference/load-job.schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, encode := range []func(any) ([]byte, error){json.Marshal, yaml.Marshal} {
		b, err := encode(job)
		if err != nil {
			t.Fatal(err)
		}
		parsed, err := Parse(b)
		if err != nil {
			t.Fatal(err)
		}
		if parsed.Source.Cosmos.Vertices[0].PropertyTypes["score"] != "float64" {
			t.Fatal("type declaration lost")
		}
	}
	b, _ := json.Marshal(job)
	var doc any
	json.Unmarshal(b, &doc)
	if err := schema.Validate(doc); err != nil {
		t.Fatal(err)
	}
	for _, types := range []map[string]string{{"missing": "float64"}, {"score": "date"}, {"score": ""}} {
		job.Source.Cosmos.Vertices[0].PropertyTypes = types
		b, _ := json.Marshal(job)
		if _, err := Parse(b); err == nil {
			t.Fatal("invalid Cosmos type accepted")
		}
	}
}

func TestCosmosGremlinTypeConfiguration(t *testing.T) {
	job := validCSVJob(t)
	job.Source = Source{Type: SourceCosmos, Namespace: "p1", Cosmos: &CosmosSource{Endpoint: "https://example.documents.azure.com:443/", Database: "p1", Credential: "default-azure", PageSize: 100, Gremlin: &CosmosGremlin{Enabled: true, Container: "graph", PartitionKeyProperty: "pk", MaxLabels: 64, MaxProperties: 128, MaxDiscoveryDocuments: 10000, PropertyTypes: map[string]string{"score": "float64", "distance_km": "float64"}}}}
	schema, err := jsonschema.NewCompiler().Compile(filepath.Join(moduleRoot(t), "docs/reference/load-job.schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, encode := range []func(any) ([]byte, error){json.Marshal, yaml.Marshal} {
		b, err := encode(job)
		if err != nil {
			t.Fatal(err)
		}
		parsed, err := Parse(b)
		if err != nil {
			t.Fatal(err)
		}
		if parsed.Source.Cosmos.Gremlin.PropertyTypes["score"] != "float64" {
			t.Fatal("type lost")
		}
	}
	b, _ := json.Marshal(job)
	var doc any
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	if err := schema.Validate(doc); err != nil {
		t.Fatal(err)
	}
	for _, types := range []map[string]string{{"id": "string"}, {"label": "string"}, {"pk": "string"}, {"_internal": "float64"}, {"": "float64"}, {"score": "date"}, {"score": ""}} {
		job.Source.Cosmos.Gremlin.PropertyTypes = types
		b, _ := json.Marshal(job)
		if _, err := Parse(b); err == nil {
			t.Fatalf("invalid declaration accepted: %v", types)
		}
	}
	if err := ValidateCosmosGremlinPropertyTypes("pk", 1, map[string]string{"a": "float64", "b": "float64"}); err == nil {
		t.Fatal("limit ignored")
	}
	for _, typ := range []string{"string", "int64", "float64", "boolean", "string[]", "int64[]", "float64[]", "boolean[]"} {
		if err := ValidateCosmosGremlinPropertyTypes("pk", 1, map[string]string{"value": typ}); err != nil {
			t.Fatal(err)
		}
	}
}
