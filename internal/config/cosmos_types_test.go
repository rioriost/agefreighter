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
