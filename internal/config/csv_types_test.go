package config

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

func TestCSVTypeConfiguration(t *testing.T) {
	job := validCSVJob(t)
	job.Source.CSV.Vertices[0].PropertyTypes = map[string]string{"age": "int64"}
	job.Source.CSV.Vertices[0].Properties["age"] = "age"
	encoded, err := json.Marshal(job)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"propertyTypes":{"age":"int64"}`) {
		t.Fatal("type declaration lost")
	}
	if _, err := Parse(encoded); err != nil {
		t.Fatal(err)
	}
	schema, err := jsonschema.NewCompiler().Compile(filepath.Join(moduleRoot(t), "docs/reference/load-job.schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	var document any
	if err := json.Unmarshal(encoded, &document); err != nil {
		t.Fatal(err)
	}
	if err := schema.Validate(document); err != nil {
		t.Fatal(err)
	}
	for _, types := range []map[string]string{{"age": "int64"}, {"age": "boolean[]"}} {
		if err := ValidateCSVPropertyTypes(job.Source.CSV.Vertices[0].Properties, types); err != nil {
			t.Fatal(err)
		}
	}
	for _, types := range []map[string]string{{"missing": "int64"}, {"age": ""}, {"age": "date"}} {
		if err := ValidateCSVPropertyTypes(job.Source.CSV.Vertices[0].Properties, types); err == nil {
			t.Fatal("invalid declaration accepted")
		}
		job.Source.CSV.Vertices[0].PropertyTypes = types
		encoded, err := json.Marshal(job)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Parse(encoded); err == nil {
			t.Fatal("parser accepted invalid property types")
		}
	}
}
