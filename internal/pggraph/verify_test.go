package pggraph

import (
	"errors"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/meta"
)

func TestInspectionValidate(t *testing.T) {
	valid := Inspection{
		Mapping: meta.PropertyGraphGeneration{State: meta.PropertyGraphActive},
		Labels: []LabelInspection{
			{Name: "Person", Kind: meta.VertexLabel, PrimaryKeys: 1, UniqueKeys: 1},
			{Name: "KNOWS", Kind: meta.EdgeLabel, PrimaryKeys: 1, UniqueKeys: 1,
				ForeignKeys: 2},
		},
		DirectedMatches: 2, UndirectedMatch: 4,
	}
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		edit func(*Inspection)
	}{
		{"inactive mapping", func(value *Inspection) { value.Mapping.State = meta.PropertyGraphLoading }},
		{"missing primary key", func(value *Inspection) { value.Labels[0].PrimaryKeys = 0 }},
		{"missing unique key", func(value *Inspection) { value.Labels[0].UniqueKeys = 0 }},
		{"wrong foreign keys", func(value *Inspection) { value.Labels[1].ForeignKeys = 1 }},
		{"missing source", func(value *Inspection) { value.Labels[1].MissingStarts = 1 }},
		{"missing destination", func(value *Inspection) { value.Labels[1].MissingEnds = 1 }},
		{"invalid undirected count", func(value *Inspection) { value.UndirectedMatch = 1 }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			value := valid
			value.Labels = append([]LabelInspection(nil), valid.Labels...)
			test.edit(&value)
			if err := value.Validate(); !errors.Is(err, ErrIntegrity) {
				t.Fatalf("Validate() error = %v", err)
			}
		})
	}
}

func TestInspectValidation(t *testing.T) {
	adapter := &Adapter{}
	validJobID := "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
	validDefinition := Definition{
		Schema: "graph_data", Graph: "supply_graph",
		Vertices: []VertexDefinition{{Table: "person", Label: "Person"}},
	}
	if _, err := adapter.Inspect(t.Context(), "bad", validDefinition); err == nil {
		t.Fatal("Inspect() accepted an invalid job ID")
	}
	if _, err := adapter.Inspect(t.Context(), validJobID, Definition{}); err == nil {
		t.Fatal("Inspect() accepted an invalid definition")
	}
	if _, err := adapter.ComputeDigests(t.Context(), validJobID, Definition{
		Schema: strings.Repeat("s", 64), Graph: "graph",
		Vertices: []VertexDefinition{{Table: "person", Label: "Person"}},
	}); err == nil {
		t.Fatal("ComputeDigests() accepted an invalid definition")
	}
}
