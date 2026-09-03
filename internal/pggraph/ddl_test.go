package pggraph

import (
	"strings"
	"testing"
)

func TestDefinitionDDL(t *testing.T) {
	definition := Definition{
		Schema: "Graph Data",
		Graph:  "Supply Graph",
		Vertices: []VertexDefinition{
			{Table: "person", Label: "Person"},
			{Table: "supplier", Label: "Supplier"},
		},
		Edges: []EdgeDefinition{
			{
				Table:            "supplies",
				Label:            "SUPPLIES",
				SourceTable:      "supplier",
				DestinationTable: "person",
			},
		},
	}
	statements, err := definition.DDL()
	if err != nil {
		t.Fatalf("DDL() error = %v", err)
	}
	if len(statements) != 6 {
		t.Fatalf("DDL() statement count = %d, want 6", len(statements))
	}
	joined := strings.Join(statements, ";\n")
	for _, expected := range []string{
		`CREATE SCHEMA IF NOT EXISTS "Graph Data"`,
		`CREATE TABLE "Graph Data"."person"`,
		`REFERENCES "Graph Data"."supplier" (id)`,
		`SET LOCAL search_path TO "Graph Data", pg_catalog`,
		`CREATE PROPERTY GRAPH "Supply Graph"`,
		`"supplies" SOURCE KEY (start_id) REFERENCES "supplier" (id)`,
		`LABEL "SUPPLIES"`,
	} {
		if !strings.Contains(joined, expected) {
			t.Errorf("DDL() missing %q:\n%s", expected, joined)
		}
	}
}

func TestDefinitionRejectsInvalidOrAmbiguousElements(t *testing.T) {
	tests := []struct {
		name       string
		definition Definition
		want       string
	}{
		{
			name:       "no vertices",
			definition: Definition{Schema: "public", Graph: "graph"},
			want:       "at least one vertex",
		},
		{
			name: "duplicate labels",
			definition: Definition{
				Schema: "public",
				Graph:  "graph",
				Vertices: []VertexDefinition{
					{Table: "a", Label: "same"},
					{Table: "b", Label: "same"},
				},
			},
			want: "duplicate label",
		},
		{
			name: "unknown endpoint",
			definition: Definition{
				Schema:   "public",
				Graph:    "graph",
				Vertices: []VertexDefinition{{Table: "a", Label: "a"}},
				Edges: []EdgeDefinition{{
					Table:            "edge",
					Label:            "edge",
					SourceTable:      "missing",
					DestinationTable: "a",
				}},
			},
			want: "is not a vertex table",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := test.definition.DDL()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("DDL() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestDefinitionFingerprintIsOrderIndependent(t *testing.T) {
	left := Definition{
		Schema: "graph_data", Graph: "supply_graph",
		Vertices: []VertexDefinition{
			{Table: "v_supplier", Label: "Supplier"},
			{Table: "v_part", Label: "Part"},
		},
		Edges: []EdgeDefinition{
			{
				Table: "e_supplies", Label: "SUPPLIES",
				SourceTable: "v_supplier", DestinationTable: "v_part",
			},
			{
				Table: "e_replaces", Label: "REPLACES",
				SourceTable: "v_part", DestinationTable: "v_part",
			},
		},
	}
	right := Definition{
		Schema: "graph_data", Graph: "supply_graph",
		Vertices: []VertexDefinition{left.Vertices[1], left.Vertices[0]},
		Edges:    []EdgeDefinition{left.Edges[1], left.Edges[0]},
	}

	leftFingerprint, err := left.Fingerprint()
	if err != nil {
		t.Fatal(err)
	}
	rightFingerprint, err := right.Fingerprint()
	if err != nil {
		t.Fatal(err)
	}
	if leftFingerprint != rightFingerprint {
		t.Fatalf("fingerprints differ: %s != %s", leftFingerprint, rightFingerprint)
	}
	leftDDL, err := left.DDL()
	if err != nil {
		t.Fatal(err)
	}
	rightDDL, err := right.DDL()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(leftDDL, "\n") != strings.Join(rightDDL, "\n") {
		t.Fatal("normalized DDL depends on source mapping order")
	}
}
