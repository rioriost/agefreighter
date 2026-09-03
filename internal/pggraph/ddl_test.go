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
		{
			name: "unknown destination",
			definition: Definition{
				Schema: "public", Graph: "graph",
				Vertices: []VertexDefinition{{Table: "a", Label: "a"}},
				Edges: []EdgeDefinition{{
					Table: "edge", Label: "edge", SourceTable: "a", DestinationTable: "missing",
				}},
			},
			want: "destination table",
		},
	}
	valid := Definition{
		Schema: "public", Graph: "graph",
		Vertices: []VertexDefinition{{Table: "person", Label: "Person"}},
		Edges: []EdgeDefinition{{
			Table: "knows", Label: "KNOWS", SourceTable: "person", DestinationTable: "person",
		}},
	}
	invalid := func(edit func(*Definition)) Definition {
		value := valid
		value.Vertices = append([]VertexDefinition(nil), valid.Vertices...)
		value.Edges = append([]EdgeDefinition(nil), valid.Edges...)
		edit(&value)
		return value
	}
	tests = append(tests,
		struct {
			name       string
			definition Definition
			want       string
		}{"invalid schema", invalid(func(value *Definition) { value.Schema = "" }), "schema"},
		struct {
			name       string
			definition Definition
			want       string
		}{"invalid graph", invalid(func(value *Definition) { value.Graph = "" }), "graph"},
		struct {
			name       string
			definition Definition
			want       string
		}{"invalid vertex table", invalid(func(value *Definition) { value.Vertices[0].Table = "" }), "table"},
		struct {
			name       string
			definition Definition
			want       string
		}{"invalid vertex label", invalid(func(value *Definition) { value.Vertices[0].Label = "" }), "label"},
		struct {
			name       string
			definition Definition
			want       string
		}{"duplicate table", invalid(func(value *Definition) { value.Edges[0].Table = "person" }), "duplicate table"},
		struct {
			name       string
			definition Definition
			want       string
		}{"invalid edge table", invalid(func(value *Definition) { value.Edges[0].Table = "" }), "table"},
		struct {
			name       string
			definition Definition
			want       string
		}{"invalid edge label", invalid(func(value *Definition) { value.Edges[0].Label = "" }), "label"},
	)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := test.definition.DDL()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("DDL() error = %v, want %q", err, test.want)
			}
		})
	}
	if _, err := (Definition{}).Fingerprint(); err == nil {
		t.Fatal("Fingerprint accepted an invalid definition")
	}
	if _, _, err := ReplacementDefinitions(Definition{},
		"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"); err == nil {
		t.Fatal("ReplacementDefinitions accepted an invalid canonical definition")
	}
	normalized := (Definition{
		Vertices: []VertexDefinition{{Table: "b", Label: "same"}, {Table: "a", Label: "same"}},
		Edges: []EdgeDefinition{
			{Table: "b", Label: "same", SourceTable: "b", DestinationTable: "b"},
			{Table: "a", Label: "same", SourceTable: "a", DestinationTable: "a"},
		},
	}).normalized()
	if normalized.Vertices[0].Table != "a" || normalized.Edges[0].Table != "a" {
		t.Fatalf("normalized same-label ordering = %#v", normalized)
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

func TestReplacementDefinitionsAreDeterministicAndDisjoint(t *testing.T) {
	canonical := Definition{
		Schema: "graph_data", Graph: "supply_graph",
		Vertices: []VertexDefinition{
			{Table: "v_supplier", Label: "Supplier"},
			{Table: "v_part", Label: "Part"},
		},
		Edges: []EdgeDefinition{{
			Table: "e_supplies", Label: "SUPPLIES",
			SourceTable: "v_supplier", DestinationTable: "v_part",
		}},
	}
	jobID := "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
	shadow, backup, err := ReplacementDefinitions(canonical, jobID)
	if err != nil {
		t.Fatal(err)
	}
	againShadow, againBackup, err := ReplacementDefinitions(canonical, jobID)
	if err != nil {
		t.Fatal(err)
	}
	if shadow.Graph != againShadow.Graph || backup.Graph != againBackup.Graph {
		t.Fatal("replacement definitions are not deterministic")
	}
	if shadow.Graph == canonical.Graph || backup.Graph == canonical.Graph ||
		shadow.Graph == backup.Graph {
		t.Fatalf("replacement graph names overlap: %q, %q, %q",
			canonical.Graph, shadow.Graph, backup.Graph)
	}
	if !strings.HasPrefix(shadow.Graph, "afs_") || !strings.HasPrefix(backup.Graph, "afb_") {
		t.Fatalf("replacement graph names = %q, %q", shadow.Graph, backup.Graph)
	}
	if len(shadow.Vertices) != 2 || len(shadow.Edges) != 1 ||
		shadow.Edges[0].SourceTable != shadow.Vertices[0].Table ||
		shadow.Edges[0].DestinationTable != shadow.Vertices[1].Table {
		t.Fatalf("replacement shadow mapping = %#v", shadow)
	}
	if _, err := shadow.DDL(); err != nil {
		t.Fatalf("shadow DDL: %v", err)
	}
	if _, _, err := ReplacementDefinitions(canonical, "bad"); err == nil {
		t.Fatal("ReplacementDefinitions accepted an invalid job ID")
	}
}
