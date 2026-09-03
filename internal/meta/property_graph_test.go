package meta

import (
	"strings"
	"testing"
)

func TestValidatePropertyGraph(t *testing.T) {
	valid := PropertyGraphGeneration{
		JobID:  "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
		Schema: "Graph Data", Graph: "Supply Graph",
		DefinitionFingerprint: strings.Repeat("a", 64),
		State:                 PropertyGraphLoading,
		Labels: []PropertyGraphLabel{
			{Name: "Person", Kind: VertexLabel, Table: "v_person"},
			{
				Name: "KNOWS", Kind: EdgeLabel, Table: "e_knows",
				StartLabel: "Person", EndLabel: "Person",
			},
		},
	}
	if err := validatePropertyGraph(valid); err != nil {
		t.Fatalf("valid property graph rejected: %v", err)
	}

	tests := []struct {
		name string
		edit func(*PropertyGraphGeneration)
		want string
	}{
		{
			name: "invalid state",
			edit: func(value *PropertyGraphGeneration) { value.State = "retired" },
			want: "unsupported property graph state",
		},
		{
			name: "duplicate table",
			edit: func(value *PropertyGraphGeneration) {
				value.Labels[1].Table = value.Labels[0].Table
			},
			want: "duplicate property graph table",
		},
		{
			name: "unknown endpoint",
			edit: func(value *PropertyGraphGeneration) { value.Labels[1].EndLabel = "Missing" },
			want: "unknown end label",
		},
		{
			name: "vertex endpoint",
			edit: func(value *PropertyGraphGeneration) { value.Labels[0].StartLabel = "Person" },
			want: "has edge endpoints",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			value := valid
			value.Labels = append([]PropertyGraphLabel(nil), valid.Labels...)
			test.edit(&value)
			err := validatePropertyGraph(value)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("validatePropertyGraph() error = %v, want %q", err, test.want)
			}
		})
	}
}
