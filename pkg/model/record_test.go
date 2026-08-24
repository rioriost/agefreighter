package model

import "testing"

func TestRecordKindsAndPositions(t *testing.T) {
	vertexPosition := SourcePosition{Connector: "csv", Offset: 10}
	edgePosition := SourcePosition{Connector: "csv", Offset: 20}
	tests := []struct {
		name     string
		record   Record
		kind     RecordKind
		position SourcePosition
		valid    bool
	}{
		{
			name: "vertex",
			record: VertexRecord(Vertex{
				Label:    "Person",
				Position: vertexPosition,
			}),
			kind:     RecordVertex,
			position: vertexPosition,
			valid:    true,
		},
		{
			name: "edge",
			record: EdgeRecord(Edge{
				Label:    "KNOWS",
				Position: edgePosition,
			}),
			kind:     RecordEdge,
			position: edgePosition,
			valid:    true,
		},
		{name: "empty", record: Record{}, kind: RecordInvalid},
		{
			name: "ambiguous",
			record: Record{
				Vertex: &Vertex{},
				Edge:   &Edge{},
			},
			kind: RecordInvalid,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.record.Kind(); got != test.kind {
				t.Fatalf("Kind() = %v, want %v", got, test.kind)
			}
			position, valid := test.record.SourcePosition()
			if valid != test.valid || position != test.position {
				t.Fatalf(
					"SourcePosition() = (%#v, %v), want (%#v, %v)",
					position,
					valid,
					test.position,
					test.valid,
				)
			}
		})
	}
}
