package age

import (
	"context"
	"io"
	"strings"
	"testing"
)

func TestCopyTextReaderFramesPartialReads(t *testing.T) {
	reader := &copyTextReader{
		rowCount: 2,
		vertexAt: func(index int, output []byte) []byte {
			return append(output, "row\tvalue\n"...)
		},
	}
	var output strings.Builder
	buffer := make([]byte, 3)
	for {
		count, err := reader.Read(buffer)
		output.Write(buffer[:count])
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Read() error = %v", err)
		}
	}
	if output.String() != "row\tvalue\nrow\tvalue\n" {
		t.Fatalf("copy text = %q", output.String())
	}
	if count, err := reader.Read(nil); count != 0 || err != nil {
		t.Fatalf("Read(nil) = %d, %v", count, err)
	}
}

func TestAppendCopyTextEscapesControlCharacters(t *testing.T) {
	got := string(appendCopyText(nil, []byte("a\\b\tc\nd\re")))
	if got != `a\\b\tc\nd\re` {
		t.Fatalf("appendCopyText() = %q", got)
	}
}

func TestCopyRowValidation(t *testing.T) {
	label := LabelCatalog{LabelName: "Person", LabelID: 3, Kind: VertexLabel}
	validID, _ := MakeGraphID(3, 1)
	wrongID, _ := MakeGraphID(4, 1)
	if err := validateVertexRows(label, []VertexRow{{ID: validID, Properties: []byte("{}")}}); err != nil {
		t.Fatalf("validateVertexRows() error = %v", err)
	}
	if err := validateVertexRows(label, []VertexRow{{ID: wrongID, Properties: []byte("{}")}}); err == nil {
		t.Fatal("validateVertexRows() accepted wrong label ID")
	}
	if err := validateVertexRows(label, []VertexRow{{ID: 0, Properties: []byte("{}")}}); err == nil {
		t.Fatal("validateVertexRows() accepted invalid graphid")
	}
	if err := validateVertexRows(label, []VertexRow{{ID: validID}}); err == nil {
		t.Fatal("validateVertexRows() accepted empty properties")
	}

	edgeLabel := LabelCatalog{LabelName: "KNOWS", LabelID: 5, Kind: EdgeLabel}
	edgeID, _ := MakeGraphID(5, 1)
	if err := validateEdgeRows(edgeLabel, []EdgeRow{{
		ID: edgeID, StartID: validID, EndID: validID, Properties: []byte("{}"),
	}}); err != nil {
		t.Fatalf("validateEdgeRows() error = %v", err)
	}
	if err := validateEdgeRows(edgeLabel, []EdgeRow{{
		ID: edgeID, StartID: 0, EndID: validID, Properties: []byte("{}"),
	}}); err == nil {
		t.Fatal("validateEdgeRows() accepted invalid endpoint")
	}
	if err := validateEdgeRows(edgeLabel, []EdgeRow{{
		ID: edgeID, StartID: validID, EndID: 0, Properties: []byte("{}"),
	}}); err == nil {
		t.Fatal("validateEdgeRows() accepted invalid end ID")
	}
	if err := validateEdgeRows(edgeLabel, []EdgeRow{{
		ID: edgeID, StartID: validID, EndID: validID,
	}}); err == nil {
		t.Fatal("validateEdgeRows() accepted empty properties")
	}
}

func TestIDBlock(t *testing.T) {
	block := IDBlock{LabelID: 3, FirstEntry: 10, LastEntry: 12}
	if block.Count() != 3 {
		t.Fatalf("Count() = %d", block.Count())
	}
	id, err := block.GraphID(2)
	if err != nil || id.EntryID() != 12 || id.LabelID() != 3 {
		t.Fatalf("GraphID(2) = %v, %v", id, err)
	}
	if _, err := block.GraphID(3); err == nil {
		t.Fatal("GraphID(3) succeeded")
	}
	if (IDBlock{}).Count() != 0 {
		t.Fatal("empty block count is not zero")
	}
}

func TestCopyOperationsRejectInvalidInputsBeforeDatabaseAccess(t *testing.T) {
	transaction := &Transaction{}
	vertexLabel := LabelCatalog{LabelName: "Person", LabelID: 3, Kind: VertexLabel}
	edgeLabel := LabelCatalog{LabelName: "KNOWS", LabelID: 5, Kind: EdgeLabel}
	vertexID, _ := MakeGraphID(vertexLabel.LabelID, 1)
	edgeID, _ := MakeGraphID(edgeLabel.LabelID, 1)
	vertex := []VertexRow{{ID: vertexID, Properties: []byte("{}")}}
	edge := []EdgeRow{{
		ID: edgeID, StartID: vertexID, EndID: vertexID, Properties: []byte("{}"),
	}}

	if _, err := transaction.CopyVertices(
		context.Background(),
		edgeLabel,
		vertex,
		DirectTextCopy,
	); err == nil {
		t.Fatal("CopyVertices() accepted edge label")
	}
	if _, err := transaction.CopyVertices(
		context.Background(),
		vertexLabel,
		vertex,
		CopyStrategy("other"),
	); err == nil {
		t.Fatal("CopyVertices() accepted unknown strategy")
	}
	if _, err := transaction.CopyEdges(
		context.Background(),
		vertexLabel,
		edge,
		DirectTextCopy,
	); err == nil {
		t.Fatal("CopyEdges() accepted vertex label")
	}
	if _, err := transaction.CopyEdges(
		context.Background(),
		edgeLabel,
		edge,
		CopyStrategy("other"),
	); err == nil {
		t.Fatal("CopyEdges() accepted unknown strategy")
	}
}

func TestRequireAffectedRows(t *testing.T) {
	if affected, err := requireAffectedRows("merge", 2, 2); err != nil || affected != 2 {
		t.Fatalf("requireAffectedRows() = %d, %v", affected, err)
	}
	if _, err := requireAffectedRows("merge", 1, 2); err == nil {
		t.Fatal("requireAffectedRows() accepted mismatched count")
	}
}
