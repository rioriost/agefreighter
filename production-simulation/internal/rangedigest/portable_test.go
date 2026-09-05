package rangedigest

import (
	"context"
	"errors"
	"io"
	"path/filepath"
	"testing"

	csvsource "github.com/rioriost/agefreighter/internal/source/csv"
	"github.com/rioriost/agefreighter/pkg/model"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
)

func TestPortableFixtureCSVCanonicalParity(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	original := filepath.Join(dir, "original")
	output := filepath.Join(dir, "portable")
	if _, err := fixture.Generate(ctx, fixture.GenerateConfig{Phase: fixture.PhaseTiny, Output: original, Seed: 20260829, Shards: 4, Workers: 2}); err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(original, "manifest.json")
	expected, err := FixtureManifest(ctx, manifestPath, 7)
	if err != nil {
		t.Fatal(err)
	}
	converted, err := portable.Export(ctx, manifestPath, output)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := portable.Export(ctx, manifestPath, output); err == nil {
		t.Fatal("existing output overwritten")
	}
	source := converted.CSVSource()
	for i := range source.Vertices {
		source.Vertices[i].Path = filepath.Join(output, source.Vertices[i].Path)
	}
	for i := range source.Edges {
		source.Edges[i].Path = filepath.Join(output, source.Edges[i].Path)
	}
	iterator, err := csvsource.NewIterator(ctx, csvsource.IteratorOptions{Namespace: "p1", Source: source})
	if err != nil {
		t.Fatal(err)
	}
	defer iterator.Close()
	builder, err := newRangeBuilder(7)
	if err != nil {
		t.Fatal(err)
	}
	kind, name := "", ""
	for {
		item, err := iterator.Next(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		record := item.Record
		var nextKind, nextName string
		var props model.Properties
		var id, start, end string
		if record.Vertex != nil {
			nextKind, nextName = "v", string(record.Vertex.Label)
			props = record.Vertex.Properties
			id = string(record.Vertex.ExternalID)
		} else {
			nextKind, nextName = "e", string(record.Edge.Label)
			props = record.Edge.Properties
			id = string(record.Edge.ExternalID)
			start = string(record.Edge.Start.ExternalID)
			end = string(record.Edge.End.ExternalID)
		}
		if kind != nextKind || name != nextName {
			if kind != "" {
				if err := builder.end(); err != nil {
					t.Fatal(err)
				}
			}
			kind, name = nextKind, nextName
			if err := builder.begin(kind, name); err != nil {
				t.Fatal(err)
			}
		}
		key := props["source_key"]
		if key.Kind != model.ValueInteger {
			t.Fatal("source key lost its integer type")
		}
		encoded, err := model.EncodeProperties(props)
		if err != nil {
			t.Fatal(err)
		}
		line := vertexLine(name, key.Integer, id, encoded)
		if kind == "e" {
			line = edgeLine(name, key.Integer, id, start, end, encoded)
		}
		if err := builder.add(key.Integer, line); err != nil {
			t.Fatal(err)
		}
	}
	if err := builder.end(); err != nil {
		t.Fatal(err)
	}
	actual := builder.result("csv", converted.FixtureRoot, "", "")
	if actual.RootSHA256 != expected.RootSHA256 || actual.RecordCount != 560 {
		t.Fatalf("canonical mismatch: %s vs %s / %d", actual.RootSHA256, expected.RootSHA256, actual.RecordCount)
	}
}
