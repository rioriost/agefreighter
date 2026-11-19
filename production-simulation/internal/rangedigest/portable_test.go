package rangedigest

import (
	"context"
	"path/filepath"
	"testing"

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
	actual, err := CSVManifest(ctx, source, converted.FixtureRoot, 7)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CompareCSV(expected, actual); err != nil || actual.RecordCount != 560 {
		t.Fatalf("CSV canonical mismatch: %d rows: %v", actual.RecordCount, err)
	}
	if _, err := Compare(expected, actual); err == nil {
		t.Fatal("a CSV source-only result cannot be counted as a target pass")
	}
}
