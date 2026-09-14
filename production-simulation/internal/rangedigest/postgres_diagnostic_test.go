package rangedigest

import (
	"bytes"
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rioriost/agefreighter/pkg/model"
	fixturemodel "github.com/rioriost/agefreighter/production-simulation/internal/fixture"
)

// Diagnostic only: never change the canonical verifier to accept this lossy
// representation. An explicit retained P1 fixture is required, and is read-only.
func TestDiagnoseP1PostgreSQLIntegralFloatCollapse(t *testing.T) {
	path := os.Getenv("AF_P1_DIAGNOSTIC_FIXTURE")
	if path == "" {
		t.Skip("set AF_P1_DIAGNOSTIC_FIXTURE to diagnose the retained P1 failure")
	}
	f, err := fixturemodel.Verify(path)
	if err != nil {
		t.Fatal(err)
	}
	if f.RootSHA256 != "f74220f6c58f0c1a62f80a567520ffcde43a2499ba48100667ee7b78ff4e2e2f" {
		t.Fatal("not the frozen P1 fixture")
	}
	b, err := newRangeBuilder(100000)
	if err != nil {
		t.Fatal(err)
	}
	collapsed := 0
	wrap := func(field string, convert func([]string) (int64, []byte, error)) func([]string) (int64, []byte, error) {
		return func(row []string) (int64, []byte, error) {
			key, line, err := convert(row)
			if err != nil {
				return key, nil, err
			}
			separator := bytes.LastIndexByte(line, 0)
			properties, _, err := canonicalJSONProperties(string(line[separator+1:]))
			if err != nil {
				return key, nil, err
			}
			v := properties[field]
			if v.Kind == model.ValueFloat && v.Float == math.Trunc(v.Float) && v.Float >= -1e9 && v.Float <= 1e9 {
				properties[field] = model.Value{Kind: model.ValueInteger, Integer: int64(v.Float)}
				collapsed++
			}
			encoded, err := model.EncodeProperties(properties)
			return key, append(append(line[:separator+1:separator+1], encoded...), '\n'), err
		}
	}
	digest := func(kind, name, fileKind, field string, convert func([]string) (int64, []byte, error)) {
		t.Helper()
		if err := b.begin(kind, name); err != nil {
			t.Fatal(err)
		}
		if _, err := digestFixtureFiles(context.Background(), filepath.Dir(path), fixturePaths(f, fileKind, name), wrap(field, convert), b); err != nil {
			t.Fatal(err)
		}
		if err := b.end(); err != nil {
			t.Fatal(err)
		}
	}
	vertices := make(map[string]fixturemodel.VertexSpec)
	for _, spec := range f.Plan.VertexSpecs {
		vertices[spec.Label] = spec
		digest("v", spec.Label, "node", "score", func(row []string) (int64, []byte, error) { return fixtureVertex(spec.Label, row) })
	}
	for _, spec := range f.Plan.EdgeSpecs {
		digest("e", spec.Type, "edge", "distance_km", func(row []string) (int64, []byte, error) { return fixtureEdge(spec, vertices, row) })
	}
	m := b.result("fixture", f.RootSHA256, "", "")
	const observed = "33196eb1524a2310b74f5313a6fa64e96ad7704118eafefa895a33f533ae6cb1"
	if m.RootSHA256 != observed || m.RecordCount != 5600000 || len(m.Leaves) != 64 {
		t.Fatalf("float collapse does not fully explain observed result: root=%s rows=%d leaves=%d", m.RootSHA256, m.RecordCount, len(m.Leaves))
	}
	t.Logf("Diagnostic reproduces failed target root over all %d records / %d ranges; collapsed %d floats. This is not qualification PASS.", m.RecordCount, len(m.Leaves), collapsed)
}
