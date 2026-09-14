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

// Diagnostic counterfactual only: never used by qualification or to accept a
// migrated graph. Compare this root with retained failures to isolate numeric
// representation differences without target credentials or target writes.
func TestDiagnosticIntegralFloatEncoding(t *testing.T) {
	path := os.Getenv("AF_P1_NUMERIC_DIAGNOSTIC_FIXTURE")
	if path == "" {
		t.Skip("optional local frozen-fixture diagnostic")
	}
	m, err := fixturemodel.Verify(path)
	if err != nil {
		t.Fatal(err)
	}
	if m.Plan.Phase != fixturemodel.PhaseP1 {
		t.Fatal("P1 only")
	}
	b, _ := newRangeBuilder(100000)
	vertices := make(map[string]fixturemodel.VertexSpec)
	var changed int64
	convert := func(key int64, line []byte, err error, fields int, property string) (int64, []byte, error) {
		if err != nil {
			return key, nil, err
		}
		parts := bytes.SplitN(line, []byte{0}, fields)
		properties, _, err := canonicalJSONProperties(string(bytes.TrimSuffix(parts[fields-1], []byte{'\n'})))
		if err != nil {
			return key, nil, err
		}
		v := properties[property]
		if v.Kind == model.ValueFloat && math.Trunc(v.Float) == v.Float && v.Float >= math.MinInt64 && v.Float < math.MaxInt64 {
			properties[property] = model.Value{Kind: model.ValueInteger, Integer: int64(v.Float)}
			changed++
		}
		encoded, err := model.EncodeProperties(properties)
		parts[fields-1] = append(encoded, '\n')
		return key, bytes.Join(parts, []byte{0}), err
	}
	for _, s := range m.Plan.VertexSpecs {
		vertices[s.Label] = s
		b.begin("v", s.Label)
		_, err := digestFixtureFiles(context.Background(), filepath.Dir(path), fixturePaths(m, "node", s.Label), func(row []string) (int64, []byte, error) {
			k, l, e := fixtureVertex(s.Label, row)
			return convert(k, l, e, 5, "score")
		}, b)
		if err != nil {
			t.Fatal(err)
		}
		b.end()
	}
	for _, s := range m.Plan.EdgeSpecs {
		b.begin("e", s.Type)
		_, err := digestFixtureFiles(context.Background(), filepath.Dir(path), fixturePaths(m, "edge", s.Type), func(row []string) (int64, []byte, error) {
			k, l, e := fixtureEdge(s, vertices, row)
			return convert(k, l, e, 7, "distance_km")
		}, b)
		if err != nil {
			t.Fatal(err)
		}
		b.end()
	}
	r := b.result("diagnostic-only", m.RootSHA256, "", "")
	t.Logf("COUNTERFACTUAL ONLY: root=%s rows=%d ranges=%d changedFloats=%d", r.RootSHA256, r.RecordCount, len(r.Leaves), changed)
	for _, leaf := range r.Leaves {
		t.Logf("%s/%s/%d %s", leaf.Kind, leaf.Name, leaf.RangeIndex, leaf.SHA256)
	}
}
