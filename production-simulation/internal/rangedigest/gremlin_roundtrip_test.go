package rangedigest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
)

func TestGremlinPortableCanonicalParity(t *testing.T) {
	ctx := context.Background()
	original := filepath.Join(t.TempDir(), "original")
	if _, err := fixture.Generate(ctx, fixture.GenerateConfig{Phase: fixture.PhaseTiny, Output: original, Seed: 20260829, Shards: 4, Workers: 2}); err != nil {
		t.Fatal(err)
	}
	input := filepath.Join(original, "manifest.json")
	expected, err := GremlinFixtureManifest(ctx, input, 100000)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := FixtureManifest(ctx, input, 100000)
	if err != nil {
		t.Fatal(err)
	}
	if raw.RootSHA256 == expected.RootSHA256 {
		t.Fatal("partition oracle must differ from raw ID oracle")
	}
	for _, mutation := range []string{"none", "partition", "endpoint", "property", "missing", "duplicate", "checksum", "metadata", "label", "trailing"} {
		t.Run(mutation, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "gremlin")
			m, err := portable.ExportGremlin(ctx, input, dir)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := portable.ExportGremlin(ctx, input, dir); err == nil {
				t.Fatal("must refuse existing directory")
			}
			if mutation == "metadata" {
				m.Tables[0].Rows++
			}
			if mutation != "none" {
				index := 0
				if mutation == "endpoint" {
					index = len(m.Plan.VertexSpecs)
				}
				path := filepath.Join(dir, m.Tables[index].Documents)
				data, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
				var doc map[string]any
				if err := json.Unmarshal([]byte(lines[0]), &doc); err != nil {
					t.Fatal(err)
				}
				switch mutation {
				case "label":
					doc["label"] = "wrong-label"
				case "partition", "checksum":
					doc["partitionKey"] = "wrong-partition"
				case "endpoint":
					doc["_sinkPartition"] = "wrong-sink"
				case "property":
					doc["name"] = []any{map[string]any{"_value": "changed"}}
				}
				b, _ := json.Marshal(doc)
				lines[0] = string(b)
				if mutation == "trailing" {
					lines[0] += " {}"
				}
				if mutation == "missing" {
					lines = lines[1:]
				}
				if mutation == "duplicate" {
					lines = append(lines, lines[0])
				}
				data = []byte(strings.Join(lines, "\n") + "\n")
				if err := os.WriteFile(path, data, 0600); err != nil {
					t.Fatal(err)
				}
				if mutation != "checksum" {
					sum := sha256.Sum256(data)
					m.Tables[index].DocumentsHash = hex.EncodeToString(sum[:])
				}
			}
			actual, err := GremlinDocumentsManifest(ctx, dir, m, 100000)
			if err == nil {
				_, err = CompareOfflineGremlin(expected, actual)
			}
			if mutation != "none" {
				if err == nil {
					t.Fatal("corruption accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if actual.RecordCount != 560 {
				t.Fatalf("rows=%d", actual.RecordCount)
			}
			t.Logf("OFFLINE ONLY: records=%d ranges=%d root=%s", actual.RecordCount, len(actual.Leaves), actual.RootSHA256)
			actual.Source = "apache-age"
			if _, err := CompareOfflineGremlin(expected, actual); err == nil {
				t.Fatal("target role accepted by offline comparator")
			}
		})
	}
}

func TestGremlinNumberTransportPreservesPrecision(t *testing.T) {
	for _, tc := range []struct{ input, want string }{{"1.0000", "1"}, {"120.00", "120"}, {"1.2500", "1.2500"}, {"9007199254740993", "9007199254740993"}, {"9007199254740993.0000", "9007199254740993"}, {"1e-3", "1e-3"}} {
		if got := normalizeGremlinNumbers(json.Number(tc.input)); got != json.Number(tc.want) {
			t.Fatalf("%s: %v", tc.input, got)
		}
	}
}

// Optional full frozen-fixture rerun; never creates or changes fixture files.
func TestGremlinFrozenP1(t *testing.T) {
	input, dir := os.Getenv("AF_P1_GREMLIN_FIXTURE"), os.Getenv("AF_P1_GREMLIN_PORTABLE")
	if input == "" && dir == "" {
		t.Skip("set both AF_P1_GREMLIN_FIXTURE and AF_P1_GREMLIN_PORTABLE for full P1")
	}
	if input == "" || dir == "" {
		t.Fatal("both frozen-fixture paths are required")
	}
	data, err := os.ReadFile(filepath.Join(dir, "portable-manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var m portable.Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if m.Plan.Phase != fixture.PhaseP1 {
		t.Fatal("requires P1")
	}
	expected, err := GremlinFixtureManifest(t.Context(), input, 100000)
	if err != nil {
		t.Fatal(err)
	}
	actual, err := GremlinDocumentsManifest(t.Context(), dir, m, 100000)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CompareOfflineGremlin(expected, actual); err != nil {
		t.Fatal(err)
	}
	if actual.RecordCount != 5_600_000 || len(actual.Leaves) != 64 {
		t.Fatal("incomplete P1 coverage")
	}
	t.Logf("OFFLINE ONLY: fixture=%s records=%d ranges=%d root=%s", expected.FixtureRoot, actual.RecordCount, len(actual.Leaves), actual.RootSHA256)
}
