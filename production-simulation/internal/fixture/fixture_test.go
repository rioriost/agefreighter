package fixture

import (
	"context"
	"encoding/csv"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestOnePercentDeltasPreserveBaseAndIdentityDomains(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "base", "manifest.json")
	base, err := Generate(t.Context(), GenerateConfig{
		Phase: PhaseP0, Output: filepath.Dir(path), Shards: 4, Workers: 2, Seed: 20260829,
	})
	if err != nil {
		t.Fatal(err)
	}
	if base.RootSHA256 != "f5c55737a9b8abe9259937920c43504351db48c56b29afd031a813e403b20b1d" {
		t.Fatal("the previously qualified P0 base fixture changed")
	}
	for _, kind := range []DeltaKind{DeltaAppend, DeltaUpsert} {
		t.Run(string(kind), func(t *testing.T) {
			output := filepath.Join(dir, string(kind))
			delta, err := GenerateDelta(t.Context(), path, output, kind)
			if err != nil {
				t.Fatal(err)
			}
			if delta.Plan.VertexTotal != base.Plan.VertexTotal/100 || delta.Plan.EdgeTotal != base.Plan.EdgeTotal/100 {
				t.Fatal("delta is not exactly one percent of each base total")
			}
			if delta.Delta.BaseRoot != base.RootSHA256 {
				t.Fatal("delta does not pin its base")
			}
			if _, err := Verify(filepath.Join(output, "manifest.json")); err != nil {
				t.Fatal(err)
			}
			seen := map[string]int64{}
			var endpointMix [4]int
			for _, entry := range delta.Files {
				if entry.Kind != "node" && entry.Kind != "edge" {
					continue
				}
				var first, offset int64
				for _, spec := range delta.Plan.VertexSpecs {
					if spec.Label == entry.Name {
						first, offset = spec.FirstKey, spec.IDOffset
					}
				}
				for _, spec := range delta.Plan.EdgeSpecs {
					if spec.Type == entry.Name {
						first, offset = spec.FirstKey, spec.IDOffset
					}
				}
				f, err := os.Open(filepath.Join(output, entry.Path))
				if err != nil {
					t.Fatal(err)
				}
				reader := csv.NewReader(f)
				for {
					row, err := reader.Read()
					if err == io.EOF {
						break
					}
					if err != nil {
						t.Fatal(err)
					}
					local := seen[entry.Name]
					if row[0] != strconv.FormatInt(first+local, 10) {
						t.Fatal("source keys are not the frozen prefix/suffix")
					}
					want, status := externalID(entry.Name, offset+local), 5
					if entry.Kind == "edge" {
						want, status = relationshipID(entry.Name, offset+local), 6
						if kind == DeltaAppend {
							start, e1 := strconv.ParseInt(row[2], 10, 64)
							end, e2 := strconv.ParseInt(row[3], 10, 64)
							if e1 != nil || e2 != nil {
								t.Fatal("invalid endpoint key")
							}
							category := 0
							if start > base.Plan.VertexTotal {
								category |= 1
							}
							if end > base.Plan.VertexTotal {
								category |= 2
							}
							endpointMix[category]++
						}
					}
					if row[1] != want || (kind == DeltaUpsert && row[status] != "updated") {
						t.Fatal("delta changed identity or failed to update a selected record")
					}
					seen[entry.Name]++
				}
				if err := f.Close(); err != nil {
					t.Fatal(err)
				}
			}
			if kind == DeltaAppend {
				for _, count := range endpointMix {
					if count == 0 {
						t.Fatal("append omitted an old/new endpoint combination")
					}
				}
			}
			again, err := GenerateDelta(t.Context(), path, filepath.Join(dir, string(kind)+"-repeat"), kind)
			if err != nil || again.RootSHA256 != delta.RootSHA256 {
				t.Fatalf("delta is not deterministic: %v", err)
			}
			if _, err := GenerateDelta(t.Context(), path, output, kind); err == nil {
				t.Fatal("an existing delta was overwritten")
			}
			delta.Delta.BaseRoot = strings.Repeat("0", 64)
			tampered := filepath.Join(output, "tampered.json")
			if err := writeManifest(tampered, delta); err != nil {
				t.Fatal(err)
			}
			if _, err := Verify(tampered); err == nil {
				t.Fatal("delta provenance is not covered by its digest")
			}
		})
	}
	after, err := Verify(path)
	if err != nil || after.RootSHA256 != base.RootSHA256 {
		t.Fatalf("base fixture changed: %v", err)
	}
	for _, phase := range []Phase{PhaseP0, PhaseP1, PhaseP2} {
		for _, kind := range []DeltaKind{DeltaAppend, DeltaUpsert} {
			delta, err := deltaPlan(phase, kind)
			if err != nil {
				t.Fatal(err)
			}
			base, _ := BuildPlan(phase)
			if delta.VertexTotal*100 != base.VertexTotal || delta.EdgeTotal*100 != base.EdgeTotal {
				t.Fatal("larger-phase delta was rounded to the wrong total")
			}
		}
	}
	for _, phase := range []Phase{PhaseTiny, PhaseP3} {
		if _, err := deltaPlan(phase, DeltaAppend); err == nil {
			t.Fatal("unapproved incremental envelope accepted")
		}
	}
}

func TestPlansHaveExactTotalsAndContiguousKeys(t *testing.T) {
	for _, phase := range []Phase{PhaseTiny, PhaseP0, PhaseP1, PhaseP2, PhaseP3} {
		plan, err := BuildPlan(phase)
		if err != nil {
			t.Fatal(err)
		}
		vertexTotal := int64(0)
		nextKey := int64(1)
		for _, item := range plan.VertexSpecs {
			if item.Count <= 0 || item.FirstKey != nextKey {
				t.Fatalf("%s vertex %#v after key %d", phase, item, nextKey)
			}
			vertexTotal += item.Count
			nextKey += item.Count
		}
		edgeTotal := int64(0)
		nextKey = 1
		for _, item := range plan.EdgeSpecs {
			if item.Count <= 0 || item.FirstKey != nextKey {
				t.Fatalf("%s edge %#v after key %d", phase, item, nextKey)
			}
			edgeTotal += item.Count
			nextKey += item.Count
		}
		if vertexTotal != plan.VertexTotal || edgeTotal != plan.EdgeTotal {
			t.Fatalf("%s totals = %d/%d", phase, vertexTotal, edgeTotal)
		}
	}
	if _, err := BuildPlan("unknown"); err == nil {
		t.Fatal("accepted unknown phase")
	}
	full, err := BuildPlan(PhaseP3)
	if err != nil {
		t.Fatal(err)
	}
	for index, item := range full.VertexSpecs {
		if item.Count != vertexModel[index].full {
			t.Fatalf("full vertex %s = %d, want %d", item.Label, item.Count, vertexModel[index].full)
		}
	}
	for index, item := range full.EdgeSpecs {
		if item.Count != edgeModel[index].full {
			t.Fatalf("full edge %s = %d, want %d", item.Type, item.Count, edgeModel[index].full)
		}
	}
}

func TestGenerateVerifyAndDetectModification(t *testing.T) {
	root := filepath.Join(t.TempDir(), "fixture")
	manifest, err := Generate(context.Background(), GenerateConfig{
		Phase: PhaseTiny, Output: root, Shards: 3, Workers: 2, Seed: 20260829,
	})
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Plan.VertexTotal != 160 || manifest.Plan.EdgeTotal != 400 || manifest.RootSHA256 == "" {
		t.Fatalf("manifest = %#v", manifest)
	}
	verified, err := Verify(filepath.Join(root, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if verified.RootSHA256 != manifest.RootSHA256 {
		t.Fatalf("root = %s, want %s", verified.RootSHA256, manifest.RootSHA256)
	}
	if _, err := Generate(context.Background(), GenerateConfig{
		Phase: PhaseTiny, Output: root, Shards: 1, Workers: 1,
	}); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("existing output error = %v", err)
	}

	var dataPath string
	for _, entry := range manifest.Files {
		if entry.Kind == "node" {
			dataPath = filepath.Join(root, filepath.FromSlash(entry.Path))
			break
		}
	}
	file, err := os.OpenFile(dataPath, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString("tampered\n"); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := Verify(filepath.Join(root, "manifest.json")); err == nil {
		t.Fatal("verification accepted a modified fixture")
	}
}

func TestGenerateValidation(t *testing.T) {
	valid := GenerateConfig{Phase: PhaseTiny, Output: filepath.Join(t.TempDir(), "x"), Shards: 1, Workers: 1}
	tests := []GenerateConfig{
		{Phase: PhaseTiny, Shards: 1, Workers: 1},
		{Phase: PhaseTiny, Output: valid.Output, Shards: 0, Workers: 1},
		{Phase: PhaseTiny, Output: valid.Output, Shards: 1, Workers: 257},
		{Phase: "bad", Output: valid.Output, Shards: 1, Workers: 1},
	}
	for index, config := range tests {
		if _, err := Generate(context.Background(), config); err == nil {
			t.Fatalf("case %d succeeded", index)
		}
	}
	if _, err := Generate(nil, valid); err == nil {
		t.Fatal("accepted nil context")
	}
}

func TestVerifyRejectsManifestTampering(t *testing.T) {
	root := filepath.Join(t.TempDir(), "fixture")
	manifest, err := Generate(context.Background(), GenerateConfig{
		Phase: PhaseTiny, Output: root, Shards: 2, Workers: 1, Seed: 9,
	})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		edit func(*Manifest)
	}{
		{
			name: "canonical plan",
			edit: func(value *Manifest) {
				value.Plan.VertexSpecs[0].Count++
			},
		},
		{
			name: "unsafe path",
			edit: func(value *Manifest) {
				value.Files[0].Path = "../outside"
				value.RootSHA256 = manifestRoot(value.Files)
			},
		},
		{
			name: "duplicate path",
			edit: func(value *Manifest) {
				value.Files[1].Path = value.Files[0].Path
				value.RootSHA256 = manifestRoot(value.Files)
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			copyValue := manifest
			copyValue.Plan.VertexSpecs = append([]VertexSpec(nil), manifest.Plan.VertexSpecs...)
			copyValue.Files = append([]FileEntry(nil), manifest.Files...)
			test.edit(&copyValue)
			path := filepath.Join(root, strings.ReplaceAll(test.name, " ", "-")+".json")
			if err := writeManifest(path, copyValue); err != nil {
				t.Fatal(err)
			}
			if _, err := Verify(path); err == nil {
				t.Fatal("accepted tampered manifest")
			}
		})
	}
}

func TestDescriptionWidthBuckets(t *testing.T) {
	tests := []struct {
		value uint64
		want  int
	}{
		{value: 0, want: 32},
		{value: 900, want: 256},
		{value: 990, want: 2048},
		{value: 999, want: 8192},
	}
	for _, test := range tests {
		if got := len(description("Product", test.value)); got != test.want {
			t.Fatalf("description(%d) length = %d, want %d", test.value, got, test.want)
		}
	}
}
