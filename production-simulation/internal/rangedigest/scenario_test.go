package rangedigest

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/source/cosmos"
	"github.com/rioriost/agefreighter/pkg/model"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
)

func TestScenarioOracleMatchesCSVAfterAppendAndUpsert(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	paths := Scenario{Base: filepath.Join(root, "base", "manifest.json"),
		Append: filepath.Join(root, "append", "manifest.json"), Upsert: filepath.Join(root, "upsert", "manifest.json")}
	base, err := fixture.Generate(ctx, fixture.GenerateConfig{
		Phase: fixture.PhaseP0, Output: filepath.Dir(paths.Base), Shards: 4, Workers: 2, Seed: 20260829,
	})
	if err != nil {
		t.Fatal(err)
	}
	for path, kind := range map[string]fixture.DeltaKind{paths.Append: fixture.DeltaAppend, paths.Upsert: fixture.DeltaUpsert} {
		if _, err := fixture.GenerateDelta(ctx, paths.Base, filepath.Dir(path), kind); err != nil {
			t.Fatal(err)
		}
	}
	const rangeRows = 10_000
	unmodified, err := ScenarioManifest(ctx, Scenario{Base: paths.Base}, rangeRows, false)
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := FixtureManifest(ctx, paths.Base, rangeRows)
	if err != nil || unmodified.RootSHA256 != legacy.RootSHA256 || unmodified.FixtureRoot != base.RootSHA256 {
		t.Fatalf("base digest changed: %v", err)
	}
	upsertOnly, err := ScenarioManifest(ctx, Scenario{Base: paths.Base, Upsert: paths.Upsert}, rangeRows, false)
	if err != nil || upsertOnly.RecordCount != 458_000 || upsertOnly.RootSHA256 == legacy.RootSHA256 {
		t.Fatalf("upsert did not modify existing records only: %v", err)
	}
	expected, err := ScenarioManifest(ctx, paths, rangeRows, false)
	if err != nil || expected.RecordCount != 462_580 {
		t.Fatalf("append plus upsert must finish at 101%%, not 102%%: %d / %v", expected.RecordCount, err)
	}
	manifests := make([]portable.Manifest, 3)
	dirs := make([]string, 3)
	for i, path := range []string{paths.Base, paths.Append, paths.Upsert} {
		dirs[i] = filepath.Join(root, "portable-"+filepath.Base(filepath.Dir(path)))
		manifests[i], err = portable.Export(ctx, path, dirs[i])
		if err != nil {
			t.Fatal(err)
		}
	}
	snapshot := filepath.Join(root, "snapshot")
	if err := os.Mkdir(snapshot, 0700); err != nil {
		t.Fatal(err)
	}
	source := manifests[0].CSVSource()
	for i, table := range manifests[0].Tables {
		rows := make(map[string][]string)
		readScenarioCSV(t, filepath.Join(dirs[2], manifests[2].Tables[i].CSV), func(row []string) {
			rows[row[1]] = row
		})
		file, err := os.Create(filepath.Join(snapshot, table.CSV))
		if err != nil {
			t.Fatal(err)
		}
		writer := csv.NewWriter(file)
		if err := writer.Write(table.Columns); err != nil {
			t.Fatal(err)
		}
		emit := func(row []string) {
			if err := writer.Write(row); err != nil {
				t.Fatal(err)
			}
		}
		// This independent test projection is identity-keyed, unlike the
		// streaming raw-fixture oracle's canonical prefix overlay.
		readScenarioCSV(t, filepath.Join(dirs[0], table.CSV), func(row []string) {
			if update, ok := rows[row[1]]; ok {
				status := 5
				if table.Kind == "edge" {
					status = 6
				}
				for column := range row {
					if column == status {
						if row[column] == update[column] {
							t.Fatal("selected upsert did not change its property")
						}
					} else if row[column] != update[column] {
						t.Fatal("property-only upsert changed an identity, endpoint or unrelated property")
					}
				}
				emit(update)
				delete(rows, row[1])
			} else {
				emit(row)
			}
		})
		if len(rows) != 0 {
			t.Fatal("upsert introduced previously nonexistent identities")
		}
		readScenarioCSV(t, filepath.Join(dirs[1], manifests[1].Tables[i].CSV), emit)
		writer.Flush()
		if err := writer.Error(); err != nil {
			t.Fatal(err)
		}
		if err := file.Close(); err != nil {
			t.Fatal(err)
		}
	}
	for i := range source.Vertices {
		source.Vertices[i].Path = filepath.Join(snapshot, source.Vertices[i].Path)
	}
	for i := range source.Edges {
		source.Edges[i].Path = filepath.Join(snapshot, source.Edges[i].Path)
	}
	actual, err := CSVManifest(ctx, source, expected.FixtureRoot, rangeRows)
	if err != nil {
		t.Fatal(err)
	}
	if actual.RootSHA256 != expected.RootSHA256 || actual.RecordCount != expected.RecordCount {
		t.Fatalf("independent CSV connector differs: %s vs %s / %d", actual.RootSHA256, expected.RootSHA256, actual.RecordCount)
	}
	withoutUpdates, err := ScenarioManifest(ctx, Scenario{Base: paths.Base, Append: paths.Append}, rangeRows, false)
	if err != nil || withoutUpdates.RootSHA256 == expected.RootSHA256 {
		t.Fatalf("oracle failed to detect a missing upsert: %v", err)
	}
	for _, invalid := range []Scenario{
		{Base: paths.Append}, {Base: paths.Base, Append: paths.Upsert}, {Base: paths.Base, Upsert: paths.Append},
	} {
		if _, err := readScenario(invalid); err == nil {
			t.Fatal("invalid scenario provenance or mode accepted")
		}
	}
	gremlinDir := filepath.Join(root, "gremlin-append")
	gm, err := portable.ExportGremlin(ctx, paths.Append, gremlinDir)
	if err != nil {
		t.Fatal(err)
	}
	added, err := fixture.Verify(paths.Append)
	if err != nil {
		t.Fatal(err)
	}
	spec := added.Plan.VertexSpecs[0]
	lines := map[int64][]byte{}
	if _, err := visitFixtureFiles(ctx, filepath.Dir(paths.Append), fixturePaths(added, "node", spec.Label),
		func(row []string) error {
			key, line, err := gremlinFixtureVertex(spec, row)
			if err == nil {
				lines[key] = line
			}
			return err
		}); err != nil {
		t.Fatal(err)
	}
	client := &gremlinFileClient{files: map[string]string{spec.Label: filepath.Join(gremlinDir, gm.Tables[0].Documents)},
		pagers: map[string]*gremlinFilePager{}}
	defer client.Close()
	it, err := cosmos.NewIterator(ctx, cosmos.IteratorOptions{Namespace: "offline", Client: client,
		Source: config.CosmosSource{Endpoint: "https://offline.invalid/", Database: "delta", PageSize: 100,
			Vertices: []config.CosmosVertexQuery{{Container: "graph", Label: spec.Label, Query: spec.Label,
				IDField: "/id", DocumentFormat: config.CosmosDocumentGremlin, PartitionKeyProperty: "partitionKey",
				MaxProperties: 128, PropertyTypes: gm.Tables[0].Types}}}})
	if err != nil {
		t.Fatal(err)
	}
	defer it.Close()
	for {
		item, err := it.Next(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		vertex := item.Record.Vertex
		key := vertex.Properties["source_key"].Integer
		encoded, err := model.EncodeProperties(vertex.Properties)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(lines[key], vertexLine(spec.Label, key, string(vertex.ExternalID), encoded)) {
			t.Fatal("appended Gremlin partition or typed properties differ from the independent key-offset oracle")
		}
		delete(lines, key)
	}
	if len(lines) != 0 {
		t.Fatal("appended Gremlin records were omitted")
	}
	base.Seed++
	encoded, err := json.Marshal(base)
	if err != nil {
		t.Fatal(err)
	}
	altered := filepath.Join(filepath.Dir(paths.Base), "altered.json")
	if err := os.WriteFile(altered, encoded, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := readScenario(Scenario{Base: altered, Append: paths.Append}); err == nil {
		t.Fatal("delta accepted with a different seed")
	}
}

func readScenarioCSV(t *testing.T, path string, visit func([]string)) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	if _, err := reader.Read(); err != nil {
		t.Fatal(err)
	}
	for {
		row, err := reader.Read()
		if errors.Is(err, io.EOF) {
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		visit(row)
	}
}

func TestScenarioHonorsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := ScenarioManifest(ctx, Scenario{Base: "missing"}, 1000, false); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation must precede input I/O: %v", err)
	}
}
