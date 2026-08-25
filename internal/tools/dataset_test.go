package tools

import (
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
)

func TestGenerateDatasetIsDeterministicAndValid(t *testing.T) {
	root := t.TempDir()
	spec := DatasetSpec{Vertices: 5, Edges: 20, Seed: 42}
	first, err := GenerateDataset(filepath.Join(root, "first"), spec)
	if err != nil {
		t.Fatalf("GenerateDataset(first) error = %v", err)
	}
	second, err := GenerateDataset(filepath.Join(root, "second"), spec)
	if err != nil {
		t.Fatalf("GenerateDataset(second) error = %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("manifests differ:\n%#v\n%#v", first, second)
	}
	for _, file := range first.Files {
		firstData := readFile(t, filepath.Join(root, "first", file.Name))
		secondData := readFile(t, filepath.Join(root, "second", file.Name))
		if !bytes.Equal(firstData, secondData) {
			t.Errorf("%s is not deterministic", file.Name)
		}
		sum := sha256.Sum256(firstData)
		if hex.EncodeToString(sum[:]) != file.SHA256 {
			t.Errorf("%s checksum = %s", file.Name, file.SHA256)
		}
	}

	var stored DatasetManifest
	manifestData := readFile(t, filepath.Join(root, "first", "manifest.json"))
	if err := json.Unmarshal(manifestData, &stored); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if !reflect.DeepEqual(stored, first) {
		t.Fatalf("stored manifest = %#v, want %#v", stored, first)
	}

	records := readCSV(t, filepath.Join(root, "first", "edges.csv"))
	if len(records) != int(spec.Edges)+1 {
		t.Fatalf("edge CSV rows = %d", len(records))
	}
	for index, record := range records[1:] {
		start, err := strconv.ParseUint(record[1], 10, 64)
		if err != nil {
			t.Fatalf("edge %d start: %v", index, err)
		}
		end, err := strconv.ParseUint(record[2], 10, 64)
		if err != nil {
			t.Fatalf("edge %d end: %v", index, err)
		}
		if start == 0 || start > spec.Vertices ||
			end == 0 || end > spec.Vertices ||
			start == end {
			t.Errorf("edge %d endpoints = %d -> %d", index, start, end)
		}
	}
}

func TestGenerateDatasetRejectsInvalidOrExistingOutput(t *testing.T) {
	root := t.TempDir()
	if _, err := GenerateDataset("", DatasetSpec{Vertices: 1}); err == nil {
		t.Fatal("GenerateDataset() accepted empty output")
	}
	if _, err := GenerateDataset(
		filepath.Join(root, "zero"),
		DatasetSpec{},
	); err == nil {
		t.Fatal("GenerateDataset() accepted zero vertices")
	}
	existing := filepath.Join(root, "existing")
	if err := os.Mkdir(existing, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := GenerateDataset(
		existing,
		DatasetSpec{Vertices: 1},
	); err == nil {
		t.Fatal("GenerateDataset() overwrote existing output")
	}
	parentFile := filepath.Join(root, "parent-file")
	if err := os.WriteFile(parentFile, []byte("not a directory"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := GenerateDataset(
		filepath.Join(parentFile, "dataset"),
		DatasetSpec{Vertices: 1},
	); err == nil {
		t.Fatal("GenerateDataset() accepted a file as its parent directory")
	}
}

func TestGenerateDatasetSupportsOneVertexAndNoEdges(t *testing.T) {
	output := filepath.Join(t.TempDir(), "dataset")
	manifest, err := GenerateDataset(output, DatasetSpec{Vertices: 1, Seed: 9})
	if err != nil {
		t.Fatalf("GenerateDataset() error = %v", err)
	}
	if manifest.Edges != 0 {
		t.Fatalf("manifest edges = %d", manifest.Edges)
	}
	records := readCSV(t, filepath.Join(output, "edges.csv"))
	if len(records) != 1 {
		t.Fatalf("edge CSV rows = %d, want header only", len(records))
	}
}

func TestDatasetWritersRefuseExistingFiles(t *testing.T) {
	path := filepath.Join(t.TempDir(), "existing")
	if err := os.WriteFile(path, []byte("preserve"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := writeCSV(path, []string{"id"}, 0, func(uint64) []string {
		return nil
	}); err == nil {
		t.Fatal("writeCSV() overwrote an existing file")
	}
	if err := writeManifest(path, DatasetManifest{}); err == nil {
		t.Fatal("writeManifest() overwrote an existing file")
	}
	if got := string(readFile(t, path)); got != "preserve" {
		t.Fatalf("existing file = %q", got)
	}
}

func TestGenerateCommand(t *testing.T) {
	output := filepath.Join(t.TempDir(), "fixture")
	var stdout bytes.Buffer
	command := NewGenerateCommand()
	command.SetOut(&stdout)
	command.SetArgs([]string{"fixture", "--output", output})
	if err := command.Execute(); err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if stdout.String() != "generated 4 vertices and 6 edges in "+output+" (seed 1)\n" {
		t.Fatalf("stdout = %q", stdout.String())
	}

	command = NewGenerateCommand()
	command.SetArgs([]string{
		"benchmark",
		"--output", filepath.Join(t.TempDir(), "benchmark"),
		"--vertices", "3",
		"--edges", "7",
		"--seed", "99",
	})
	if err := command.Execute(); err != nil {
		t.Fatalf("benchmark Execute() error = %v", err)
	}

	command = NewGenerateCommand()
	command.SetArgs([]string{"fixture"})
	if err := command.Execute(); err == nil {
		t.Fatal("fixture command accepted a missing output path")
	}
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v", path, err)
	}
	return data
}

func readCSV(t *testing.T, path string) [][]string {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("Open(%s) error = %v", path, err)
	}
	defer file.Close()
	records, err := csv.NewReader(file).ReadAll()
	if err != nil {
		t.Fatalf("ReadAll(%s) error = %v", path, err)
	}
	return records
}
