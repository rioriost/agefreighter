package tools

import (
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"strconv"
)

const DatasetFormatVersion = 1

type DatasetSpec struct {
	Vertices uint64
	Edges    uint64
	Seed     uint64
}

type DatasetFile struct {
	Name   string `json:"name"`
	Rows   uint64 `json:"rows"`
	SHA256 string `json:"sha256"`
}

type DatasetManifest struct {
	FormatVersion int           `json:"formatVersion"`
	Seed          uint64        `json:"seed"`
	Vertices      uint64        `json:"vertices"`
	Edges         uint64        `json:"edges"`
	Files         []DatasetFile `json:"files"`
}

func GenerateDataset(outputDirectory string, spec DatasetSpec) (DatasetManifest, error) {
	if outputDirectory == "" {
		return DatasetManifest{}, errors.New("output directory is required")
	}
	if spec.Vertices == 0 {
		return DatasetManifest{}, errors.New("vertex count must be positive")
	}
	outputDirectory = filepath.Clean(outputDirectory)
	if _, err := os.Stat(outputDirectory); err == nil {
		return DatasetManifest{}, fmt.Errorf("output path already exists: %s", outputDirectory)
	} else if !errors.Is(err, os.ErrNotExist) {
		return DatasetManifest{}, fmt.Errorf("inspect output path %s: %w", outputDirectory, err)
	}

	parent := filepath.Dir(outputDirectory)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return DatasetManifest{}, fmt.Errorf("create output parent %s: %w", parent, err)
	}
	temporary, err := os.MkdirTemp(parent, "."+filepath.Base(outputDirectory)+"-")
	if err != nil {
		return DatasetManifest{}, fmt.Errorf("create temporary dataset directory: %w", err)
	}
	keepTemporary := false
	defer func() {
		if !keepTemporary {
			_ = os.RemoveAll(temporary)
		}
	}()

	vertexChecksum, err := writeCSV(
		filepath.Join(temporary, "vertices.csv"),
		[]string{"external_id", "name", "score", "active"},
		spec.Vertices,
		func(index uint64) []string {
			id := index + 1
			return []string{
				strconv.FormatUint(id, 10),
				fmt.Sprintf("person-%d", id),
				strconv.FormatUint((id*37+spec.Seed)%1000, 10),
				strconv.FormatBool((id+spec.Seed)%2 == 0),
			}
		},
	)
	if err != nil {
		return DatasetManifest{}, err
	}
	edgeChecksum, err := writeCSV(
		filepath.Join(temporary, "edges.csv"),
		[]string{"external_id", "start_external_id", "end_external_id", "since"},
		spec.Edges,
		func(index uint64) []string {
			id := index + 1
			start := (index*17+spec.Seed)%spec.Vertices + 1
			end := (index*31+spec.Seed+1)%spec.Vertices + 1
			if spec.Vertices > 1 && end == start {
				end = end%spec.Vertices + 1
			}
			return []string{
				strconv.FormatUint(id, 10),
				strconv.FormatUint(start, 10),
				strconv.FormatUint(end, 10),
				strconv.FormatUint(2000+(id+spec.Seed)%25, 10),
			}
		},
	)
	if err != nil {
		return DatasetManifest{}, err
	}

	manifest := DatasetManifest{
		FormatVersion: DatasetFormatVersion,
		Seed:          spec.Seed,
		Vertices:      spec.Vertices,
		Edges:         spec.Edges,
		Files: []DatasetFile{
			{Name: "vertices.csv", Rows: spec.Vertices, SHA256: vertexChecksum},
			{Name: "edges.csv", Rows: spec.Edges, SHA256: edgeChecksum},
		},
	}
	if err := writeManifest(filepath.Join(temporary, "manifest.json"), manifest); err != nil {
		return DatasetManifest{}, err
	}
	if err := os.Rename(temporary, outputDirectory); err != nil {
		return DatasetManifest{}, fmt.Errorf("publish dataset to %s: %w", outputDirectory, err)
	}
	keepTemporary = true
	return manifest, nil
}

func writeCSV(
	path string,
	header []string,
	rows uint64,
	rowAt func(uint64) []string,
) (string, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return "", fmt.Errorf("create %s: %w", path, err)
	}
	digest := sha256.New()
	writer := csv.NewWriter(io.MultiWriter(file, digest))
	if err := writer.Write(header); err != nil {
		_ = file.Close()
		return "", fmt.Errorf("write header to %s: %w", path, err)
	}
	for index := uint64(0); index < rows; index++ {
		if err := writer.Write(rowAt(index)); err != nil {
			_ = file.Close()
			return "", fmt.Errorf("write row %d to %s: %w", index, path, err)
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		_ = file.Close()
		return "", fmt.Errorf("flush %s: %w", path, err)
	}
	if err := file.Close(); err != nil {
		return "", fmt.Errorf("close %s: %w", path, err)
	}
	return sumHex(digest), nil
}

func writeManifest(path string, manifest DatasetManifest) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("create dataset manifest: %w", err)
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(manifest); err != nil {
		_ = file.Close()
		return fmt.Errorf("write dataset manifest: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close dataset manifest: %w", err)
	}
	return nil
}

func sumHex(digest hash.Hash) string {
	return hex.EncodeToString(digest.Sum(nil))
}
