package fixture

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
)

func Verify(manifestPath string) (Manifest, error) {
	file, err := os.Open(manifestPath)
	if err != nil {
		return Manifest{}, fmt.Errorf("open manifest: %w", err)
	}
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var manifest Manifest
	err = decoder.Decode(&manifest)
	if err == nil {
		var extra any
		if extraErr := decoder.Decode(&extra); !errors.Is(extraErr, io.EOF) {
			err = errors.New("manifest must contain one JSON value")
		}
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return Manifest{}, fmt.Errorf("decode manifest: %w", err)
	}
	if manifest.Version != ManifestVersion {
		return Manifest{}, fmt.Errorf("unsupported manifest version %d", manifest.Version)
	}
	if manifest.Shards < 1 || manifest.RootSHA256 == "" {
		return Manifest{}, errors.New("manifest is incomplete")
	}
	canonicalPlan, err := BuildPlan(manifest.Plan.Phase)
	if err != nil || !reflect.DeepEqual(manifest.Plan, canonicalPlan) {
		return Manifest{}, errors.New("manifest plan does not match the canonical phase model")
	}
	if manifest.RootSHA256 != manifestRoot(manifest.Files) {
		return Manifest{}, errors.New("manifest root digest does not match its file table")
	}
	if err := verifyCardinality(manifest); err != nil {
		return Manifest{}, err
	}
	root := filepath.Dir(manifestPath)
	seenPaths := make(map[string]struct{}, len(manifest.Files))
	for _, entry := range manifest.Files {
		cleanPath := filepath.Clean(filepath.FromSlash(entry.Path))
		if filepath.IsAbs(cleanPath) || cleanPath == "." || cleanPath == ".." ||
			strings.HasPrefix(cleanPath, ".."+string(filepath.Separator)) {
			return Manifest{}, fmt.Errorf("unsafe manifest path %q", entry.Path)
		}
		if _, duplicate := seenPaths[cleanPath]; duplicate {
			return Manifest{}, fmt.Errorf("duplicate manifest path %q", entry.Path)
		}
		seenPaths[cleanPath] = struct{}{}
		path := filepath.Join(root, cleanPath)
		info, err := os.Stat(path)
		if err != nil {
			return Manifest{}, fmt.Errorf("stat %s: %w", entry.Path, err)
		}
		if info.Size() != entry.Bytes {
			return Manifest{}, fmt.Errorf("size mismatch for %s", entry.Path)
		}
		digest, err := fileDigest(path)
		if err != nil {
			return Manifest{}, err
		}
		if digest != entry.SHA256 {
			return Manifest{}, fmt.Errorf("SHA-256 mismatch for %s", entry.Path)
		}
	}
	return manifest, nil
}

func verifyCardinality(manifest Manifest) error {
	vertexRows := map[string]int64{}
	edgeRows := map[string]int64{}
	for _, entry := range manifest.Files {
		switch entry.Kind {
		case "node":
			vertexRows[entry.Name] += entry.Rows
		case "edge":
			edgeRows[entry.Name] += entry.Rows
		case "node-header", "edge-header":
			if entry.Rows != 1 {
				return fmt.Errorf("header %s must contain one row", entry.Path)
			}
		default:
			return fmt.Errorf("unknown file kind %q", entry.Kind)
		}
	}
	vertexTotal := int64(0)
	for _, spec := range manifest.Plan.VertexSpecs {
		if vertexRows[spec.Label] != spec.Count {
			return fmt.Errorf("vertex row count mismatch for %s", spec.Label)
		}
		vertexTotal += spec.Count
	}
	edgeTotal := int64(0)
	for _, spec := range manifest.Plan.EdgeSpecs {
		if edgeRows[spec.Type] != spec.Count {
			return fmt.Errorf("edge row count mismatch for %s", spec.Type)
		}
		edgeTotal += spec.Count
	}
	if vertexTotal != manifest.Plan.VertexTotal || edgeTotal != manifest.Plan.EdgeTotal {
		return errors.New("plan totals do not match type totals")
	}
	return nil
}

func fileDigest(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open %s: %w", path, err)
	}
	digest := sha256.New()
	_, copyErr := io.Copy(digest, file)
	closeErr := file.Close()
	if copyErr != nil {
		return "", fmt.Errorf("hash %s: %w", path, copyErr)
	}
	if closeErr != nil {
		return "", fmt.Errorf("close %s: %w", path, closeErr)
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}
