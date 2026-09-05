// Package portable adapts the immutable simulation fixture to ordinary CSV,
// relational COPY and Cosmos documents. It does not provision or contact Azure.
package portable

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
)

type Table struct {
	Name          string            `json:"name"`
	Kind          string            `json:"kind"`
	StartLabel    string            `json:"startLabel,omitempty"`
	EndLabel      string            `json:"endLabel,omitempty"`
	Columns       []string          `json:"columns"`
	Types         map[string]string `json:"types"`
	Rows          int64             `json:"rows"`
	CSV           string            `json:"csv"`
	CSVHash       string            `json:"csvSha256"`
	Documents     string            `json:"documents"`
	DocumentsHash string            `json:"documentsSha256"`
}

type Manifest struct {
	Version     int          `json:"version"`
	FixtureRoot string       `json:"fixtureRoot"`
	Plan        fixture.Plan `json:"plan"`
	Tables      []Table      `json:"tables"`
}

// Export writes only to a new directory, retaining partial output on failure.
func Export(ctx context.Context, manifestPath, output string) (Manifest, error) {
	if ctx == nil || output == "" {
		return Manifest{}, errors.New("context and new output directory are required")
	}
	input, err := fixture.Verify(manifestPath)
	if err != nil {
		return Manifest{}, err
	}
	if err := ctx.Err(); err != nil {
		return Manifest{}, err
	}
	if err := os.Mkdir(output, 0700); err != nil {
		return Manifest{}, err
	}
	result := Manifest{Version: 1, FixtureRoot: input.RootSHA256, Plan: input.Plan}
	vertices := map[string]fixture.VertexSpec{}
	for _, v := range input.Plan.VertexSpecs {
		vertices[v.Label] = v
	}
	for _, v := range input.Plan.VertexSpecs {
		result.Tables = append(result.Tables, Table{Name: v.Label, Kind: "node", Rows: v.Count,
			Columns: []string{"source_key", "external_id", "name", "region", "created_at", "status", "score", "active", "tags", "quantities", "description"},
			Types:   map[string]string{"source_key": "int64", "score": "float64", "active": "boolean", "tags": "string[]", "quantities": "int64[]"}})
	}
	for _, e := range input.Plan.EdgeSpecs {
		result.Tables = append(result.Tables, Table{Name: e.Type, Kind: "edge", StartLabel: e.Start, EndLabel: e.End, Rows: e.Count,
			Columns: []string{"source_key", "relationship_id", "start_id", "end_id", "occurred_at", "quantity", "status", "distance_km", "notes"},
			Types:   map[string]string{"source_key": "int64", "quantity": "int64", "distance_km": "float64"}})
	}
	for index := range result.Tables {
		if err := exportTable(ctx, filepath.Dir(manifestPath), output, input.Files, &result.Tables[index], vertices); err != nil {
			return Manifest{}, err
		}
	}
	// Verify the input again: never certify output from changed source shards.
	if _, err := fixture.Verify(manifestPath); err != nil {
		return Manifest{}, err
	}
	if err := writeJSON(filepath.Join(output, "portable-manifest.json"), result); err != nil {
		return Manifest{}, err
	}
	if err := writeJSON(filepath.Join(output, "csv-source.json"), result.CSVSource()); err != nil {
		return Manifest{}, err
	}
	return result, nil
}

func (manifest Manifest) CSVSource() config.CSVSource {
	header, null := true, `\N`
	source := config.CSVSource{Defaults: config.DelimitedOptions{Delimiter: ",", Quote: `"`, Escape: `"`, Header: &header, Encoding: "utf-8", NullValue: &null}}
	for _, table := range manifest.Tables {
		props := map[string]string{}
		for _, name := range table.Columns {
			if name != "start_id" && name != "end_id" {
				props[name] = name
			}
		}
		if table.Kind == "node" {
			source.Vertices = append(source.Vertices, config.CSVVertex{Label: table.Name, Path: table.CSV, IDColumn: "external_id", Properties: props, PropertyTypes: table.Types})
		} else {
			source.Edges = append(source.Edges, config.CSVEdge{Label: table.Name, Path: table.CSV, ExternalIDColumn: "relationship_id",
				Start: config.EndpointMapping{Label: table.StartLabel, Field: "start_id"}, End: config.EndpointMapping{Label: table.EndLabel, Field: "end_id"}, Properties: props, PropertyTypes: table.Types})
		}
	}
	return source
}

func exportTable(ctx context.Context, root, output string, entries []fixture.FileEntry, table *Table, vertices map[string]fixture.VertexSpec) error {
	table.CSV = table.Name + ".csv"
	table.Documents = table.Name + ".jsonl"
	cf, err := os.OpenFile(filepath.Join(output, table.CSV), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer cf.Close()
	df, err := os.OpenFile(filepath.Join(output, table.Documents), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer df.Close()
	ch, dh := sha256.New(), sha256.New()
	cb, db := bufio.NewWriterSize(io.MultiWriter(cf, ch), 256<<10), bufio.NewWriterSize(io.MultiWriter(df, dh), 256<<10)
	cw, dw := csv.NewWriter(cb), json.NewEncoder(db)
	if err := cw.Write(table.Columns); err != nil {
		return err
	}
	paths := []string{}
	for _, entry := range entries {
		if entry.Kind == table.Kind && entry.Name == table.Name {
			paths = append(paths, entry.Path)
		}
	}
	slices.Sort(paths)
	var count int64
	for _, path := range paths {
		err := convertFile(ctx, filepath.Join(root, path), func(row []string) error {
			if len(row) != len(table.Columns) {
				return errors.New("fixture column count mismatch")
			}
			if table.Kind == "node" {
				if row[5] == "" {
					row[5] = `\N`
				}
				tags, err := json.Marshal(strings.Split(row[8], ";"))
				if err != nil {
					return err
				}
				row[8] = string(tags)
				parts := strings.Split(row[9], ";")
				numbers := make([]int64, len(parts))
				for i, part := range parts {
					value, err := strconv.ParseInt(part, 10, 64)
					if err != nil {
						return err
					}
					numbers[i] = value
				}
				quantities, err := json.Marshal(numbers)
				if err != nil {
					return err
				}
				row[9] = string(quantities)
			} else {
				for i, label := range []string{table.StartLabel, table.EndLabel} {
					key, err := strconv.ParseInt(row[i+2], 10, 64)
					if err != nil {
						return err
					}
					v := vertices[label]
					if key < v.FirstKey || key >= v.FirstKey+v.Count {
						return errors.New("fixture endpoint outside label range")
					}
					row[i+2] = fmt.Sprintf("%s-%012d", strings.ToLower(label), key-v.FirstKey+1)
				}
			}
			if err := cw.Write(row); err != nil {
				return err
			}
			doc := map[string]any{"id": row[1], "label": table.Name, "kind": table.Kind, "partitionKey": table.Name + "-" + strconv.FormatInt(count%64, 10)}
			for index, name := range table.Columns {
				var value any = row[index]
				if row[index] == `\N` {
					value = nil
				} else if typ := table.Types[name]; typ != "" {
					// All typed fixture cells are JSON numbers, booleans or arrays.
					decoder := json.NewDecoder(strings.NewReader(row[index]))
					decoder.UseNumber()
					if err := decoder.Decode(&value); err != nil {
						return err
					}
				}
				doc[name] = value
			}
			if err := dw.Encode(doc); err != nil {
				return err
			}
			count++
			return nil
		})
		if err != nil {
			return err
		}
	}
	if count != table.Rows {
		return fmt.Errorf("%s row count %d expected %d", table.Name, count, table.Rows)
	}
	cw.Flush()
	if err := cw.Error(); err != nil {
		return err
	}
	if err := cb.Flush(); err != nil {
		return err
	}
	if err := db.Flush(); err != nil {
		return err
	}
	if err := cf.Sync(); err != nil {
		return err
	}
	if err := df.Sync(); err != nil {
		return err
	}
	table.CSVHash = hex.EncodeToString(ch.Sum(nil))
	table.DocumentsHash = hex.EncodeToString(dh.Sum(nil))
	return nil
}

func convertFile(ctx context.Context, path string, visit func([]string) error) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	r := csv.NewReader(f)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		row, err := r.Read()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if err := visit(row); err != nil {
			return err
		}
	}
}

func writeJSON(path string, value any) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	err = json.NewEncoder(f).Encode(value)
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}
