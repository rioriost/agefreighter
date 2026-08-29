package rangedigest

import (
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/rioriost/agefreighter/pkg/model"
	fixturemodel "github.com/rioriost/agefreighter/production-simulation/internal/fixture"
)

func FixtureManifest(ctx context.Context, manifestPath string, rangeRows int64) (Manifest, error) {
	if ctx == nil {
		return Manifest{}, errors.New("context is required")
	}
	fixtureManifest, err := fixturemodel.Verify(manifestPath)
	if err != nil {
		return Manifest{}, err
	}
	builder, err := newRangeBuilder(rangeRows)
	if err != nil {
		return Manifest{}, err
	}
	root := filepath.Dir(manifestPath)
	for _, spec := range fixtureManifest.Plan.VertexSpecs {
		if err := builder.begin("v", spec.Label); err != nil {
			return Manifest{}, err
		}
		paths := fixturePaths(fixtureManifest, "node", spec.Label)
		rows, err := digestFixtureFiles(ctx, root, paths, func(row []string) (int64, []byte, error) {
			return fixtureVertex(spec.Label, row)
		}, builder)
		if err != nil {
			return Manifest{}, err
		}
		if rows != spec.Count {
			return Manifest{}, fmt.Errorf("vertex %q rows=%d expected=%d", spec.Label, rows, spec.Count)
		}
		if err := builder.end(); err != nil {
			return Manifest{}, err
		}
	}
	vertices := make(map[string]fixturemodel.VertexSpec, len(fixtureManifest.Plan.VertexSpecs))
	for _, spec := range fixtureManifest.Plan.VertexSpecs {
		vertices[spec.Label] = spec
	}
	for _, spec := range fixtureManifest.Plan.EdgeSpecs {
		if err := builder.begin("e", spec.Type); err != nil {
			return Manifest{}, err
		}
		paths := fixturePaths(fixtureManifest, "edge", spec.Type)
		rows, err := digestFixtureFiles(ctx, root, paths, func(row []string) (int64, []byte, error) {
			return fixtureEdge(spec, vertices, row)
		}, builder)
		if err != nil {
			return Manifest{}, err
		}
		if rows != spec.Count {
			return Manifest{}, fmt.Errorf("edge %q rows=%d expected=%d", spec.Type, rows, spec.Count)
		}
		if err := builder.end(); err != nil {
			return Manifest{}, err
		}
	}
	return builder.result("fixture", fixtureManifest.RootSHA256, "", ""), nil
}

func fixturePaths(manifest fixturemodel.Manifest, kind, name string) []string {
	var paths []string
	for _, entry := range manifest.Files {
		if entry.Kind == kind && entry.Name == name {
			paths = append(paths, entry.Path)
		}
	}
	slices.Sort(paths)
	return paths
}

func digestFixtureFiles(
	ctx context.Context,
	root string,
	paths []string,
	convert func([]string) (int64, []byte, error),
	builder *rangeBuilder,
) (int64, error) {
	var rows int64
	for _, relative := range paths {
		file, err := os.Open(filepath.Join(root, filepath.FromSlash(relative)))
		if err != nil {
			return rows, fmt.Errorf("open %s: %w", relative, err)
		}
		reader := csv.NewReader(file)
		for {
			if err := ctx.Err(); err != nil {
				_ = file.Close()
				return rows, err
			}
			row, readErr := reader.Read()
			if errors.Is(readErr, io.EOF) {
				break
			}
			if readErr != nil {
				_ = file.Close()
				return rows, fmt.Errorf("read %s: %w", relative, readErr)
			}
			key, canonical, convertErr := convert(row)
			if convertErr != nil {
				_ = file.Close()
				return rows, fmt.Errorf("convert %s row %d: %w", relative, rows+1, convertErr)
			}
			if err := builder.add(key, canonical); err != nil {
				_ = file.Close()
				return rows, err
			}
			rows++
		}
		if err := file.Close(); err != nil {
			return rows, fmt.Errorf("close %s: %w", relative, err)
		}
	}
	return rows, nil
}

func fixtureVertex(label string, row []string) (int64, []byte, error) {
	if len(row) != 11 {
		return 0, nil, fmt.Errorf("vertex has %d columns", len(row))
	}
	key, err := strconv.ParseInt(row[0], 10, 64)
	if err != nil {
		return 0, nil, err
	}
	score, err := strconv.ParseFloat(row[6], 64)
	if err != nil {
		return 0, nil, err
	}
	active, err := strconv.ParseBool(row[7])
	if err != nil {
		return 0, nil, err
	}
	created, err := canonicalTimestamp(row[4])
	if err != nil {
		return 0, nil, err
	}
	quantities, err := integerList(row[9])
	if err != nil {
		return 0, nil, err
	}
	properties := model.Properties{
		"source_key":  {Kind: model.ValueInteger, Integer: key},
		"external_id": {Kind: model.ValueString, String: row[1]},
		"name":        {Kind: model.ValueString, String: row[2]},
		"region":      {Kind: model.ValueString, String: row[3]},
		"created_at":  {Kind: model.ValueString, String: created},
		"score":       {Kind: model.ValueFloat, Float: score},
		"active":      {Kind: model.ValueBoolean, Boolean: active},
		"tags":        {Kind: model.ValueList, List: stringList(row[8])},
		"quantities":  {Kind: model.ValueList, List: quantities},
		"description": {Kind: model.ValueString, String: row[10]},
	}
	// Neo4j's bulk importer treats an empty, unquoted CSV field as null and does
	// not create that property. Mirror the imported source graph, not the CSV
	// transport representation, in the independent expected digest.
	if row[5] != "" {
		properties["status"] = model.Value{Kind: model.ValueString, String: row[5]}
	}
	encoded, err := model.EncodeProperties(properties)
	if err != nil {
		return 0, nil, err
	}
	return key, vertexLine(label, key, row[1], encoded), nil
}

func fixtureEdge(
	spec fixturemodel.EdgeSpec,
	vertices map[string]fixturemodel.VertexSpec,
	row []string,
) (int64, []byte, error) {
	if len(row) != 9 {
		return 0, nil, fmt.Errorf("edge has %d columns", len(row))
	}
	key, err := strconv.ParseInt(row[0], 10, 64)
	if err != nil {
		return 0, nil, err
	}
	startKey, err := strconv.ParseInt(row[2], 10, 64)
	if err != nil {
		return 0, nil, err
	}
	endKey, err := strconv.ParseInt(row[3], 10, 64)
	if err != nil {
		return 0, nil, err
	}
	quantity, err := strconv.ParseInt(row[5], 10, 64)
	if err != nil {
		return 0, nil, err
	}
	distance, err := strconv.ParseFloat(row[7], 64)
	if err != nil {
		return 0, nil, err
	}
	occurred, err := canonicalTimestamp(row[4])
	if err != nil {
		return 0, nil, err
	}
	startID, err := fixtureExternalID(vertices[spec.Start], startKey)
	if err != nil {
		return 0, nil, err
	}
	endID, err := fixtureExternalID(vertices[spec.End], endKey)
	if err != nil {
		return 0, nil, err
	}
	properties := model.Properties{
		"source_key":      {Kind: model.ValueInteger, Integer: key},
		"relationship_id": {Kind: model.ValueString, String: row[1]},
		"occurred_at":     {Kind: model.ValueString, String: occurred},
		"quantity":        {Kind: model.ValueInteger, Integer: quantity},
		"status":          {Kind: model.ValueString, String: row[6]},
		"distance_km":     {Kind: model.ValueFloat, Float: distance},
		"notes":           {Kind: model.ValueString, String: row[8]},
	}
	encoded, err := model.EncodeProperties(properties)
	if err != nil {
		return 0, nil, err
	}
	return key, edgeLine(spec.Type, key, row[1], startID, endID, encoded), nil
}

func canonicalTimestamp(value string) (string, error) {
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return "", fmt.Errorf("parse timestamp: %w", err)
	}
	return parsed.Format(time.RFC3339Nano), nil
}

func stringList(value string) []model.Value {
	if value == "" {
		return []model.Value{}
	}
	parts := strings.Split(value, ";")
	result := make([]model.Value, len(parts))
	for index, item := range parts {
		result[index] = model.Value{Kind: model.ValueString, String: item}
	}
	return result
}

func integerList(value string) ([]model.Value, error) {
	if value == "" {
		return []model.Value{}, nil
	}
	parts := strings.Split(value, ";")
	result := make([]model.Value, len(parts))
	for index, item := range parts {
		parsed, err := strconv.ParseInt(item, 10, 64)
		if err != nil {
			return nil, err
		}
		result[index] = model.Value{Kind: model.ValueInteger, Integer: parsed}
	}
	return result, nil
}

func fixtureExternalID(spec fixturemodel.VertexSpec, key int64) (string, error) {
	local := key - spec.FirstKey
	if local < 0 || local >= spec.Count {
		return "", fmt.Errorf("endpoint key %d is outside %s", key, spec.Label)
	}
	return strings.ToLower(spec.Label) + "-" + fmt.Sprintf("%012d", local+1), nil
}
