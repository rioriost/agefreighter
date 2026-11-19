package rangedigest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
)

type Scenario struct {
	Base   string
	Append string
	Upsert string
}

type scenarioInput struct {
	base, added, updated fixture.Manifest
	final                fixture.Manifest
}

func readScenario(paths Scenario) (scenarioInput, error) {
	base, err := fixture.Verify(paths.Base)
	if err != nil {
		return scenarioInput{}, err
	}
	if base.Delta != nil {
		return scenarioInput{}, errors.New("scenario requires an original base fixture")
	}
	input := scenarioInput{base: base, final: base}
	// Final counts change, but the original fixture's slices must stay immutable.
	input.final.Plan.VertexSpecs = append([]fixture.VertexSpec(nil), base.Plan.VertexSpecs...)
	input.final.Plan.EdgeSpecs = append([]fixture.EdgeSpec(nil), base.Plan.EdgeSpecs...)
	for _, layer := range []struct {
		path string
		kind fixture.DeltaKind
		out  *fixture.Manifest
	}{{paths.Append, fixture.DeltaAppend, &input.added}, {paths.Upsert, fixture.DeltaUpsert, &input.updated}} {
		if layer.path == "" {
			continue
		}
		delta, err := fixture.Verify(layer.path)
		if err != nil {
			return scenarioInput{}, err
		}
		if delta.Delta == nil || delta.Delta.Kind != layer.kind ||
			delta.Delta.BaseRoot != base.RootSHA256 || delta.Seed != base.Seed || delta.Plan.Phase != base.Plan.Phase {
			return scenarioInput{}, errors.New("scenario delta does not match the pinned base and mode")
		}
		*layer.out = delta
	}
	if paths.Append != "" {
		input.final.Plan.VertexTotal += input.added.Plan.VertexTotal
		input.final.Plan.EdgeTotal += input.added.Plan.EdgeTotal
		for i := range input.final.Plan.VertexSpecs {
			input.final.Plan.VertexSpecs[i].Count += input.added.Plan.VertexSpecs[i].Count
		}
		for i := range input.final.Plan.EdgeSpecs {
			input.final.Plan.EdgeSpecs[i].Count += input.added.Plan.EdgeSpecs[i].Count
		}
	}
	if paths.Append != "" || paths.Upsert != "" {
		hash := sha256.Sum256([]byte("scenario-v1\x00" + base.RootSHA256 + "\x00" + input.added.RootSHA256 + "\x00" + input.updated.RootSHA256))
		input.final.RootSHA256 = hex.EncodeToString(hash[:])
	}
	return input, nil
}

// ScenarioManifest independently overlays the immutable fixture rows, not
// connector output or target metadata. Only one CSV row is held at a time.
func ScenarioManifest(ctx context.Context, paths Scenario, rangeRows int64, gremlin bool) (Manifest, error) {
	if ctx == nil {
		return Manifest{}, errors.New("context is required")
	}
	if err := ctx.Err(); err != nil {
		return Manifest{}, err
	}
	input, err := readScenario(paths)
	if err != nil {
		return Manifest{}, err
	}
	builder, err := newRangeBuilder(rangeRows)
	if err != nil {
		return Manifest{}, err
	}
	for i, spec := range input.base.Plan.VertexSpecs {
		convert := func(spec fixture.VertexSpec) func([]string) (int64, []byte, error) {
			return func(row []string) (int64, []byte, error) {
				if gremlin {
					return gremlinFixtureVertex(spec, row)
				}
				return fixtureVertex(spec.Label, row)
			}
		}
		appended := spec
		if paths.Append != "" {
			appended = input.added.Plan.VertexSpecs[i]
		}
		if err := digestScenarioLabel(ctx, paths, input, "v", spec.Label, spec.FirstKey, spec.Count,
			appended.FirstKey, convert(spec), convert(appended), builder); err != nil {
			return Manifest{}, err
		}
	}
	vertices := map[string]fixture.VertexSpec{}
	for _, spec := range input.base.Plan.VertexSpecs {
		vertices[spec.Label] = spec
	}
	for i, spec := range input.base.Plan.EdgeSpecs {
		convert := func(row []string) (int64, []byte, error) {
			if gremlin {
				return gremlinFixtureEdge(spec, vertices, row)
			}
			return fixtureEdge(spec, vertices, row)
		}
		appendFirst := spec.FirstKey
		convertAppend := convert
		if paths.Append != "" {
			appendFirst = input.added.Plan.EdgeSpecs[i].FirstKey
			convertAppend = func(row []string) (int64, []byte, error) {
				if len(row) != 9 {
					return 0, nil, errors.New("append edge column count differs")
				}
				endpoints := make(map[string]fixture.VertexSpec, len(vertices))
				for name, vertex := range vertices {
					endpoints[name] = vertex
				}
				for column, name := range []string{spec.Start, spec.End} {
					key, err := strconv.ParseInt(row[column+2], 10, 64)
					if err != nil {
						return 0, nil, err
					}
					for _, added := range input.added.Plan.VertexSpecs {
						if added.Label == name && key >= added.FirstKey && key < added.FirstKey+added.Count {
							endpoints[name] = added
						}
					}
				}
				if gremlin {
					return gremlinFixtureEdge(spec, endpoints, row)
				}
				return fixtureEdge(spec, endpoints, row)
			}
		}
		if err := digestScenarioLabel(ctx, paths, input, "e", spec.Type, spec.FirstKey, spec.Count,
			appendFirst, convert, convertAppend, builder); err != nil {
			return Manifest{}, err
		}
	}
	result := builder.result("fixture", input.final.RootSHA256, "", "")
	if gremlin {
		result.CanonicalVersion = GremlinCanonicalVersion
	}
	return result, nil
}

func digestScenarioLabel(ctx context.Context, paths Scenario, input scenarioInput, kind, name string,
	first, baseCount, appendFirst int64, convert, convertAppend func([]string) (int64, []byte, error), builder *rangeBuilder,
) error {
	if err := builder.begin(kind, name); err != nil {
		return err
	}
	fileKind, width := "node", 12
	if kind == "e" {
		fileKind, width = "edge", 15
	}
	var updated int64
	if paths.Upsert != "" {
		var err error
		updated, err = visitFixtureFiles(ctx, filepath.Dir(paths.Upsert), fixturePaths(input.updated, fileKind, name),
			func(row []string) error {
				if err := scenarioIdentity(row, name, first+updated, updated, width); err != nil {
					return err
				}
				key, canonical, err := convert(row)
				if err == nil {
					err = builder.add(key, canonical)
				}
				updated++
				return err
			})
		if err != nil {
			return err
		}
		if updated != scenarioRows(input.updated, fileKind, name) {
			return errors.New("upsert row count differs from its manifest")
		}
	}
	var baseRows int64
	_, err := visitFixtureFiles(ctx, filepath.Dir(paths.Base), fixturePaths(input.base, fileKind, name), func(row []string) error {
		skip := baseRows < updated
		baseRows++
		if skip {
			return nil
		}
		key, canonical, err := convert(row)
		if err != nil {
			return err
		}
		return builder.add(key, canonical)
	})
	if err != nil {
		return err
	}
	if baseRows != baseCount {
		return errors.New("base row count differs from its manifest")
	}
	if paths.Append != "" {
		var count int64
		_, err := visitFixtureFiles(ctx, filepath.Dir(paths.Append), fixturePaths(input.added, fileKind, name), func(row []string) error {
			if err := scenarioIdentity(row, name, appendFirst+count, baseCount+count, width); err != nil {
				return err
			}
			key, canonical, err := convertAppend(row)
			if err == nil {
				err = builder.add(key, canonical)
			}
			count++
			return err
		})
		if err != nil {
			return err
		}
		if count != scenarioRows(input.added, fileKind, name) {
			return errors.New("append row count differs from its manifest")
		}
	}
	return builder.end()
}

func scenarioRows(manifest fixture.Manifest, kind, name string) int64 {
	var total int64
	for _, entry := range manifest.Files {
		if entry.Kind == kind && entry.Name == name {
			total += entry.Rows
		}
	}
	return total
}

func scenarioIdentity(row []string, name string, key, ordinal int64, width int) error {
	if len(row) < 2 || row[0] != strconv.FormatInt(key, 10) ||
		row[1] != fmt.Sprintf("%s-%0*d", strings.ToLower(name), width, ordinal+1) {
		return errors.New("delta key/identity is not the frozen base prefix or appended suffix")
	}
	return nil
}
