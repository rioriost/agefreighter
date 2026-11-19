package fixture

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
)

type DeltaKind string

const (
	DeltaAppend DeltaKind = "append"
	DeltaUpsert DeltaKind = "upsert"
)

type Delta struct {
	Kind     DeltaKind `json:"kind"`
	BaseRoot string    `json:"baseRootSha256"`
}

// GenerateDelta preserves its base and writes a separately selectable 1% input.
// Appends mix old/new endpoints; upserts retain keys and endpoints.
func GenerateDelta(ctx context.Context, basePath, output string, kind DeltaKind) (Manifest, error) {
	if ctx == nil {
		return Manifest{}, errors.New("context is required")
	}
	base, err := Verify(basePath)
	if err != nil {
		return Manifest{}, err
	}
	if base.Delta != nil {
		return Manifest{}, errors.New("a delta must reference a base fixture, not another delta")
	}
	plan, err := deltaPlan(base.Plan.Phase, kind)
	if err != nil {
		return Manifest{}, err
	}
	return generate(ctx, GenerateConfig{
		Phase: base.Plan.Phase, Output: output, Shards: base.Shards, Workers: 2, Seed: base.Seed,
		delta: &Delta{Kind: kind, BaseRoot: base.RootSHA256}, base: basePath,
	}, plan)
}

func deltaPlan(phase Phase, kind DeltaKind) (Plan, error) {
	switch phase {
	case PhaseP0, PhaseP1, PhaseP2:
	default:
		return Plan{}, errors.New("one-percent deltas are limited to approved P0-P2 incremental scenarios")
	}
	if kind != DeltaAppend && kind != DeltaUpsert {
		return Plan{}, fmt.Errorf("unknown delta kind %q", kind)
	}
	base, err := BuildPlan(phase)
	if err != nil {
		return Plan{}, err
	}
	weights := make([]int64, len(base.VertexSpecs))
	for i, spec := range base.VertexSpecs {
		weights[i] = spec.Count
	}
	counts := allocate(weights, base.VertexTotal/100)
	result := Plan{Phase: phase, VertexTotal: base.VertexTotal / 100, EdgeTotal: base.EdgeTotal / 100}
	for i, spec := range base.VertexSpecs {
		next := spec
		next.Count = counts[i]
		if kind == DeltaAppend {
			next.FirstKey += base.VertexTotal
			next.IDOffset = spec.Count
		}
		result.VertexSpecs = append(result.VertexSpecs, next)
	}
	weights = make([]int64, len(base.EdgeSpecs))
	for i, spec := range base.EdgeSpecs {
		weights[i] = spec.Count
	}
	counts = allocate(weights, base.EdgeTotal/100)
	for i, spec := range base.EdgeSpecs {
		next := spec
		next.Count = counts[i]
		if kind == DeltaAppend {
			next.FirstKey += base.EdgeTotal
			next.IDOffset = spec.Count
		}
		result.EdgeSpecs = append(result.EdgeSpecs, next)
	}
	return result, nil
}

// EndpointPlan describes base references; append deltas add their own ranges.
func (manifest Manifest) EndpointPlan() (Plan, error) {
	if manifest.Delta != nil {
		return BuildPlan(manifest.Plan.Phase)
	}
	return manifest.Plan, nil
}

func manifestDigest(manifest Manifest) string {
	files := manifestRoot(manifest.Files)
	if manifest.Delta == nil {
		return files
	}
	digest := sha256.Sum256([]byte("delta-v1\x00" + string(manifest.Delta.Kind) + "\x00" + manifest.Delta.BaseRoot + "\x00" + files))
	return hex.EncodeToString(digest[:])
}
