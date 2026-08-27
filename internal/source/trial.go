package source

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"

	"github.com/rioriost/agefreighter/pkg/model"
)

type TrialOptions struct {
	MaxVerticesPerLabel int64
	MaxVertices         int64
	MaxEdges            int64
	MaxBytes            int64
	IncludeLabels       []model.Label
}

type TrialSummary struct {
	VerticesPerLabel map[string]int64 `json:"verticesPerLabel"`
	TotalVertices    int64            `json:"totalVertices"`
	TotalEdges       int64            `json:"totalEdges"`
	TotalBytes       int64            `json:"totalBytes"`
	SkippedVertices  int64            `json:"skippedVertices"`
	SkippedEdges     int64            `json:"skippedEdges"`
	LimitsReached    []string         `json:"limitsReached"`
}

type TrialIterator struct {
	inner         Iterator
	options       TrialOptions
	includeLabels map[model.Label]bool
	selected      map[endpointIdentity]struct{}
	vertexCounts  map[model.Label]int64
	totalVertices int64
	totalEdges    int64
	totalBytes    int64
	skippedVertex int64
	skippedEdge   int64
	edgesStarted  bool
	done          bool
	reached       map[string]bool
}

type endpointIdentity struct {
	label      model.Label
	namespace  model.Namespace
	externalID model.ExternalID
}

func NewTrialIterator(
	inner Iterator,
	options TrialOptions,
) (*TrialIterator, error) {
	if inner == nil {
		return nil, errors.New("trial source iterator is required")
	}
	if options.MaxVerticesPerLabel <= 0 ||
		options.MaxVertices <= 0 ||
		options.MaxEdges <= 0 ||
		options.MaxBytes <= 0 {
		return nil, errors.New("trial limits must be positive")
	}
	if options.MaxVerticesPerLabel > options.MaxVertices {
		return nil, errors.New(
			"trial vertex per-label limit exceeds total vertex limit",
		)
	}
	include := make(map[model.Label]bool, len(options.IncludeLabels))
	for _, label := range options.IncludeLabels {
		if label == "" {
			return nil, errors.New("trial include label must not be empty")
		}
		if include[label] {
			return nil, fmt.Errorf("duplicate trial include label %q", label)
		}
		include[label] = true
	}
	return &TrialIterator{
		inner:         inner,
		options:       options,
		includeLabels: include,
		selected:      make(map[endpointIdentity]struct{}),
		vertexCounts:  make(map[model.Label]int64),
		reached:       make(map[string]bool),
	}, nil
}

func (iterator *TrialIterator) Next(ctx context.Context) (Item, error) {
	if iterator.done {
		return Item{}, io.EOF
	}
	for {
		item, err := iterator.inner.Next(ctx)
		if err != nil {
			return Item{}, err
		}
		if item.SizeBytes < 0 {
			return Item{}, errors.New(
				"trial source returned a negative record size",
			)
		}
		sampleBytes := item.SampleBytes
		if sampleBytes == 0 {
			sampleBytes = item.SizeBytes
		}
		if sampleBytes < 0 {
			return Item{}, errors.New(
				"trial source returned a negative sample record size",
			)
		}
		switch item.Record.Kind() {
		case model.RecordVertex:
			selected, err := iterator.selectVertex(item, sampleBytes)
			if err != nil {
				return Item{}, err
			}
			if iterator.done {
				return Item{}, io.EOF
			}
			if selected {
				return item, nil
			}
		case model.RecordEdge:
			selected := iterator.selectEdge(item, sampleBytes)
			if iterator.done {
				return Item{}, io.EOF
			}
			if selected {
				return item, nil
			}
		default:
			return Item{}, errors.New("trial source returned an invalid record")
		}
	}
}

func (iterator *TrialIterator) Close() error {
	return iterator.inner.Close()
}

func (iterator *TrialIterator) Summary() TrialSummary {
	perLabel := make(map[string]int64, len(iterator.vertexCounts))
	for label, count := range iterator.vertexCounts {
		perLabel[string(label)] = count
	}
	reached := make([]string, 0, len(iterator.reached))
	for limit := range iterator.reached {
		reached = append(reached, limit)
	}
	slices.Sort(reached)
	return TrialSummary{
		VerticesPerLabel: perLabel,
		TotalVertices:    iterator.totalVertices,
		TotalEdges:       iterator.totalEdges,
		TotalBytes:       iterator.totalBytes,
		SkippedVertices:  iterator.skippedVertex,
		SkippedEdges:     iterator.skippedEdge,
		LimitsReached:    reached,
	}
}

func (iterator *TrialIterator) selectVertex(
	item Item,
	sampleBytes int64,
) (bool, error) {
	if iterator.edgesStarted {
		return false, errors.New(
			"trial source returned a vertex after edge iteration started",
		)
	}
	vertex := item.Record.Vertex
	if len(iterator.includeLabels) != 0 &&
		!iterator.includeLabels[vertex.Label] {
		iterator.skippedVertex++
		return false, nil
	}
	if iterator.vertexCounts[vertex.Label] >=
		iterator.options.MaxVerticesPerLabel {
		iterator.skippedVertex++
		iterator.reached["maxVerticesPerLabel"] = true
		return false, nil
	}
	if iterator.totalVertices >= iterator.options.MaxVertices {
		iterator.skippedVertex++
		iterator.reached["maxVertices"] = true
		return false, nil
	}
	if !iterator.fits(sampleBytes) {
		iterator.skippedVertex++
		iterator.reached["maxBytes"] = true
		iterator.done = true
		return false, nil
	}
	iterator.vertexCounts[vertex.Label]++
	iterator.totalVertices++
	iterator.totalBytes += sampleBytes
	if iterator.vertexCounts[vertex.Label] ==
		iterator.options.MaxVerticesPerLabel {
		iterator.reached["maxVerticesPerLabel"] = true
	}
	if iterator.totalVertices == iterator.options.MaxVertices {
		iterator.reached["maxVertices"] = true
	}
	iterator.selected[endpointIdentity{
		label:      vertex.Label,
		namespace:  vertex.Namespace,
		externalID: vertex.ExternalID,
	}] = struct{}{}
	return true, nil
}

func (iterator *TrialIterator) selectEdge(
	item Item,
	sampleBytes int64,
) bool {
	iterator.edgesStarted = true
	edge := item.Record.Edge
	if !iterator.hasEndpoint(edge.Start) ||
		!iterator.hasEndpoint(edge.End) {
		iterator.skippedEdge++
		return false
	}
	if iterator.totalEdges >= iterator.options.MaxEdges {
		iterator.skippedEdge++
		iterator.reached["maxEdges"] = true
		iterator.done = true
		return false
	}
	if !iterator.fits(sampleBytes) {
		iterator.skippedEdge++
		iterator.reached["maxBytes"] = true
		iterator.done = true
		return false
	}
	iterator.totalEdges++
	iterator.totalBytes += sampleBytes
	if iterator.totalEdges == iterator.options.MaxEdges {
		iterator.reached["maxEdges"] = true
	}
	return true
}

func (iterator *TrialIterator) hasEndpoint(endpoint model.Endpoint) bool {
	_, found := iterator.selected[endpointIdentity{
		label:      endpoint.Label,
		namespace:  endpoint.Namespace,
		externalID: endpoint.ExternalID,
	}]
	return found
}

func (iterator *TrialIterator) fits(size int64) bool {
	return size <= iterator.options.MaxBytes-iterator.totalBytes
}
