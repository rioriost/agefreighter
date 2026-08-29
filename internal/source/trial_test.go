package source

import (
	"context"
	"errors"
	"io"
	"reflect"
	"testing"

	"github.com/rioriost/agefreighter/pkg/model"
)

func TestTrialIteratorPreservesEndpointClosureAndLimits(t *testing.T) {
	inner := &trialSliceIterator{items: []Item{
		trialVertex("Person", "p1", 10),
		trialVertex("Person", "p2", 10),
		trialVertex("Person", "p3", 10),
		trialVertex("Company", "c1", 10),
		trialVertex("Company", "c2", 10),
		trialEdge("WORKS_AT", "e1", "Person", "p1", "Company", "c1", 10),
		trialEdge("WORKS_AT", "e2", "Person", "p3", "Company", "c1", 10),
		trialEdge("WORKS_AT", "e3", "Person", "p2", "Company", "c2", 10),
	}}
	iterator, err := NewTrialIterator(inner, TrialOptions{
		MaxVerticesPerLabel: 2,
		MaxVertices:         4,
		MaxEdges:            1,
		MaxBytes:            100,
	})
	if err != nil {
		t.Fatalf("NewTrialIterator() error = %v", err)
	}

	records := collectTrialRecords(t, iterator)

	if got := trialRecordIDs(records); !reflect.DeepEqual(
		got,
		[]string{"p1", "p2", "c1", "c2", "e1"},
	) {
		t.Fatalf("record IDs = %v", got)
	}
	summary := iterator.Summary()
	if summary.TotalVertices != 4 ||
		summary.TotalEdges != 1 ||
		summary.SkippedVertices != 1 ||
		summary.SkippedEdges != 2 ||
		summary.TotalBytes != 50 ||
		!reflect.DeepEqual(
			summary.LimitsReached,
			[]string{"maxEdges", "maxVertices", "maxVerticesPerLabel"},
		) {
		t.Fatalf("Summary() = %#v", summary)
	}
	if !inner.closed {
		t.Fatal("Close() was not delegated")
	}
}

func TestTrialIteratorFiltersLabelsAndBytes(t *testing.T) {
	inner := &trialSliceIterator{items: []Item{
		trialVertex("Person", "p1", 10),
		trialVertex("Company", "c1", 10),
		trialVertex("Person", "p2", 11),
		trialEdge("KNOWS", "e1", "Person", "p1", "Person", "p2", 1),
	}}
	iterator, err := NewTrialIterator(inner, TrialOptions{
		MaxVerticesPerLabel: 10,
		MaxVertices:         10,
		MaxEdges:            10,
		MaxBytes:            20,
		IncludeLabels:       []model.Label{"Person"},
	})
	if err != nil {
		t.Fatalf("NewTrialIterator() error = %v", err)
	}

	records := collectTrialRecords(t, iterator)

	if got := trialRecordIDs(records); !reflect.DeepEqual(got, []string{"p1"}) {
		t.Fatalf("record IDs = %v", got)
	}
	summary := iterator.Summary()
	if summary.TotalVertices != 1 ||
		summary.SkippedVertices != 2 ||
		summary.SkippedEdges != 0 ||
		!reflect.DeepEqual(summary.LimitsReached, []string{"maxBytes"}) {
		t.Fatalf("Summary() = %#v", summary)
	}
}

func TestTrialIteratorUsesLogicalSampleBytes(t *testing.T) {
	first := trialVertex("Person", "p1", 1_000)
	first.SampleBytes = 10
	second := trialVertex("Person", "p2", 1_000)
	second.SampleBytes = 10
	iterator, err := NewTrialIterator(&trialSliceIterator{
		items: []Item{first, second},
	}, TrialOptions{
		MaxVerticesPerLabel: 2,
		MaxVertices:         2,
		MaxEdges:            1,
		MaxBytes:            20,
	})
	if err != nil {
		t.Fatalf("NewTrialIterator() error = %v", err)
	}

	records := collectTrialRecords(t, iterator)

	if len(records) != 2 || iterator.Summary().TotalBytes != 20 {
		t.Fatalf(
			"records = %d, summary = %#v",
			len(records),
			iterator.Summary(),
		)
	}
}

func TestTrialIteratorRejectsInvalidContracts(t *testing.T) {
	valid := TrialOptions{
		MaxVerticesPerLabel: 1,
		MaxVertices:         1,
		MaxEdges:            1,
		MaxBytes:            1,
	}
	if _, err := NewTrialIterator(nil, valid); err == nil {
		t.Fatal("NewTrialIterator() accepted nil source")
	}
	invalid := valid
	invalid.MaxEdges = 0
	if _, err := NewTrialIterator(&trialSliceIterator{}, invalid); err == nil {
		t.Fatal("NewTrialIterator() accepted zero edge limit")
	}
	invalid = valid
	invalid.MaxVerticesPerLabel = 2
	if _, err := NewTrialIterator(&trialSliceIterator{}, invalid); err == nil {
		t.Fatal("NewTrialIterator() accepted inconsistent vertex limits")
	}
	invalid = valid
	invalid.IncludeLabels = []model.Label{"Person", "Person"}
	if _, err := NewTrialIterator(&trialSliceIterator{}, invalid); err == nil {
		t.Fatal("NewTrialIterator() accepted duplicate include label")
	}
	invalid = valid
	invalid.IncludeLabels = []model.Label{""}
	if _, err := NewTrialIterator(&trialSliceIterator{}, invalid); err == nil {
		t.Fatal("NewTrialIterator() accepted an empty include label")
	}
}

func TestTrialIteratorPropagatesErrorsAndOrderingViolations(t *testing.T) {
	sourceErr := errors.New("source failed")
	iterator, err := NewTrialIterator(&trialSliceIterator{err: sourceErr}, TrialOptions{
		MaxVerticesPerLabel: 1,
		MaxVertices:         1,
		MaxEdges:            1,
		MaxBytes:            1,
	})
	if err != nil {
		t.Fatalf("NewTrialIterator() error = %v", err)
	}
	if _, err := iterator.Next(t.Context()); !errors.Is(err, sourceErr) {
		t.Fatalf("Next() error = %v", err)
	}

	iterator, err = NewTrialIterator(&trialSliceIterator{items: []Item{
		trialEdge("KNOWS", "e1", "Person", "p1", "Person", "p2", 1),
		trialVertex("Person", "p1", 1),
	}}, TrialOptions{
		MaxVerticesPerLabel: 2,
		MaxVertices:         2,
		MaxEdges:            2,
		MaxBytes:            2,
	})
	if err != nil {
		t.Fatalf("NewTrialIterator() error = %v", err)
	}
	if _, err := iterator.Next(t.Context()); err == nil ||
		err.Error() != "trial source returned a vertex after edge iteration started" {
		t.Fatalf("Next() error = %v", err)
	}
}

func TestTrialIteratorRemainingContractBranches(t *testing.T) {
	options := TrialOptions{
		MaxVerticesPerLabel: 2,
		MaxVertices:         2,
		MaxEdges:            1,
		MaxBytes:            10,
	}
	for _, item := range []Item{
		{Record: model.VertexRecord(model.Vertex{Label: "Person"}), SizeBytes: -1},
		{
			Record:      model.VertexRecord(model.Vertex{Label: "Person"}),
			SizeBytes:   1,
			SampleBytes: -1,
		},
		{Record: model.Record{}, SizeBytes: 1},
	} {
		iterator, err := NewTrialIterator(
			&trialSliceIterator{items: []Item{item}},
			options,
		)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := iterator.Next(t.Context()); err == nil {
			t.Fatalf("Next() accepted invalid item %#v", item)
		}
	}

	iterator, err := NewTrialIterator(&trialSliceIterator{items: []Item{
		trialVertex("Person", "p1", 1),
		trialVertex("Company", "c1", 1),
		trialVertex("Other", "o1", 1),
	}}, options)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := iterator.Next(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := iterator.Next(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := iterator.Next(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("vertex limit Next() error = %v", err)
	}

	iterator, err = NewTrialIterator(&trialSliceIterator{items: []Item{
		trialVertex("Person", "p1", 4),
		trialVertex("Person", "p2", 4),
		trialEdge("KNOWS", "e1", "Person", "p1", "Person", "p2", 3),
	}}, TrialOptions{
		MaxVerticesPerLabel: 2,
		MaxVertices:         2,
		MaxEdges:            1,
		MaxBytes:            10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := iterator.Next(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := iterator.Next(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := iterator.Next(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("edge byte limit Next() error = %v", err)
	}
	if _, err := iterator.Next(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("done Next() error = %v", err)
	}
}

func collectTrialRecords(
	t *testing.T,
	iterator *TrialIterator,
) []model.Record {
	t.Helper()
	var records []model.Record
	for {
		item, err := iterator.Next(t.Context())
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		records = append(records, item.Record)
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	return records
}

func trialRecordIDs(records []model.Record) []string {
	ids := make([]string, 0, len(records))
	for _, record := range records {
		if record.Vertex != nil {
			ids = append(ids, string(record.Vertex.ExternalID))
		} else {
			ids = append(ids, string(record.Edge.ExternalID))
		}
	}
	return ids
}

func trialVertex(label, id string, size int64) Item {
	return Item{
		Record: model.VertexRecord(model.Vertex{
			Label:      model.Label(label),
			Namespace:  "crm",
			ExternalID: model.ExternalID(id),
		}),
		SizeBytes: size,
	}
}

func trialEdge(
	label string,
	id string,
	startLabel string,
	startID string,
	endLabel string,
	endID string,
	size int64,
) Item {
	return Item{
		Record: model.EdgeRecord(model.Edge{
			Label:      model.Label(label),
			Namespace:  "crm",
			ExternalID: model.ExternalID(id),
			Start: model.Endpoint{
				Label:      model.Label(startLabel),
				Namespace:  "crm",
				ExternalID: model.ExternalID(startID),
			},
			End: model.Endpoint{
				Label:      model.Label(endLabel),
				Namespace:  "crm",
				ExternalID: model.ExternalID(endID),
			},
		}),
		SizeBytes: size,
	}
}

type trialSliceIterator struct {
	items  []Item
	index  int
	err    error
	closed bool
}

func (iterator *trialSliceIterator) Next(context.Context) (Item, error) {
	if iterator.index < len(iterator.items) {
		item := iterator.items[iterator.index]
		iterator.index++
		return item, nil
	}
	if iterator.err != nil {
		return Item{}, iterator.err
	}
	return Item{}, io.EOF
}

func (iterator *trialSliceIterator) Close() error {
	iterator.closed = true
	return nil
}
