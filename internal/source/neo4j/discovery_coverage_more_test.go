package neo4j

import (
	"errors"
	"strings"
	"testing"

	sourcecontract "github.com/rioriost/agefreighter/internal/source"
)

func TestDiscoveryRemainingOrchestrationBranches(t *testing.T) {
	t.Run("unlabeled conflict", func(t *testing.T) {
		options := *discoverySource().Discovery
		client := &fakeClient{streams: []RecordStream{
			discoveryStream(record(map[string]any{"label": unlabeledTargetLabel}, "label")),
			discoveryStream(record(map[string]any{"count": int64(1)}, "count")),
		}}
		if _, err := discoverLabels(t.Context(), client, options, nil); err == nil ||
			!strings.Contains(err.Error(), "conflicts") {
			t.Fatalf("discoverLabels() error = %v", err)
		}
	})

	t.Run("unlabeled exceeds label maximum", func(t *testing.T) {
		options := *discoverySource().Discovery
		options.MaxLabels = 1
		client := &fakeClient{streams: []RecordStream{
			discoveryStream(record(map[string]any{"label": "Person"}, "label")),
			discoveryStream(record(map[string]any{"count": int64(1)}, "count")),
		}}
		if _, err := discoverLabels(t.Context(), client, options, nil); err == nil ||
			!strings.Contains(err.Error(), "maximum is 1") {
			t.Fatalf("discoverLabels() error = %v", err)
		}
	})

	t.Run("edge missing stable property", func(t *testing.T) {
		source := discoverySource()
		client := &fakeClient{streams: []RecordStream{
			discoveryStream(record(map[string]any{"label": "Person"}, "label")),
			discoveryStream(record(map[string]any{"count": int64(0)}, "count")),
			discoveryStream(
				record(map[string]any{"property": "seq"}, "property"),
				record(map[string]any{"property": "vid"}, "property"),
			),
			discoveryStream(record(map[string]any{
				"relationshipType": "KNOWS",
			}, "relationshipType")),
			discoveryStream(record(map[string]any{
				"startLabels": []string{"Person"},
				"endLabels":   []string{"Person"},
			}, "startLabels", "endLabels")),
			discoveryStream(record(map[string]any{"property": "other"}, "property")),
		}}
		if _, err := DiscoverMappings(t.Context(), source, client); err == nil ||
			!strings.Contains(err.Error(), "stable property") {
			t.Fatalf("DiscoverMappings() error = %v", err)
		}
	})
}

func TestDiscoveryBudgetBranches(t *testing.T) {
	t.Run("strings can process", func(t *testing.T) {
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{Rows: 1},
		)
		client := &fakeClient{streams: []RecordStream{discoveryStream(
			record(map[string]any{"label": "A"}, "label"),
			record(map[string]any{"label": "B"}, "label"),
		)}}
		if _, err := discoverStrings(
			t.Context(), client, "query", "label", "", 10, budget,
			sourcecontract.ProfileBudgetUsage{},
		); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("discoverStrings() error = %v", err)
		}
	})

	t.Run("strings decoded bytes", func(t *testing.T) {
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{DecodedInputBytes: 1},
		)
		client := &fakeClient{streams: []RecordStream{
			discoveryStream(record(map[string]any{"label": "A"}, "label")),
		}}
		if _, err := discoverStrings(
			t.Context(), client, "query", "label", "", 10, budget,
			sourcecontract.ProfileBudgetUsage{},
		); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("discoverStrings() error = %v", err)
		}
	})

	t.Run("strings catalog charge", func(t *testing.T) {
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{Labels: 1},
		)
		client := &fakeClient{streams: []RecordStream{discoveryStream(
			record(map[string]any{"label": "A"}, "label"),
			record(map[string]any{"label": "B"}, "label"),
		)}}
		if _, err := discoverStrings(
			t.Context(), client, "query", "label", "", 10, budget,
			sourcecontract.ProfileBudgetUsage{Labels: 1},
		); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("discoverStrings() error = %v", err)
		}
	})

	t.Run("count full", func(t *testing.T) {
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{Rows: 1},
		)
		if err := budget.Charge(sourcecontract.ProfileBudgetUsage{Rows: 1}); err != nil {
			t.Fatal(err)
		}
		if _, err := discoverCount(
			t.Context(), &fakeClient{}, "query", budget,
		); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("discoverCount() error = %v", err)
		}
	})

	t.Run("count decoded bytes", func(t *testing.T) {
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{DecodedInputBytes: 1},
		)
		client := &fakeClient{streams: []RecordStream{
			discoveryStream(record(map[string]any{"count": int64(1)}, "count")),
		}}
		if _, err := discoverCount(
			t.Context(), client, "query", budget,
		); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("discoverCount() error = %v", err)
		}
	})

	t.Run("count can process", func(t *testing.T) {
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{Rows: 1},
		)
		client := &fakeClient{streams: []RecordStream{
			discoveryStream(record(map[string]any{"count": int64(1)}, "count")),
		}}
		if _, err := discoverCount(
			t.Context(), client, "query", budget,
		); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("discoverCount() error = %v", err)
		}
	})

	t.Run("count extra row charge", func(t *testing.T) {
		first := record(map[string]any{"count": int64(1)}, "count")
		size, err := estimateRecordSize(first, int64(^uint64(0)>>1))
		if err != nil {
			t.Fatal(err)
		}
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{DecodedInputBytes: size + 1},
		)
		client := &fakeClient{streams: []RecordStream{discoveryStream(
			first,
			record(map[string]any{"count": int64(2)}, "count"),
		)}}
		if _, err := discoverCount(
			t.Context(), client, "query", budget,
		); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("discoverCount() error = %v", err)
		}
	})
}

func TestEndpointDiscoveryRemainingBranches(t *testing.T) {
	labels := []discoveredLabel{
		{source: "A", target: "A"},
		{source: "B", target: "B"},
	}
	t.Run("full", func(t *testing.T) {
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{Labels: 1},
		)
		if err := budget.Charge(sourcecontract.ProfileBudgetUsage{Labels: 1}); err != nil {
			t.Fatal(err)
		}
		if _, err := discoverEndpointPairs(
			t.Context(), &fakeClient{}, "R", labels, budget,
		); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("discoverEndpointPairs() error = %v", err)
		}
	})

	t.Run("query", func(t *testing.T) {
		client := &fakeClient{queryErr: errors.New("query failed")}
		if _, err := discoverEndpointPairs(
			t.Context(), client, "R", labels, nil,
		); err == nil {
			t.Fatal("discoverEndpointPairs() ignored query error")
		}
	})

	t.Run("can process", func(t *testing.T) {
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{Rows: 1},
		)
		client := &fakeClient{streams: []RecordStream{discoveryStream(
			endpointRecord("A", "A"),
			endpointRecord("A", "B"),
		)}}
		if _, err := discoverEndpointPairs(
			t.Context(), client, "R", labels, budget,
		); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("discoverEndpointPairs() error = %v", err)
		}
	})

	t.Run("decoded bytes", func(t *testing.T) {
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{DecodedInputBytes: 1},
		)
		client := &fakeClient{streams: []RecordStream{
			discoveryStream(endpointRecord("A", "A")),
		}}
		if _, err := discoverEndpointPairs(
			t.Context(), client, "R", labels, budget,
		); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("discoverEndpointPairs() error = %v", err)
		}
	})

	for _, test := range []struct {
		name   string
		record Record
	}{
		{"start", record(map[string]any{"endLabels": []string{"A"}}, "endLabels")},
		{"end", record(map[string]any{"startLabels": []string{"A"}}, "startLabels")},
	} {
		t.Run(test.name, func(t *testing.T) {
			client := &fakeClient{streams: []RecordStream{
				discoveryStream(test.record),
			}}
			if _, err := discoverEndpointPairs(
				t.Context(), client, "R", labels, nil,
			); err == nil {
				t.Fatal("discoverEndpointPairs() accepted incomplete endpoints")
			}
		})
	}

	t.Run("label charge", func(t *testing.T) {
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{Labels: 1},
		)
		client := &fakeClient{streams: []RecordStream{discoveryStream(
			endpointRecord("A", "A"),
			endpointRecord("A", "B"),
		)}}
		if _, err := discoverEndpointPairs(
			t.Context(), client, "R", labels, budget,
		); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("discoverEndpointPairs() error = %v", err)
		}
	})

	t.Run("close", func(t *testing.T) {
		client := &fakeClient{streams: []RecordStream{
			&fakeStream{closeErr: errors.New("close failed")},
		}}
		if _, err := discoverEndpointPairs(
			t.Context(), client, "R", labels, nil,
		); err == nil {
			t.Fatal("discoverEndpointPairs() ignored close error")
		}
	})

	if value, selected, err := endpointPrimaryLabel(
		record(map[string]any{"labels": []string{}}, "labels"),
		"labels",
		labels,
	); err != nil || selected || value != "" {
		t.Fatalf("endpointPrimaryLabel(empty) = %q, %t, %v", value, selected, err)
	}
	if got := primaryLabelPredicate(
		"n", unlabeledTargetLabel, labels,
	); got != "size(labels(n)) = 0" {
		t.Fatalf("primaryLabelPredicate() = %q", got)
	}
}

func endpointRecord(start, end string) Record {
	return record(map[string]any{
		"startLabels": []string{start},
		"endLabels":   []string{end},
	}, "startLabels", "endLabels")
}
