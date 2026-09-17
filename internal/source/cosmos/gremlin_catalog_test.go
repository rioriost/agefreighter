package cosmos

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

type catalogQueryFunc func(string, string, []Parameter, QueryOptions) (Pager, error)

func (f catalogQueryFunc) NewQueryPager(c, q string, p []Parameter, o QueryOptions) (Pager, error) {
	return f(c, q, p, o)
}

func TestExactCatalogFindsLateLabelsAndEndpointCombinations(t *testing.T) {
	// Model the service's filter, not a pre-deduplicated scripted result. Each
	// final catalog entry is behind more duplicate rows than the discovery cap.
	source := gremlinSource()
	source.Gremlin.MaxDiscoveryDocuments = 10_000
	labels := []any{}
	edges := []any{}
	for _, label := range []string{"AppPerson", "AppOrganization", "AppLate"} {
		for range 10_100 {
			labels = append(labels, label)
		}
	}
	for _, end := range []string{"AppPerson", "AppOrganization", "AppLate"} {
		for range 10_100 {
			edges = append(edges, map[string]any{"label": "APP_LINK", "startLabel": "AppPerson", "endLabel": end})
		}
	}
	calls := 0
	client := catalogQueryFunc(func(_ string, query string, params []Parameter, options QueryOptions) (Pager, error) {
		calls++
		if options.PageSizeHint != 1 || options.HasContinuationToken || options.ContinuationToken != "" {
			t.Fatalf("new exclusion query must start at beginning: %#v", options)
		}
		if strings.Contains(query, "DISTINCT") || strings.Contains(query, "ORDER BY") || strings.Contains(query, "TOP ") {
			t.Fatal("must use gateway-compatible filtering")
		}
		documents := labels
		projection := "c.label"
		if strings.HasPrefix(query, gremlinEdgeMappingsQuery) {
			documents = edges
			projection = `{"label":c.label,"startLabel":c._vertexLabel,"endLabel":c._sinkLabel}`
		}
		known := []any{}
		for _, p := range params {
			if p.Name == "@afKnownCatalog" {
				known = p.Value.([]any)
			}
		}
		if len(known) > 0 && !strings.HasSuffix(query, " AND NOT ARRAY_CONTAINS(@afKnownCatalog, "+projection+")") {
			t.Fatalf("missing exclusion: %s", query)
		}
		for _, document := range documents {
			found := false
			for _, previous := range known {
				found = found || reflect.DeepEqual(document, previous)
			}
			if !found {
				raw, _ := json.Marshal(document)
				return &fakePager{page: fakePage{items: [][]byte{raw}, hasContinuation: true, continuationToken: "unused"}}, nil
			}
		}
		return &fakePager{}, nil
	})
	resolved, err := InterpretGremlinDocuments(t.Context(), source, client)
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved.Vertices) != 3 || len(resolved.Edges) != 3 || calls != 8 {
		t.Fatalf("vertices=%d edges=%d calls=%d", len(resolved.Vertices), len(resolved.Edges), calls)
	}
	if resolved.Vertices[0].Label != "AppLate" || resolved.Edges[0].End.Label != "AppLate" {
		t.Fatalf("late catalog entry missing: %#v", resolved)
	}
}

func TestExactCatalogDrainsEmptyPagesAndFreezesParameters(t *testing.T) {
	client := newFakeClient()
	client.script("graph", "query",
		fakePage{hasContinuation: true, continuationToken: "a"},
		fakePage{items: [][]byte{jsonItem(`"first"`)}, hasContinuation: true, continuationToken: "b"},
	)
	filtered := "query AND NOT ARRAY_CONTAINS(@afKnownCatalog, c.label)"
	client.script("graph", filtered,
		fakePage{hasContinuation: true, continuationToken: "a"},
		fakePage{items: [][]byte{jsonItem(`"last"`)}, hasContinuation: true, continuationToken: "b"},
		fakePage{hasContinuation: true, continuationToken: "a"},
		fakePage{},
	)
	count := 0
	err := visitExactGremlinCatalog(t.Context(), client, "graph", "query", nil, 10, "c.label", func([]byte) error { count++; return nil })
	if err != nil || count != 2 || client.callCount() != 6 {
		t.Fatalf("count=%d calls=%d err=%v", count, client.callCount(), err)
	}
	for _, index := range []int{0, 2, 4} {
		if client.callAt(index).options.HasContinuationToken {
			t.Fatal("continuation reused for a changed predicate")
		}
		if client.callAt(index+1).options.ContinuationToken != "a" {
			t.Fatal("empty-page continuation not drained")
		}
	}
	if len(client.callAt(2).parameters[0].Value.([]any)) != 1 || len(client.callAt(4).parameters[0].Value.([]any)) != 2 {
		t.Fatal("recorded parameters mutated")
	}
}

func TestExactCatalogFailsClosed(t *testing.T) {
	marker := errors.New("request failed")
	for _, tc := range []struct {
		name     string
		pages    []fakePage
		limit    int
		want     error
		contains string
	}{
		{"open", []fakePage{{newQueryPagerErr: marker}}, 10, marker, ""},
		{"fetch", []fakePage{{nextPageErr: marker}}, 10, marker, ""},
		{"malformed", []fakePage{{items: [][]byte{jsonItem(`{`)}}}, 10, nil, "decode Cosmos"},
		{"document cap", []fakePage{{items: [][]byte{jsonItem(`"x"`), jsonItem(`"x"`)}}}, 1, ErrDiscoveryLimit, ""},
		{"empty token", []fakePage{{hasContinuation: true}}, 10, nil, "repeated continuation"},
		{"cycle", []fakePage{{hasContinuation: true, continuationToken: "a"}, {hasContinuation: true, continuationToken: "b"}, {hasContinuation: true, continuationToken: "a"}}, 10, nil, "repeated continuation"},
		{"page cap", []fakePage{{hasContinuation: true, continuationToken: "a"}, {hasContinuation: true, continuationToken: "b"}}, 1, ErrDiscoveryLimit, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := newFakeClient()
			client.script("graph", "query", tc.pages...)
			err := visitExactGremlinCatalog(t.Context(), client, "graph", "query", nil, tc.limit, "c.label", func([]byte) error { return nil })
			if err == nil || (tc.want != nil && !errors.Is(err, tc.want)) || !strings.Contains(err.Error(), tc.contains) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	err := visitExactGremlinCatalog(ctx, newFakeClient(), "graph", "query", nil, 10, "c.label", nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestExactCatalogBoundsIgnoredEntriesAndPropagatesVisitorFailure(t *testing.T) {
	calls := 0
	client := catalogQueryFunc(func(_ string, _ string, _ []Parameter, _ QueryOptions) (Pager, error) {
		calls++
		raw, _ := json.Marshal(calls)
		return &fakePager{page: fakePage{items: [][]byte{raw}, hasContinuation: true, continuationToken: "next"}}, nil
	})
	err := visitExactGremlinCatalog(t.Context(), client, "graph", "query", nil, 10_000, "c.label", func([]byte) error { return nil })
	if !errors.Is(err, ErrDiscoveryLimit) || calls != maxGremlinMappings+1 {
		t.Fatalf("ignored catalog entries were not bounded: calls=%d error=%v", calls, err)
	}
	marker := errors.New("invalid catalog entry")
	err = visitExactGremlinCatalog(t.Context(), client, "graph", "query", nil, 10, "c.label", func([]byte) error { return marker })
	if !errors.Is(err, marker) {
		t.Fatal(err)
	}
}
