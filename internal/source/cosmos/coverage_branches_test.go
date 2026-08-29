package cosmos

import (
	"context"
	"errors"
	"io"
	"math"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
)

func TestCosmosIteratorOptionBranches(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient()
	valid := IteratorOptions{
		Namespace: "ns",
		Source:    source,
		Client:    client,
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	tests := []struct {
		name    string
		ctx     context.Context
		options IteratorOptions
	}{
		{"nil context", nil, valid},
		{"canceled context", ctx, valid},
		{"missing namespace", t.Context(), func() IteratorOptions {
			value := valid
			value.Namespace = ""
			return value
		}()},
		{"missing client", t.Context(), func() IteratorOptions {
			value := valid
			value.Client = nil
			return value
		}()},
		{"negative reject limit", t.Context(), func() IteratorOptions {
			value := valid
			value.RejectLimit = -1
			return value
		}()},
		{"missing malformed handler", t.Context(), func() IteratorOptions {
			value := valid
			value.RejectLimit = 1
			return value
		}()},
		{"negative record limit", t.Context(), func() IteratorOptions {
			value := valid
			value.MaxRecordBytes = -1
			return value
		}()},
		{"negative property limit", t.Context(), func() IteratorOptions {
			value := valid
			value.MaxProperties = -1
			return value
		}()},
		{"zero page size", t.Context(), func() IteratorOptions {
			value := valid
			value.Source.PageSize = 0
			return value
		}()},
		{"large page size", t.Context(), func() IteratorOptions {
			value := valid
			value.Source.PageSize = 1001
			return value
		}()},
		{"invalid mapping", t.Context(), func() IteratorOptions {
			value := valid
			value.Source.Vertices = append(
				[]config.CosmosVertexQuery(nil),
				value.Source.Vertices...,
			)
			value.Source.Vertices[0].IDField = "invalid"
			return value
		}()},
		{"invalid resume token", t.Context(), func() IteratorOptions {
			value := valid
			value.AfterToken = "invalid"
			return value
		}()},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := NewIterator(test.ctx, test.options); err == nil {
				t.Fatal("NewIterator() accepted invalid options")
			}
		})
	}

	observer := newFakeClient()
	observer.addThrottled(3)
	iterator, err := NewIterator(t.Context(), IteratorOptions{
		Namespace: "ns", Source: source, Client: observer,
	})
	if err != nil {
		t.Fatal(err)
	}
	if iterator.throttleObserved != 3 {
		t.Fatalf("initial throttled count = %d", iterator.throttleObserved)
	}
}

func TestCosmosIteratorRuntimeBranches(t *testing.T) {
	t.Run("closed and canceled", func(t *testing.T) {
		iterator := &Iterator{closed: true}
		if _, err := iterator.Next(t.Context()); err == nil {
			t.Fatal("Next() accepted a closed iterator")
		}
		iterator.closed = false
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := iterator.Next(ctx); !errors.Is(err, context.Canceled) {
			t.Fatalf("Next() error = %v", err)
		}
	})

	t.Run("budget before second item", func(t *testing.T) {
		source := baseVertexSource()
		client := newFakeClient()
		client.script("people", "SELECT * FROM c", fakePage{items: [][]byte{
			jsonItem(`{"id":"p1"}`), jsonItem(`{"id":"p2"}`),
		}})
		budget := sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{Rows: 1})
		iterator, err := NewIterator(t.Context(), IteratorOptions{
			Namespace: "ns", Source: source, Client: client, ProfileBudget: budget,
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := iterator.Next(t.Context()); err != nil {
			t.Fatal(err)
		}
		if _, err := iterator.Next(t.Context()); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("second Next() error = %v", err)
		}
	})

	t.Run("decoded byte charge", func(t *testing.T) {
		source := baseVertexSource()
		client := newFakeClient()
		client.script("people", "SELECT * FROM c", fakePage{
			items: [][]byte{jsonItem(`{"id":"p1"}`)},
		})
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{DecodedInputBytes: 1},
		)
		iterator, err := NewIterator(t.Context(), IteratorOptions{
			Namespace: "ns", Source: source, Client: client, ProfileBudget: budget,
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := iterator.Next(t.Context()); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("Next() error = %v", err)
		}
	})

	t.Run("quarantine failure", func(t *testing.T) {
		source := baseVertexSource()
		client := newFakeClient()
		client.script("people", "SELECT * FROM c", fakePage{
			items: [][]byte{jsonItem(`{"missing":"id"}`)},
		})
		iterator, err := NewIterator(t.Context(), IteratorOptions{
			Namespace: "ns", Source: source, Client: client,
			RejectLimit: 1,
			OnMalformed: func(context.Context, MalformedRecord) error {
				return errors.New("quarantine failed")
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := iterator.Next(t.Context()); err == nil ||
			!strings.Contains(err.Error(), "write Cosmos quarantine") {
			t.Fatalf("Next() error = %v", err)
		}
	})

	t.Run("sync without observer", func(t *testing.T) {
		client := &queryClientOnly{client: newFakeClient()}
		iterator := &Iterator{options: IteratorOptions{Client: client}}
		iterator.syncThrottled()
	})

	t.Run("resume reject limit", func(t *testing.T) {
		source := baseVertexSource()
		mappings, err := buildMappings(t.Context(), "ns", source, 1024)
		if err != nil {
			t.Fatal(err)
		}
		fingerprint, err := bindFingerprint(
			source.Endpoint, source.Database, "ns", int32(source.PageSize), mappings,
		)
		if err != nil {
			t.Fatal(err)
		}
		token := formatResumeToken(resumeState{
			fingerprint: fingerprint, mappingIndex: 0,
			mappingKind: vertexMapping, rejected: 1,
		})
		if _, err := NewIterator(t.Context(), IteratorOptions{
			Namespace: "ns", Source: source, Client: newFakeClient(),
			AfterToken: token,
		}); err == nil {
			t.Fatal("NewIterator() accepted a resume rejection count above the limit")
		}
	})
}

func TestCosmosFetchAndPropertyBranches(t *testing.T) {
	mapping := compiledMapping{
		kind: vertexMapping, container: "people", label: "Person",
		query: "SELECT * FROM c",
	}
	tests := []struct {
		name   string
		ctx    context.Context
		client *fakeClient
		budget *sourcecontract.ProfileBudget
	}{
		{"canceled", func() context.Context {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			return ctx
		}(), newFakeClient(), nil},
		{"full budget", t.Context(), newFakeClient(), func() *sourcecontract.ProfileBudget {
			budget := sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{Pages: 1})
			_ = budget.Charge(sourcecontract.ProfileBudgetUsage{Pages: 1})
			return budget
		}()},
		{"open pager", t.Context(), func() *fakeClient {
			client := newFakeClient()
			client.script("people", "SELECT * FROM c", fakePage{
				newQueryPagerErr: errors.New("open failed"),
			})
			return client
		}(), nil},
		{"fetch page", t.Context(), func() *fakeClient {
			client := newFakeClient()
			client.script("people", "SELECT * FROM c", fakePage{
				nextPageErr: errors.New("fetch failed"),
			})
			return client
		}(), nil},
		{"page budget", t.Context(), func() *fakeClient {
			client := newFakeClient()
			client.script("people", "SELECT * FROM c", fakePage{
				items: [][]byte{jsonItem(`{"id":"p1"}`)},
			})
			return client
		}(), sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{RawInputBytes: 1},
		)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			iterator := &Iterator{
				options: IteratorOptions{
					Source: config.CosmosSource{PageSize: 10},
					Client: test.client, ProfileBudget: test.budget,
				},
			}
			if _, err := iterator.fetchPage(
				test.ctx, mapping, false, "",
			); err == nil {
				t.Fatal("fetchPage() succeeded")
			}
		})
	}

	propertyPointer, err := parsePointer("/value")
	if err != nil {
		t.Fatal(err)
	}
	properties := []compiledProperty{{name: "value", pointer: propertyPointer}}
	iterator := &Iterator{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, _, err := iterator.buildProperties(
		ctx, map[string]any{"value": "x"}, properties,
	); !errors.Is(err, context.Canceled) {
		t.Fatalf("buildProperties(canceled) error = %v", err)
	}
	if _, _, _, err := iterator.buildProperties(
		t.Context(), map[string]any{}, properties,
	); err == nil {
		t.Fatal("buildProperties() accepted a missing property")
	}
	if _, _, _, err := iterator.buildProperties(
		t.Context(), map[string]any{"value": struct{}{}}, properties,
	); err == nil {
		t.Fatal("buildProperties() accepted an unsupported value")
	}
	iterator.options.PreencodeProperties = true
	if values, encoded, size, err := iterator.buildProperties(
		t.Context(), map[string]any{"value": "x"}, properties,
	); err != nil || values != nil || len(encoded) == 0 || size == 0 {
		t.Fatalf("buildProperties(preencoded) = %#v, %q, %d, %v", values, encoded, size, err)
	}
	if got := saturatingAdd(math.MaxInt64, 1); got != math.MaxInt64 {
		t.Fatalf("saturatingAdd() = %d", got)
	}

	iterator = &Iterator{
		options: IteratorOptions{
			Source: config.CosmosSource{PageSize: 10},
			Client: newFakeClient(),
		},
		mappings: []compiledMapping{mapping},
	}
	if err := iterator.advancePage(t.Context(), "missing"); err == nil {
		t.Fatal("advancePage() ignored a fetch failure")
	}

	id, _ := parsePointer("/id")
	start, _ := parsePointer("/start")
	end, _ := parsePointer("/end")
	value, _ := parsePointer("/value")
	vertex := compiledMapping{
		kind: vertexMapping, label: "Person", namespace: "ns", idField: id,
		properties: []compiledProperty{{name: "value", pointer: value}},
	}
	edge := compiledMapping{
		kind: edgeMapping, label: "KNOWS", namespace: "ns",
		hasExternalID: true, externalIDField: id,
		start: config.EndpointMapping{Label: "Person"}, startField: start,
		end: config.EndpointMapping{Label: "Person"}, endField: end,
		properties: []compiledProperty{{name: "value", pointer: value}},
	}
	iterator = &Iterator{options: IteratorOptions{MaxRecordBytes: 4}}
	for _, test := range []struct {
		name    string
		mapping compiledMapping
		raw     string
	}{
		{"oversized", vertex, `{"id":"p1"}`},
		{"invalid JSON", vertex, `{`},
		{"missing edge ID", edge, `{"start":"p1","end":"p2","value":"x"}`},
		{"missing edge start", edge, `{"id":"e1","end":"p2","value":"x"}`},
		{"missing edge end", edge, `{"id":"e1","start":"p1","value":"x"}`},
		{"missing edge property", edge, `{"id":"e1","start":"p1","end":"p2"}`},
	} {
		t.Run("decode "+test.name, func(t *testing.T) {
			limit := iterator.options.MaxRecordBytes
			if test.name != "oversized" {
				iterator.options.MaxRecordBytes = 1024
			}
			_, _, err := iterator.decodeRecord(
				t.Context(), test.mapping, []byte(test.raw),
			)
			iterator.options.MaxRecordBytes = limit
			if err == nil {
				t.Fatal("decodeRecord() succeeded")
			}
		})
	}
}

func TestGremlinDiscoveryAdditionalBranches(t *testing.T) {
	if _, err := InterpretGremlinDocumentsBounded(
		nil, gremlinSource(), newFakeClient(), nil,
	); err == nil {
		t.Fatal("InterpretGremlinDocumentsBounded() accepted nil context")
	}

	t.Run("unbounded label discovery error", func(t *testing.T) {
		source := gremlinSource()
		client := newFakeClient()
		client.script(
			source.Gremlin.Container,
			gremlinVertexLabelsQuery+" AND STARTSWITH(c.label, @labelPrefix)",
			fakePage{newQueryPagerErr: errors.New("open failed")},
		)
		if _, err := InterpretGremlinDocuments(t.Context(), source, client); err == nil {
			t.Fatal("InterpretGremlinDocuments() ignored discovery failure")
		}
	})

	boundedCases := []struct {
		name string
		item string
	}{
		{"invalid JSON", "{"},
		{"non-object", `"label"`},
		{"invalid label", `{"label":7}`},
		{"invalid edge mapping", `{"isEdge":true,"label":"APP_KNOWS","startLabel":7,"endLabel":"AppPerson"}`},
	}
	for _, test := range boundedCases {
		t.Run(test.name, func(t *testing.T) {
			source := gremlinSource()
			client := newFakeClient()
			client.script(source.Gremlin.Container, gremlinCatalogQuery, fakePage{
				items: [][]byte{jsonItem(test.item)},
			})
			budget := sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{
				Rows: 10, Pages: 10, RawInputBytes: 1 << 20,
				DecodedInputBytes: 1 << 20, Labels: 10,
			})
			if _, err := InterpretGremlinDocumentsBounded(
				t.Context(), source, client, budget,
			); err == nil {
				t.Fatal("InterpretGremlinDocumentsBounded() accepted invalid discovery data")
			}
		})
	}

	t.Run("out-of-prefix catalog rows", func(t *testing.T) {
		source := gremlinSource()
		client := newFakeClient()
		client.script(source.Gremlin.Container, gremlinCatalogQuery, fakePage{
			items: [][]byte{
				jsonItem(`{"label":"Outside"}`),
				jsonItem(`{"label":"AppPerson"}`),
				jsonItem(`{"isEdge":true,"label":"OTHER","startLabel":"AppPerson","endLabel":"AppPerson"}`),
			},
		})

		t.Run("bounded no labels", func(t *testing.T) {
			source := gremlinSource()
			client := newFakeClient()
			client.script(source.Gremlin.Container, gremlinCatalogQuery, fakePage{
				items: [][]byte{jsonItem(`{"label":"Outside"}`)},
			})
			budget := sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{
				Rows: 10, Pages: 10, RawInputBytes: 1 << 20,
				DecodedInputBytes: 1 << 20, Labels: 10,
			})
			if _, err := InterpretGremlinDocumentsBounded(
				t.Context(), source, client, budget,
			); err == nil {
				t.Fatal("InterpretGremlinDocumentsBounded() accepted an empty selected catalog")
			}
		})

		t.Run("bounded label charge", func(t *testing.T) {
			source := gremlinSource()
			client := newFakeClient()
			client.script(source.Gremlin.Container, gremlinCatalogQuery, fakePage{
				items: [][]byte{
					jsonItem(`{"label":"AppA"}`),
					jsonItem(`{"label":"AppB"}`),
				},
			})
			budget := sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{
				Rows: 10, Pages: 10, RawInputBytes: 1 << 20,
				DecodedInputBytes: 1 << 20, Labels: 1,
			})
			if _, err := InterpretGremlinDocumentsBounded(
				t.Context(), source, client, budget,
			); !errors.Is(err, sourcecontract.ErrProfileBudget) {
				t.Fatalf("InterpretGremlinDocumentsBounded() error = %v", err)
			}
		})

		t.Run("bounded edge charge", func(t *testing.T) {
			source := gremlinSource()
			client := newFakeClient()
			client.script(source.Gremlin.Container, gremlinCatalogQuery, fakePage{
				items: [][]byte{
					jsonItem(`{"label":"AppPerson"}`),
					jsonItem(`{"isEdge":true,"label":"APP_KNOWS","startLabel":"AppPerson","endLabel":"AppPerson"}`),
				},
			})
			budget := sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{
				Rows: 10, Pages: 10, RawInputBytes: 1 << 20,
				DecodedInputBytes: 1 << 20, Labels: 1,
			})
			if _, err := InterpretGremlinDocumentsBounded(
				t.Context(), source, client, budget,
			); !errors.Is(err, sourcecontract.ErrProfileBudget) {
				t.Fatalf("InterpretGremlinDocumentsBounded() error = %v", err)
			}
		})
		budget := sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{
			Rows: 10, Pages: 10, RawInputBytes: 1 << 20,
			DecodedInputBytes: 1 << 20, Labels: 10,
		})
		resolved, err := InterpretGremlinDocumentsBounded(
			t.Context(), source, client, budget,
		)
		if err != nil || len(resolved.Vertices) != 1 || len(resolved.Edges) != 0 {
			t.Fatalf("resolved = %#v, %v", resolved, err)
		}
	})

	t.Run("visit callback error", func(t *testing.T) {
		client := newFakeClient()
		client.script("graph", "query", fakePage{
			items: [][]byte{jsonItem(`{}`)},
		})
		err := visitGremlinDiscovery(
			t.Context(), client, "graph", "query", nil, 10, 10, nil,
			func([]byte) error { return errors.New("visit failed") },
		)
		if err == nil || !strings.Contains(err.Error(), "visit failed") {
			t.Fatalf("visitGremlinDiscovery() error = %v", err)
		}
	})

	t.Run("visit canceled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if err := visitGremlinDiscovery(
			ctx, newFakeClient(), "graph", "query", nil, 10, 10, nil,
			func([]byte) error { return nil },
		); !errors.Is(err, context.Canceled) {
			t.Fatalf("visitGremlinDiscovery() error = %v", err)
		}
	})

	t.Run("visit page budget", func(t *testing.T) {
		client := newFakeClient()
		client.script("graph", "query", fakePage{
			items: [][]byte{jsonItem(`{}`)},
		})
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{RawInputBytes: 1},
		)
		if err := visitGremlinDiscovery(
			t.Context(), client, "graph", "query", nil, 10, 10, budget,
			func([]byte) error { return nil },
		); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("visitGremlinDiscovery() error = %v", err)
		}
	})

	t.Run("visit row budget", func(t *testing.T) {
		client := newFakeClient()
		client.script("graph", "query", fakePage{
			items: [][]byte{jsonItem(`{}`), jsonItem(`{}`)},
		})

		t.Run("visit decoded byte charge", func(t *testing.T) {
			client := newFakeClient()
			client.script("graph", "query", fakePage{
				items: [][]byte{jsonItem(`{}`)},
			})
			budget := sourcecontract.NewProfileBudget(
				sourcecontract.ProfileBudgetLimits{DecodedInputBytes: 1},
			)
			if err := visitGremlinDiscovery(
				t.Context(), client, "graph", "query", nil, 10, 10, budget,
				func([]byte) error { return nil },
			); !errors.Is(err, sourcecontract.ErrProfileBudget) {
				t.Fatalf("visitGremlinDiscovery() error = %v", err)
			}
		})

		t.Run("visit label dimension full", func(t *testing.T) {
			budget := sourcecontract.NewProfileBudget(
				sourcecontract.ProfileBudgetLimits{Labels: 1},
			)
			if err := budget.Charge(sourcecontract.ProfileBudgetUsage{Labels: 1}); err != nil {
				t.Fatal(err)
			}
			if err := visitGremlinDiscovery(
				t.Context(), newFakeClient(), "graph", "query", nil, 10, 10, budget,
				func([]byte) error { return nil },
			); !errors.Is(err, sourcecontract.ErrProfileBudget) {
				t.Fatalf("visitGremlinDiscovery() error = %v", err)
			}
		})

		t.Run("unbounded edge discovery failures", func(t *testing.T) {
			options := *gremlinSource().Gremlin
			source := gremlinSource()
			query := gremlinEdgeMappingsQuery +
				" AND STARTSWITH(c.label, @relationshipTypePrefix)" +
				" AND STARTSWITH(c._vertexLabel, @labelPrefix)" +
				" AND STARTSWITH(c._sinkLabel, @labelPrefix)"
			for _, item := range []string{
				"{",
				`"not-object"`,
				`{"label":"APP_KNOWS","startLabel":7,"endLabel":"AppPerson"}`,
			} {
				client := newFakeClient()
				client.script(options.Container, query, fakePage{
					items: [][]byte{jsonItem(item)},
				})
				if _, err := discoverGremlinEdges(
					t.Context(), client, source, options, []string{"AppPerson"},
				); err == nil {
					t.Fatalf("discoverGremlinEdges(%q) succeeded", item)
				}
			}

			client := newFakeClient()
			client.script(options.Container, query, fakePage{items: [][]byte{
				jsonItem(`{"label":"APP_A","startLabel":"Missing","endLabel":"AppPerson"}`),
				jsonItem(`{"label":"APP_B","startLabel":"AppPerson","endLabel":"Missing"}`),
			}})
			edges, err := discoverGremlinEdges(
				t.Context(), client, source, options, []string{"AppPerson"},
			)
			if err != nil || len(edges) != 0 {
				t.Fatalf("discoverGremlinEdges() = %#v, %v", edges, err)
			}

			options.MaxLabels = 1
			client = newFakeClient()
			client.script(options.Container, query, fakePage{items: [][]byte{
				jsonItem(`{"label":"APP_A","startLabel":"AppPerson","endLabel":"AppPerson"}`),
				jsonItem(`{"label":"APP_B","startLabel":"AppPerson","endLabel":"AppPerson"}`),
			}})
			if _, err := discoverGremlinEdges(
				t.Context(), client, source, options, []string{"AppPerson"},
			); err == nil {
				t.Fatal("discoverGremlinEdges() accepted too many relationship types")
			}
		})

		t.Run("unbounded interpretation edge failure", func(t *testing.T) {
			source := gremlinSource()
			labelQuery := gremlinVertexLabelsQuery +
				" AND STARTSWITH(c.label, @labelPrefix)"
			edgeQuery := gremlinEdgeMappingsQuery +
				" AND STARTSWITH(c.label, @relationshipTypePrefix)" +
				" AND STARTSWITH(c._vertexLabel, @labelPrefix)" +
				" AND STARTSWITH(c._sinkLabel, @labelPrefix)"
			client := newFakeClient()
			client.script(source.Gremlin.Container, labelQuery, fakePage{
				items: [][]byte{jsonItem(`"AppPerson"`)},
			})
			client.script(source.Gremlin.Container, edgeQuery, fakePage{
				items: [][]byte{jsonItem(`{`)},
			})
			if _, err := InterpretGremlinDocuments(
				t.Context(), source, client,
			); err == nil {
				t.Fatal("InterpretGremlinDocuments() ignored edge discovery failure")
			}
		})

		t.Run("unbounded invalid label JSON", func(t *testing.T) {
			source := gremlinSource()
			query := gremlinVertexLabelsQuery +
				" AND STARTSWITH(c.label, @labelPrefix)"
			client := newFakeClient()
			client.script(source.Gremlin.Container, query, fakePage{
				items: [][]byte{jsonItem(`{`)},
			})
			if _, err := discoverGremlinLabels(
				t.Context(), client, source, *source.Gremlin,
			); err == nil {
				t.Fatal("discoverGremlinLabels() accepted invalid JSON")
			}
		})
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{Rows: 1},
		)
		if err := visitGremlinDiscovery(
			t.Context(), client, "graph", "query", nil, 10, 10, budget,
			func([]byte) error { return nil },
		); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("visitGremlinDiscovery() error = %v", err)
		}
	})

	t.Run("decode edge mapping", func(t *testing.T) {
		for _, document := range []map[string]any{
			{"label": 1, "startLabel": "A", "endLabel": "B"},
			{"label": "R", "startLabel": "", "endLabel": "B"},
			{"label": "R", "startLabel": "A", "endLabel": "\n"},
		} {
			if _, err := decodeGremlinEdgeMapping(document); err == nil {
				t.Fatalf("decodeGremlinEdgeMapping(%#v) succeeded", document)
			}
		}
	})

	for _, value := range []string{"", strings.Repeat("x", 257), string([]byte{0xff}), "bad\nname"} {
		if validGremlinName(value) {
			t.Fatalf("validGremlinName(%q) = true", value)
		}
	}
}

type queryClientOnly struct {
	client *fakeClient
}

func (client *queryClientOnly) NewQueryPager(
	container, query string,
	parameters []Parameter,
	options QueryOptions,
) (Pager, error) {
	return client.client.NewQueryPager(container, query, parameters, options)
}

var _ QueryClient = (*queryClientOnly)(nil)
var _ io.Closer = (*fakeClient)(nil)
