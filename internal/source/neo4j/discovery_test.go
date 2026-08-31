package neo4j

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
)

func TestDiscoverMappingsBuildsDeterministicMappings(t *testing.T) {
	client := &fakeClient{streams: []RecordStream{
		discoveryStream(record(map[string]any{"label": "AppPerson"}, "label")),
		discoveryStream(
			record(map[string]any{"property": "vid"}, "property"),
			record(map[string]any{"property": "seq"}, "property"),
			record(map[string]any{"property": "name"}, "property"),
		),
		discoveryStream(record(
			map[string]any{"relationshipType": "APP_KNOWS"},
			"relationshipType",
		)),
		discoveryStream(record(map[string]any{
			"startLabels": []any{"AppPerson"},
			"endLabels":   []any{"AppPerson"},
		}, "startLabels", "endLabels")),
		discoveryStream(
			record(map[string]any{"property": "eid"}, "property"),
			record(map[string]any{"property": "seq"}, "property"),
			record(map[string]any{"property": "since"}, "property"),
		),
	}}
	source := discoverySource()
	source.Discovery.LabelPrefix = "App"
	source.Discovery.RelationshipTypePrefix = "APP_"

	resolved, err := DiscoverMappings(context.Background(), source, client)
	if err != nil {
		t.Fatalf("DiscoverMappings() error = %v", err)
	}
	if resolved.Discovery != nil {
		t.Fatalf("resolved discovery = %#v, want nil", resolved.Discovery)
	}
	if len(resolved.Vertices) != 1 || len(resolved.Edges) != 1 {
		t.Fatalf("resolved mappings = %#v, %#v", resolved.Vertices, resolved.Edges)
	}
	vertex := resolved.Vertices[0]
	if vertex.Label != "AppPerson" ||
		vertex.IDField != "__id" ||
		vertex.KeyField != "__key" ||
		vertex.Properties["name"] != "__property_0000" ||
		!strings.Contains(vertex.Query, "MATCH (n:`AppPerson`)") ||
		!strings.Contains(vertex.Query, "`seq` > $afterKey") ||
		strings.Contains(vertex.Query, "$afterKey IS NULL") ||
		!strings.Contains(vertex.InitialQuery, "`seq` IS NOT NULL") ||
		!strings.Contains(vertex.Query, "ORDER BY __key") ||
		strings.Contains(vertex.Query, "$pageRows") {
		t.Fatalf("vertex mapping = %#v", vertex)
	}
	edge := resolved.Edges[0]
	if edge.Label != "APP_KNOWS" ||
		edge.Start.Label != "AppPerson" ||
		edge.End.Label != "AppPerson" ||
		edge.ExternalIDField != "__id" ||
		edge.Properties["since"] != "__property_0002" ||
		!strings.Contains(edge.Query, "[r:`APP_KNOWS`]") ||
		!strings.Contains(edge.InitialQuery, "`seq` IS NOT NULL") ||
		!strings.Contains(edge.Query, "ORDER BY __key") ||
		strings.Contains(edge.Query, "$pageRows") {
		t.Fatalf("edge mapping = %#v", edge)
	}
	if _, err := buildMappings(
		context.Background(),
		"crm",
		resolved,
		10,
	); err != nil {
		t.Fatalf("generated mappings do not compile: %v", err)
	}
}

func TestDiscoverMappingsSupportsUnlabeledVertices(t *testing.T) {
	client := &fakeClient{streams: []RecordStream{
		discoveryStream(),
		discoveryStream(record(map[string]any{"count": int64(2)}, "count")),
		discoveryStream(
			record(map[string]any{"property": "vid"}, "property"),
			record(map[string]any{"property": "seq"}, "property"),
		),
		discoveryStream(),
	}}

	resolved, err := DiscoverMappings(
		context.Background(),
		discoverySource(),
		client,
	)
	if err != nil {
		t.Fatalf("DiscoverMappings() error = %v", err)
	}
	if len(resolved.Vertices) != 1 ||
		resolved.Vertices[0].Label != unlabeledTargetLabel ||
		!strings.Contains(
			resolved.Vertices[0].Query,
			"WHERE (size(labels(n)) = 0)",
		) {
		t.Fatalf("unlabeled mapping = %#v", resolved.Vertices)
	}
}

func TestDiscoverMappingsPartitionsMultiLabelVertices(t *testing.T) {
	client := &fakeClient{streams: []RecordStream{
		discoveryStream(
			record(map[string]any{"label": "Role"}, "label"),
			record(map[string]any{"label": "Person"}, "label"),
		),
		discoveryStream(record(map[string]any{"count": int64(0)}, "count")),
		discoveryStream(
			record(map[string]any{"property": "vid"}, "property"),
			record(map[string]any{"property": "seq"}, "property"),
		),
		discoveryStream(
			record(map[string]any{"property": "vid"}, "property"),
			record(map[string]any{"property": "seq"}, "property"),
		),
		discoveryStream(record(
			map[string]any{"relationshipType": "KNOWS"},
			"relationshipType",
		)),
		discoveryStream(record(map[string]any{
			"startLabels": []any{"Role", "Person"},
			"endLabels":   []any{"Role"},
		}, "startLabels", "endLabels")),
		discoveryStream(
			record(map[string]any{"property": "eid"}, "property"),
			record(map[string]any{"property": "seq"}, "property"),
		),
	}}

	resolved, err := DiscoverMappings(
		context.Background(),
		discoverySource(),
		client,
	)
	if err != nil {
		t.Fatalf("DiscoverMappings() error = %v", err)
	}
	if len(resolved.Edges) != 1 ||
		resolved.Edges[0].Start.Label != "Person" ||
		resolved.Edges[0].End.Label != "Role" ||
		!strings.Contains(
			resolved.Edges[0].Query,
			"a:`Person`",
		) {
		t.Fatalf("multi-label edge mapping = %#v", resolved.Edges)
	}
	if !strings.Contains(
		resolved.Vertices[0].Query,
		"n:`Person`",
	) || !strings.Contains(
		resolved.Vertices[1].Query,
		"n:`Role` AND NOT n:`Person`",
	) {
		t.Fatalf("partitioned vertex mappings = %#v", resolved.Vertices)
	}
	if len(client.queries) < 4 ||
		!strings.Contains(
			client.queries[3],
			"WHERE n:`Role` AND NOT n:`Person`",
		) {
		t.Fatalf("partitioned property discovery queries = %#v", client.queries)
	}
}

func TestDiscoverMappingsOmitsEmptyPrimaryLabelPartition(t *testing.T) {
	client := &fakeClient{streams: []RecordStream{
		discoveryStream(
			record(map[string]any{"label": "Role"}, "label"),
			record(map[string]any{"label": "Person"}, "label"),
		),
		discoveryStream(record(map[string]any{"count": int64(0)}, "count")),
		discoveryStream(
			record(map[string]any{"property": "vid"}, "property"),
			record(map[string]any{"property": "seq"}, "property"),
		),
		discoveryStream(),
		discoveryStream(record(map[string]any{"count": int64(0)}, "count")),
		discoveryStream(),
	}}

	resolved, err := DiscoverMappings(
		context.Background(),
		discoverySource(),
		client,
	)
	if err != nil {
		t.Fatalf("DiscoverMappings() error = %v", err)
	}
	if len(resolved.Vertices) != 1 ||
		resolved.Vertices[0].Label != "Person" {
		t.Fatalf("resolved vertices = %#v", resolved.Vertices)
	}
	if len(client.queries) < 5 ||
		!strings.Contains(
			client.queries[4],
			"WHERE n:`Role` AND NOT n:`Person`",
		) {
		t.Fatalf("partition count queries = %#v", client.queries)
	}
}

func TestDiscoverMappingsSkipsEdgesOutsideLabelPrefix(t *testing.T) {
	client := &fakeClient{streams: []RecordStream{
		discoveryStream(record(map[string]any{"label": "AppPerson"}, "label")),
		discoveryStream(
			record(map[string]any{"property": "vid"}, "property"),
			record(map[string]any{"property": "seq"}, "property"),
		),
		discoveryStream(record(
			map[string]any{"relationshipType": "APP_KNOWS"},
			"relationshipType",
		)),
		discoveryStream(record(map[string]any{
			"startLabels": []any{"AppPerson"},
			"endLabels":   []any{"External"},
		}, "startLabels", "endLabels")),
	}}
	source := discoverySource()
	source.Discovery.LabelPrefix = "App"

	resolved, err := DiscoverMappings(context.Background(), source, client)
	if err != nil {
		t.Fatalf("DiscoverMappings() error = %v", err)
	}
	if len(resolved.Edges) != 0 {
		t.Fatalf("out-of-scope edge mappings = %#v", resolved.Edges)
	}
}

func TestDiscoverMappingsRejectsInvalidMetadataAndLimits(t *testing.T) {
	tests := []struct {
		name    string
		client  *fakeClient
		options func(*config.Neo4jDiscovery)
		want    string
	}{
		{
			name: "label limit",
			client: &fakeClient{streams: []RecordStream{discoveryStream(
				record(map[string]any{"label": "A"}, "label"),
				record(map[string]any{"label": "B"}, "label"),
			)}},
			options: func(options *config.Neo4jDiscovery) {
				options.MaxLabels = 1
			},
			want: "more than 1",
		},
		{
			name: "missing stable property",
			client: &fakeClient{streams: []RecordStream{
				discoveryStream(record(map[string]any{"label": "Person"}, "label")),
				discoveryStream(record(map[string]any{"count": int64(0)}, "count")),
				discoveryStream(record(map[string]any{"property": "name"}, "property")),
			}},
			options: func(*config.Neo4jDiscovery) {},
			want:    `stable property "seq"`,
		},
		{
			name: "invalid discovered label",
			client: &fakeClient{streams: []RecordStream{
				discoveryStream(record(map[string]any{"label": "bad\nlabel"}, "label")),
			}},
			options: func(*config.Neo4jDiscovery) {},
			want:    "invalid label",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := discoverySource()
			test.options(source.Discovery)
			_, err := DiscoverMappings(context.Background(), source, test.client)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("DiscoverMappings() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestDiscoverMappingsPropagatesStreamFailures(t *testing.T) {
	source := discoverySource()
	for _, test := range []struct {
		name   string
		stream *fakeStream
	}{
		{
			name:   "next",
			stream: &fakeStream{nextErr: errors.New("read failed")},
		},
		{
			name:   "close",
			stream: &fakeStream{closeErr: errors.New("close failed")},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := DiscoverMappings(
				context.Background(),
				source,
				&fakeClient{streams: []RecordStream{test.stream}},
			)
			if err == nil {
				t.Fatal("DiscoverMappings() error = nil")
			}
		})
	}
}

func TestDiscoveryDoesNotStartQueryForExhaustedCatalogDimension(t *testing.T) {
	tests := []struct {
		name       string
		usage      sourcecontract.ProfileBudgetUsage
		exhausted  func(context.Context, Client, *sourcecontract.ProfileBudget) error
		applicable func(context.Context, Client, *sourcecontract.ProfileBudget) error
	}{
		{
			name:  "labels",
			usage: sourcecontract.ProfileBudgetUsage{Labels: 1},
			exhausted: func(ctx context.Context, client Client, budget *sourcecontract.ProfileBudget) error {
				_, err := discoverStrings(
					ctx, client, "labels", "label", "", 10, budget,
					sourcecontract.ProfileBudgetUsage{Labels: 1},
				)
				return err
			},
			applicable: func(ctx context.Context, client Client, budget *sourcecontract.ProfileBudget) error {
				_, err := discoverProperties(ctx, client, "properties", 10, budget)
				return err
			},
		},
		{
			name:  "properties",
			usage: sourcecontract.ProfileBudgetUsage{Properties: 1},
			exhausted: func(ctx context.Context, client Client, budget *sourcecontract.ProfileBudget) error {
				_, err := discoverProperties(ctx, client, "properties", 10, budget)
				return err
			},
			applicable: func(ctx context.Context, client Client, budget *sourcecontract.ProfileBudget) error {
				_, err := discoverStrings(
					ctx, client, "labels", "label", "", 10, budget,
					sourcecontract.ProfileBudgetUsage{Labels: 1},
				)
				return err
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			budget := sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{
				Rows: 10, Pages: 10, Labels: 1, Properties: 1,
			})
			if err := budget.Charge(test.usage); err != nil {
				t.Fatal(err)
			}
			client := &fakeClient{streams: []RecordStream{discoveryStream()}}
			if err := test.exhausted(t.Context(), client, budget); !errors.Is(
				err, sourcecontract.ErrProfileBudget,
			) {
				t.Fatalf("exhausted discovery error = %v", err)
			}
			if len(client.queries) != 0 {
				t.Fatalf("exhausted discovery started %d queries", len(client.queries))
			}
			if err := test.applicable(t.Context(), client, budget); err != nil {
				t.Fatalf("other catalog discovery error = %v", err)
			}
			if len(client.queries) != 1 {
				t.Fatalf("other catalog discovery started %d queries, want 1", len(client.queries))
			}
		})
	}
}

func TestDiscoveryIdentifierEscaping(t *testing.T) {
	if got := quoteCypherIdentifier("a`b"); got != "`a``b`" {
		t.Fatalf("quoteCypherIdentifier() = %q", got)
	}
	if !validDiscoveredIdentifier("a`b") ||
		validDiscoveredIdentifier("") ||
		validDiscoveredIdentifier("a\x00b") {
		t.Fatal("validDiscoveredIdentifier() returned an unexpected result")
	}
}

func TestDiscoverMappingsValidationAndOrchestrationFailures(t *testing.T) {
	source := discoverySource()
	for _, test := range []struct {
		name   string
		ctx    context.Context
		source config.Neo4jSource
		client Client
		want   string
	}{
		{"nil context", nil, source, &fakeClient{}, "context is required"},
		{"nil client", t.Context(), source, nil, "client is required"},
		{"missing discovery", t.Context(), config.Neo4jSource{}, &fakeClient{}, "configuration is required"},
		{"disabled discovery", t.Context(), config.Neo4jSource{Discovery: &config.Neo4jDiscovery{}}, &fakeClient{}, "configuration is required"},
		{"no labels", t.Context(), source, &fakeClient{streams: []RecordStream{
			discoveryStream(),
			discoveryStream(record(map[string]any{"count": int64(0)}, "count")),
		}}, "no matching vertices"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := DiscoverMappingsBounded(test.ctx, test.source, test.client, nil)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("DiscoverMappingsBounded() error = %v, want %q", err, test.want)
			}
		})
	}

	failures := []struct {
		name    string
		streams []RecordStream
		want    string
	}{
		{
			name: "unlabeled count",
			streams: []RecordStream{
				discoveryStream(),
				&fakeStream{nextErr: errors.New("count failed")},
			},
			want: "discover unlabeled",
		},
		{
			name: "vertex properties",
			streams: []RecordStream{
				discoveryStream(record(map[string]any{"label": "Person"}, "label")),
				discoveryStream(record(map[string]any{"count": int64(0)}, "count")),
				&fakeStream{nextErr: errors.New("property failed")},
			},
			want: `vertex label "Person"`,
		},
		{
			name: "empty vertex count",
			streams: []RecordStream{
				discoveryStream(record(map[string]any{"label": "Person"}, "label")),
				discoveryStream(record(map[string]any{"count": int64(0)}, "count")),
				discoveryStream(),
				&fakeStream{nextErr: errors.New("partition failed")},
			},
			want: "count Neo4j vertex",
		},
		{
			name: "relationship catalog",
			streams: []RecordStream{
				discoveryStream(record(map[string]any{"label": "Person"}, "label")),
				discoveryStream(record(map[string]any{"count": int64(0)}, "count")),
				discoveryStream(
					record(map[string]any{"property": "seq"}, "property"),
					record(map[string]any{"property": "vid"}, "property"),
				),
				&fakeStream{nextErr: errors.New("relationship failed")},
			},
			want: "relationship types",
		},
		{
			name: "relationship endpoints",
			streams: []RecordStream{
				discoveryStream(record(map[string]any{"label": "Person"}, "label")),
				discoveryStream(record(map[string]any{"count": int64(0)}, "count")),
				discoveryStream(
					record(map[string]any{"property": "seq"}, "property"),
					record(map[string]any{"property": "vid"}, "property"),
				),
				discoveryStream(record(map[string]any{"relationshipType": "KNOWS"}, "relationshipType")),
				&fakeStream{nextErr: errors.New("endpoint failed")},
			},
			want: "endpoint failed",
		},
		{
			name: "relationship properties",
			streams: []RecordStream{
				discoveryStream(record(map[string]any{"label": "Person"}, "label")),
				discoveryStream(record(map[string]any{"count": int64(0)}, "count")),
				discoveryStream(
					record(map[string]any{"property": "seq"}, "property"),
					record(map[string]any{"property": "vid"}, "property"),
				),
				discoveryStream(record(map[string]any{"relationshipType": "KNOWS"}, "relationshipType")),
				discoveryStream(record(map[string]any{
					"startLabels": []string{"Person"}, "endLabels": []string{"Person"},
				}, "startLabels", "endLabels")),
				&fakeStream{nextErr: errors.New("edge property failed")},
			},
			want: `relationship type "KNOWS"`,
		},
	}
	for _, test := range failures {
		t.Run(test.name, func(t *testing.T) {
			_, err := DiscoverMappings(t.Context(), source, &fakeClient{streams: test.streams})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("DiscoverMappings() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestDiscoverCountRowContracts(t *testing.T) {
	tests := []struct {
		name   string
		client *fakeClient
		want   int64
		err    string
	}{
		{"valid", &fakeClient{streams: []RecordStream{
			discoveryStream(record(map[string]any{"count": int64(3)}, "count")),
		}}, 3, ""},
		{"query", &fakeClient{queryErr: errors.New("query failed")}, 0, "query failed"},
		{"no row", &fakeClient{streams: []RecordStream{discoveryStream()}}, 0, "EOF"},
		{"missing count", &fakeClient{streams: []RecordStream{
			discoveryStream(record(map[string]any{"other": int64(1)}, "other")),
		}}, 0, "invalid count"},
		{"wrong count type", &fakeClient{streams: []RecordStream{
			discoveryStream(record(map[string]any{"count": 1}, "count")),
		}}, 0, "invalid count"},
		{"negative", &fakeClient{streams: []RecordStream{
			discoveryStream(record(map[string]any{"count": int64(-1)}, "count")),
		}}, 0, "invalid count"},
		{"multiple rows", &fakeClient{streams: []RecordStream{
			discoveryStream(
				record(map[string]any{"count": int64(1)}, "count"),
				record(map[string]any{"count": int64(2)}, "count"),
			),
		}}, 0, "multiple rows"},
		{"extra row error", &fakeClient{streams: []RecordStream{
			&fakeStream{
				records: []Record{record(map[string]any{"count": int64(1)}, "count")},
				nextErr: errors.New("extra failed"),
			},
		}}, 0, "extra failed"},
		{"close", &fakeClient{streams: []RecordStream{
			&fakeStream{
				records:  []Record{record(map[string]any{"count": int64(1)}, "count")},
				closeErr: errors.New("close failed"),
			},
		}}, 0, "close failed"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := discoverCount(t.Context(), test.client, "count", nil)
			if test.err == "" {
				if err != nil || got != test.want {
					t.Fatalf("discoverCount() = %d, %v, want %d", got, err, test.want)
				}
			} else if err == nil || !strings.Contains(err.Error(), test.err) {
				t.Fatalf("discoverCount() error = %v, want %q", err, test.err)
			}
		})
	}
}

func TestDiscoverStringsContractsAndEndpointHelpers(t *testing.T) {
	values, err := discoverStrings(
		t.Context(),
		&fakeClient{streams: []RecordStream{discoveryStream(
			record(map[string]any{"label": "B"}, "label"),
			record(map[string]any{"label": "skip"}, "label"),
			record(map[string]any{"label": "A"}, "label"),
			record(map[string]any{"label": "B"}, "label"),
		)}},
		"labels", "label", "", 4, nil,
		sourcecontract.ProfileBudgetUsage{Labels: 1},
	)
	if err != nil || strings.Join(values, ",") != "A,B,skip" {
		t.Fatalf("discoverStrings() = %v, %v", values, err)
	}
	for _, test := range []struct {
		name   string
		client *fakeClient
		max    int
		want   string
	}{
		{"query", &fakeClient{queryErr: errors.New("query failed")}, 1, "query failed"},
		{"next", &fakeClient{streams: []RecordStream{&fakeStream{nextErr: errors.New("next failed")}}}, 1, "next failed"},
		{"missing", &fakeClient{streams: []RecordStream{
			discoveryStream(record(map[string]any{"other": "A"}, "other")),
		}}, 1, "invalid label"},
		{"wrong type", &fakeClient{streams: []RecordStream{
			discoveryStream(record(map[string]any{"label": 1}, "label")),
		}}, 1, "invalid label"},
		{"limit", &fakeClient{streams: []RecordStream{discoveryStream(
			record(map[string]any{"label": "A"}, "label"),
			record(map[string]any{"label": "B"}, "label"),
		)}}, 1, "more than 1"},
		{"close", &fakeClient{streams: []RecordStream{
			&fakeStream{closeErr: errors.New("close failed")},
		}}, 1, "close failed"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := discoverStrings(
				t.Context(), test.client, "labels", "label", "", test.max, nil,
				sourcecontract.ProfileBudgetUsage{Labels: 1},
			)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("discoverStrings() error = %v, want %q", err, test.want)
			}
		})
	}

	labels := []discoveredLabel{
		{source: "A", target: "A"}, {source: "B", target: "B"},
		{target: unlabeledTargetLabel},
	}
	for _, test := range []struct {
		name     string
		record   Record
		field    string
		want     string
		selected bool
		err      string
	}{
		{"missing", record(map[string]any{}, "other"), "labels", "", false, "omitted"},
		{"invalid", record(map[string]any{"labels": "A"}, "labels"), "labels", "", false, "invalid"},
		{"unlabeled", record(map[string]any{"labels": []string{}}, "labels"), "labels", unlabeledTargetLabel, true, ""},
		{"outside", record(map[string]any{"labels": []any{"C"}}, "labels"), "labels", "", false, ""},
		{"primary sorted", record(map[string]any{"labels": []any{"B", "A"}}, "labels"), "labels", "A", true, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, selected, err := endpointPrimaryLabel(test.record, test.field, labels)
			if test.err != "" {
				if err == nil || !strings.Contains(err.Error(), test.err) {
					t.Fatalf("endpointPrimaryLabel() error = %v", err)
				}
				return
			}
			if err != nil || got != test.want || selected != test.selected {
				t.Fatalf("endpointPrimaryLabel() = %q, %t, %v", got, selected, err)
			}
		})
	}
	for _, value := range []any{[]string{"A"}, []any{"A", "B"}} {
		if got, err := stringList(value); err != nil || len(got) == 0 {
			t.Errorf("stringList(%#v) = %#v, %v", value, got, err)
		}
	}
	for _, value := range []any{[]string{"bad\n"}, []any{1}, "A"} {
		if _, err := stringList(value); err == nil {
			t.Errorf("stringList(%#v) succeeded", value)
		}
	}
	if got := vertexPartitionCountQuery(discoveredLabel{}, labels); got != discoverUnlabeledQuery {
		t.Fatalf("unlabeled partition query = %q", got)
	}
}

func discoverySource() config.Neo4jSource {
	return config.Neo4jSource{
		Discovery: &config.Neo4jDiscovery{
			Enabled:           true,
			VertexKeyProperty: "seq",
			VertexIDProperty:  "vid",
			EdgeKeyProperty:   "seq",
			EdgeIDProperty:    "eid",
			MaxLabels:         10,
			MaxProperties:     10,
		},
	}
}

func discoveryStream(records ...Record) *fakeStream {
	return &fakeStream{records: records}
}
