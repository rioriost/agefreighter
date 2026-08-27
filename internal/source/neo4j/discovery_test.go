package neo4j

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
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
		!strings.Contains(vertex.Query, "ORDER BY __key") {
		t.Fatalf("vertex mapping = %#v", vertex)
	}
	edge := resolved.Edges[0]
	if edge.Label != "APP_KNOWS" ||
		edge.Start.Label != "AppPerson" ||
		edge.End.Label != "AppPerson" ||
		edge.ExternalIDField != "__id" ||
		edge.Properties["since"] != "__property_0002" ||
		!strings.Contains(edge.Query, "[r:`APP_KNOWS`]") {
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
