package cosmos

import (
	"context"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

func TestInterpretGremlinDocumentsBuildsSortedMappings(t *testing.T) {
	source := gremlinSource()
	labelQuery := gremlinVertexLabelsQuery +
		" AND STARTSWITH(c.label, @labelPrefix)"
	edgeQuery := gremlinEdgeMappingsQuery +
		" AND STARTSWITH(c.label, @relationshipTypePrefix)" +
		" AND STARTSWITH(c._vertexLabel, @labelPrefix)" +
		" AND STARTSWITH(c._sinkLabel, @labelPrefix)"
	client := newFakeClient()
	client.script(
		source.Gremlin.Container,
		labelQuery,
		fakePage{
			items:             [][]byte{jsonItem(`"AppPerson"`)},
			hasContinuation:   true,
			continuationToken: "next-labels",
		},
		fakePage{items: [][]byte{jsonItem(`"AppOrganization"`)}},
	)
	client.script(
		source.Gremlin.Container,
		edgeQuery,
		fakePage{items: [][]byte{
			jsonItem(`{"label":"APP_WORKS_AT","startLabel":"AppPerson","endLabel":"AppOrganization"}`),
			jsonItem(`{"label":"APP_KNOWS","startLabel":"AppPerson","endLabel":"AppPerson"}`),
		}},
	)

	resolved, err := InterpretGremlinDocuments(
		context.Background(),
		source,
		client,
	)
	if err != nil {
		t.Fatalf("InterpretGremlinDocuments() error = %v", err)
	}
	if resolved.Gremlin != nil ||
		len(resolved.Vertices) != 2 ||
		resolved.Vertices[0].Label != "AppOrganization" ||
		resolved.Vertices[1].Label != "AppPerson" {
		t.Fatalf("resolved vertices = %#v", resolved.Vertices)
	}
	if len(resolved.Edges) != 2 ||
		resolved.Edges[0].Label != "APP_KNOWS" ||
		resolved.Edges[1].Label != "APP_WORKS_AT" {
		t.Fatalf("resolved edges = %#v", resolved.Edges)
	}
	for _, vertex := range resolved.Vertices {
		if vertex.DocumentFormat != config.CosmosDocumentGremlin ||
			vertex.PartitionKeyProperty != "pk" ||
			vertex.MaxProperties != 10 ||
			strings.Contains(vertex.Query, "ORDER BY") {
			t.Fatalf("generated vertex mapping = %#v", vertex)
		}
	}
	for _, edge := range resolved.Edges {
		if edge.DocumentFormat != config.CosmosDocumentGremlin ||
			edge.ExternalIDField != "/id" ||
			edge.Start.Field != "/_vertexId" ||
			edge.End.Field != "/_sink" {
			t.Fatalf("generated edge mapping = %#v", edge)
		}
	}
	if _, err := buildMappings(
		context.Background(),
		"graph",
		resolved,
		1_024,
	); err != nil {
		t.Fatalf("generated mappings do not compile: %v", err)
	}
	if client.callCount() != 3 ||
		!client.callAt(1).options.HasContinuationToken ||
		client.callAt(1).options.ContinuationToken != "next-labels" {
		t.Fatalf("discovery calls = %#v", client.calls)
	}
}

func TestInterpretGremlinDocumentsSkipsOutsideEndpoints(t *testing.T) {
	source := gremlinSource()
	source.Gremlin.LabelPrefix = ""
	source.Gremlin.RelationshipTypePrefix = ""
	client := newFakeClient()
	client.script(
		source.Gremlin.Container,
		gremlinVertexLabelsQuery,
		fakePage{items: [][]byte{jsonItem(`"Person"`)}},
	)
	client.script(
		source.Gremlin.Container,
		gremlinEdgeMappingsQuery,
		fakePage{items: [][]byte{
			jsonItem(`{"label":"KNOWS","startLabel":"Person","endLabel":"Missing"}`),
		}},
	)

	resolved, err := InterpretGremlinDocuments(
		context.Background(),
		source,
		client,
	)
	if err != nil {
		t.Fatalf("InterpretGremlinDocuments() error = %v", err)
	}
	if len(resolved.Edges) != 0 {
		t.Fatalf("out-of-scope edges = %#v", resolved.Edges)
	}
}

func TestInterpretGremlinDocumentsRejectsInvalidDiscovery(t *testing.T) {
	tests := []struct {
		name   string
		source config.CosmosSource
		client QueryClient
		want   string
	}{
		{
			name:   "missing options",
			source: config.CosmosSource{},
			client: newFakeClient(),
			want:   "configuration",
		},
		{
			name:   "missing client",
			source: gremlinSource(),
			want:   "client",
		},
		{
			name:   "no labels",
			source: gremlinSource(),
			client: scriptedGremlinLabels(),
			want:   "no matching vertices",
		},
		{
			name: "label limit",
			source: func() config.CosmosSource {
				value := gremlinSource()
				value.Gremlin.MaxLabels = 1
				return value
			}(),
			client: scriptedGremlinLabels(`"A"`, `"B"`),
			want:   "more than 1",
		},
		{
			name:   "invalid label",
			source: gremlinSource(),
			client: scriptedGremlinLabels(`7`),
			want:   "invalid vertex label",
		},
		{
			name: "document scan limit",
			source: func() config.CosmosSource {
				value := gremlinSource()
				value.Gremlin.MaxDiscoveryDocuments = 1
				return value
			}(),
			client: scriptedGremlinLabels(`"AppPerson"`, `"AppPerson"`),
			want:   "scanned more than 1 documents",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := InterpretGremlinDocuments(
				context.Background(),
				test.source,
				test.client,
			)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf(
					"InterpretGremlinDocuments() error = %v, want %q",
					err,
					test.want,
				)
			}
		})
	}
}

func TestGremlinDiscoveryRejectsRepeatedContinuation(t *testing.T) {
	source := gremlinSource()
	client := newFakeClient()
	client.script(
		source.Gremlin.Container,
		gremlinVertexLabelsQuery+
			" AND STARTSWITH(c.label, @labelPrefix)",
		fakePage{
			items:             [][]byte{jsonItem(`"AppPerson"`)},
			hasContinuation:   true,
			continuationToken: "same",
		},
		fakePage{
			items:             [][]byte{jsonItem(`"AppPerson"`)},
			hasContinuation:   true,
			continuationToken: "same",
		},
	)

	_, err := InterpretGremlinDocuments(context.Background(), source, client)
	if err == nil || !strings.Contains(err.Error(), "repeated continuation") {
		t.Fatalf("InterpretGremlinDocuments() error = %v", err)
	}
}

func TestIteratorInterpretsGremlinDocuments(t *testing.T) {
	options := *gremlinSource().Gremlin
	vertex, err := gremlinVertexQuery(options, "AppPerson")
	if err != nil {
		t.Fatal(err)
	}
	edge, err := gremlinEdgeQuery(options, gremlinEdgeMapping{
		label: "APP_KNOWS",
		start: "AppPerson",
		end:   "AppPerson",
	})
	if err != nil {
		t.Fatal(err)
	}
	source := gremlinSource()
	source.Gremlin = nil
	source.Vertices = []config.CosmosVertexQuery{vertex}
	source.Edges = []config.CosmosEdgeQuery{edge}
	client := newFakeClient()
	client.script(
		options.Container,
		vertex.Query,
		fakePage{items: [][]byte{jsonItem(`{
			"id":"v1",
			"label":"AppPerson",
			"pk":"source",
			"name":[{"id":"vp1","_value":"Ada","_meta":{}}],
			"tags":[
				{"id":"vp2","_value":"math","_meta":{}},
				{"id":"vp3","_value":"code","_meta":{}}
			],
			"active":true,
			"_rid":"internal"
		}`)}},
	)
	client.script(
		options.Container,
		edge.Query,
		fakePage{items: [][]byte{jsonItem(`{
			"id":"e1",
			"label":"APP_KNOWS",
			"pk":"source",
			"_isEdge":true,
			"_vertexId":"v1",
			"_vertexLabel":"AppPerson",
			"_sink":"v2",
			"_sinkLabel":"AppPerson",
			"_sinkPartition":"target",
			"weight":1.5,
			"_internal":"ignored"
		}`)}},
	)
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "crm",
		Source:    source,
		Client:    client,
	})
	if err != nil {
		t.Fatalf("NewIterator() error = %v", err)
	}
	defer iterator.Close()

	records, err := drainAll(t, iterator)
	if err != nil {
		t.Fatalf("drainAll() error = %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("records = %#v", records)
	}
	vertexRecord := records[0].Vertex
	if vertexRecord.ExternalID != `["source","v1"]` ||
		vertexRecord.Properties["name"].String != "Ada" ||
		vertexRecord.Properties["active"].Kind != model.ValueBoolean ||
		len(vertexRecord.Properties["tags"].List) != 2 {
		t.Fatalf("vertex = %#v", vertexRecord)
	}
	if _, exists := vertexRecord.Properties["pk"]; exists {
		t.Fatalf("partition key leaked into properties: %#v", vertexRecord.Properties)
	}
	edgeRecord := records[1].Edge
	if edgeRecord.ExternalID != `["source","e1"]` ||
		edgeRecord.Start.ExternalID != `["source","v1"]` ||
		edgeRecord.End.ExternalID != `["target","v2"]` ||
		edgeRecord.Properties["weight"].Float != 1.5 {
		t.Fatalf("edge = %#v", edgeRecord)
	}
	if _, exists := edgeRecord.Properties["_internal"]; exists {
		t.Fatalf("system field leaked into properties: %#v", edgeRecord.Properties)
	}
}

func TestIteratorRejectsMalformedGremlinDocuments(t *testing.T) {
	options := *gremlinSource().Gremlin
	vertex, err := gremlinVertexQuery(options, "AppPerson")
	if err != nil {
		t.Fatal(err)
	}
	source := gremlinSource()
	source.Gremlin = nil
	source.Vertices = []config.CosmosVertexQuery{vertex}
	tests := []struct {
		name     string
		document string
		want     string
	}{
		{
			name:     "edge in vertex query",
			document: `{"id":"v1","label":"AppPerson","pk":"p","_isEdge":true}`,
			want:     "_isEdge",
		},
		{
			name:     "wrong label",
			document: `{"id":"v1","label":"Other","pk":"p"}`,
			want:     "label",
		},
		{
			name:     "missing partition",
			document: `{"id":"v1","label":"AppPerson"}`,
			want:     "partition key",
		},
		{
			name: "malformed property wrapper",
			document: `{
				"id":"v1","label":"AppPerson","pk":"p",
				"name":[{"id":"vp1","_meta":{}}]
			}`,
			want: "_value",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := newFakeClient()
			client.script(
				options.Container,
				vertex.Query,
				fakePage{items: [][]byte{jsonItem(test.document)}},
			)
			iterator, err := NewIterator(
				context.Background(),
				IteratorOptions{
					Namespace: "crm",
					Source:    source,
					Client:    client,
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			defer iterator.Close()
			_, err = iterator.Next(context.Background())
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Next() error = %v, want %q", err, test.want)
			}
		})
	}
}

func gremlinSource() config.CosmosSource {
	return config.CosmosSource{
		Endpoint:   "https://example.documents.azure.com:443/",
		Credential: "default-azure",
		Database:   "graph",
		PageSize:   2,
		Gremlin: &config.CosmosGremlin{
			Enabled:                true,
			Container:              "graph",
			PartitionKeyProperty:   "pk",
			LabelPrefix:            "App",
			RelationshipTypePrefix: "APP_",
			MaxLabels:              10,
			MaxProperties:          10,
			MaxDiscoveryDocuments:  100,
		},
	}
}

func scriptedGremlinLabels(items ...string) *fakeClient {
	client := newFakeClient()
	raw := make([][]byte, len(items))
	for index, item := range items {
		raw[index] = jsonItem(item)
	}
	client.script(
		"graph",
		gremlinVertexLabelsQuery+
			" AND STARTSWITH(c.label, @labelPrefix)",
		fakePage{items: raw},
	)
	return client
}
