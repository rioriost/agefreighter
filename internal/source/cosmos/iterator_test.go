package cosmos

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

func baseVertexSource() config.CosmosSource {
	return config.CosmosSource{
		Endpoint: "https://example.documents.azure.com:443/",
		Database: "graphdb",
		PageSize: 100,
		Vertices: []config.CosmosVertexQuery{
			{
				Container: "people",
				Label:     "Person",
				Query:     "SELECT * FROM c",
				IDField:   "/id",
			},
		},
	}
}

func drainAll(t *testing.T, iterator *Iterator) ([]model.Record, error) {
	t.Helper()
	var records []model.Record
	for {
		item, err := iterator.Next(context.Background())
		if errors.Is(err, io.EOF) {
			return records, nil
		}
		if err != nil {
			return records, err
		}
		records = append(records, item.Record)
	}
}

func TestIteratorMultiPageMultiMappingOrder(t *testing.T) {
	source := config.CosmosSource{
		Endpoint: "https://example.documents.azure.com:443/",
		Database: "graphdb",
		PageSize: 10,
		Vertices: []config.CosmosVertexQuery{
			{Container: "people", Label: "Person", Query: "SELECT * FROM c", IDField: "/id"},
			{Container: "orgs", Label: "Organization", Query: "SELECT * FROM c", IDField: "/id"},
		},
		Edges: []config.CosmosEdgeQuery{
			{Container: "works", Label: "WORKS_AT", Query: "SELECT * FROM c",
				Start: config.EndpointMapping{Label: "Person", Field: "/fromId"},
				End:   config.EndpointMapping{Label: "Organization", Field: "/toId"}},
		},
	}
	client := newFakeClient()
	client.script("people", "SELECT * FROM c",
		fakePage{items: [][]byte{jsonItem(`{"id":"p1"}`), jsonItem(`{"id":"p2"}`)}, hasContinuation: true, continuationToken: "p-cont-1"},
		fakePage{items: [][]byte{jsonItem(`{"id":"p3"}`)}},
	)
	client.script("orgs", "SELECT * FROM c",
		fakePage{items: [][]byte{jsonItem(`{"id":"o1"}`)}},
	)
	client.script("works", "SELECT * FROM c",
		fakePage{items: [][]byte{jsonItem(`{"fromId":"p1","toId":"o1"}`)}},
	)

	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()

	records, err := drainAll(t, iterator)
	if err != nil {
		t.Fatalf("drainAll: %v", err)
	}
	if len(records) != 5 {
		t.Fatalf("got %d records, want 5 (3 people + 1 org + 1 edge)", len(records))
	}
	wantVertexIDs := []string{"p1", "p2", "p3", "o1"}
	for index, want := range wantVertexIDs {
		if records[index].Kind() != model.RecordVertex || string(records[index].Vertex.ExternalID) != want {
			t.Errorf("records[%d] = %+v, want vertex %q", index, records[index], want)
		}
	}
	if records[3].Vertex.Label != "Organization" {
		t.Errorf("records[3].Label = %q, want Organization", records[3].Vertex.Label)
	}
	edge := records[4]
	if edge.Kind() != model.RecordEdge || edge.Edge.Start.ExternalID != "p1" || edge.Edge.End.ExternalID != "o1" {
		t.Errorf("records[4] = %+v, want edge p1->o1", edge)
	}

	if client.callCount() != 4 {
		t.Fatalf("fakeClient recorded %d calls, want 4 (2 people pages + 1 orgs + 1 works)", client.callCount())
	}
}

func TestIteratorEdgeAfterVertices(t *testing.T) {
	source := baseVertexSource()
	source.Edges = []config.CosmosEdgeQuery{
		{Container: "works", Label: "WORKS_AT", Query: "SELECT * FROM c",
			ExternalIDField: "/id",
			Start:           config.EndpointMapping{Label: "Person", Field: "/fromId"},
			End:             config.EndpointMapping{Label: "Person", Field: "/toId"}},
	}
	client := newFakeClient()
	client.script("people", "SELECT * FROM c",
		fakePage{items: [][]byte{jsonItem(`{"id":"p1","name":"Ada"}`)}},
	)
	client.script("works", "SELECT * FROM c",
		fakePage{items: [][]byte{jsonItem(`{"id":"e1","fromId":"p1","toId":"p1"}`)}},
	)
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()
	records, err := drainAll(t, iterator)
	if err != nil {
		t.Fatalf("drainAll: %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("got %d records, want 2", len(records))
	}
	if records[0].Kind() != model.RecordVertex {
		t.Error("expected first record to be a vertex")
	}
	if records[1].Kind() != model.RecordEdge {
		t.Error("expected second record to be an edge")
	}
	edge := records[1].Edge
	if edge.ExternalID != "e1" || edge.Start.ExternalID != "p1" || edge.End.ExternalID != "p1" {
		t.Errorf("edge = %+v, unexpected mapping", edge)
	}
}

func TestIteratorQueryParametersPassedThrough(t *testing.T) {
	source := baseVertexSource()
	kindValue, err := config.NewCosmosParamValue("person")
	if err != nil {
		t.Fatalf("NewCosmosParamValue: %v", err)
	}
	ageValue, err := config.NewCosmosParamValue(int64(42))
	if err != nil {
		t.Fatalf("NewCosmosParamValue: %v", err)
	}
	source.Vertices[0].Parameters = []config.CosmosQueryParameter{
		{Name: "@kind", Value: kindValue},
		{Name: "@age", Value: ageValue},
	}
	client := newFakeClient()
	client.script("people", "SELECT * FROM c", fakePage{items: [][]byte{jsonItem(`{"id":"p1"}`)}})

	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()
	if _, err := drainAll(t, iterator); err != nil {
		t.Fatalf("drainAll: %v", err)
	}
	if client.callCount() != 1 {
		t.Fatalf("got %d calls, want 1", client.callCount())
	}
	call := client.callAt(0)
	if len(call.parameters) != 2 {
		t.Fatalf("got %d parameters, want 2", len(call.parameters))
	}
	if call.parameters[0].Name != "@kind" || call.parameters[0].Value != "person" {
		t.Errorf("parameters[0] = %+v, want {@kind person}", call.parameters[0])
	}
	if call.parameters[1].Name != "@age" || call.parameters[1].Value != int64(42) {
		t.Errorf("parameters[1] = %+v, want {@age 42}", call.parameters[1])
	}
	if call.options.PageSizeHint != int32(source.PageSize) {
		t.Errorf("PageSizeHint = %d, want %d", call.options.PageSizeHint, source.PageSize)
	}
}

func TestIteratorNestedPointersAndEscaping(t *testing.T) {
	source := baseVertexSource()
	source.Vertices[0].Properties = map[string]string{
		"deep":  "/nested/deep/1",
		"slash": "/a~1b",
		"tilde": "/a~0b",
	}
	client := newFakeClient()
	document := `{"id":"p1","nested":{"deep":["zero","one"]},"a/b":"slash-value","a~b":"tilde-value"}`
	client.script("people", "SELECT * FROM c", fakePage{items: [][]byte{jsonItem(document)}})

	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()
	records, err := drainAll(t, iterator)
	if err != nil {
		t.Fatalf("drainAll: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("got %d records, want 1", len(records))
	}
	properties := records[0].Vertex.Properties
	if properties["deep"].String != "one" {
		t.Errorf("deep = %+v, want one", properties["deep"])
	}
	if properties["slash"].String != "slash-value" {
		t.Errorf("slash = %+v, want slash-value", properties["slash"])
	}
	if properties["tilde"].String != "tilde-value" {
		t.Errorf("tilde = %+v, want tilde-value", properties["tilde"])
	}
}

func TestIteratorAllJSONValueKinds(t *testing.T) {
	source := baseVertexSource()
	source.Vertices[0].Properties = map[string]string{
		"n": "/n", "b": "/b", "i": "/i", "f": "/f", "s": "/s", "l": "/l", "o": "/o",
	}
	client := newFakeClient()
	document := `{"id":"p1","n":null,"b":true,"i":7,"f":1.5,"s":"hi","l":[1,2],"o":{"x":1}}`
	client.script("people", "SELECT * FROM c", fakePage{items: [][]byte{jsonItem(document)}})

	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()
	records, err := drainAll(t, iterator)
	if err != nil {
		t.Fatalf("drainAll: %v", err)
	}
	properties := records[0].Vertex.Properties
	if properties["n"].Kind != model.ValueNull {
		t.Error("n: expected null")
	}
	if properties["b"].Kind != model.ValueBoolean || !properties["b"].Boolean {
		t.Error("b: expected true")
	}
	if properties["i"].Kind != model.ValueInteger || properties["i"].Integer != 7 {
		t.Error("i: expected integer 7")
	}
	if properties["f"].Kind != model.ValueFloat || properties["f"].Float != 1.5 {
		t.Error("f: expected float 1.5")
	}
	if properties["s"].Kind != model.ValueString || properties["s"].String != "hi" {
		t.Error("s: expected string hi")
	}
	if properties["l"].Kind != model.ValueList || len(properties["l"].List) != 2 {
		t.Error("l: expected a 2-element list")
	}
	if properties["o"].Kind != model.ValueObject || properties["o"].Object["x"].Integer != 1 {
		t.Error("o: expected object with x=1")
	}
}

func TestIteratorInt64BoundariesAndOverflow(t *testing.T) {
	source := baseVertexSource()
	source.Vertices[0].Properties = map[string]string{"v": "/v"}
	client := newFakeClient()
	client.script("people", "SELECT * FROM c",
		fakePage{items: [][]byte{
			jsonItem(`{"id":"p1","v":9223372036854775807}`),
			jsonItem(`{"id":"p2","v":-9223372036854775808}`),
		}},
	)
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()
	records, err := drainAll(t, iterator)
	if err != nil {
		t.Fatalf("drainAll: %v", err)
	}
	if records[0].Vertex.Properties["v"].Integer != 9223372036854775807 {
		t.Errorf("p1.v = %d", records[0].Vertex.Properties["v"].Integer)
	}
	if records[1].Vertex.Properties["v"].Integer != -9223372036854775808 {
		t.Errorf("p2.v = %d", records[1].Vertex.Properties["v"].Integer)
	}
}

func TestIteratorInt64OverflowIsMalformed(t *testing.T) {
	source := baseVertexSource()
	source.Vertices[0].Properties = map[string]string{"v": "/v"}
	client := newFakeClient()
	client.script("people", "SELECT * FROM c",
		fakePage{items: [][]byte{jsonItem(`{"id":"p1","v":99999999999999999999999999}`)}},
	)
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()
	if _, err := iterator.Next(context.Background()); err == nil {
		t.Fatal("expected overflow to be rejected in fail mode")
	}
}

func TestIteratorMalformedFailModeStopsImmediately(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient()
	client.script("people", "SELECT * FROM c",
		fakePage{items: [][]byte{jsonItem(`{"name":"missing-id"}`), jsonItem(`{"id":"p2"}`)}},
	)
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()
	_, err = iterator.Next(context.Background())
	if err == nil {
		t.Fatal("expected malformed record to fail immediately without a handler")
	}
}

func TestIteratorQuarantineAndRejectLimit(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient()
	client.script("people", "SELECT * FROM c",
		fakePage{items: [][]byte{
			jsonItem(`{"bad":1}`),
			jsonItem(`{"id":"p2"}`),
			jsonItem(`{"bad":2}`),
			jsonItem(`{"id":"p3"}`),
		}},
	)
	var quarantined []MalformedRecord
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
		RejectLimit: 1,
		OnMalformed: func(_ context.Context, malformed MalformedRecord) error {
			quarantined = append(quarantined, malformed)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()

	// The first malformed record is quarantined transparently (within the
	// limit of 1), so this call surfaces the next good record, p2.
	item, err := iterator.Next(context.Background())
	if err != nil {
		t.Fatalf("Next (1st malformed quarantined, then p2): %v", err)
	}
	if string(item.Record.Vertex.ExternalID) != "p2" {
		t.Fatalf("got %q, want p2", item.Record.Vertex.ExternalID)
	}
	// The second malformed record exceeds the reject limit.
	if _, err := iterator.Next(context.Background()); err == nil {
		t.Fatal("expected the second malformed record to exceed the reject limit")
	}
	if len(quarantined) != 1 {
		t.Fatalf("got %d quarantined records, want 1", len(quarantined))
	}
	if quarantined[0].Position.Connector != "cosmos-nosql" {
		t.Errorf("quarantined position connector = %q", quarantined[0].Position.Connector)
	}
	rejectedCount, position := iterator.RejectionCheckpoint()
	if rejectedCount != 2 {
		t.Errorf("RejectionCheckpoint count = %d, want 2", rejectedCount)
	}
	if position.Token == "" {
		t.Error("RejectionCheckpoint: expected a non-empty token")
	}
}

func TestIteratorResumeMidPage(t *testing.T) {
	source := baseVertexSource()
	page := fakePage{items: [][]byte{
		jsonItem(`{"id":"p1"}`), jsonItem(`{"id":"p2"}`), jsonItem(`{"id":"p3"}`), jsonItem(`{"id":"p4"}`),
	}}
	client := newFakeClient()
	client.script("people", "SELECT * FROM c", page, page) // one copy consumed per iterator instance

	first, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	item1, err := first.Next(context.Background())
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	item2, err := first.Next(context.Background())
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if string(item1.Record.Vertex.ExternalID) != "p1" || string(item2.Record.Vertex.ExternalID) != "p2" {
		t.Fatalf("unexpected first two records: %q, %q", item1.Record.Vertex.ExternalID, item2.Record.Vertex.ExternalID)
	}
	resumeToken := item2.Record.Vertex.Position.Token
	if err := first.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	second, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client, AfterToken: resumeToken,
	})
	if err != nil {
		t.Fatalf("NewIterator (resume): %v", err)
	}
	defer second.Close()
	records, err := drainAll(t, second)
	if err != nil {
		t.Fatalf("drainAll (resume): %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("got %d resumed records, want 2", len(records))
	}
	if string(records[0].Vertex.ExternalID) != "p3" || string(records[1].Vertex.ExternalID) != "p4" {
		t.Errorf("resumed records = %q, %q, want p3, p4", records[0].Vertex.ExternalID, records[1].Vertex.ExternalID)
	}
}

func TestIteratorResumeAtMappingBoundary(t *testing.T) {
	source := baseVertexSource()
	source.Vertices = append(source.Vertices, config.CosmosVertexQuery{
		Container: "orgs", Label: "Organization", Query: "SELECT * FROM c", IDField: "/id",
	})
	firstPage := fakePage{items: [][]byte{jsonItem(`{"id":"p1"}`), jsonItem(`{"id":"p2"}`)}}
	client := newFakeClient()
	client.script("people", "SELECT * FROM c", firstPage, firstPage)
	client.script("orgs", "SELECT * FROM c", fakePage{items: [][]byte{jsonItem(`{"id":"o1"}`)}})

	first, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	if _, err := first.Next(context.Background()); err != nil {
		t.Fatalf("Next: %v", err)
	}
	item2, err := first.Next(context.Background())
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	resumeToken := item2.Record.Vertex.Position.Token
	if err := first.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	second, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client, AfterToken: resumeToken,
	})
	if err != nil {
		t.Fatalf("NewIterator (resume): %v", err)
	}
	defer second.Close()
	records, err := drainAll(t, second)
	if err != nil {
		t.Fatalf("drainAll (resume): %v", err)
	}
	if len(records) != 1 || string(records[0].Vertex.ExternalID) != "o1" {
		t.Fatalf("resumed records = %+v, want a single Organization o1", records)
	}
}

func TestIteratorResumeFingerprintMismatch(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient()
	client.script("people", "SELECT * FROM c", fakePage{items: [][]byte{jsonItem(`{"id":"p1"}`)}})
	token := formatResumeToken(resumeState{fingerprint: "wrong", mappingIndex: 0, mappingKind: vertexMapping})
	_, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client, AfterToken: token,
	})
	if err == nil || !strings.Contains(err.Error(), "fingerprint") {
		t.Fatalf("expected a fingerprint mismatch error, got %v", err)
	}
}

func TestIteratorResumeMappingKindMismatch(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient()
	client.script("people", "SELECT * FROM c", fakePage{items: [][]byte{jsonItem(`{"id":"p1"}`)}})

	mappings, err := buildMappings(context.Background(), "ns", source, 1024)
	if err != nil {
		t.Fatalf("buildMappings: %v", err)
	}
	fingerprint, err := bindFingerprint(source.Endpoint, source.Database, "ns", int32(source.PageSize), mappings)
	if err != nil {
		t.Fatalf("bindFingerprint: %v", err)
	}
	token := formatResumeToken(resumeState{fingerprint: fingerprint, mappingIndex: 0, mappingKind: edgeMapping})
	_, err = NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client, AfterToken: token,
	})
	if err == nil || !strings.Contains(err.Error(), "kind") {
		t.Fatalf("expected a mapping kind mismatch error, got %v", err)
	}
}

func TestIteratorResumeMappingIndexOutOfRange(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient()
	client.script("people", "SELECT * FROM c", fakePage{items: [][]byte{jsonItem(`{"id":"p1"}`)}})

	mappings, err := buildMappings(context.Background(), "ns", source, 1024)
	if err != nil {
		t.Fatalf("buildMappings: %v", err)
	}
	fingerprint, err := bindFingerprint(source.Endpoint, source.Database, "ns", int32(source.PageSize), mappings)
	if err != nil {
		t.Fatalf("bindFingerprint: %v", err)
	}
	token := formatResumeToken(resumeState{fingerprint: fingerprint, mappingIndex: 5, mappingKind: vertexMapping})
	_, err = NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client, AfterToken: token,
	})
	if err == nil || !strings.Contains(err.Error(), "range") {
		t.Fatalf("expected a mapping index range error, got %v", err)
	}
}

func TestIteratorResumeSkipExceedsPage(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient()
	client.script("people", "SELECT * FROM c", fakePage{items: [][]byte{jsonItem(`{"id":"p1"}`)}})

	mappings, err := buildMappings(context.Background(), "ns", source, 1024)
	if err != nil {
		t.Fatalf("buildMappings: %v", err)
	}
	fingerprint, err := bindFingerprint(source.Endpoint, source.Database, "ns", int32(source.PageSize), mappings)
	if err != nil {
		t.Fatalf("bindFingerprint: %v", err)
	}
	token := formatResumeToken(resumeState{
		fingerprint: fingerprint, mappingIndex: 0, mappingKind: vertexMapping, consumed: 99,
	})
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client, AfterToken: token,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()
	if _, err := iterator.Next(context.Background()); err == nil || !strings.Contains(err.Error(), "skip") {
		t.Fatalf("expected a skip error, got %v", err)
	}
}

func TestIteratorEmptyPagesWithContinuationContinue(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient()
	client.script("people", "SELECT * FROM c",
		fakePage{items: nil, hasContinuation: true, continuationToken: "cont-1"},
		fakePage{items: nil, hasContinuation: true, continuationToken: "cont-2"},
		fakePage{items: [][]byte{jsonItem(`{"id":"p1"}`)}},
	)
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()
	records, err := drainAll(t, iterator)
	if err != nil {
		t.Fatalf("drainAll: %v", err)
	}
	if len(records) != 1 || string(records[0].Vertex.ExternalID) != "p1" {
		t.Fatalf("records = %+v, want a single p1", records)
	}
	if client.callCount() != 3 {
		t.Fatalf("got %d calls, want 3 (two empty pages plus the final page)", client.callCount())
	}
}

func TestIteratorCancellationIsPrompt(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient() // no scripted pages; a fetch attempt would fail loudly
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := iterator.Next(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("Next with a cancelled context = %v, want context.Canceled", err)
	}
	if client.callCount() != 0 {
		t.Errorf("expected no fetch attempts once the context was already cancelled, got %d", client.callCount())
	}
}

func TestIteratorCloseIsIdempotent(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient()
	client.script("people", "SELECT * FROM c", fakePage{items: [][]byte{jsonItem(`{"id":"p1"}`)}})
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := iterator.Next(context.Background()); err == nil {
		t.Fatal("expected Next to fail after Close")
	}
}

func TestIteratorCloseInvokesInjectedClientCloserExactlyOnce(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient()
	client.script("people", "SELECT * FROM c", fakePage{items: [][]byte{jsonItem(`{"id":"p1"}`)}})
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if client.closeCount != 1 {
		t.Fatalf("client.closeCount = %d, want 1 (Close must delegate to the injected Closer exactly once)", client.closeCount)
	}
}

func TestIteratorTelemetryAndThrottleWithoutTokenLeakage(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient()
	client.script("people", "SELECT * FROM c",
		fakePage{
			items: [][]byte{jsonItem(`{"id":"p1"}`)}, hasContinuation: true,
			continuationToken: "super-secret-continuation-value", requestCharge: 2.5, failedRequestCount: 1,
		},
		fakePage{items: [][]byte{jsonItem(`{"id":"p2"}`)}, requestCharge: 1.5},
	)
	client.addThrottled(3)

	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()
	if _, err := drainAll(t, iterator); err != nil {
		t.Fatalf("drainAll: %v", err)
	}

	telemetry := iterator.Telemetry()
	if telemetry.Pages != 2 {
		t.Errorf("Pages = %d, want 2", telemetry.Pages)
	}
	if telemetry.RequestCharge != 4.0 {
		t.Errorf("RequestCharge = %v, want 4.0", telemetry.RequestCharge)
	}
	if telemetry.FailedRequestAttempts != 1 {
		t.Errorf("FailedRequestAttempts = %d, want 1", telemetry.FailedRequestAttempts)
	}
	if telemetry.ThrottledRequests != 3 {
		t.Errorf("ThrottledRequests = %d, want 3", telemetry.ThrottledRequests)
	}
	if telemetry.ContinuationDigest != "" {
		t.Errorf("ContinuationDigest = %q, want empty after the final page", telemetry.ContinuationDigest)
	}
	if strings.Contains(fmt.Sprintf("%+v", telemetry), "super-secret-continuation-value") {
		t.Fatal("telemetry snapshot must never contain the raw continuation token")
	}
}

func TestIteratorTelemetryContinuationDigestIsTruncatedHash(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient()
	client.script("people", "SELECT * FROM c",
		fakePage{items: [][]byte{jsonItem(`{"id":"p1"}`)}, hasContinuation: true, continuationToken: "cont-value"},
	)
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer iterator.Close()
	if _, err := iterator.Next(context.Background()); err != nil {
		t.Fatalf("Next: %v", err)
	}
	digest := iterator.Telemetry().ContinuationDigest
	expected := truncatedDigest("cont-value")
	if digest != expected {
		t.Errorf("ContinuationDigest = %q, want %q", digest, expected)
	}
	if digest == "cont-value" || strings.Contains(digest, "cont-value") {
		t.Errorf("ContinuationDigest %q looks like it leaks the raw token", digest)
	}
}

func TestIteratorSizeBytesScalesWithRetainedPage(t *testing.T) {
	smallSource := baseVertexSource()
	smallClient := newFakeClient()
	smallClient.script("people", "SELECT * FROM c", fakePage{items: [][]byte{jsonItem(`{"id":"p1"}`)}})
	smallIterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: smallSource, Client: smallClient,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer smallIterator.Close()
	smallItem, err := smallIterator.Next(context.Background())
	if err != nil {
		t.Fatalf("Next: %v", err)
	}

	largeSource := baseVertexSource()
	largeClient := newFakeClient()
	largeItems := make([][]byte, 0, 200)
	for i := 0; i < 200; i++ {
		largeItems = append(largeItems, jsonItem(fmt.Sprintf(`{"id":"p%d"}`, i)))
	}
	largeClient.script("people", "SELECT * FROM c", fakePage{items: largeItems})
	largeIterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: largeSource, Client: largeClient,
	})
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	defer largeIterator.Close()
	largeItem, err := largeIterator.Next(context.Background())
	if err != nil {
		t.Fatalf("Next: %v", err)
	}

	if largeItem.SizeBytes <= smallItem.SizeBytes {
		t.Errorf(
			"SizeBytes did not scale with retained page size: small=%d large=%d",
			smallItem.SizeBytes, largeItem.SizeBytes,
		)
	}
}

func TestIteratorRejectsInvalidOptions(t *testing.T) {
	source := baseVertexSource()
	client := newFakeClient()

	if _, err := NewIterator(context.Background(), IteratorOptions{Source: source, Client: client}); err == nil {
		t.Error("expected missing namespace to be rejected")
	}
	if _, err := NewIterator(context.Background(), IteratorOptions{Namespace: "ns", Source: source}); err == nil {
		t.Error("expected missing client to be rejected")
	}
	if _, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: source, Client: client, RejectLimit: 1,
	}); err == nil {
		t.Error("expected a positive reject limit without a handler to be rejected")
	}
	badPageSize := baseVertexSource()
	badPageSize.PageSize = 0
	if _, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "ns", Source: badPageSize, Client: client,
	}); err == nil {
		t.Error("expected an invalid page size to be rejected")
	}
}
