package neo4j

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	neotypes "github.com/neo4j/neo4j-go-driver/v6/neo4j/dbtype"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

func TestMappingDefensiveValidationBranches(t *testing.T) {
	base := testSource()
	tests := []struct {
		name string
		edit func(*config.Neo4jSource)
	}{
		{"vertex label", func(source *config.Neo4jSource) { source.Vertices[0].Label = "" }},
		{"vertex id", func(source *config.Neo4jSource) { source.Vertices[0].IDField = "" }},
		{"vertex properties", func(source *config.Neo4jSource) {
			source.Vertices[0].Properties = map[string]string{"a": "a", "b": "b"}
		}},
		{"edge label", func(source *config.Neo4jSource) {
			source.Vertices = nil
			source.Edges = []config.EdgeQuery{{}}
		}},
		{"edge start", func(source *config.Neo4jSource) {
			source.Vertices = nil
			source.Edges = []config.EdgeQuery{validEdge()}
			source.Edges[0].Start.Label = ""
		}},
		{"edge end", func(source *config.Neo4jSource) {
			source.Vertices = nil
			source.Edges = []config.EdgeQuery{validEdge()}
			source.Edges[0].End.Field = ""
		}},
		{"edge query", func(source *config.Neo4jSource) {
			source.Vertices = nil
			source.Edges = []config.EdgeQuery{validEdge()}
			source.Edges[0].Query = ""
		}},
		{"edge properties", func(source *config.Neo4jSource) {
			source.Vertices = nil
			source.Edges = []config.EdgeQuery{validEdge()}
			source.Edges[0].Properties = map[string]string{"a": "a", "b": "b"}
		}},
		{"none", func(source *config.Neo4jSource) { source.Vertices = nil }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := base
			source.Vertices = append([]config.VertexQuery(nil), base.Vertices...)
			test.edit(&source)
			if _, err := buildMappings(context.Background(), "people", source, 1); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := buildMappings(ctx, "people", base, 10); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled build = %v", err)
	}
	if err := validateEndpoint(
		config.EndpointMapping{Label: "L", Field: "f"}, "", "endpoint",
	); err == nil {
		t.Fatal("accepted endpoint without namespace")
	}
	if vertexMapping.String() != "vertex" || edgeMapping.String() != "edge" ||
		mappingKind(99).String() != "unknown" {
		t.Fatal("unexpected mapping kind string")
	}
}

func validEdge() config.EdgeQuery {
	return config.EdgeQuery{
		Label: "REL", KeyField: "k",
		Query: "MATCH ()-[r]->() WHERE $afterKey IS NULL OR r.k > $afterKey " +
			"RETURN r.k AS k, r.s AS s, r.e AS e ORDER BY k",
		Start: config.EndpointMapping{Label: "A", Field: "s"},
		End:   config.EndpointMapping{Label: "B", Field: "e"},
	}
}

func TestIteratorErrorAndResumeBranches(t *testing.T) {
	t.Run("nil next context", func(t *testing.T) {
		iterator := newTestIterator(t, testSource(), &fakeClient{})
		if _, err := iterator.Next(nil); err == nil {
			t.Fatal("accepted nil context")
		}
	})
	t.Run("malformed without handler", func(t *testing.T) {
		iterator := newTestIterator(t, testSource(), &fakeClient{
			streams: []RecordStream{&fakeStream{records: []Record{
				record(map[string]any{"k": int64(1), "name": "bad"}, "k", "name"),
			}}},
		})
		if _, err := iterator.Next(context.Background()); err == nil {
			t.Fatal("accepted malformed record")
		}
	})
	t.Run("reject limit", func(t *testing.T) {
		iterator, err := NewIterator(context.Background(), IteratorOptions{
			Namespace: "people", Source: testSource(),
			Client: &fakeClient{streams: []RecordStream{&fakeStream{records: []Record{
				record(map[string]any{"k": int64(1)}, "k"),
			}}}},
			OnMalformed:    func(context.Context, MalformedRecord) error { return nil },
			MaxRecordBytes: 1000, MaxProperties: 10,
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := iterator.Next(context.Background()); err == nil ||
			!strings.Contains(err.Error(), "reject limit") {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("handler failure", func(t *testing.T) {
		iterator, err := NewIterator(context.Background(), IteratorOptions{
			Namespace: "people", Source: testSource(),
			Client: &fakeClient{streams: []RecordStream{&fakeStream{records: []Record{
				record(map[string]any{"k": int64(1)}, "k"),
			}}}},
			RejectLimit: 1, MaxRecordBytes: 1000, MaxProperties: 10,
			OnMalformed: func(context.Context, MalformedRecord) error {
				return errors.New("sink secret")
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		errText := ""
		if _, err := iterator.Next(context.Background()); err != nil {
			errText = err.Error()
		}
		if errText == "" || strings.Contains(errText, "secret") {
			t.Fatalf("handler error = %q", errText)
		}
	})
	t.Run("stream close failure", func(t *testing.T) {
		iterator := newTestIterator(t, testSource(), &fakeClient{
			streams: []RecordStream{&fakeStream{closeErr: errors.New("raw close")}},
		})
		if _, err := iterator.Next(context.Background()); err == nil ||
			strings.Contains(err.Error(), "raw close") {
			t.Fatalf("close error = %v", err)
		}
	})

	key := int64(1)
	base := resumeState{
		fingerprint: strings.Repeat("ab", 32), mappingIndex: 0,
		mappingKind: vertexMapping, consumed: 1, lastKey: &key,
	}
	token, _ := formatResumeToken(base)
	options := IteratorOptions{
		Namespace: "people", Source: testSource(), Client: &fakeClient{},
		AfterToken: token, MaxRecordBytes: 1000, MaxProperties: 10,
	}
	if _, err := NewIterator(context.Background(), options); err == nil ||
		!strings.Contains(err.Error(), "fingerprint") {
		t.Fatalf("fingerprint mismatch = %v", err)
	}
	mappings, _ := buildMappings(context.Background(), "people", options.Source, 10)
	fingerprint, _ := bindFingerprint(options.Source, "people", mappings)
	base.fingerprint = fingerprint
	base.mappingIndex = 9
	options.AfterToken, _ = formatResumeToken(base)
	if _, err := NewIterator(context.Background(), options); err == nil {
		t.Fatal("accepted out-of-range mapping")
	}
	base.mappingIndex = 0
	base.mappingKind = edgeMapping
	options.AfterToken, _ = formatResumeToken(base)
	if _, err := NewIterator(context.Background(), options); err == nil {
		t.Fatal("accepted mismatched mapping kind")
	}
	base.mappingKind = vertexMapping
	base.rejected = 2
	options.AfterToken, _ = formatResumeToken(base)
	if _, err := NewIterator(context.Background(), options); err == nil {
		t.Fatal("accepted excessive rejected count")
	}
}

func TestSizeHelpersAndRareConversions(t *testing.T) {
	values := []model.Value{
		{Kind: model.ValueString, String: "x"},
		{Kind: model.ValueList, List: []model.Value{{Kind: model.ValueInteger}}},
		{Kind: model.ValueObject, Object: map[string]model.Value{
			"x": {Kind: model.ValueBoolean},
		}},
		{Kind: model.ValueInteger},
	}
	for _, value := range values {
		if estimateValueSize(value) <= 0 {
			t.Fatalf("bad estimate for %#v", value)
		}
	}
	if saturatingAdd(math.MaxInt64, 1) != math.MaxInt64 ||
		saturatingAdd(1, 2) != 3 {
		t.Fatal("saturatingAdd failed")
	}
	if _, err := stringValue(string([]byte{0xff})); err == nil {
		t.Fatal("accepted invalid temporal string")
	}
	for _, raw := range []any{
		true, int64(1), timeFixture(), neotypes.Node{Props: map[string]any{"x": 1}},
		neotypes.Relationship{Props: map[string]any{"x": 1}}, []byte("x"),
	} {
		if estimateRawSize(raw, 0, 1000) <= 0 {
			t.Fatalf("bad raw estimate for %T", raw)
		}
	}
	if estimateRawSize("x", model.MaxPropertyDepth+1, 0) <= 0 ||
		estimateRawSize("x", 0, -1) <= 0 {
		t.Fatal("depth/budget estimates failed")
	}
	huge := resumeState{
		fingerprint: strings.Repeat("a", maxResumeTokenBytes),
		mappingKind: vertexMapping, consumed: 1, lastKey: pointerTo(int64(1)),
	}
	if _, err := formatResumeToken(huge); err == nil {
		t.Fatal("accepted oversized resume token")
	}
	if cloneKey(nil) != nil {
		t.Fatal("cloneKey(nil) != nil")
	}
}

func timeFixture() any {
	return struct{ X int }{X: 1}
}

func pointerTo[T any](value T) *T {
	return &value
}
