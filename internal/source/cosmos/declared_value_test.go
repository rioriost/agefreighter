package cosmos

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/pkg/model"
)

func TestDeclaredCosmosValues(t *testing.T) {
	for _, tc := range []struct {
		raw, typ string
		kind     model.ValueKind
		encoded  string
	}{
		{"1", "float64", model.ValueFloat, "1.0"}, {"1.5", "float64", model.ValueFloat, "1.5"},
		{"1.0", "int64", model.ValueInteger, "1"}, {"1e3", "int64", model.ValueInteger, "1000"},
		{"9223372036854775807", "int64", model.ValueInteger, "9223372036854775807"},
		{"-9223372036854775808", "int64", model.ValueInteger, "-9223372036854775808"},
		{"null", "float64", model.ValueNull, "null"}, {"[1,2.5,null]", "float64[]", model.ValueList, "[1.0,2.5,null]"},
		{"true", "boolean", model.ValueBoolean, "true"}, {`"value"`, "string", model.ValueString, `"value"`},
		{"1", "", model.ValueInteger, "1"}, {"1.0", "", model.ValueFloat, "1.0"},
	} {
		t.Run(tc.raw+tc.typ, func(t *testing.T) {
			raw, err := decodeDocument([]byte(tc.raw))
			if err != nil {
				t.Fatal(err)
			}
			v, err := convertDeclaredValue(raw, tc.typ)
			if err != nil {
				t.Fatal(err)
			}
			b, err := model.EncodeProperties(model.Properties{"v": v})
			if err != nil || v.Kind != tc.kind || string(b) != `{"v":`+tc.encoded+`}` {
				t.Fatalf("got %s %v", b, err)
			}
		})
	}
	for _, tc := range []struct{ raw, typ string }{
		{"1.5", "int64"}, {"9223372036854775808", "int64"}, {"-9223372036854775809", "int64"},
		{"9007199254740993", "float64"}, {"1e99999", "float64"}, {"1e-99999", "int64"}, {"1e-400", "float64"},
		{`"secret-number"`, "float64"}, {"true", "int64"}, {"1", "string"}, {"1", "boolean"}, {"1", "int64[]"}, {"[[1]]", "int64[]"}, {"[1,1.5]", "int64[]"},
	} {
		raw, err := decodeDocument([]byte(tc.raw))
		if err != nil {
			t.Fatal(err)
		}
		_, err = convertDeclaredValue(raw, tc.typ)
		if err == nil || strings.Contains(err.Error(), "secret-number") {
			t.Fatalf("accepted or leaked %s as %s", tc.raw, tc.typ)
		}
	}
	if _, err := convertDeclaredValue(json.Number(strings.Repeat("9", 129)), "int64"); err == nil {
		t.Fatal("unbounded number")
	}
}

func TestDeclaredCosmosFingerprintAndIterator(t *testing.T) {
	s := testSource()
	s.Vertices[0].Properties = map[string]string{"score": "/score"}
	s.Edges[0].Properties = map[string]string{"distance_km": "/distance"}
	fingerprint := func() string {
		v, err := bindFingerprint(s.Endpoint, s.Database, "ns", 100, compileTestMappings(t, s))
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	legacy := fingerprint()
	s.Vertices[0].PropertyTypes = map[string]string{}
	if fingerprint() != legacy {
		t.Fatal("empty declarations changed legacy fingerprint")
	}
	s.Vertices[0].PropertyTypes = map[string]string{"score": "float64"}
	typedVertex := fingerprint()
	if typedVertex == legacy {
		t.Fatal("vertex type not fingerprinted")
	}
	s.Edges[0].PropertyTypes = map[string]string{"distance_km": "float64"}
	if fingerprint() == typedVertex {
		t.Fatal("edge type not fingerprinted")
	}
	oldToken := formatResumeToken(resumeState{fingerprint: legacy, mappingKind: vertexMapping})
	if _, err := NewIterator(context.Background(), IteratorOptions{Namespace: "ns", Source: s, Client: newFakeClient(), AfterToken: oldToken}); err == nil || !strings.Contains(err.Error(), "fingerprint changed") {
		t.Fatalf("changed types must reject resume: %v", err)
	}
	client := newFakeClient()
	client.script("people", s.Vertices[0].Query, fakePage{items: [][]byte{jsonItem(`{"id":"p1","score":1}`)}})
	client.script("friendships", s.Edges[0].Query, fakePage{items: [][]byte{jsonItem(`{"id":"e1","fromId":"p1","toId":"p1","distance":2}`)}})
	i, err := NewIterator(context.Background(), IteratorOptions{Namespace: "ns", Source: s, Client: client, PreencodeProperties: true})
	if err != nil {
		t.Fatal(err)
	}
	defer i.Close()
	records, err := drainAll(t, i)
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 || string(records[0].Vertex.EncodedProperties) != `{"score":1.0}` || string(records[1].Edge.EncodedProperties) != `{"distance_km":2.0}` {
		t.Fatal("typed mapping lost in iterator")
	}
	s.Vertices[0].PropertyTypes["unmapped"] = "float64"
	if _, err := buildMappings(context.Background(), "ns", s, 1024); err == nil {
		t.Fatal("unmapped type accepted")
	}
}
