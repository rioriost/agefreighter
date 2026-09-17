package cosmos

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
)

// A bounded transport regression, not a live Cosmos or P1 qualification.
// JSON carries no schema distinction between an integer and an integral float.
func TestGremlinDeclaredTypesSurviveInterpretation(t *testing.T) {
	options := *gremlinSource().Gremlin
	if err := json.Unmarshal([]byte(`{"propertyTypes":{"score":"float64","distance_km":"float64"}}`), &options); err != nil {
		t.Fatal(err)
	}
	v, err := gremlinVertexQuery(options, "Person")
	if err != nil {
		t.Fatal(err)
	}
	e, err := gremlinEdgeQuery(options, gremlinEdgeMapping{label: "LINK", start: "Person", end: "Person"})
	if err != nil {
		t.Fatal(err)
	}
	s := gremlinSource()
	s.Gremlin = nil
	s.Vertices = []config.CosmosVertexQuery{v}
	s.Edges = []config.CosmosEdgeQuery{e}
	c := newFakeClient()
	c.script(options.Container, v.Query, fakePage{items: [][]byte{jsonItem(`{"id":"v1","label":"Person","pk":"p","score":[{"_value":1}],"source_key":[{"_value":1}]}`)}})
	c.script(options.Container, e.Query, fakePage{items: [][]byte{jsonItem(`{"id":"e1","label":"LINK","pk":"p","_isEdge":true,"_vertexId":"v1","_vertexLabel":"Person","_sink":"v1","_sinkLabel":"Person","_sinkPartition":"p","distance_km":2}`)}})
	i, err := NewIterator(t.Context(), IteratorOptions{Namespace: "p1", Source: s, Client: c})
	if err != nil {
		t.Fatal(err)
	}
	defer i.Close()
	records, err := drainAll(t, i)
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 {
		t.Fatalf("records=%d", len(records))
	}
	if got := records[0].Vertex.Properties["score"]; got.Kind != model.ValueFloat || got.Float != 1 {
		t.Fatalf("declared float lost: %#v", got)
	}
	if got := records[1].Edge.Properties["distance_km"]; got.Kind != model.ValueFloat || got.Float != 2 {
		t.Fatalf("edge declared float lost: %#v", got)
	}
	if records[0].Vertex.Properties["source_key"].Kind != model.ValueInteger {
		t.Fatal("undeclared integer changed")
	}
	if records[0].Vertex.ExternalID != `["p","v1"]` || records[1].Edge.Start.ExternalID != records[0].Vertex.ExternalID {
		t.Fatal("partition-qualified identity changed")
	}
}

func TestGremlinTypeFingerprintAndResume(t *testing.T) {
	s := gremlinSource()
	v, err := gremlinVertexQuery(*s.Gremlin, "Person")
	if err != nil {
		t.Fatal(err)
	}
	s.Gremlin = nil
	s.Vertices = []config.CosmosVertexQuery{v}
	fingerprint := func() string {
		t.Helper()
		f, err := bindFingerprint(s.Endpoint, s.Database, "p1", 2, compileTestMappings(t, s))
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	legacy := fingerprint()
	s.Vertices[0].PropertyTypes = map[string]string{}
	if fingerprint() != legacy {
		t.Fatal("empty declaration changed legacy resume identity")
	}
	s.Vertices[0].PropertyTypes = map[string]string{"score": "float64"}
	typed := fingerprint()
	if typed == legacy {
		t.Fatal("declaration not bound to fingerprint")
	}
	old := formatResumeToken(resumeState{fingerprint: legacy, mappingKind: vertexMapping})
	if _, err := NewIterator(t.Context(), IteratorOptions{Namespace: "p1", Source: s, Client: newFakeClient(), AfterToken: old}); err == nil || !strings.Contains(err.Error(), "fingerprint changed") {
		t.Fatalf("old checkpoint accepted: %v", err)
	}
	s.Vertices[0].PropertyTypes["score"] = "int64"
	if fingerprint() == typed {
		t.Fatal("changed type not bound")
	}
	m := compileTestMappings(t, s)
	s.Vertices[0].PropertyTypes["score"] = "string"
	if m[0].propertyTypes["score"] != "int64" {
		t.Fatal("compiled declarations alias caller state")
	}
}

func TestGremlinDeclaredValuesFailClosed(t *testing.T) {
	m := compiledMapping{kind: vertexMapping, label: "Person", namespace: "p1", documentFormat: config.CosmosDocumentGremlin, partitionKeyProperty: "pk", maxProperties: 10, propertyTypes: map[string]string{"score": "float64", "values": "float64[]", "optional": "float64", "absent": "float64"}}
	for _, preencoded := range []bool{false, true} {
		i := &Iterator{options: IteratorOptions{PreencodeProperties: preencoded}}
		doc, err := decodeDocument([]byte(`{"id":"v1","label":"Person","pk":"p","score":[{"_value":1}],"values":[{"_value":[1,2.5,null]}],"optional":[{"_value":null}]}`))
		if err != nil {
			t.Fatal(err)
		}
		r, _, err := i.decodeGremlinRecord(t.Context(), m, doc)
		if err != nil {
			t.Fatal(err)
		}
		encoded := r.Vertex.EncodedProperties
		if !preencoded {
			encoded, err = model.EncodeProperties(r.Vertex.Properties)
			if err != nil {
				t.Fatal(err)
			}
		}
		if string(encoded) != `{"optional":null,"score":1.0,"values":[1.0,2.5,null]}` {
			t.Fatalf("properties=%s", encoded)
		}
		for _, raw := range []string{`"not-a-number"`, `9007199254740993`, `1e999`} {
			doc, err = decodeDocument([]byte(`{"id":"v1","label":"Person","pk":"p","score":[{"_value":` + raw + `}]}`))
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err = i.decodeGremlinRecord(t.Context(), m, doc); err == nil || strings.Contains(err.Error(), "not-a-number") {
				t.Fatalf("incompatible value accepted or leaked: %v", err)
			}
		}
	}
}

func TestGremlinTypesValidateBeforeDiscovery(t *testing.T) {
	s := gremlinSource()
	s.Gremlin.PropertyTypes = map[string]string{"score": "date"}
	c := newFakeClient()
	if _, err := InterpretGremlinDocuments(t.Context(), s, c); err == nil {
		t.Fatal("invalid type accepted")
	}
	if c.callCount() != 0 {
		t.Fatal("queried before type validation")
	}
}

func TestGremlinDiscoveryRetainsTypesInSnapshot(t *testing.T) {
	for _, bounded := range []bool{false, true} {
		s := gremlinSource()
		s.Gremlin.LabelPrefix = ""
		s.Gremlin.RelationshipTypePrefix = ""
		s.Gremlin.PropertyTypes = map[string]string{"score": "float64"}
		c := newFakeClient()
		var budget *sourcecontract.ProfileBudget
		if bounded {
			c.script(s.Gremlin.Container, gremlinCatalogQuery, fakePage{items: [][]byte{jsonItem(`{"label":"AppPerson"}`), jsonItem(`{"isEdge":true,"label":"APP_KNOWS","startLabel":"AppPerson","endLabel":"AppPerson"}`)}})
			budget = sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{Rows: 10, Pages: 10, RawInputBytes: 1 << 20, DecodedInputBytes: 1 << 20, RequestCharge: 100, Labels: 10})
		} else {
			c.script(s.Gremlin.Container, gremlinVertexLabelsQuery, fakePage{items: [][]byte{jsonItem(`"AppPerson"`)}})
			c.script(s.Gremlin.Container, gremlinEdgeMappingsQuery, fakePage{items: [][]byte{jsonItem(`{"label":"APP_KNOWS","startLabel":"AppPerson","endLabel":"AppPerson"}`)}})
		}
		resolved, err := InterpretGremlinDocumentsBounded(t.Context(), s, c, budget)
		if err != nil {
			t.Fatal(err)
		}
		if len(resolved.Vertices) != 1 || len(resolved.Edges) != 1 {
			t.Fatal("unexpected discovery")
		}
		b, err := json.Marshal(resolved)
		if err != nil {
			t.Fatal(err)
		}
		var restored config.CosmosSource
		if err = json.Unmarshal(b, &restored); err != nil {
			t.Fatal(err)
		}
		if restored.Gremlin != nil || restored.Vertices[0].PropertyTypes["score"] != "float64" || restored.Edges[0].PropertyTypes["score"] != "float64" {
			t.Fatal("snapshot lost declarations")
		}
		resolved.Vertices[0].PropertyTypes["score"] = "int64"
		if s.Gremlin.PropertyTypes["score"] != "float64" || resolved.Edges[0].PropertyTypes["score"] != "float64" {
			t.Fatal("discovered maps alias")
		}
		if _, err := buildMappings(t.Context(), "p1", restored, 1024); err != nil {
			t.Fatal(err)
		}
	}
}
