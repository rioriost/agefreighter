package cosmos

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

func TestGremlinDocumentValidationBranches(t *testing.T) {
	vertex := compiledMapping{
		kind: vertexMapping, label: "Person", namespace: "crm",
		documentFormat:       config.CosmosDocumentGremlin,
		partitionKeyProperty: "pk", maxProperties: 2,
	}
	edge := compiledMapping{
		kind: edgeMapping, label: "KNOWS", namespace: "crm",
		start:                config.EndpointMapping{Label: "Person"},
		end:                  config.EndpointMapping{Label: "Person"},
		documentFormat:       config.CosmosDocumentGremlin,
		partitionKeyProperty: "pk", maxProperties: 2,
	}
	iterator := &Iterator{options: IteratorOptions{}}
	tests := []struct {
		name     string
		mapping  compiledMapping
		document any
	}{
		{"non-object", vertex, []any{"not-object"}},
		{"missing label", vertex, map[string]any{"id": "v1", "pk": "p"}},
		{"empty label", vertex, map[string]any{"id": "v1", "label": "", "pk": "p"}},
		{"wrong label", vertex, map[string]any{"id": "v1", "label": "Other", "pk": "p"}},
		{"missing ID", vertex, map[string]any{"label": "Person", "pk": "p"}},
		{"invalid partition", vertex, map[string]any{
			"id": "v1", "label": "Person", "pk": []any{"bad"},
		}},
		{"too many properties", vertex, map[string]any{
			"id": "v1", "label": "Person", "pk": "p",
			"a": 1, "b": 2, "c": 3,
		}},
		{"missing edge marker", edge, map[string]any{
			"id": "e1", "label": "KNOWS", "pk": "p",
		}},
		{"false edge marker", edge, map[string]any{
			"id": "e1", "label": "KNOWS", "pk": "p", "_isEdge": false,
		}},
		{"missing start label", edge, map[string]any{
			"id": "e1", "label": "KNOWS", "pk": "p", "_isEdge": true,
		}},
		{"missing end label", edge, map[string]any{
			"id": "e1", "label": "KNOWS", "pk": "p", "_isEdge": true,
			"_vertexLabel": "Person",
		}},
		{"wrong endpoint label", edge, map[string]any{
			"id": "e1", "label": "KNOWS", "pk": "p", "_isEdge": true,
			"_vertexLabel": "Other", "_sinkLabel": "Person",
		}},
		{"missing start ID", edge, map[string]any{
			"id": "e1", "label": "KNOWS", "pk": "p", "_isEdge": true,
			"_vertexLabel": "Person", "_sinkLabel": "Person",
		}},
		{"missing end ID", edge, map[string]any{
			"id": "e1", "label": "KNOWS", "pk": "p", "_isEdge": true,
			"_vertexLabel": "Person", "_sinkLabel": "Person",
			"_vertexId": "v1",
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, _, err := iterator.decodeGremlinRecord(
				t.Context(), test.mapping, test.document,
			); err == nil {
				t.Fatal("decodeGremlinRecord() accepted invalid document")
			}
		})
	}
}

func TestGremlinDocumentPropertyBranches(t *testing.T) {
	mapping := compiledMapping{
		kind: vertexMapping, label: "Person", namespace: "crm",
		documentFormat:       config.CosmosDocumentGremlin,
		partitionKeyProperty: "pk", maxProperties: 10,
	}
	document := map[string]any{
		"id": "v1", "label": "Person", "pk": json.Number("7"),
		"name": []any{map[string]any{"_value": "Ada"}},
	}
	iterator := &Iterator{options: IteratorOptions{PreencodeProperties: true}}
	record, size, err := iterator.decodeGremlinRecord(t.Context(), mapping, document)
	if err != nil {
		t.Fatalf("decodeGremlinRecord() error = %v", err)
	}
	if record.Kind() != model.RecordVertex || record.Vertex.Properties != nil ||
		len(record.Vertex.EncodedProperties) == 0 || size == 0 {
		t.Fatalf("decodeGremlinRecord() = %#v, size %d", record, size)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, _, err := iterator.gremlinProperties(ctx, document, mapping); !errors.Is(
		err, context.Canceled,
	) {
		t.Fatalf("gremlinProperties() error = %v", err)
	}

	for _, raw := range []any{
		[]any{map[string]any{"_value": "wrapped"}, "plain"},
		[]any{map[string]any{"id": "wrapper"}},
	} {
		if _, err := unwrapGremlinVertexProperty(raw); err == nil {
			t.Fatalf("unwrapGremlinVertexProperty(%#v) succeeded", raw)
		}
	}
	unwrapped, err := unwrapGremlinVertexProperty([]any{
		map[string]any{"_value": "a"},
		map[string]any{"_value": "b"},
	})
	if err != nil || len(unwrapped.([]any)) != 2 {
		t.Fatalf("unwrapGremlinVertexProperty() = %#v, %v", unwrapped, err)
	}
	if value, err := unwrapGremlinVertexProperty([]any{"a", "b"}); err != nil ||
		len(value.([]any)) != 2 {
		t.Fatalf("unwrapped plain list = %#v, %v", value, err)
	}

	if _, err := gremlinDocumentID(
		map[string]any{"id": "v1"}, "id", "pk",
	); err == nil || !strings.Contains(err.Error(), "missing partition") {
		t.Fatalf("gremlinDocumentID() error = %v", err)
	}
	for _, partition := range []any{"p", true, json.Number("1")} {
		if _, err := gremlinDocumentID(
			map[string]any{"id": "v1", "pk": partition}, "id", "pk",
		); err != nil {
			t.Fatalf("gremlinDocumentID(%#v) error = %v", partition, err)
		}
	}
}
