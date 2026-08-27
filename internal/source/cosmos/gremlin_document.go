package cosmos

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/rioriost/agefreighter/pkg/model"
)

var gremlinStructuralFields = map[string]struct{}{
	"id":                {},
	"label":             {},
	"_rid":              {},
	"_self":             {},
	"_etag":             {},
	"_attachments":      {},
	"_ts":               {},
	"_isEdge":           {},
	"_vertexId":         {},
	"_vertexLabel":      {},
	"_sink":             {},
	"_sinkLabel":        {},
	"_sinkPartition":    {},
	"_isPkEdgeProperty": {},
}

func (iterator *Iterator) decodeGremlinRecord(
	ctx context.Context,
	mapping compiledMapping,
	document any,
) (model.Record, int64, error) {
	object, ok := document.(map[string]any)
	if !ok {
		return model.Record{}, 0, errors.New(
			"Cosmos Gremlin document must be a JSON object",
		)
	}
	if err := validateGremlinDocumentKind(object, mapping.kind); err != nil {
		return model.Record{}, 0, err
	}
	label, err := gremlinRequiredString(object, "label")
	if err != nil {
		return model.Record{}, 0, err
	}
	if label != string(mapping.label) {
		return model.Record{}, 0, errors.New(
			"Cosmos Gremlin document label does not match its mapping",
		)
	}
	properties, encoded, propertiesSize, err := iterator.gremlinProperties(
		ctx,
		object,
		mapping,
	)
	if err != nil {
		return model.Record{}, 0, err
	}
	if mapping.kind == vertexMapping {
		externalID, err := gremlinDocumentID(
			object,
			"id",
			mapping.partitionKeyProperty,
		)
		if err != nil {
			return model.Record{}, 0, err
		}
		vertex := model.Vertex{
			Label:             mapping.label,
			Namespace:         mapping.namespace,
			ExternalID:        externalID,
			Properties:        properties,
			EncodedProperties: encoded,
		}
		size := saturatingAdd(propertiesSize, vertexBaseSize)
		size = saturatingAdd(size, int64(
			len(vertex.Label)+len(vertex.Namespace)+len(vertex.ExternalID),
		))
		return model.VertexRecord(vertex), size, nil
	}

	startLabel, err := gremlinRequiredString(object, "_vertexLabel")
	if err != nil {
		return model.Record{}, 0, err
	}
	endLabel, err := gremlinRequiredString(object, "_sinkLabel")
	if err != nil {
		return model.Record{}, 0, err
	}
	if startLabel != mapping.start.Label || endLabel != mapping.end.Label {
		return model.Record{}, 0, errors.New(
			"Cosmos Gremlin edge endpoint labels do not match its mapping",
		)
	}
	externalID, err := gremlinDocumentID(
		object,
		"id",
		mapping.partitionKeyProperty,
	)
	if err != nil {
		return model.Record{}, 0, err
	}
	startID, err := gremlinDocumentID(
		object,
		"_vertexId",
		mapping.partitionKeyProperty,
	)
	if err != nil {
		return model.Record{}, 0, err
	}
	endID, err := gremlinDocumentID(
		object,
		"_sink",
		"_sinkPartition",
	)
	if err != nil {
		return model.Record{}, 0, err
	}
	edge := model.Edge{
		Label:      mapping.label,
		Namespace:  mapping.namespace,
		ExternalID: externalID,
		Start: model.Endpoint{
			Label:      model.Label(startLabel),
			Namespace:  mapping.namespace,
			ExternalID: startID,
		},
		End: model.Endpoint{
			Label:      model.Label(endLabel),
			Namespace:  mapping.namespace,
			ExternalID: endID,
		},
		Properties:        properties,
		EncodedProperties: encoded,
	}
	size := saturatingAdd(propertiesSize, edgeBaseSize)
	size = saturatingAdd(size, int64(
		len(edge.Label)+len(edge.Namespace)+len(edge.ExternalID)+
			len(edge.Start.Label)+len(edge.Start.Namespace)+len(edge.Start.ExternalID)+
			len(edge.End.Label)+len(edge.End.Namespace)+len(edge.End.ExternalID),
	))
	return model.EdgeRecord(edge), size, nil
}

func validateGremlinDocumentKind(
	document map[string]any,
	kind mappingKind,
) error {
	raw, found := document["_isEdge"]
	if kind == vertexMapping {
		if found {
			return errors.New(
				"Cosmos Gremlin vertex document contains _isEdge",
			)
		}
		return nil
	}
	isEdge, ok := raw.(bool)
	if !found || !ok || !isEdge {
		return errors.New(
			"Cosmos Gremlin edge document must contain boolean _isEdge=true",
		)
	}
	return nil
}

func gremlinRequiredString(
	document map[string]any,
	field string,
) (string, error) {
	value, ok := document[field].(string)
	if !ok || value == "" {
		return "", fmt.Errorf(
			"Cosmos Gremlin document field %q must be a non-empty string",
			field,
		)
	}
	return value, nil
}

func gremlinDocumentID(
	document map[string]any,
	idField string,
	partitionField string,
) (model.ExternalID, error) {
	id, err := gremlinRequiredString(document, idField)
	if err != nil {
		return "", err
	}
	partition, found := document[partitionField]
	if !found {
		return "", fmt.Errorf(
			"Cosmos Gremlin document is missing partition key field %q",
			partitionField,
		)
	}
	switch partition.(type) {
	case string, bool, json.Number:
	default:
		return "", fmt.Errorf(
			"Cosmos Gremlin partition key field %q must be a string, number, or boolean",
			partitionField,
		)
	}
	encoded, err := json.Marshal([]any{partition, id})
	if err != nil {
		return "", fmt.Errorf("encode Cosmos Gremlin composite identity: %w", err)
	}
	return model.ExternalID(encoded), nil
}

func (iterator *Iterator) gremlinProperties(
	ctx context.Context,
	document map[string]any,
	mapping compiledMapping,
) (model.Properties, []byte, int64, error) {
	names := make([]string, 0, len(document))
	for name := range document {
		if _, structural := gremlinStructuralFields[name]; structural ||
			name == mapping.partitionKeyProperty ||
			strings.HasPrefix(name, "_") {
			continue
		}
		names = append(names, name)
	}
	slices.Sort(names)
	if len(names) > mapping.maxProperties {
		return nil, nil, 0, fmt.Errorf(
			"Cosmos Gremlin document has %d properties, maximum is %d",
			len(names),
			mapping.maxProperties,
		)
	}
	properties := make(model.Properties, len(names))
	var size int64
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return nil, nil, 0, err
		}
		raw := document[name]
		if mapping.kind == vertexMapping {
			var err error
			raw, err = unwrapGremlinVertexProperty(raw)
			if err != nil {
				return nil, nil, 0, fmt.Errorf(
					"Cosmos Gremlin vertex property %q: %w",
					name,
					err,
				)
			}
		}
		value, err := convertValue(raw, 0)
		if err != nil {
			return nil, nil, 0, fmt.Errorf(
				"Cosmos Gremlin property %q: %w",
				name,
				err,
			)
		}
		properties[name] = value
		size = saturatingAdd(
			size,
			saturatingAdd(estimateValueSize(value), int64(len(name))),
		)
	}
	if !iterator.options.PreencodeProperties {
		return properties, nil, size, nil
	}
	encoded, err := model.EncodeProperties(properties)
	if err != nil {
		return nil, nil, 0, fmt.Errorf(
			"encode Cosmos Gremlin properties: %w",
			err,
		)
	}
	return nil, encoded, int64(len(encoded)), nil
}

func unwrapGremlinVertexProperty(raw any) (any, error) {
	values, ok := raw.([]any)
	if !ok || len(values) == 0 {
		return raw, nil
	}
	unwrapped := make([]any, len(values))
	wrapped := false
	for _, item := range values {
		property, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if _, exists := property["_value"]; exists {
			wrapped = true
		}
		if _, exists := property["_meta"]; exists {
			wrapped = true
		}
		if _, exists := property["id"]; exists {
			wrapped = true
		}
	}
	if !wrapped {
		return raw, nil
	}
	for index, item := range values {
		property, ok := item.(map[string]any)
		if !ok {
			return nil, errors.New(
				"property mixes wrapped and unwrapped values",
			)
		}
		value, ok := property["_value"]
		if !ok {
			return nil, errors.New(
				"property wrapper is missing _value",
			)
		}
		unwrapped[index] = value
	}
	if len(unwrapped) == 1 {
		return unwrapped[0], nil
	}
	return unwrapped, nil
}
