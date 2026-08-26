package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"

	"go.yaml.in/yaml/v3"
)

// maxCosmosParamDepth bounds recursion when decoding a Cosmos query
// parameter value, mirroring the nesting limit applied to source documents
// and canonical property encoding.
const maxCosmosParamDepth = 100

// CosmosParamValue holds a strict JSON value bound to a named Cosmos DB
// query parameter. It decodes the same way from YAML or JSON input and
// preserves the distinction between integers and floating point numbers,
// rejecting non-finite floats, integer overflow, and value shapes that are
// not representable as plain JSON (for example YAML tags such as
// timestamps or binary blobs).
//
// The decoded value is one of: nil, bool, int64, float64, string,
// []any (each element following these same rules), or
// map[string]any (each value following these same rules).
type CosmosParamValue struct {
	value any
}

// Native returns the decoded Go value backing this parameter.
func (value CosmosParamValue) Native() any {
	return value.value
}

// NewCosmosParamValue builds a CosmosParamValue from a native Go value
// (nil, bool, int64, float64, string, []any, or map[string]any, with the
// same shape recursively), primarily for programmatic construction and
// tests outside YAML/JSON decoding.
func NewCosmosParamValue(native any) (CosmosParamValue, error) {
	normalized, err := normalizeNativeParamValue(native, 0)
	if err != nil {
		return CosmosParamValue{}, err
	}
	return CosmosParamValue{value: normalized}, nil
}

func normalizeNativeParamValue(raw any, depth int) (any, error) {
	if depth > maxCosmosParamDepth {
		return nil, fmt.Errorf("Cosmos query parameter value nesting exceeds %d", maxCosmosParamDepth)
	}
	switch typed := raw.(type) {
	case nil:
		return nil, nil
	case bool:
		return typed, nil
	case string:
		return typed, nil
	case int:
		return int64(typed), nil
	case int64:
		return typed, nil
	case float64:
		if math.IsNaN(typed) || math.IsInf(typed, 0) {
			return nil, errors.New("Cosmos query parameter value must be a finite number")
		}
		return typed, nil
	case []any:
		list := make([]any, len(typed))
		for index, item := range typed {
			decoded, err := normalizeNativeParamValue(item, depth+1)
			if err != nil {
				return nil, err
			}
			list[index] = decoded
		}
		return list, nil
	case map[string]any:
		object := make(map[string]any, len(typed))
		for key, item := range typed {
			decoded, err := normalizeNativeParamValue(item, depth+1)
			if err != nil {
				return nil, err
			}
			object[key] = decoded
		}
		return object, nil
	default:
		return nil, fmt.Errorf("Cosmos query parameter value has unsupported shape %T", raw)
	}
}

func (value *CosmosParamValue) UnmarshalYAML(node *yaml.Node) error {
	decoded, err := decodeYAMLParamValue(node, 0)
	if err != nil {
		return err
	}
	value.value = decoded
	return nil
}

func (value CosmosParamValue) MarshalYAML() (any, error) {
	return value.value, nil
}

func (value *CosmosParamValue) UnmarshalJSON(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var raw any
	if err := decoder.Decode(&raw); err != nil {
		return fmt.Errorf("decode Cosmos query parameter value: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return errors.New("Cosmos query parameter value has trailing content")
	}
	decoded, err := normalizeJSONParamValue(raw, 0)
	if err != nil {
		return err
	}
	value.value = decoded
	return nil
}

func (value CosmosParamValue) MarshalJSON() ([]byte, error) {
	encoded, err := json.Marshal(value.value)
	if err != nil {
		return nil, fmt.Errorf("encode Cosmos query parameter value: %w", err)
	}
	return encoded, nil
}

func decodeYAMLParamValue(node *yaml.Node, depth int) (any, error) {
	if depth > maxCosmosParamDepth {
		return nil, fmt.Errorf("Cosmos query parameter value nesting exceeds %d", maxCosmosParamDepth)
	}
	if node == nil {
		return nil, errors.New("Cosmos query parameter value is required")
	}
	switch node.Kind {
	case yaml.DocumentNode:
		if len(node.Content) != 1 {
			return nil, errors.New("Cosmos query parameter value must contain exactly one document")
		}
		return decodeYAMLParamValue(node.Content[0], depth)
	case yaml.AliasNode:
		return decodeYAMLParamValue(node.Alias, depth)
	case yaml.ScalarNode:
		return decodeYAMLParamScalar(node)
	case yaml.SequenceNode:
		list := make([]any, 0, len(node.Content))
		for _, child := range node.Content {
			decoded, err := decodeYAMLParamValue(child, depth+1)
			if err != nil {
				return nil, err
			}
			list = append(list, decoded)
		}
		return list, nil
	case yaml.MappingNode:
		if len(node.Content)%2 != 0 {
			return nil, errors.New("Cosmos query parameter value has an incomplete mapping")
		}
		object := make(map[string]any, len(node.Content)/2)
		for index := 0; index < len(node.Content); index += 2 {
			keyNode := node.Content[index]
			if keyNode.Kind != yaml.ScalarNode || keyNode.Tag != "!!str" {
				return nil, errors.New("Cosmos query parameter value object keys must be strings")
			}
			if _, exists := object[keyNode.Value]; exists {
				return nil, fmt.Errorf(
					"Cosmos query parameter value has duplicate object key %q",
					keyNode.Value,
				)
			}
			decoded, err := decodeYAMLParamValue(node.Content[index+1], depth+1)
			if err != nil {
				return nil, err
			}
			object[keyNode.Value] = decoded
		}
		return object, nil
	default:
		return nil, errors.New("Cosmos query parameter value has an unsupported shape")
	}
}

func decodeYAMLParamScalar(node *yaml.Node) (any, error) {
	switch node.Tag {
	case "!!null":
		return nil, nil
	case "!!bool":
		var parsed bool
		if err := node.Decode(&parsed); err != nil {
			return nil, fmt.Errorf("invalid Cosmos query parameter boolean: %w", err)
		}
		return parsed, nil
	case "!!int":
		parsed, err := strconv.ParseInt(node.Value, 0, 64)
		if err != nil {
			return nil, fmt.Errorf("Cosmos query parameter integer %q overflows int64 or is invalid", node.Value)
		}
		return parsed, nil
	case "!!float":
		parsed, err := strconv.ParseFloat(node.Value, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid Cosmos query parameter number %q: %w", node.Value, err)
		}
		if math.IsNaN(parsed) || math.IsInf(parsed, 0) {
			return nil, errors.New("Cosmos query parameter value must be a finite number")
		}
		return parsed, nil
	case "!!str":
		return node.Value, nil
	default:
		return nil, fmt.Errorf("Cosmos query parameter value has unsupported tag %q", node.Tag)
	}
}

func normalizeJSONParamValue(raw any, depth int) (any, error) {
	if depth > maxCosmosParamDepth {
		return nil, fmt.Errorf("Cosmos query parameter value nesting exceeds %d", maxCosmosParamDepth)
	}
	switch typed := raw.(type) {
	case nil:
		return nil, nil
	case bool:
		return typed, nil
	case string:
		return typed, nil
	case json.Number:
		text := typed.String()
		if !strings.ContainsAny(text, ".eE") {
			parsed, err := strconv.ParseInt(text, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("Cosmos query parameter integer %q overflows int64", text)
			}
			return parsed, nil
		}
		parsed, err := strconv.ParseFloat(text, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid Cosmos query parameter number %q: %w", text, err)
		}
		if math.IsNaN(parsed) || math.IsInf(parsed, 0) {
			return nil, errors.New("Cosmos query parameter value must be a finite number")
		}
		return parsed, nil
	case []any:
		list := make([]any, len(typed))
		for index, item := range typed {
			decoded, err := normalizeJSONParamValue(item, depth+1)
			if err != nil {
				return nil, err
			}
			list[index] = decoded
		}
		return list, nil
	case map[string]any:
		object := make(map[string]any, len(typed))
		for key, item := range typed {
			decoded, err := normalizeJSONParamValue(item, depth+1)
			if err != nil {
				return nil, err
			}
			object[key] = decoded
		}
		return object, nil
	default:
		return nil, fmt.Errorf("Cosmos query parameter value has unsupported shape %T", raw)
	}
}
