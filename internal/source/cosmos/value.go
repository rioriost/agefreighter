package cosmos

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/rioriost/agefreighter/pkg/model"
)

// maxDocumentDepth bounds recursion when converting a decoded Cosmos
// document into model.Value, matching the nesting limit already enforced by
// the canonical property encoder in pkg/model.
const maxDocumentDepth = model.MaxPropertyDepth

// decodeDocument decodes a single Cosmos document using json.Decoder with
// UseNumber (to preserve integers exactly) and rejects any trailing content
// after the top-level JSON value.
func decodeDocument(raw []byte) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var document any
	if err := decoder.Decode(&document); err != nil {
		return nil, fmt.Errorf("decode Cosmos document: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return nil, errors.New("Cosmos document has trailing content")
	}
	return document, nil
}

// convertValue recursively converts a value produced by decodeDocument into
// a model.Value, preserving signed int64 integers exactly, rejecting
// integer overflow, non-finite floats, invalid UTF-8, unsupported shapes,
// and excessive nesting.
func convertValue(raw any, depth int) (model.Value, error) {
	if depth > maxDocumentDepth {
		return model.Value{}, fmt.Errorf("Cosmos document nesting exceeds %d", maxDocumentDepth)
	}
	switch typed := raw.(type) {
	case nil:
		return model.Value{Kind: model.ValueNull}, nil
	case bool:
		return model.Value{Kind: model.ValueBoolean, Boolean: typed}, nil
	case string:
		if !utf8.ValidString(typed) {
			return model.Value{}, errors.New("Cosmos document string is not valid UTF-8")
		}
		return model.Value{Kind: model.ValueString, String: typed}, nil
	case json.Number:
		text := typed.String()
		if !strings.ContainsAny(text, ".eE") {
			parsed, err := strconv.ParseInt(text, 10, 64)
			if err != nil {
				return model.Value{}, errors.New("Cosmos document integer overflows int64")
			}
			return model.Value{Kind: model.ValueInteger, Integer: parsed}, nil
		}
		parsed, err := strconv.ParseFloat(text, 64)
		if err != nil {
			return model.Value{}, fmt.Errorf("invalid Cosmos document number: %w", err)
		}
		if math.IsNaN(parsed) || math.IsInf(parsed, 0) {
			return model.Value{}, errors.New("Cosmos document number must be finite")
		}
		return model.Value{Kind: model.ValueFloat, Float: parsed}, nil
	case []any:
		list := make([]model.Value, len(typed))
		for index, item := range typed {
			value, err := convertValue(item, depth+1)
			if err != nil {
				return model.Value{}, err
			}
			list[index] = value
		}
		return model.Value{Kind: model.ValueList, List: list}, nil
	case map[string]any:
		object := make(map[string]model.Value, len(typed))
		for key, item := range typed {
			if !utf8.ValidString(key) {
				return model.Value{}, errors.New("Cosmos document property name is not valid UTF-8")
			}
			value, err := convertValue(item, depth+1)
			if err != nil {
				return model.Value{}, err
			}
			object[key] = value
		}
		return model.Value{Kind: model.ValueObject, Object: object}, nil
	default:
		return model.Value{}, fmt.Errorf("Cosmos document has an unsupported value shape %T", raw)
	}
}

// resolveRequiredString resolves pointer against document and requires the
// result to be a non-empty, valid-UTF-8 JSON string, as required for vertex
// and edge identities and endpoint field mappings. The error messages never
// include the resolved value or the document itself.
func resolveRequiredString(document any, target pointer, what string) (string, error) {
	value, ok := target.resolve(document)
	if !ok {
		return "", fmt.Errorf("Cosmos document is missing %s", what)
	}
	text, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("Cosmos document %s must be a JSON string", what)
	}
	if text == "" {
		return "", fmt.Errorf("Cosmos document %s must not be empty", what)
	}
	if !utf8.ValidString(text) {
		return "", fmt.Errorf("Cosmos document %s is not valid UTF-8", what)
	}
	return text, nil
}
