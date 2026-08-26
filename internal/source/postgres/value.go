package postgres

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/rioriost/agefreighter/pkg/model"
)

const (
	keyNumber = "number"
)

type keyValue struct {
	kind   string
	text   string
	native any
}

func decodeObject(raw []byte) (map[string]any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var document map[string]any
	if err := decoder.Decode(&document); err != nil {
		return nil, fmt.Errorf("decode PostgreSQL row: %w", err)
	}
	if document == nil {
		return nil, errors.New("PostgreSQL row must be a JSON object")
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return nil, errors.New("PostgreSQL row has trailing content")
	}
	return document, nil
}

func convertValue(raw any, depth int) (model.Value, error) {
	if depth > model.MaxPropertyDepth {
		return model.Value{}, fmt.Errorf(
			"PostgreSQL value nesting exceeds %d", model.MaxPropertyDepth,
		)
	}
	switch value := raw.(type) {
	case nil:
		return model.Value{Kind: model.ValueNull}, nil
	case bool:
		return model.Value{Kind: model.ValueBoolean, Boolean: value}, nil
	case string:
		if !utf8.ValidString(value) {
			return model.Value{}, errors.New("PostgreSQL string is not valid UTF-8")
		}
		return model.Value{Kind: model.ValueString, String: value}, nil
	case json.Number:
		text := value.String()
		if !strings.ContainsAny(text, ".eE") {
			integer, err := strconv.ParseInt(text, 10, 64)
			if err != nil {
				return model.Value{}, errors.New("PostgreSQL integer overflows int64")
			}
			return model.Value{Kind: model.ValueInteger, Integer: integer}, nil
		}
		floating, err := strconv.ParseFloat(text, 64)
		if err != nil || math.IsNaN(floating) || math.IsInf(floating, 0) {
			return model.Value{}, errors.New("PostgreSQL number must be finite")
		}
		return model.Value{Kind: model.ValueFloat, Float: floating}, nil
	case []any:
		list := make([]model.Value, len(value))
		for index, item := range value {
			converted, err := convertValue(item, depth+1)
			if err != nil {
				return model.Value{}, err
			}
			list[index] = converted
		}
		return model.Value{Kind: model.ValueList, List: list}, nil
	case map[string]any:
		object := make(map[string]model.Value, len(value))
		for name, item := range value {
			if !utf8.ValidString(name) {
				return model.Value{}, errors.New("PostgreSQL object name is not valid UTF-8")
			}
			converted, err := convertValue(item, depth+1)
			if err != nil {
				return model.Value{}, err
			}
			object[name] = converted
		}
		return model.Value{Kind: model.ValueObject, Object: object}, nil
	default:
		return model.Value{}, fmt.Errorf("PostgreSQL value has unsupported shape %T", raw)
	}
}

func resolveExternalID(document map[string]any, field, what string) (model.ExternalID, error) {
	raw, ok := document[field]
	if !ok {
		return "", fmt.Errorf("PostgreSQL row is missing %s", what)
	}
	switch value := raw.(type) {
	case string:
		if value == "" {
			return "", fmt.Errorf("PostgreSQL row %s must not be empty", what)
		}
		if !utf8.ValidString(value) {
			return "", fmt.Errorf("PostgreSQL row %s is not valid UTF-8", what)
		}
		return model.ExternalID(value), nil
	case json.Number:
		text := value.String()
		if strings.ContainsAny(text, ".eE") {
			number, err := strconv.ParseFloat(text, 64)
			if err != nil || math.IsNaN(number) || math.IsInf(number, 0) {
				return "", fmt.Errorf("PostgreSQL row %s must be finite", what)
			}
		} else if _, ok := new(big.Int).SetString(text, 10); !ok {
			return "", fmt.Errorf("PostgreSQL row %s integer is invalid", what)
		}
		return model.ExternalID(text), nil
	case nil:
		return "", fmt.Errorf("PostgreSQL row %s must not be null", what)
	default:
		return "", fmt.Errorf("PostgreSQL row %s must be a string or number", what)
	}
}

func extractKey(raw []byte, field string) (keyValue, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	var document map[string]json.RawMessage
	if err := decoder.Decode(&document); err != nil {
		return keyValue{}, errors.New("decode PostgreSQL keyset row")
	}
	encoded, ok := document[field]
	if !ok {
		return keyValue{}, errors.New("PostgreSQL keyset row is missing keyField")
	}
	valueDecoder := json.NewDecoder(bytes.NewReader(encoded))
	valueDecoder.UseNumber()
	var value any
	if err := valueDecoder.Decode(&value); err != nil {
		return keyValue{}, errors.New("decode PostgreSQL keyset key")
	}
	switch typed := value.(type) {
	case json.Number:
		return parseNumberKey(typed.String())
	case nil:
		return keyValue{}, errors.New("PostgreSQL keyset key must not be null")
	default:
		return keyValue{}, errors.New(
			"PostgreSQL keyset key must be a signed 64-bit integer",
		)
	}
}

func parseNumberKey(text string) (keyValue, error) {
	if len(text) > maxResumeKeyBytes {
		return keyValue{}, errors.New("PostgreSQL keyset numeric key is too large")
	}
	if strings.ContainsAny(text, ".eE") {
		return keyValue{}, errors.New(
			"PostgreSQL keyset key must be a signed 64-bit integer",
		)
	}
	integer, err := strconv.ParseInt(text, 10, 64)
	if err != nil {
		return keyValue{}, errors.New("PostgreSQL keyset integer key overflows int64")
	}
	return keyValue{
		kind: keyNumber, text: text, native: integer,
	}, nil
}

func compareKeys(previous, current keyValue) (int, error) {
	if previous.kind != current.kind {
		return 0, errors.New("PostgreSQL keyset key type changed between rows")
	}
	if previous.kind != keyNumber {
		return 0, errors.New("PostgreSQL keyset key type is invalid")
	}
	left := previous.native.(int64)
	right := current.native.(int64)
	switch {
	case left < right:
		return -1, nil
	case left > right:
		return 1, nil
	default:
		return 0, nil
	}
}
