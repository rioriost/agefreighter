package model

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

// MaxPropertyDepth bounds recursion when encoding nested properties. It is
// exported so other packages that build on Value/Properties (for example
// query-parameter or source-document decoders) can enforce the same nesting
// limit as the canonical property encoder below.
const MaxPropertyDepth = 100

// ErrInvalidValue is returned when a Value cannot be canonically encoded,
// for example due to non-finite floats, invalid UTF-8, unknown kinds, or
// excessive nesting.
var ErrInvalidValue = errors.New("invalid property value")

// EncodeProperties canonically encodes properties as a JSON object with
// lexicographically sorted keys. The encoding is deterministic so it can be
// used as a fast-path payload for downstream consumers (for example the AGE
// COPY loader) without re-deriving it from Value trees.
func EncodeProperties(properties Properties) ([]byte, error) {
	var output bytes.Buffer
	if err := encodeObject(&output, properties, 0); err != nil {
		return nil, err
	}
	return output.Bytes(), nil
}

func encodeValue(output *bytes.Buffer, value Value, depth int) error {
	if depth > MaxPropertyDepth {
		return fmt.Errorf("%w: nesting exceeds %d", ErrInvalidValue, MaxPropertyDepth)
	}
	switch value.Kind {
	case ValueNull:
		output.WriteString("null")
	case ValueBoolean:
		output.WriteString(strconv.FormatBool(value.Boolean))
	case ValueInteger:
		output.WriteString(strconv.FormatInt(value.Integer, 10))
	case ValueFloat:
		if math.IsNaN(value.Float) || math.IsInf(value.Float, 0) {
			return fmt.Errorf("%w: non-finite float", ErrInvalidValue)
		}
		text := strconv.FormatFloat(value.Float, 'g', -1, 64)
		if !strings.ContainsAny(text, ".eE") {
			text += ".0"
		}
		output.WriteString(text)
	case ValueString:
		if !utf8.ValidString(value.String) {
			return fmt.Errorf("%w: string is not valid UTF-8", ErrInvalidValue)
		}
		encoded, err := json.Marshal(value.String)
		if err != nil {
			return fmt.Errorf("%w: encode string: %v", ErrInvalidValue, err)
		}
		output.Write(encoded)
	case ValueList:
		output.WriteByte('[')
		for index, item := range value.List {
			if index > 0 {
				output.WriteByte(',')
			}
			if err := encodeValue(output, item, depth+1); err != nil {
				return err
			}
		}
		output.WriteByte(']')
	case ValueObject:
		return encodeObject(output, value.Object, depth+1)
	default:
		return fmt.Errorf("%w: unknown value kind %d", ErrInvalidValue, value.Kind)
	}
	return nil
}

func encodeObject(output *bytes.Buffer, object map[string]Value, depth int) error {
	if depth > MaxPropertyDepth {
		return fmt.Errorf("%w: nesting exceeds %d", ErrInvalidValue, MaxPropertyDepth)
	}
	keys := make([]string, 0, len(object))
	for key := range object {
		keys = append(keys, key)
	}
	slices.Sort(keys)

	output.WriteByte('{')
	for index, key := range keys {
		if index > 0 {
			output.WriteByte(',')
		}
		if !utf8.ValidString(key) {
			return fmt.Errorf("%w: property name is not valid UTF-8", ErrInvalidValue)
		}
		encodedKey, err := json.Marshal(key)
		if err != nil {
			return fmt.Errorf("%w: encode property name: %v", ErrInvalidValue, err)
		}
		output.Write(encodedKey)
		output.WriteByte(':')
		if err := encodeValue(output, object[key], depth+1); err != nil {
			return fmt.Errorf("property %q: %w", key, err)
		}
	}
	output.WriteByte('}')
	return nil
}
