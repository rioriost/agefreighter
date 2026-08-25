package age

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

	"github.com/rioriost/agefreighter/pkg/model"
)

const maxPropertyDepth = 100

var ErrInvalidProperty = errors.New("invalid property value")

func EncodeProperties(properties model.Properties) ([]byte, error) {
	var output bytes.Buffer
	if err := encodeObject(&output, properties, 0); err != nil {
		return nil, err
	}
	return output.Bytes(), nil
}

func encodeValue(output *bytes.Buffer, value model.Value, depth int) error {
	if depth > maxPropertyDepth {
		return fmt.Errorf("%w: nesting exceeds %d", ErrInvalidProperty, maxPropertyDepth)
	}
	switch value.Kind {
	case model.ValueNull:
		output.WriteString("null")
	case model.ValueBoolean:
		output.WriteString(strconv.FormatBool(value.Boolean))
	case model.ValueInteger:
		output.WriteString(strconv.FormatInt(value.Integer, 10))
	case model.ValueFloat:
		if math.IsNaN(value.Float) || math.IsInf(value.Float, 0) {
			return fmt.Errorf("%w: non-finite float", ErrInvalidProperty)
		}
		text := strconv.FormatFloat(value.Float, 'g', -1, 64)
		if !strings.ContainsAny(text, ".eE") {
			text += ".0"
		}
		output.WriteString(text)
	case model.ValueString:
		if !utf8.ValidString(value.String) {
			return fmt.Errorf("%w: string is not valid UTF-8", ErrInvalidProperty)
		}
		encoded, err := json.Marshal(value.String)
		if err != nil {
			return fmt.Errorf("%w: encode string: %v", ErrInvalidProperty, err)
		}
		output.Write(encoded)
	case model.ValueList:
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
	case model.ValueObject:
		return encodeObject(output, value.Object, depth+1)
	default:
		return fmt.Errorf("%w: unknown value kind %d", ErrInvalidProperty, value.Kind)
	}
	return nil
}

func encodeObject(output *bytes.Buffer, object map[string]model.Value, depth int) error {
	if depth > maxPropertyDepth {
		return fmt.Errorf("%w: nesting exceeds %d", ErrInvalidProperty, maxPropertyDepth)
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
			return fmt.Errorf("%w: property name is not valid UTF-8", ErrInvalidProperty)
		}
		encodedKey, err := json.Marshal(key)
		if err != nil {
			return fmt.Errorf("%w: encode property name: %v", ErrInvalidProperty, err)
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
