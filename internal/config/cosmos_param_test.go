package config

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"go.yaml.in/yaml/v3"
)

func decodeYAMLParam(t *testing.T, text string) (CosmosParamValue, error) {
	t.Helper()
	var value CosmosParamValue
	err := yaml.Unmarshal([]byte(text), &value)
	return value, err
}

func TestCosmosParamValueYAMLScalarKinds(t *testing.T) {
	tests := []struct {
		name string
		text string
		want any
	}{
		{"null", "null", nil},
		{"bool", "true", true},
		{"int", "42", int64(42)},
		{"negative int", "-9223372036854775808", int64(math.MinInt64)},
		{"max int", "9223372036854775807", int64(math.MaxInt64)},
		{"hex int", "0x1A", int64(26)},
		{"float", "2.5", 2.5},
		{"string", "hello", "hello"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			value, err := decodeYAMLParam(t, test.text)
			if err != nil {
				t.Fatalf("Unmarshal(%q) error = %v", test.text, err)
			}
			if value.Native() != test.want {
				t.Fatalf("Unmarshal(%q) = %#v, want %#v", test.text, value.Native(), test.want)
			}
		})
	}
}

func TestCosmosParamValueYAMLRejectsNonFiniteAndUnsupportedShapes(t *testing.T) {
	tests := []string{
		".inf",
		"-.inf",
		".nan",
		"2018-01-01",               // timestamp tag, unsupported shape
		"!!binary |\n  ZGF0YQ==\n", // binary tag, unsupported shape
	}
	for _, text := range tests {
		if _, err := decodeYAMLParam(t, text); err == nil {
			t.Errorf("Unmarshal(%q) error = nil, want rejection", text)
		}
	}
}

func TestCosmosParamValueYAMLNestedShapes(t *testing.T) {
	value, err := decodeYAMLParam(t, "- 1\n- two\n- true\n- null\n")
	if err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	list, ok := value.Native().([]any)
	if !ok || len(list) != 4 || list[0] != int64(1) || list[1] != "two" || list[2] != true || list[3] != nil {
		t.Fatalf("Unmarshal() list = %#v", value.Native())
	}

	value, err = decodeYAMLParam(t, "name: Ada\nage: 36\n")
	if err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	object, ok := value.Native().(map[string]any)
	if !ok || object["name"] != "Ada" || object["age"] != int64(36) {
		t.Fatalf("Unmarshal() object = %#v", value.Native())
	}
}

func TestCosmosParamValueYAMLRejectsNonStringKeys(t *testing.T) {
	if _, err := decodeYAMLParam(t, "1: a\n2: b\n"); err == nil {
		t.Fatal("Unmarshal() error = nil, want non-string key rejection")
	}
}

func TestCosmosParamValueJSONRoundTrip(t *testing.T) {
	tests := []string{
		`null`, `true`, `false`, `42`, `-7`, `2.5`, `"text"`,
		`[1,"two",true,null]`, `{"a":1,"b":{"c":2}}`,
	}
	for _, text := range tests {
		var value CosmosParamValue
		if err := json.Unmarshal([]byte(text), &value); err != nil {
			t.Fatalf("UnmarshalJSON(%q) error = %v", text, err)
		}
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatalf("MarshalJSON(%q) error = %v", text, err)
		}
		var reDecoded any
		if err := json.Unmarshal(encoded, &reDecoded); err != nil {
			t.Fatalf("re-decode(%q) error = %v", encoded, err)
		}
	}
}

func TestCosmosParamValueJSONPreservesIntegerExactly(t *testing.T) {
	var value CosmosParamValue
	if err := json.Unmarshal([]byte(`9223372036854775807`), &value); err != nil {
		t.Fatalf("UnmarshalJSON() error = %v", err)
	}
	if value.Native() != int64(math.MaxInt64) {
		t.Fatalf("Native() = %#v, want MaxInt64", value.Native())
	}
}

func TestCosmosParamValueJSONRejectsOverflowAndNonFinite(t *testing.T) {
	tests := []string{
		`99999999999999999999999999999999`,
		`1e400`,
	}
	for _, text := range tests {
		var value CosmosParamValue
		if err := json.Unmarshal([]byte(text), &value); err == nil {
			t.Errorf("UnmarshalJSON(%q) error = nil, want rejection", text)
		}
	}
}

func TestCosmosParamValueJSONRejectsTrailingContent(t *testing.T) {
	var value CosmosParamValue
	err := json.Unmarshal([]byte(`1 2`), &value)
	if err == nil {
		t.Fatal("UnmarshalJSON() error = nil, want trailing content rejection")
	}
}

func TestCosmosParamValueJSONRejectsUnsupportedTopLevelType(t *testing.T) {
	if _, err := normalizeJSONParamValue(make(chan int), 0); err == nil {
		t.Fatal("normalizeJSONParamValue() error = nil, want unsupported type rejection")
	}
}

func TestNewCosmosParamValueNativeShapes(t *testing.T) {
	for _, native := range []any{
		nil,
		true,
		"person",
		7,
		int64(8),
		2.5,
		[]any{1, "two", false},
		map[string]any{"nested": []any{int64(1)}},
	} {
		value, err := NewCosmosParamValue(native)
		if err != nil {
			t.Errorf("NewCosmosParamValue(%#v) error = %v", native, err)
			continue
		}
		if _, err := value.MarshalYAML(); err != nil {
			t.Errorf("MarshalYAML(%#v) error = %v", native, err)
		}
	}
	for _, native := range []any{
		math.NaN(),
		make(chan int),
		[]any{make(chan int)},
		map[string]any{"bad": make(chan int)},
	} {
		if _, err := NewCosmosParamValue(native); err == nil {
			t.Errorf("NewCosmosParamValue(%T) accepted invalid value", native)
		}
	}

	var nested any = int64(1)
	for range maxCosmosParamDepth + 2 {
		nested = []any{nested}
	}
	if _, err := NewCosmosParamValue(nested); err == nil {
		t.Fatal("NewCosmosParamValue accepted excessive nesting")
	}
}

func TestCosmosParamValueDepthLimit(t *testing.T) {
	var builder strings.Builder
	depth := maxCosmosParamDepth + 5
	for range depth {
		builder.WriteString("[")
	}
	builder.WriteString("1")
	for range depth {
		builder.WriteString("]")
	}
	var value CosmosParamValue
	if err := json.Unmarshal([]byte(builder.String()), &value); err == nil {
		t.Fatal("UnmarshalJSON() error = nil, want depth-limit rejection")
	}

	yamlBuilder := strings.Builder{}
	for range depth {
		yamlBuilder.WriteString("[")
	}
	yamlBuilder.WriteString("1")
	for range depth {
		yamlBuilder.WriteString("]")
	}
	if _, err := decodeYAMLParam(t, yamlBuilder.String()); err == nil {
		t.Fatal("yaml.Unmarshal() error = nil, want depth-limit rejection")
	}
}
