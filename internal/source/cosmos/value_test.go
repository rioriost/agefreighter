package cosmos

import (
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/pkg/model"
)

func TestDecodeDocumentRejectsTrailingContent(t *testing.T) {
	_, err := decodeDocument([]byte(`{"a":1}{"b":2}`))
	if err == nil {
		t.Fatal("expected trailing content to be rejected")
	}
}

func TestDecodeDocumentAcceptsObject(t *testing.T) {
	document, err := decodeDocument([]byte(`{"a":1,"b":[1,2,3]}`))
	if err != nil {
		t.Fatalf("decodeDocument: %v", err)
	}
	object, ok := document.(map[string]any)
	if !ok {
		t.Fatalf("decodeDocument: got %T, want map[string]any", document)
	}
	if _, ok := object["a"]; !ok {
		t.Error("decodeDocument: missing key a")
	}
}

func TestConvertValueAllKinds(t *testing.T) {
	document, err := decodeDocument([]byte(`{
		"n": null,
		"b": true,
		"i": 42,
		"f": 1.5,
		"s": "hello",
		"l": [1, "two", false, null],
		"o": {"x": 1}
	}`))
	if err != nil {
		t.Fatalf("decodeDocument: %v", err)
	}
	object := document.(map[string]any)

	check := func(key string, kind model.ValueKind) model.Value {
		t.Helper()
		value, err := convertValue(object[key], 0)
		if err != nil {
			t.Fatalf("convertValue(%q): %v", key, err)
		}
		if value.Kind != kind {
			t.Fatalf("convertValue(%q).Kind = %v, want %v", key, value.Kind, kind)
		}
		return value
	}

	check("n", model.ValueNull)
	if v := check("b", model.ValueBoolean); !v.Boolean {
		t.Error("expected boolean true")
	}
	if v := check("i", model.ValueInteger); v.Integer != 42 {
		t.Errorf("integer = %d, want 42", v.Integer)
	}
	if v := check("f", model.ValueFloat); v.Float != 1.5 {
		t.Errorf("float = %v, want 1.5", v.Float)
	}
	if v := check("s", model.ValueString); v.String != "hello" {
		t.Errorf("string = %q, want hello", v.String)
	}
	if v := check("l", model.ValueList); len(v.List) != 4 {
		t.Errorf("list length = %d, want 4", len(v.List))
	}
	if v := check("o", model.ValueObject); v.Object["x"].Integer != 1 {
		t.Errorf("object.x = %v, want 1", v.Object["x"])
	}
}

func TestConvertValueInt64BoundariesExact(t *testing.T) {
	cases := []int64{math.MaxInt64, math.MinInt64, 0, -1, 1}
	for _, expected := range cases {
		raw := strconv.FormatInt(expected, 10)
		document, err := decodeDocument([]byte(`{"v":` + raw + `}`))
		if err != nil {
			t.Fatalf("decodeDocument(%s): %v", raw, err)
		}
		value, err := convertValue(document.(map[string]any)["v"], 0)
		if err != nil {
			t.Fatalf("convertValue(%s): %v", raw, err)
		}
		if value.Kind != model.ValueInteger || value.Integer != expected {
			t.Errorf("convertValue(%s) = %+v, want exact integer %d", raw, value, expected)
		}
	}
}

func TestConvertValueRejectsInt64Overflow(t *testing.T) {
	cases := []string{
		"9223372036854775808",  // MaxInt64 + 1
		"-9223372036854775809", // MinInt64 - 1
		"99999999999999999999999999999999999999",
	}
	for _, raw := range cases {
		document, err := decodeDocument([]byte(`{"v":` + raw + `}`))
		if err != nil {
			t.Fatalf("decodeDocument(%s): %v", raw, err)
		}
		if _, err := convertValue(document.(map[string]any)["v"], 0); err == nil {
			t.Errorf("convertValue(%s): expected overflow rejection", raw)
		}
	}
}

func TestConvertValueRejectsNonFiniteFloats(t *testing.T) {
	// encoding/json cannot itself decode NaN/Inf literals, so build the
	// json.Number values directly to exercise convertValue's own guard.
	document, err := decodeDocument([]byte(`{"v":1e400}`))
	if err != nil {
		t.Fatalf("decodeDocument: %v", err)
	}
	if _, err := convertValue(document.(map[string]any)["v"], 0); err == nil {
		t.Error("convertValue: expected overflowing float literal to be rejected as non-finite")
	}
}

func TestConvertValueRejectsUnsupportedShape(t *testing.T) {
	if _, err := convertValue(struct{}{}, 0); err == nil {
		t.Error("convertValue: expected unsupported shape to be rejected")
	}
}

func TestConvertValueRejectsInvalidUTF8String(t *testing.T) {
	if _, err := convertValue(string([]byte{0xff, 0xfe}), 0); err == nil {
		t.Error("convertValue: expected invalid UTF-8 string to be rejected")
	}
}

func TestConvertValueRejectsInvalidUTF8ObjectKey(t *testing.T) {
	object := map[string]any{string([]byte{0xff}): "value"}
	if _, err := convertValue(object, 0); err == nil {
		t.Error("convertValue: expected invalid UTF-8 object key to be rejected")
	}
}

func TestConvertValueRejectsExcessiveNesting(t *testing.T) {
	var raw strings.Builder
	for i := 0; i <= model.MaxPropertyDepth+2; i++ {
		raw.WriteByte('[')
	}
	raw.WriteString("1")
	for i := 0; i <= model.MaxPropertyDepth+2; i++ {
		raw.WriteByte(']')
	}
	document, err := decodeDocument([]byte(raw.String()))
	if err != nil {
		t.Fatalf("decodeDocument: %v", err)
	}
	if _, err := convertValue(document, 0); err == nil {
		t.Error("convertValue: expected excessive nesting to be rejected")
	}
}

func TestResolveRequiredStringRules(t *testing.T) {
	document := map[string]any{
		"id":     "abc",
		"empty":  "",
		"number": 1.0,
	}
	idPointer, _ := parsePointer("/id")
	if value, err := resolveRequiredString(document, idPointer, "idField"); err != nil || value != "abc" {
		t.Fatalf("resolveRequiredString(/id) = %q, %v", value, err)
	}

	emptyPointer, _ := parsePointer("/empty")
	if _, err := resolveRequiredString(document, emptyPointer, "idField"); err == nil {
		t.Error("resolveRequiredString: expected empty string to be rejected")
	}

	numberPointer, _ := parsePointer("/number")
	if _, err := resolveRequiredString(document, numberPointer, "idField"); err == nil {
		t.Error("resolveRequiredString: expected non-string value to be rejected")
	}

	missingPointer, _ := parsePointer("/missing")
	if _, err := resolveRequiredString(document, missingPointer, "idField"); err == nil {
		t.Error("resolveRequiredString: expected missing pointer to be rejected")
	}
}
