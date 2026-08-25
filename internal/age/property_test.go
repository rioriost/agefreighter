package age

import (
	"errors"
	"math"
	"testing"
	"unicode/utf8"

	"github.com/rioriost/agefreighter/pkg/model"
)

func TestEncodePropertiesCanonicalMapping(t *testing.T) {
	properties := model.Properties{
		"string": {Kind: model.ValueString, String: "line\n\"quoted\""},
		"int":    {Kind: model.ValueInteger, Integer: -7},
		"float":  {Kind: model.ValueFloat, Float: 2},
		"bool":   {Kind: model.ValueBoolean, Boolean: true},
		"null":   {Kind: model.ValueNull},
		"list": {
			Kind: model.ValueList,
			List: []model.Value{
				{Kind: model.ValueInteger, Integer: 1},
				{Kind: model.ValueString, String: "two"},
			},
		},
		"object": {
			Kind: model.ValueObject,
			Object: map[string]model.Value{
				"nested": {Kind: model.ValueBoolean},
			},
		},
	}
	encoded, err := EncodeProperties(properties)
	if err != nil {
		t.Fatalf("EncodeProperties() error = %v", err)
	}
	const expected = `{"bool":true,"float":2.0,"int":-7,"list":[1,"two"],"null":null,"object":{"nested":false},"string":"line\n\"quoted\""}`
	if string(encoded) != expected {
		t.Fatalf("EncodeProperties() = %s\nwant %s", encoded, expected)
	}
}

func TestEncodePropertiesRejectsUnsupportedValues(t *testing.T) {
	tests := []model.Value{
		{Kind: model.ValueFloat, Float: math.NaN()},
		{Kind: model.ValueFloat, Float: math.Inf(1)},
		{Kind: model.ValueKind(99)},
		{Kind: model.ValueString, String: string([]byte{0xff})},
	}
	for _, value := range tests {
		if _, err := EncodeProperties(model.Properties{"bad": value}); !errors.Is(err, ErrInvalidProperty) {
			t.Errorf("EncodeProperties(%#v) error = %v", value, err)
		}
	}

	deep := model.Value{Kind: model.ValueNull}
	for range maxPropertyDepth + 2 {
		deep = model.Value{
			Kind: model.ValueList,
			List: []model.Value{deep},
		}
	}
	if _, err := EncodeProperties(model.Properties{"deep": deep}); !errors.Is(err, ErrInvalidProperty) {
		t.Fatalf("deep EncodeProperties() error = %v", err)
	}
	if _, err := EncodeProperties(model.Properties{
		string([]byte{0xff}): {Kind: model.ValueNull},
	}); !errors.Is(err, ErrInvalidProperty) {
		t.Fatalf("invalid key EncodeProperties() error = %v", err)
	}
}

func TestEncodeEmptyProperties(t *testing.T) {
	encoded, err := EncodeProperties(nil)
	if err != nil || string(encoded) != "{}" {
		t.Fatalf("EncodeProperties(nil) = %q, %v", encoded, err)
	}
}

func FuzzEncodeStringProperty(f *testing.F) {
	f.Add("plain")
	f.Add("tabs\tnewlines\nquotes\"slashes\\")
	f.Fuzz(func(t *testing.T, value string) {
		_, err := EncodeProperties(model.Properties{
			"value": {Kind: model.ValueString, String: value},
		})
		if !utf8.ValidString(value) {
			if !errors.Is(err, ErrInvalidProperty) {
				t.Fatalf("invalid UTF-8 error = %v", err)
			}
			return
		}
		if err != nil {
			t.Fatalf("EncodeProperties() error = %v", err)
		}
	})
}
