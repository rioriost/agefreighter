package model

import (
	"errors"
	"math"
	"testing"
	"unicode/utf8"
)

func TestEncodePropertiesCanonicalMapping(t *testing.T) {
	properties := Properties{
		"string": {Kind: ValueString, String: "line\n\"quoted\""},
		"int":    {Kind: ValueInteger, Integer: -7},
		"float":  {Kind: ValueFloat, Float: 2},
		"bool":   {Kind: ValueBoolean, Boolean: true},
		"null":   {Kind: ValueNull},
		"list": {
			Kind: ValueList,
			List: []Value{
				{Kind: ValueInteger, Integer: 1},
				{Kind: ValueString, String: "two"},
			},
		},
		"object": {
			Kind: ValueObject,
			Object: map[string]Value{
				"nested": {Kind: ValueBoolean},
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
	tests := []Value{
		{Kind: ValueFloat, Float: math.NaN()},
		{Kind: ValueFloat, Float: math.Inf(1)},
		{Kind: ValueKind(99)},
		{Kind: ValueString, String: string([]byte{0xff})},
	}
	for _, value := range tests {
		if _, err := EncodeProperties(Properties{"bad": value}); !errors.Is(err, ErrInvalidValue) {
			t.Errorf("EncodeProperties(%#v) error = %v", value, err)
		}
	}

	deep := Value{Kind: ValueNull}
	for range MaxPropertyDepth + 2 {
		deep = Value{Kind: ValueList, List: []Value{deep}}
	}
	if _, err := EncodeProperties(Properties{"deep": deep}); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("deep EncodeProperties() error = %v", err)
	}

	deepObject := Value{Kind: ValueNull}
	for range MaxPropertyDepth + 2 {
		deepObject = Value{Kind: ValueObject, Object: map[string]Value{"next": deepObject}}
	}
	if _, err := EncodeProperties(
		Properties{"deep": deepObject},
	); !errors.Is(err, ErrInvalidValue) {
		t.Fatalf("deep object EncodeProperties() error = %v", err)
	}
	if _, err := EncodeProperties(Properties{
		string([]byte{0xff}): {Kind: ValueNull},
	}); !errors.Is(err, ErrInvalidValue) {
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
		_, err := EncodeProperties(Properties{
			"value": {Kind: ValueString, String: value},
		})
		if !utf8.ValidString(value) {
			if !errors.Is(err, ErrInvalidValue) {
				t.Fatalf("invalid UTF-8 error = %v", err)
			}
			return
		}
		if err != nil {
			t.Fatalf("EncodeProperties() error = %v", err)
		}
	})
}
