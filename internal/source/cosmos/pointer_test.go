package cosmos

import "testing"

func TestParsePointerRejectsInvalidSyntax(t *testing.T) {
	cases := []string{"", "id", "/a~", "/a~2", "/a~"}
	for _, raw := range cases {
		if _, err := parsePointer(raw); err == nil {
			t.Errorf("parsePointer(%q): expected error, got nil", raw)
		}
	}
}

func TestParsePointerAcceptsRootAndEscapes(t *testing.T) {
	cases := []string{"/", "/a", "/a/b", "/a~0b", "/a~1b", "/0", "/a~01"}
	for _, raw := range cases {
		if _, err := parsePointer(raw); err != nil {
			t.Errorf("parsePointer(%q): unexpected error: %v", raw, err)
		}
	}
}

func TestPointerResolveObjectAndArray(t *testing.T) {
	document := map[string]any{
		"id": "vertex-1",
		"nested": map[string]any{
			"deep": []any{"zero", "one", map[string]any{"leaf": 42.0}},
		},
		"a/b": "slash-key",
		"a~b": "tilde-key",
	}

	cases := []struct {
		raw      string
		expected any
	}{
		{"/id", "vertex-1"},
		{"/nested/deep/0", "zero"},
		{"/nested/deep/1", "one"},
		{"/nested/deep/2/leaf", 42.0},
		{"/a~1b", "slash-key"},
		{"/a~0b", "tilde-key"},
	}
	for _, testCase := range cases {
		ptr, err := parsePointer(testCase.raw)
		if err != nil {
			t.Fatalf("parsePointer(%q): %v", testCase.raw, err)
		}
		value, ok := ptr.resolve(document)
		if !ok {
			t.Fatalf("resolve(%q): expected value, got none", testCase.raw)
		}
		if value != testCase.expected {
			t.Errorf("resolve(%q) = %v, want %v", testCase.raw, value, testCase.expected)
		}
	}
}

func TestPointerResolveMissing(t *testing.T) {
	document := map[string]any{"id": "present"}
	cases := []string{"/missing", "/id/nested", "/nested/0"}
	for _, raw := range cases {
		ptr, err := parsePointer(raw)
		if err != nil {
			t.Fatalf("parsePointer(%q): %v", raw, err)
		}
		if _, ok := ptr.resolve(document); ok {
			t.Errorf("resolve(%q): expected missing, got a value", raw)
		}
	}
}

func TestPointerResolveArrayOutOfRange(t *testing.T) {
	document := map[string]any{"list": []any{"a", "b"}}
	ptr, err := parsePointer("/list/5")
	if err != nil {
		t.Fatalf("parsePointer: %v", err)
	}
	if _, ok := ptr.resolve(document); ok {
		t.Error("resolve: expected out-of-range array index to be missing")
	}
	ptr, err = parsePointer("/list/-1")
	if err != nil {
		t.Fatalf("parsePointer: %v", err)
	}
	if _, ok := ptr.resolve(document); ok {
		t.Error("resolve: expected negative array index to be missing")
	}
}
