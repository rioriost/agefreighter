package pggraph

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/meta"
)

func TestLogicalRecordDigestCanonicalizesPropertyOrder(t *testing.T) {
	leftRange, left, err := vertexRecordDigest(
		"Person", "neo4j", "p1", []byte(`{"b":[true,null],"a":{"n":1}}`))
	if err != nil {
		t.Fatal(err)
	}
	rightRange, right, err := vertexRecordDigest(
		"Person", "neo4j", "p1", []byte(` { "a" : { "n" : 1 }, "b" : [true, null] } `))
	if err != nil {
		t.Fatal(err)
	}
	if left != right || leftRange != rightRange || len(left) != 64 {
		t.Fatalf("canonical digests differ: %d/%s != %d/%s",
			leftRange, left, rightRange, right)
	}
	_, edge, err := edgeRecordDigest(
		"KNOWS", "neo4j", "e1", "Person", "neo4j", "p1",
		"Person", "neo4j", "p2", []byte(`{"a":1}`))
	if err != nil {
		t.Fatal(err)
	}
	if edge == left {
		t.Fatal("vertex and edge logical records have the same digest")
	}
}

func TestCanonicalJSONRejectsInvalidProperties(t *testing.T) {
	for _, value := range [][]byte{
		nil, []byte(`null`), []byte(`[]`), []byte(`{"a":1} trailing`),
		[]byte(`{"a":1}{"b":2}`),
	} {
		if _, err := canonicalJSON(value); err == nil {
			t.Fatalf("canonicalJSON(%q) succeeded", value)
		}
	}
	if _, _, err := logicalRecordDigest(canonicalRecord{Properties: []byte(`bad`)}); err == nil {
		t.Fatal("logicalRecordDigest() accepted invalid properties")
	}
}

func TestCanonicalJSONMatchesPostgreSQLNumericRendering(t *testing.T) {
	for input, want := range map[string]string{
		`{"v":1e0}`:            `{"v":1}`,
		`{"v":-1.25e1}`:        `{"v":-12.5}`,
		`{"v":-0.0}`:           `{"v":0.0}`,
		`{"v":1.2300e2}`:       `{"v":123.00}`,
		`{"v":1.2300e-2}`:      `{"v":0.012300}`,
		`{"v":[1e-3,0.1e2]}`:   `{"v":[0.001,10]}`,
		`{"v":{"nested":0e5}}`: `{"v":{"nested":0}}`,
	} {
		got, err := canonicalJSON([]byte(input))
		if err != nil || string(got) != want {
			t.Fatalf("canonicalJSON(%s) = %s, %v; want %s", input, got, err, want)
		}
	}
	for _, value := range []string{"1e", "1e999999", "1e-999999"} {
		if _, err := normalizePostgreSQLNumeric(value); err == nil {
			t.Fatalf("normalizePostgreSQLNumeric(%q) succeeded", value)
		}
	}
}

func TestNormalizeJSONValuePropagatesInvalidNestedNumbers(t *testing.T) {
	for name, value := range map[string]any{
		"number": json.Number("1e"),
		"array":  []any{json.Number("1e")},
		"object": map[string]any{"value": json.Number("1e")},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := normalizeJSONValue(value); err == nil {
				t.Fatal("normalizeJSONValue() accepted an invalid PostgreSQL numeric")
			}
		})
	}
}

func TestCompareDigestsDetectsEveryBaselineChange(t *testing.T) {
	rangeValue := meta.PropertyGraphDigestRange{
		JobID: "job", LabelName: "Person", Kind: meta.VertexLabel,
		RangeID: 9, Rows: 2, Digest: strings.Repeat("a", 64),
	}
	actual := DigestSet{Root: strings.Repeat("b", 64), Rows: 2,
		Ranges: []meta.PropertyGraphDigestRange{rangeValue}}
	if err := CompareDigests(actual.Root, actual.Rows,
		[]meta.PropertyGraphDigestRange{rangeValue}, actual); err != nil {
		t.Fatalf("CompareDigests(equal): %v", err)
	}
	tests := []struct {
		name     string
		root     string
		rows     int64
		expected []meta.PropertyGraphDigestRange
		actual   DigestSet
	}{
		{"missing root", "", 2, []meta.PropertyGraphDigestRange{rangeValue}, actual},
		{"root", strings.Repeat("c", 64), 2, []meta.PropertyGraphDigestRange{rangeValue}, actual},
		{"rows", actual.Root, 3, []meta.PropertyGraphDigestRange{rangeValue}, actual},
		{"range count", actual.Root, 2, nil, actual},
		{"range value", actual.Root, 2, []meta.PropertyGraphDigestRange{{
			JobID: "job", LabelName: "Person", Kind: meta.VertexLabel,
			RangeID: 9, Rows: 2, Digest: strings.Repeat("d", 64),
		}}, actual},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := CompareDigests(test.root, test.rows, test.expected, test.actual)
			if !errors.Is(err, ErrIntegrity) {
				t.Fatalf("CompareDigests() error = %v", err)
			}
		})
	}
}

func TestValidateStoredDigestRange(t *testing.T) {
	if err := validateStoredRange(42, 42); err != nil {
		t.Fatal(err)
	}
	for _, stored := range []int{-1, 41, DigestRangeCount} {
		if err := validateStoredRange(stored, 42); !errors.Is(err, ErrIntegrity) {
			t.Fatalf("validateStoredRange(%d) = %v", stored, err)
		}
	}
}

func TestCompareDigestRangesByLabel(t *testing.T) {
	left := meta.PropertyGraphDigestRange{Kind: meta.VertexLabel, LabelName: "A"}
	right := meta.PropertyGraphDigestRange{Kind: meta.VertexLabel, LabelName: "B"}
	if compareDigestRanges(left, right) >= 0 {
		t.Fatal("compareDigestRanges() did not order labels")
	}
}

func TestCompareDigestRangesByKindAndRange(t *testing.T) {
	vertex := meta.PropertyGraphDigestRange{
		Kind: meta.VertexLabel, LabelName: "Person", RangeID: 2,
	}
	edge := meta.PropertyGraphDigestRange{
		Kind: meta.EdgeLabel, LabelName: "Person", RangeID: 1,
	}
	if compareDigestRanges(vertex, edge) >= 0 {
		t.Fatal("compareDigestRanges() did not order kinds")
	}
	if compareDigestRanges(vertex, meta.PropertyGraphDigestRange{
		Kind: meta.VertexLabel, LabelName: "Person", RangeID: 3,
	}) >= 0 {
		t.Fatal("compareDigestRanges() did not order ranges")
	}
}
