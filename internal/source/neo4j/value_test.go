package neo4j

import (
	"math"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	neotypes "github.com/neo4j/neo4j-go-driver/v6/neo4j/dbtype"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

func TestConvertValueScalarAndCompositeTypes(t *testing.T) {
	now := time.Date(2026, 2, 3, 4, 5, 6, 700, time.FixedZone("offset", 3600))
	tests := []struct {
		name string
		raw  any
		kind model.ValueKind
	}{
		{"null", nil, model.ValueNull},
		{"bool", true, model.ValueBoolean},
		{"int64", int64(-4), model.ValueInteger},
		{"int", int(3), model.ValueInteger},
		{"int8", int8(3), model.ValueInteger},
		{"int16", int16(3), model.ValueInteger},
		{"int32", int32(3), model.ValueInteger},
		{"float64", 1.25, model.ValueFloat},
		{"string", "hello", model.ValueString},
		{"list", []any{int64(1), "two"}, model.ValueList},
		{"object", map[string]any{"one": int64(1)}, model.ValueObject},
		{"time", now, model.ValueString},
		{"date", neotypes.Date(now), model.ValueString},
		{"time-zone", neotypes.Time(now), model.ValueString},
		{"local-datetime", neotypes.LocalDateTime(now), model.ValueString},
		{"local-time", neotypes.LocalTime(now), model.ValueString},
		{"duration", neotypes.Duration{Months: 1, Days: 2, Seconds: 3, Nanos: 4}, model.ValueObject},
		{"point2d", neotypes.Point2D{SpatialRefId: 4326, X: 1, Y: 2}, model.ValueObject},
		{"point3d", neotypes.Point3D{SpatialRefId: 4979, X: 1, Y: 2, Z: 3}, model.ValueObject},
		{"node", neotypes.Node{Labels: []string{"A"}, Props: map[string]any{"x": int64(1)}}, model.ValueObject},
		{"relationship", neotypes.Relationship{Type: "SECRET", Props: map[string]any{"x": int64(1)}}, model.ValueObject},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			value, err := convertValue(test.raw, 0, config.Neo4jMultiLabelConfigured)
			if err != nil {
				t.Fatal(err)
			}
			if value.Kind != test.kind {
				t.Fatalf("kind = %v", value.Kind)
			}
		})
	}
	temporal, _ := convertValue(now, 0, config.Neo4jMultiLabelConfigured)
	if temporal.String != "2026-02-03T04:05:06.0000007+01:00" {
		t.Fatalf("time = %q", temporal.String)
	}
	duration, _ := convertValue(
		neotypes.Duration{Months: 1, Days: 2, Seconds: 3, Nanos: 4},
		0, config.Neo4jMultiLabelConfigured,
	)
	if duration.Object["nanoseconds"].Integer != 4 || len(duration.Object) != 4 {
		t.Fatalf("duration = %#v", duration)
	}
	node, _ := convertValue(neotypes.Node{
		Id: 999, ElementId: "internal", Labels: []string{"Secret"},
		Props: map[string]any{"safe": "yes"},
	}, 0, config.Neo4jMultiLabelConfigured)
	if _, leaked := node.Object["id"]; leaked || node.Object["safe"].String != "yes" {
		t.Fatalf("node leaked metadata: %#v", node)
	}
	relationship, _ := convertValue(neotypes.Relationship{
		Id: 999, Type: "INTERNAL", Props: map[string]any{"safe": "yes"},
	}, 0, config.Neo4jMultiLabelConfigured)
	if _, leaked := relationship.Object["type"]; leaked {
		t.Fatalf("relationship leaked metadata: %#v", relationship)
	}
}

func TestConvertValueRejectsUnsupportedAndMalformed(t *testing.T) {
	invalidUTF8 := string([]byte{0xff})
	tests := []struct {
		name string
		raw  any
	}{
		{"nan", math.NaN()},
		{"infinity", math.Inf(1)},
		{"float32", float32(1.25)},
		{"invalid string", invalidUTF8},
		{"bytes", []byte("bytes")},
		{"path", neotypes.Path{}},
		{"unsigned", uint64(1)},
		{"bad point", neotypes.Point2D{X: math.NaN()}},
		{"bad point3", neotypes.Point3D{Z: math.Inf(-1)}},
		{"bad object name", map[string]any{invalidUTF8: true}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := convertValue(
				test.raw, 0, config.Neo4jMultiLabelConfigured,
			); err == nil {
				t.Fatal("expected conversion error")
			}
		})
	}
	deep := any("leaf")
	for range model.MaxPropertyDepth + 2 {
		deep = []any{deep}
	}
	if _, err := convertValue(deep, 0, config.Neo4jMultiLabelConfigured); err == nil {
		t.Fatal("expected nesting error")
	}
	if utf8.ValidString(invalidUTF8) {
		t.Fatal("test setup produced valid UTF-8")
	}
}

func TestMultiLabelPolicy(t *testing.T) {
	node := neotypes.Node{
		Labels: []string{"One", "Two"}, Props: map[string]any{"name": "configured"},
	}
	value, err := convertValue(node, 0, config.Neo4jMultiLabelConfigured)
	if err != nil || value.Object["name"].String != "configured" {
		t.Fatalf("configured = %#v, %v", value, err)
	}
	if _, err := convertValue(node, 0, config.Neo4jMultiLabelReject); err == nil {
		t.Fatal("reject policy accepted a multi-label node")
	}
}

func TestExternalIDsAndKeys(t *testing.T) {
	for _, value := range []any{"text", int64(-2), int(3), int8(4), int16(5), int32(6), 1.25} {
		got, err := resolveExternalID(record(map[string]any{"id": value}, "id"), "id", "idField")
		if err != nil || got == "" {
			t.Fatalf("external ID for %#v = %q, %v", value, got, err)
		}
	}
	for _, value := range []any{"", nil, math.NaN(), math.Inf(1), float32(1), uint64(1), []byte("x")} {
		if _, err := resolveExternalID(
			record(map[string]any{"id": value}, "id"), "id", "idField",
		); err == nil {
			t.Fatalf("accepted external ID %#v", value)
		}
	}
	if _, err := resolveExternalID(record(map[string]any{}, "other"), "id", "idField"); err == nil {
		t.Fatal("accepted missing external ID")
	}
	for _, value := range []any{int64(1), int(2), int8(3), int16(4), int32(5)} {
		if _, err := extractKey(record(map[string]any{"key": value}, "key"), "key"); err != nil {
			t.Fatalf("key %#v: %v", value, err)
		}
	}
	for _, value := range []any{nil, 1.0, uint64(1), "1"} {
		if _, err := extractKey(record(map[string]any{"key": value}, "key"), "key"); err == nil {
			t.Fatalf("accepted key %#v", value)
		}
	}
	if _, err := extractKey(record(map[string]any{}, "other"), "key"); err == nil {
		t.Fatal("accepted missing key")
	}
}

func TestBoundedRecordEstimation(t *testing.T) {
	value := strings.Repeat("x", 1000)
	size, err := estimateRecordSize(record(map[string]any{
		"a": []any{map[string]any{"b": value}},
	}, "a"), 100)
	if err != nil || size <= 100 {
		t.Fatalf("estimate = %d, %v", size, err)
	}
	if _, err := estimateRecordSize(record(
		map[string]any{"a": 1}, "a", "a",
	), 1000); err == nil {
		t.Fatal("accepted duplicate columns")
	}
	if _, err := estimateRecordSize(fakeRecord{
		keys: []string{"a"}, values: map[string]any{},
	}, 1000); err == nil {
		t.Fatal("accepted inconsistent columns")
	}
	path := neotypes.Path{
		Nodes:         []neotypes.Node{{Props: map[string]any{"x": value}}},
		Relationships: []neotypes.Relationship{{Props: map[string]any{"y": value}}},
	}
	if size := estimateRawSize(path, 0, 100); size <= 100 {
		t.Fatalf("path estimate = %d", size)
	}
}
