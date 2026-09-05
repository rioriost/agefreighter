package csv

import (
	"context"
	"encoding/csv"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

func TestTypedCSVPropertiesBothPaths(t *testing.T) {
	path := filepath.Join(t.TempDir(), "typed.csv")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	w := csv.NewWriter(f)
	w.WriteAll([][]string{{"id", "integer", "float", "bool", "strings", "integers", "floats", "bools", "nil", "text"},
		{"a", "9223372036854775807", "2", "true", `["日本語",""]`, `[1,-2]`, `[1,2.5]`, `[true,false]`, `\N`, "001"}})
	if err := w.Error(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	props := map[string]string{}
	for _, name := range []string{"integer", "float", "bool", "strings", "integers", "floats", "bools", "nil", "text"} {
		props[name] = name
	}
	types := map[string]string{"integer": "int64", "float": "float64", "bool": "boolean", "strings": "string[]", "integers": "int64[]", "floats": "float64[]", "bools": "boolean[]", "nil": "int64"}
	header, null := true, `\N`
	want := `{"bool":true,"bools":[true,false],"float":2.0,"floats":[1.0,2.5],"integer":9223372036854775807,"integers":[1,-2],"nil":null,"strings":["日本語",""],"text":"001"}`
	for _, preencode := range []bool{false, true} {
		iterator, err := NewIterator(context.Background(), IteratorOptions{Namespace: "p1", PreencodeProperties: preencode,
			Source: config.CSVSource{Defaults: config.DelimitedOptions{Header: &header, NullValue: &null}, Vertices: []config.CSVVertex{{Label: "V", Path: path, IDColumn: "id", Properties: props, PropertyTypes: types}}}})
		if err != nil {
			t.Fatal(err)
		}
		record, err := iterator.Next(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		got := record.Record.Vertex.EncodedProperties
		if !preencode {
			got, err = model.EncodeProperties(record.Record.Vertex.Properties)
			if err != nil {
				t.Fatal(err)
			}
		}
		if string(got) != want {
			t.Fatalf("preencode=%v: %s", preencode, got)
		}
		if err := iterator.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestTypedCSVRejectsInvalidValuesWithoutLeakingCell(t *testing.T) {
	for _, test := range []struct{ kind, value string }{
		{"int64", "9223372036854775808"}, {"float64", "NaN"}, {"float64", "Inf"},
		{"boolean", "1"}, {"boolean", "TRUE"}, {"int64[]", `[1,"2"]`},
		{"string[]", `[null]`}, {"string[]", `[1]`}, {"int64[]", `null`},
		{"boolean[]", `[true,null]`}, {"string[]", `["ok"] trailing`}, {"secret", "credential-secret"},
	} {
		if _, err := decodeCSVValue(test.value, test.kind); err == nil || strings.Contains(err.Error(), test.value) {
			t.Errorf("kind=%s: invalid or leaking error %v", test.kind, err)
		}
	}
}

func TestCSVTypeFingerprintAndValidation(t *testing.T) {
	a := fileMapping{properties: map[string]string{"p": "column"}}
	if err := bindFingerprint(&a); err != nil {
		t.Fatal(err)
	}
	legacy := string(a.fingerprintInput)
	if strings.Contains(legacy, "propertyTypes") {
		t.Fatal("omitted types changed legacy fingerprint shape")
	}
	a.propertyTypes = map[string]string{"p": "int64"}
	if err := bindFingerprint(&a); err != nil {
		t.Fatal(err)
	}
	if string(a.fingerprintInput) == legacy {
		t.Fatal("type change must invalidate resume")
	}
	a.propertyTypes = map[string]string{"missing": "string"}
	if err := bindFingerprint(&a); err == nil {
		t.Fatal("unmapped type accepted")
	}
	a.propertyTypes = map[string]string{"p": "guess"}
	if err := bindFingerprint(&a); err == nil {
		t.Fatal("unknown type accepted")
	}
}
