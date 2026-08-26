package postgres

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/csv"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

func TestBuildMappingsOrderAndProperties(t *testing.T) {
	source := config.PostgreSQLSource{
		ReadMode: config.PostgreSQLReadCursor, FetchRows: 10,
		Vertices: []config.VertexQuery{{
			Label: "Person", Query: "SELECT id, a, b FROM people ORDER BY id",
			IDField: "id", Properties: map[string]string{"z": "b", "a": "a"},
		}},
		Edges: []config.EdgeQuery{{
			Label: "KNOWS", Query: "SELECT source, target FROM knows ORDER BY source",
			Start: config.EndpointMapping{Label: "Person", Field: "source"},
			End:   config.EndpointMapping{Label: "Person", Field: "target"},
		}},
	}
	mappings, err := buildMappings(t.Context(), "crm", source, 10)
	if err != nil {
		t.Fatalf("buildMappings() error = %v", err)
	}
	if len(mappings) != 2 ||
		mappings[0].kind != vertexMapping ||
		mappings[1].kind != edgeMapping {
		t.Fatalf("mapping order = %#v", mappings)
	}
	if mappings[0].properties[0].name != "a" ||
		mappings[0].properties[1].name != "z" {
		t.Fatalf("properties are not sorted: %#v", mappings[0].properties)
	}
}

func TestBuildMappingsDefensiveValidation(t *testing.T) {
	base := config.PostgreSQLSource{
		ReadMode: config.PostgreSQLReadCursor, FetchRows: 10,
		Vertices: []config.VertexQuery{{
			Label: "Person", Query: "SELECT id FROM people ORDER BY id", IDField: "id",
		}},
	}
	tests := []struct {
		name string
		edit func(*config.PostgreSQLSource)
		want string
	}{
		{"empty query", func(source *config.PostgreSQLSource) {
			source.Vertices[0].Query = " "
		}, "query is required"},
		{"semicolon", func(source *config.PostgreSQLSource) {
			source.Vertices[0].Query = "SELECT 1;"
		}, "without a semicolon"},
		{"key outside keyset", func(source *config.PostgreSQLSource) {
			source.Vertices[0].KeyField = "id"
		}, "only valid in keyset"},
		{"missing keyset parameters", func(source *config.PostgreSQLSource) {
			source.ReadMode = config.PostgreSQLReadKeyset
			source.Vertices[0].KeyField = "id"
		}, "must use $1 and $2"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := base
			source.Vertices = append([]config.VertexQuery(nil), base.Vertices...)
			test.edit(&source)
			_, err := buildMappings(t.Context(), "crm", source, 10)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("buildMappings() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestResumeTokenRoundTripAndValidation(t *testing.T) {
	key, err := parseNumberKey("125")
	if err != nil {
		t.Fatal(err)
	}
	state := resumeState{
		fingerprint: strings.Repeat("ab", 32), mappingIndex: 2,
		mappingKind: edgeMapping, consumed: 17, rejected: 3, key: &key,
	}
	token, err := formatResumeToken(state)
	if err != nil {
		t.Fatalf("formatResumeToken() error = %v", err)
	}
	parsed, err := parseResumeToken(token)
	if err != nil {
		t.Fatalf("parseResumeToken() error = %v", err)
	}
	if parsed.mappingIndex != state.mappingIndex ||
		parsed.mappingKind != state.mappingKind ||
		parsed.consumed != state.consumed ||
		parsed.rejected != state.rejected ||
		parsed.key == nil || parsed.key.text != "125" {
		t.Fatalf("round trip = %#v", parsed)
	}

	payload := resumeTokenPayload{
		Version: 1, Fingerprint: strings.Repeat("ab", 32),
		MappingKind: "vertex",
	}
	raw, _ := json.Marshal(payload)
	raw = raw[:len(raw)-1]
	raw = append(raw, []byte(`,"unknown":true}`)...)
	unknown := resumeTokenPrefix + base64.RawURLEncoding.EncodeToString(raw)
	if _, err := parseResumeToken(unknown); err == nil {
		t.Fatal("parseResumeToken() accepted an unknown field")
	}
}

func TestValueAndIdentityConversion(t *testing.T) {
	document, err := decodeObject([]byte(
		`{"id":42,"null":null,"bool":true,"int":-7,"float":1.5,` +
			`"string":"x","list":[1,"a"],"object":{"nested":false}}`,
	))
	if err != nil {
		t.Fatal(err)
	}
	id, err := resolveExternalID(document, "id", "idField")
	if err != nil || id != "42" {
		t.Fatalf("resolveExternalID() = %q, %v", id, err)
	}
	for _, field := range []string{"null", "bool", "int", "float", "string", "list", "object"} {
		if _, err := convertValue(document[field], 0); err != nil {
			t.Fatalf("convertValue(%s) error = %v", field, err)
		}
	}
	if _, err := resolveExternalID(document, "null", "idField"); err == nil {
		t.Fatal("resolveExternalID() accepted null")
	}
	if _, err := convertValue(json.Number("9223372036854775808"), 0); err == nil {
		t.Fatal("convertValue() accepted overflowing int64")
	}
	if _, err := convertValue(json.Number("1e999"), 0); err == nil {
		t.Fatal("convertValue() accepted a non-finite float")
	}
}

func TestDecodeRecordAndPreencode(t *testing.T) {
	iterator := &Iterator{options: IteratorOptions{
		MaxRecordBytes: 1024, PreencodeProperties: true,
	}}
	mapping := compiledMapping{
		kind: vertexMapping, label: "Person", namespace: "crm", idField: "id",
		properties: []compiledProperty{{name: "active", field: "enabled"}},
	}
	record, _, err := iterator.decodeRecord(
		t.Context(), mapping, []byte(`{"id":7,"enabled":true}`),
	)
	if err != nil {
		t.Fatalf("decodeRecord() error = %v", err)
	}
	if record.Kind() != model.RecordVertex ||
		record.Vertex.ExternalID != "7" ||
		record.Vertex.Properties != nil ||
		string(record.Vertex.EncodedProperties) != `{"active":true}` {
		t.Fatalf("record = %#v", record.Vertex)
	}
}

func TestCopyCSVDecodingAndBounds(t *testing.T) {
	want := []byte(`{"text":"quote \" backslash \\ tab \t newline \n"}`)
	var encoded bytes.Buffer
	writer := csv.NewWriter(&encoded)
	if err := writer.Write([]string{string(want)}); err != nil {
		t.Fatal(err)
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		t.Fatal(err)
	}
	wire := bytes.TrimSuffix(encoded.Bytes(), []byte{'\n'})
	raw, err := decodeCopyCSV(wire)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, want) {
		t.Fatalf("decodeCopyCSV() = %q, want %q", raw, want)
	}
	if _, err := decodeCopyCSV([]byte("one,two")); err == nil {
		t.Fatal("decodeCopyCSV() accepted more than one field")
	}
	if boundedScannerSize(100) != 1224 {
		t.Fatalf("boundedScannerSize(100) = %d", boundedScannerSize(100))
	}
}

func TestRowJSONQueryPreservesTrailingLineComment(t *testing.T) {
	query := rowJSONQuery("SELECT id FROM people ORDER BY id -- stable")
	if !strings.Contains(query, "-- stable\n) AS af_row") {
		t.Fatalf("rowJSONQuery() did not terminate the line comment: %q", query)
	}
}

func TestInt64KeyComparisonAndRejection(t *testing.T) {
	integerTwo, err := parseNumberKey("2")
	if err != nil {
		t.Fatal(err)
	}
	integerTen, err := parseNumberKey("10")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := integerTwo.native.(int64); !ok {
		t.Fatalf("integer native type = %T, want int64", integerTwo.native)
	}
	if comparison, err := compareKeys(integerTwo, integerTen); err != nil || comparison >= 0 {
		t.Fatalf("numeric compare 2 < 10 = %d, %v", comparison, err)
	}
	if _, err := compareKeys(
		keyValue{kind: "unsupported", text: "1", native: int64(1)}, integerTwo,
	); err == nil {
		t.Fatal("compareKeys() accepted an unsupported key kind")
	}
	if _, err := parseNumberKey("9223372036854775808"); err == nil {
		t.Fatal("parseNumberKey() accepted overflowing int64")
	}
	if _, err := parseNumberKey("1e9999"); err == nil {
		t.Fatal("parseNumberKey() accepted exponent notation")
	}
}

func TestSnapshotIdentifierValidation(t *testing.T) {
	if !validSnapshotID("00000003-0000001B-1") {
		t.Fatal("valid snapshot identifier rejected")
	}
	for _, value := range []string{
		"", "00000003-0000001B-1'; SELECT 1", "abc", "00000003_0000001B_1",
	} {
		if validSnapshotID(value) {
			t.Fatalf("invalid snapshot identifier %q accepted", value)
		}
	}
}

func TestContextAndHandlerErrors(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := buildMappings(
		cancelled, "crm",
		config.PostgreSQLSource{Vertices: []config.VertexQuery{{Label: "x"}}},
		1,
	); !errors.Is(err, context.Canceled) {
		t.Fatalf("buildMappings() error = %v", err)
	}
}
