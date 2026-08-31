package neo4j

import (
	"context"
	"encoding/base64"
	"errors"
	"strings"
	"testing"

	neodriver "github.com/neo4j/neo4j-go-driver/v6/neo4j"
	"github.com/rioriost/agefreighter/internal/config"
)

func TestMappingValidationAndOrdering(t *testing.T) {
	source := testSource()
	source.Vertices[0].Properties = map[string]string{"z": "z_field", "a": "a_field"}
	mappings, err := buildMappings(context.Background(), "people", source, 2)
	if err != nil {
		t.Fatal(err)
	}
	if mappings[0].properties[0].name != "a" || mappings[0].properties[1].name != "z" {
		t.Fatalf("properties = %#v", mappings[0].properties)
	}
	tests := []struct {
		name  string
		query string
		key   string
	}{
		{"empty", "", "k"},
		{"key", "RETURN $afterKey AS k ORDER BY k", ""},
		{"parameter", "RETURN 1 AS k ORDER BY k", "k"},
		{"order", "RETURN $afterKey AS k", "k"},
		{"skip", "RETURN $afterKey AS k ORDER BY k SKIP 1", "k"},
		{"offset", "RETURN $afterKey AS k ORDER BY k OFFSET 1", "k"},
		{"limit", "RETURN $afterKey AS k ORDER BY k LIMIT 1", "k"},
		{"unbound page limit", "RETURN $afterKey AS k ORDER BY k LIMIT $other", "k"},
		{"union", "RETURN $afterKey AS k UNION RETURN 2 AS k ORDER BY k", "k"},
		{"collect", "RETURN collect($afterKey) AS k ORDER BY k", "k"},
		{"semicolon", "RETURN $afterKey AS k ORDER BY k;", "k"},
	}
	if err := validateQuery(
		"RETURN $afterKey AS k ORDER BY k LIMIT $pageRows", "k", "mapping",
	); err != nil {
		t.Fatalf("rejected bounded keyset query: %v", err)
	}
	if err := validateInitialQuery(
		"RETURN 1 AS k ORDER BY k LIMIT $pageRows", "k", "mapping",
	); err != nil {
		t.Fatalf("rejected bounded initial query: %v", err)
	}
	if err := validateInitialQuery(
		"RETURN 1 AS k ORDER BY k LIMIT 1", "k", "mapping",
	); err == nil {
		t.Fatal("accepted unbounded initial query")
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := validateQuery(test.query, test.key, "mapping"); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
	if _, err := buildMappings(context.Background(), "", source, 10); err == nil {
		t.Fatal("accepted empty namespace")
	}
	if _, err := compileProperties(map[string]string{"": "x"}, 10); err == nil {
		t.Fatal("accepted empty property name")
	}
	if _, err := compileProperties(map[string]string{"x": ""}, 10); err == nil {
		t.Fatal("accepted empty field")
	}
	if _, err := compileProperties(map[string]string{"x": "x"}, 0); err == nil {
		t.Fatal("accepted too many properties")
	}
}

func TestFingerprintBindsIdentityAndOrderedConfiguration(t *testing.T) {
	source := testSource()
	source.Password = &config.SecretRef{Env: "PASSWORD_ONE"}
	mappings, err := buildMappings(context.Background(), "people", source, 10)
	if err != nil {
		t.Fatal(err)
	}
	first, err := bindFingerprint(source, "people", mappings)
	if err != nil {
		t.Fatal(err)
	}
	passwordChanged := source
	passwordChanged.Password = &config.SecretRef{Env: "PASSWORD_TWO"}
	passwordFingerprint, _ := bindFingerprint(passwordChanged, "people", mappings)
	if first != passwordFingerprint {
		t.Fatal("password affected fingerprint")
	}
	changes := []func(*config.Neo4jSource){
		func(source *config.Neo4jSource) { source.SourceID = "other" },
		func(source *config.Neo4jSource) { source.URI = "neo4j://other.invalid" },
		func(source *config.Neo4jSource) { source.Database = "other" },
		func(source *config.Neo4jSource) { source.Username = "other" },
		func(source *config.Neo4jSource) { source.FetchRows++ },
		func(source *config.Neo4jSource) { source.MultiLabelPolicy = config.Neo4jMultiLabelReject },
	}
	for index, change := range changes {
		changed := source
		change(&changed)
		fingerprint, _ := bindFingerprint(changed, "people", mappings)
		if fingerprint == first {
			t.Fatalf("identity change %d did not affect fingerprint", index)
		}
	}
	if namespaceFingerprint, _ := bindFingerprint(source, "other", mappings); namespaceFingerprint == first {
		t.Fatal("namespace did not affect fingerprint")
	}
	changedMappings := append([]compiledMapping(nil), mappings...)
	changedMappings[0].query += " "
	if fingerprint, _ := bindFingerprint(source, "people", changedMappings); fingerprint == first {
		t.Fatal("mapping did not affect fingerprint")
	}
	changedMappings = append([]compiledMapping(nil), mappings...)
	changedMappings[0].initialQuery = "RETURN 1"
	if fingerprint, _ := bindFingerprint(source, "people", changedMappings); fingerprint == first {
		t.Fatal("initial query did not affect fingerprint")
	}
}

func TestResumeTokenRoundTripAndValidation(t *testing.T) {
	key := int64(-9)
	state := resumeState{
		fingerprint: strings.Repeat("ab", 32), mappingIndex: 2,
		mappingKind: edgeMapping, consumed: 3, rejected: 1, lastKey: &key,
	}
	token, err := formatResumeToken(state)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := parseResumeToken(token)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.mappingIndex != 2 || parsed.mappingKind != edgeMapping ||
		parsed.lastKey == nil || *parsed.lastKey != -9 || parsed.rejected != 1 {
		t.Fatalf("parsed = %#v", parsed)
	}
	badPayloads := []string{
		`{"v":2,"fp":"` + strings.Repeat("ab", 32) + `","mi":0,"mk":"vertex","n":1,"k":1}`,
		`{"v":1,"fp":"bad","mi":0,"mk":"vertex","n":1,"k":1}`,
		`{"v":1,"fp":"` + strings.Repeat("ab", 32) + `","mi":-1,"mk":"vertex","n":1,"k":1}`,
		`{"v":1,"fp":"` + strings.Repeat("ab", 32) + `","mi":0,"mk":"bad","n":1,"k":1}`,
		`{"v":1,"fp":"` + strings.Repeat("ab", 32) + `","mi":0,"mk":"vertex","n":0}`,
		`{"v":1,"fp":"` + strings.Repeat("ab", 32) + `","mi":0,"mk":"vertex","n":1,"r":-1,"k":1}`,
		`{"v":1,"fp":"` + strings.Repeat("ab", 32) + `","mi":0,"mk":"vertex","n":1}`,
		`{"v":1,"fp":"` + strings.Repeat("ab", 32) + `","mi":0,"mk":"vertex","n":1,"k":1,"extra":true}`,
	}
	for index, payload := range badPayloads {
		bad := resumeTokenPrefix + base64.RawURLEncoding.EncodeToString([]byte(payload))
		if _, err := parseResumeToken(bad); err == nil {
			t.Fatalf("accepted bad payload %d", index)
		}
	}
	for _, token := range []string{
		"bad", resumeTokenPrefix + "!", resumeTokenPrefix +
			base64.RawURLEncoding.EncodeToString([]byte(`{}`)) + "trailing",
	} {
		if _, err := parseResumeToken(token); err == nil {
			t.Fatalf("accepted token %q", token)
		}
	}
	if _, err := parseResumeToken(strings.Repeat("x", maxResumeTokenBytes+1)); err == nil {
		t.Fatal("accepted oversized token")
	}
}

func TestSafeErrorsNeverExposeServerMessages(t *testing.T) {
	server := &neodriver.Neo4jError{
		Code: "Neo.ClientError.Statement.SyntaxError", Msg: "secret value",
	}
	err := safeError(context.Background(), "run Neo4j query", server)
	if !strings.Contains(err.Error(), server.Code) || strings.Contains(err.Error(), server.Msg) {
		t.Fatalf("safe server error = %v", err)
	}
	unsafeCode := &neodriver.Neo4jError{Code: "bad code secret", Msg: "raw"}
	err = safeError(context.Background(), "run", unsafeCode)
	if strings.Contains(err.Error(), "secret") || strings.Contains(err.Error(), "raw") {
		t.Fatalf("unsafe code leaked: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := safeError(ctx, "run", errors.New("raw")); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation = %v", err)
	}
	if err := safeError(nil, "run", context.DeadlineExceeded); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("deadline = %v", err)
	}
}

func TestSDKConstructorValidation(t *testing.T) {
	if _, err := NewSDKClient(nil, "", "", "", "", 0); err == nil {
		t.Fatal("accepted nil context")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewSDKClient(ctx, "", "", "", "", 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled constructor = %v", err)
	}
	if _, err := NewSDKClient(context.Background(), "", "", "", "", 1); err == nil {
		t.Fatal("accepted empty database")
	}
	if _, err := NewSDKClient(context.Background(), "", "neo4j", "", "", 0); err == nil {
		t.Fatal("accepted invalid fetch rows")
	}
	if _, err := NewSDKClient(context.Background(), "", "neo4j", "", "", 100_001); err == nil {
		t.Fatal("accepted excessive fetch rows")
	}
}
