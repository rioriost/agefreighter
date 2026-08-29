package report

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

func TestGoldenReportContracts(t *testing.T) {
	document := validDocument()
	document.Outcome = OutcomeIncomplete
	document.Checks = append(document.Checks, Check{
		ID: "telemetry", Status: CheckUnavailable,
		Summary: "not recorded by metadata schema v14",
	})
	document.Sections = []Section{{
		Title: "Job",
		Fields: []Field{
			{Name: "sourceType", Value: "csv", Status: CheckPass},
			{Name: "telemetry", Value: "requires schema v15", Status: CheckUnavailable},
		},
	}}
	for _, format := range []Format{FormatJSON, FormatMarkdown} {
		format := format
		t.Run(string(format), func(t *testing.T) {
			got, err := Render(document, format)
			if err != nil {
				t.Fatalf("Render() error = %v", err)
			}
			path := filepath.Join("testdata", "migration-report.golden."+string(format))
			if os.Getenv("UPDATE_REPORT_GOLDEN") == "1" {
				if err := os.MkdirAll("testdata", 0o755); err != nil {
					t.Fatalf("MkdirAll() error = %v", err)
				}
				if err := os.WriteFile(path, got, 0o600); err != nil {
					t.Fatalf("WriteFile() error = %v", err)
				}
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("ReadFile(%s) error = %v", path, err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("%s golden mismatch:\n%s", format, got)
			}
		})
	}
}

func TestGoldenVerificationReportContracts(t *testing.T) {
	document := validDocument()
	document.Command = "verify"
	document.Target = nil
	document.Sections = []Section{
		{
			Title: "Bounded integrity",
			Fields: []Field{{
				Name:   "e.KNOWS",
				Value:  "limit=100,identityCoverage=full,identityRowsChecked=1,physicalRowsChecked=1,reversePhysicalCoverage=checked,missingPhysicalRows=0,orphanPhysicalRows=0,missingEndpointRows=0,changedEndpointRows=0,identityTruncated=false,physicalTruncated=false",
				Status: CheckPass,
			}},
		},
		{
			Title: "Per-label counts",
			Fields: []Field{{
				Name:   "v.Person",
				Value:  "counterCompleteness=complete,counterProvenance=v17-lifecycle,identityCoverage=full,acceptedRows=2,committedRows=2,livePhysicalRows=2,liveIdentityRows=2,storedPhysicalComparison=verified,physicalIdentityEquality=verified,committedBytes=unavailable,rejectedRows=0",
				Status: CheckPass,
			}},
		},
	}
	for _, format := range []Format{FormatJSON, FormatMarkdown} {
		t.Run(string(format), func(t *testing.T) {
			got, err := Render(document, format)
			if err != nil {
				t.Fatalf("Render() error = %v", err)
			}
			path := filepath.Join("testdata", "verification-report.golden."+string(format))
			if os.Getenv("UPDATE_REPORT_GOLDEN") == "1" {
				if err := os.WriteFile(path, got, 0o600); err != nil {
					t.Fatalf("WriteFile() error = %v", err)
				}
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("ReadFile(%s) error = %v", path, err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("%s golden mismatch:\n%s", format, got)
			}
		})
	}
}

func TestGoldenSourceProfileReportContracts(t *testing.T) {
	document := New("profile", time.Date(2026, 8, 28, 0, 0, 0, 0, time.UTC))
	document.Outcome = OutcomeIncomplete
	document.Checks = []Check{
		{
			ID: "source-read", Status: CheckUnknown,
			Summary: "source profile was truncated by a configured bound",
			Detail:  "limit=rows",
		},
		{
			ID: "source-version", Status: CheckUnavailable,
			Summary: "source version is not exposed by the connector iterator",
		},
	}
	document.Warnings = []Finding{{
		Code:    "PROFILE_TRUNCATED",
		Message: "reported counts and statistics are lower-bound observations from a bounded prefix",
	}}
	document.IncompleteChecks = []string{"source-profile"}
	document.Sections = []Section{
		{
			Title: "Source",
			Fields: []Field{
				{Name: "connector", Value: "csv", Status: CheckPass},
				{Name: "mode", Value: "sample", Status: CheckPass},
			},
		},
		{
			Title: "Vertex labels",
			Fields: []Field{{
				Name:   "001",
				Value:  "label=Person,sampledRows=2,countRange=2..unknown,countMethod=observed-bounded-prefix,configuredProperties=1",
				Status: CheckPass,
			}},
		},
	}
	for _, format := range []Format{FormatJSON, FormatMarkdown} {
		t.Run(string(format), func(t *testing.T) {
			got, err := Render(document, format)
			if err != nil {
				t.Fatalf("Render() error = %v", err)
			}
			path := filepath.Join("testdata", "source-profile.golden."+string(format))
			if os.Getenv("UPDATE_REPORT_GOLDEN") == "1" {
				if err := os.WriteFile(path, got, 0o600); err != nil {
					t.Fatalf("WriteFile() error = %v", err)
				}
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("ReadFile(%s) error = %v", path, err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("%s golden mismatch:\n%s", format, got)
			}
		})
	}
}

func TestMigrationReportJSONSchemaContract(t *testing.T) {
	data, err := os.ReadFile("../../docs/reference/migration-report.schema.json")
	if err != nil {
		t.Fatalf("ReadFile(schema) error = %v", err)
	}
	var schema map[string]any
	if err := json.Unmarshal(data, &schema); err != nil {
		t.Fatalf("schema JSON error = %v", err)
	}
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("schema properties are missing")
	}
	version, ok := properties["schemaVersion"].(map[string]any)
	if !ok || version["const"] != float64(SchemaVersion) {
		t.Fatalf("schema version contract = %#v", version)
	}
	command, ok := properties["command"].(map[string]any)
	if !ok || command["const"] != "report" {
		t.Fatalf("schema command contract = %#v", command)
	}
}

func TestVerificationReportJSONSchemaContract(t *testing.T) {
	data, err := os.ReadFile("../../docs/reference/verification-report.schema.json")
	if err != nil {
		t.Fatalf("ReadFile(schema) error = %v", err)
	}
	var schema map[string]any
	if err := json.Unmarshal(data, &schema); err != nil {
		t.Fatalf("schema JSON error = %v", err)
	}
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("schema properties are missing")
	}
	version, ok := properties["schemaVersion"].(map[string]any)
	if !ok || version["const"] != float64(SchemaVersion) {
		t.Fatalf("schema version contract = %#v", version)
	}
	command, ok := properties["command"].(map[string]any)
	if !ok || command["const"] != "verify" {
		t.Fatalf("schema command contract = %#v", command)
	}
}

func TestDoctorReportJSONSchemaContract(t *testing.T) {
	data, err := os.ReadFile("../../docs/reference/doctor-report.schema.json")
	if err != nil {
		t.Fatalf("ReadFile(schema) error = %v", err)
	}
	var schema map[string]any
	if err := json.Unmarshal(data, &schema); err != nil {
		t.Fatalf("schema JSON error = %v", err)
	}
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("schema properties are missing")
	}
	version, ok := properties["schemaVersion"].(map[string]any)
	if !ok || version["const"] != float64(SchemaVersion) {
		t.Fatalf("schema version contract = %#v", version)
	}
	command, ok := properties["command"].(map[string]any)
	if !ok || command["const"] != "doctor" {
		t.Fatalf("schema command contract = %#v", command)
	}
	required, ok := schema["required"].([]any)
	if !ok {
		t.Fatal("schema required contract is missing")
	}
	for _, name := range required {
		if name == "job" {
			t.Fatal("doctor schema requires a migration job ID")
		}
	}
}

func TestSourceProfileJSONSchemaContract(t *testing.T) {
	data, err := os.ReadFile("../../docs/reference/source-profile.schema.json")
	if err != nil {
		t.Fatalf("ReadFile(schema) error = %v", err)
	}

	var schema map[string]any
	if err := json.Unmarshal(data, &schema); err != nil {
		t.Fatalf("schema JSON error = %v", err)
	}
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("schema properties are missing")
	}
	version, ok := properties["schemaVersion"].(map[string]any)
	if !ok || version["const"] != float64(SchemaVersion) {
		t.Fatalf("schema version contract = %#v", version)
	}
	command, ok := properties["command"].(map[string]any)
	if !ok || command["const"] != "profile" {
		t.Fatalf("schema command contract = %#v", command)
	}
	for _, forbidden := range []string{"job", "target"} {
		if _, found := properties[forbidden]; found {
			t.Fatalf("source-only profile schema defines %q", forbidden)
		}
	}
}

func TestOptimizerJSONSchemaContract(t *testing.T) {
	data, err := os.ReadFile("../../docs/reference/optimizer-report.schema.json")
	if err != nil {
		t.Fatalf("ReadFile(schema) error = %v", err)
	}
	var schema map[string]any
	if err := json.Unmarshal(data, &schema); err != nil {
		t.Fatalf("schema JSON error = %v", err)
	}
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("schema properties are missing")
	}
	version, ok := properties["schemaVersion"].(map[string]any)
	if !ok || version["const"] != float64(SchemaVersion) {
		t.Fatalf("schema version contract = %#v", version)
	}
	command, ok := properties["command"].(map[string]any)
	if !ok || command["const"] != "optimize" {
		t.Fatalf("schema command contract = %#v", command)
	}
	if _, found := properties["job"]; found {
		t.Fatal("optimizer schema exposes a raw migration job identity")
	}
	compiled, err := jsonschema.NewCompiler().Compile(
		"../../docs/reference/optimizer-report.schema.json",
	)
	if err != nil {
		t.Fatalf("Compile(schema) error = %v", err)
	}
	golden, err := os.ReadFile("../app/testdata/optimizer.golden.json")
	if err != nil {
		t.Fatalf("ReadFile(golden) error = %v", err)
	}
	var document any
	if err := json.Unmarshal(golden, &document); err != nil {
		t.Fatalf("golden JSON error = %v", err)
	}
	if err := compiled.Validate(document); err != nil {
		t.Fatalf("optimizer golden schema validation error = %v", err)
	}
}

func TestRenderIsDeterministicAndCanonical(t *testing.T) {
	document := validDocument()
	document.Checks = []Check{
		{ID: "z-last", Status: CheckPass, Summary: "last"},
		{ID: "a-first", Status: CheckWarning, Summary: "first"},
	}
	document.Warnings = []Finding{
		{Code: "Z_CODE", Message: "last"},
		{Code: "A_CODE", Message: "first"},
	}
	document.Sections = []Section{
		{
			Title: "Jobs",
			Fields: []Field{
				{Name: "status", Value: "committed", Status: CheckPass},
				{Name: "name", Value: "people", Status: CheckPass},
			},
		},
		{
			Title: "Batches",
			Fields: []Field{
				{Name: "count", Value: "2", Status: CheckPass},
			},
		},
	}
	firstJSON, err := Render(document, FormatJSON)
	if err != nil {
		t.Fatalf("Render(JSON) error = %v", err)
	}
	firstMarkdown, err := Render(document, FormatMarkdown)
	if err != nil {
		t.Fatalf("Render(Markdown) error = %v", err)
	}

	document.Checks[0], document.Checks[1] = document.Checks[1], document.Checks[0]
	document.Warnings[0], document.Warnings[1] = document.Warnings[1], document.Warnings[0]
	document.Sections[0], document.Sections[1] = document.Sections[1], document.Sections[0]
	secondJSON, err := Render(document, FormatJSON)
	if err != nil {
		t.Fatalf("second Render(JSON) error = %v", err)
	}
	secondMarkdown, err := Render(document, FormatMarkdown)
	if err != nil {
		t.Fatalf("second Render(Markdown) error = %v", err)
	}
	if !bytes.Equal(firstJSON, secondJSON) {
		t.Fatalf("JSON output is not deterministic:\n%s\n%s", firstJSON, secondJSON)
	}
	if !bytes.Equal(firstMarkdown, secondMarkdown) {
		t.Fatalf("Markdown output is not deterministic:\n%s\n%s", firstMarkdown, secondMarkdown)
	}
	if bytes.Index(firstJSON, []byte(`"id": "a-first"`)) >
		bytes.Index(firstJSON, []byte(`"id": "z-last"`)) {
		t.Fatalf("checks were not canonicalized:\n%s", firstJSON)
	}
}

func TestUnavailableAndUnknownCannotPass(t *testing.T) {
	for _, status := range []CheckStatus{CheckUnknown, CheckUnavailable} {
		t.Run(string(status), func(t *testing.T) {
			document := validDocument()
			document.Checks = []Check{{
				ID: "metadata", Status: status, Summary: "not inspectable",
			}}
			if _, err := Render(document, FormatJSON); err == nil {
				t.Fatalf("passing report accepted %s check", status)
			}
			document.Outcome = OutcomeIncomplete
			if _, err := Render(document, FormatJSON); err != nil {
				t.Fatalf("incomplete report rejected: %v", err)
			}
		})
	}
}

func TestReportValidationRejectsInvalidAndOversizedData(t *testing.T) {
	tests := []struct {
		name   string
		change func(*Document)
	}{
		{
			name: "schema",
			change: func(document *Document) {
				document.SchemaVersion++
			},
		},
		{
			name: "command",
			change: func(document *Document) {
				document.Command = "load"
			},
		},
		{
			name: "duplicate check",
			change: func(document *Document) {
				document.Checks = append(document.Checks, document.Checks[0])
			},
		},
		{
			name: "oversized text",
			change: func(document *Document) {
				document.Checks[0].Detail = strings.Repeat("x", MaxTextBytes+1)
			},
		},
		{
			name: "failed without evidence",
			change: func(document *Document) {
				document.Outcome = OutcomeFail
			},
		},
		{
			name: "missing report job",
			change: func(document *Document) {
				document.Job = nil
			},
		},
		{
			name: "missing report target",
			change: func(document *Document) {
				document.Target = nil
			},
		},
		{
			name: "empty passing report",
			change: func(document *Document) {
				document.Checks = nil
				document.Sections = nil
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			document := validDocument()
			test.change(&document)
			if _, err := Render(document, FormatJSON); err == nil {
				t.Fatal("Render() accepted invalid report")
			}
		})
	}
	if _, err := Render(validDocument(), Format("yaml")); err == nil {
		t.Fatal("Render() accepted unsupported format")
	}
}

func TestRenderVerificationReportUsesSharedContract(t *testing.T) {
	document := validDocument()
	document.Command = "verify"
	document.Target = nil
	document.Sections = []Section{{
		Title: "Per-label counts",
		Fields: []Field{{
			Name:   "v.Person",
			Value:  "identityCoverage=full,acceptedRows=2,committedRows=2,livePhysicalRows=2,liveIdentityRows=2,physicalIdentityEquality=verified",
			Status: CheckPass,
		}},
	}}
	for _, format := range []Format{FormatJSON, FormatMarkdown} {
		first, err := Render(document, format)
		if err != nil {
			t.Fatalf("Render(%s) error = %v", format, err)
		}
		second, err := Render(document, format)
		if err != nil || string(first) != string(second) {
			t.Fatalf("Render(%s) is not deterministic: %v", format, err)
		}
		if strings.Contains(string(first), "external-id") {
			t.Fatalf("Render(%s) disclosed raw identity data", format)
		}
	}
}

func TestRenderedOutputIsBounded(t *testing.T) {
	document := validDocument()
	document.Sections = make([]Section, 3)
	for sectionIndex := range document.Sections {
		section := Section{
			Title:  fmt.Sprintf("section-%d", sectionIndex),
			Fields: make([]Field, MaxFieldsPerSection),
		}
		for fieldIndex := range section.Fields {
			section.Fields[fieldIndex] = Field{
				Name:   fmt.Sprintf("field-%d", fieldIndex),
				Value:  strings.Repeat("x", MaxTextBytes),
				Status: CheckPass,
			}
		}
		document.Sections[sectionIndex] = section
	}
	if _, err := Render(document, FormatJSON); err == nil {
		t.Fatal("Render() emitted output beyond its byte limit")
	}
}

func TestMarkdownEscapesReportValues(t *testing.T) {
	document := validDocument()
	document.Checks[0].Summary = "unsafe | *value* <tag>\nnext ![track](https://example.invalid)\x1b[31mspoof"
	output, err := Render(document, FormatMarkdown)
	if err != nil {
		t.Fatalf("Render() error = %v", err)
	}
	for _, expected := range []string{
		`\|`,
		`\*value\*`,
		`&lt;tag&gt;`,
		" next",
		`\!\[track\]\(https://example.invalid\)`,
	} {
		if !bytes.Contains(output, []byte(expected)) {
			t.Fatalf("Markdown missing %q:\n%s", expected, output)
		}
	}
	if bytes.Contains(output, []byte{0x1b}) {
		t.Fatalf("Markdown retained a terminal escape character:\n%s", output)
	}
}

func FuzzReportDecode(f *testing.F) {
	seed, err := Render(validDocument(), FormatJSON)
	if err != nil {
		f.Fatalf("render fuzz seed: %v", err)
	}
	f.Add(seed)
	f.Add([]byte(`{"schemaVersion":2}`))
	f.Add([]byte(`{}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		document, err := Decode(data)
		if err != nil {
			return
		}
		if _, err := Render(document, FormatJSON); err != nil {
			t.Fatalf("decoded report did not render: %v", err)
		}
		if _, err := Render(document, FormatMarkdown); err != nil {
			t.Fatalf("decoded report did not render as Markdown: %v", err)
		}
	})
}

func validDocument() Document {
	document := New("report", time.Date(2026, 8, 28, 6, 0, 0, 0, time.FixedZone("JST", 9*60*60)))
	document.Outcome = OutcomePass
	document.Job = &Job{
		ID:                "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
		ConfigFingerprint: strings.Repeat("a", 64),
	}
	document.Target = &Target{
		PostgreSQL: VersionValue{Value: "17.9", Status: CheckPass},
		AGE:        VersionValue{Value: "1.6.0", Status: CheckPass},
	}
	document.Checks = []Check{{
		ID: "metadata", Status: CheckPass, Summary: "metadata is current",
	}}
	return document
}
