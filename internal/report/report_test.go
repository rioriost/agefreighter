package report

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"
)

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
	document.Checks[0].Summary = "unsafe | *value* <tag>\nnext ![track](https://example.invalid)"
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
