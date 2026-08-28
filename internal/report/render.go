package report

import (
	"encoding/json"
	"fmt"
	"strings"
)

type Format string

const (
	FormatJSON     Format = "json"
	FormatMarkdown Format = "markdown"
)

func Render(document Document, format Format) ([]byte, error) {
	canonicalDocument, err := canonical(document)
	if err != nil {
		return nil, err
	}
	var output []byte
	switch format {
	case FormatJSON:
		output, err = json.MarshalIndent(canonicalDocument, "", "  ")
		if err != nil {
			return nil, fmt.Errorf("encode report JSON: %w", err)
		}
		output = append(output, '\n')
	case FormatMarkdown:
		output = []byte(renderMarkdown(canonicalDocument))
	default:
		return nil, fmt.Errorf("unsupported report format %q", format)
	}
	if len(output) > MaxOutputBytes {
		return nil, fmt.Errorf("rendered report exceeds %d bytes", MaxOutputBytes)
	}
	return output, nil
}

func renderMarkdown(document Document) string {
	var output strings.Builder
	output.WriteString("# agefreighter ")
	output.WriteString(escapeMarkdown(document.Command))
	output.WriteString(" report\n\n")
	fmt.Fprintf(&output, "- Schema version: %d\n", document.SchemaVersion)
	fmt.Fprintf(
		&output,
		"- agefreighter version: %s\n",
		escapeMarkdown(document.AgefreighterVersion),
	)
	fmt.Fprintf(
		&output,
		"- Generated at: %s\n",
		document.GeneratedAt.Format("2006-01-02T15:04:05.999999999Z07:00"),
	)
	fmt.Fprintf(&output, "- Outcome: **%s**\n", document.Outcome)
	if document.Job != nil {
		fmt.Fprintf(&output, "- Job ID: `%s`\n", document.Job.ID)
		fmt.Fprintf(
			&output,
			"- Configuration fingerprint: `%s`\n",
			document.Job.ConfigFingerprint,
		)
	}
	if document.Target != nil {
		output.WriteString("\n## Target versions\n\n")
		output.WriteString("| Component | Version | Status |\n")
		output.WriteString("|---|---|---|\n")
		writeMarkdownRow(
			&output,
			"PostgreSQL",
			document.Target.PostgreSQL.Value,
			string(document.Target.PostgreSQL.Status),
		)
		writeMarkdownRow(
			&output,
			"Apache AGE",
			document.Target.AGE.Value,
			string(document.Target.AGE.Status),
		)
	}
	if len(document.Checks) > 0 {
		output.WriteString("\n## Checks\n\n")
		output.WriteString("| Check | Status | Summary | Detail |\n")
		output.WriteString("|---|---|---|---|\n")
		for _, check := range document.Checks {
			writeMarkdownRow(
				&output,
				check.ID,
				string(check.Status),
				check.Summary,
				check.Detail,
			)
		}
	}
	writeFindingSection(&output, "Warnings", document.Warnings)
	writeFindingSection(&output, "Errors", document.Errors)
	if len(document.IncompleteChecks) > 0 {
		output.WriteString("\n## Incomplete checks\n\n")
		for _, check := range document.IncompleteChecks {
			fmt.Fprintf(&output, "- %s\n", escapeMarkdown(check))
		}
	}
	for _, section := range document.Sections {
		output.WriteString("\n## ")
		output.WriteString(escapeMarkdown(section.Title))
		output.WriteString("\n\n")
		output.WriteString("| Field | Value | Status |\n")
		output.WriteString("|---|---|---|\n")
		for _, field := range section.Fields {
			writeMarkdownRow(
				&output,
				field.Name,
				field.Value,
				string(field.Status),
			)
		}
	}
	return output.String()
}

func writeFindingSection(
	output *strings.Builder,
	title string,
	findings []Finding,
) {
	if len(findings) == 0 {
		return
	}
	output.WriteString("\n## ")
	output.WriteString(title)
	output.WriteString("\n\n")
	for _, finding := range findings {
		fmt.Fprintf(
			output,
			"- **%s:** %s\n",
			escapeMarkdown(finding.Code),
			escapeMarkdown(finding.Message),
		)
	}
}

func writeMarkdownRow(output *strings.Builder, values ...string) {
	output.WriteString("|")
	for _, value := range values {
		output.WriteString(" ")
		output.WriteString(escapeMarkdown(value))
		output.WriteString(" |")
	}
	output.WriteByte('\n')
}

func escapeMarkdown(value string) string {
	value = strings.ReplaceAll(value, "\r", " ")
	value = strings.ReplaceAll(value, "\n", " ")
	value = strings.ReplaceAll(value, `\`, `\\`)
	value = strings.ReplaceAll(value, "|", `\|`)
	value = strings.ReplaceAll(value, "`", "\\`")
	value = strings.ReplaceAll(value, "*", "\\*")
	value = strings.ReplaceAll(value, "_", "\\_")
	value = strings.ReplaceAll(value, "!", "\\!")
	value = strings.ReplaceAll(value, "[", "\\[")
	value = strings.ReplaceAll(value, "]", "\\]")
	value = strings.ReplaceAll(value, "(", "\\(")
	value = strings.ReplaceAll(value, ")", "\\)")
	value = strings.ReplaceAll(value, "<", "&lt;")
	value = strings.ReplaceAll(value, ">", "&gt;")
	return value
}
