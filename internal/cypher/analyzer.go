// Package cypher provides bounded, deterministic structural analysis for the
// command-line compatibility checker and optimizer workload evidence.
package cypher

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

const (
	SchemaVersion    = 1
	TargetAGEVersion = "1.6"

	MaxFiles         = 64
	MaxFileBytes     = 1 << 20
	MaxTotalBytes    = 8 << 20
	MaxQueries       = 1024
	MaxTokens        = 8192
	MaxDepth         = 128
	MaxFindings      = 4096
	MaxEvidenceRunes = 96
	MaxPathRunes     = 256
	MaxOutputBytes   = 4 << 20

	FileOperationTimeout = 2 * time.Second
	AnalysisTimeout      = 30 * time.Second
)

type Format string

const (
	FormatJSON     Format = "json"
	FormatMarkdown Format = "markdown"
)

type Classification string

const (
	Compatible                 Classification = "compatible"
	CompatibleWithManualChange Classification = "compatible-with-manual-change"
	Unsupported                Classification = "unsupported"
	Unknown                    Classification = "unknown"
)

type Severity string

const (
	SeverityInfo    Severity = "info"
	SeverityWarning Severity = "warning"
	SeverityError   Severity = "error"
	SeverityUnknown Severity = "unknown"
)

type Options struct{}

type Report struct {
	SchemaVersion int           `json:"schemaVersion"`
	Command       string        `json:"command"`
	Target        Target        `json:"target"`
	Limits        Limits        `json:"limits"`
	Files         []FileSummary `json:"files"`
	Queries       []Query       `json:"queries"`
	Summary       Summary       `json:"summary"`
}

type Target struct {
	AGE string `json:"age"`
}

type Limits struct {
	MaxFiles       int `json:"maxFiles"`
	MaxFileBytes   int `json:"maxFileBytes"`
	MaxTotalBytes  int `json:"maxTotalBytes"`
	MaxQueries     int `json:"maxQueries"`
	MaxTokens      int `json:"maxTokensPerQuery"`
	MaxDepth       int `json:"maxTokenDepth"`
	MaxOutputBytes int `json:"maxOutputBytes"`
}

type FileSummary struct {
	Path    string `json:"path"`
	Bytes   int64  `json:"bytes"`
	Queries int    `json:"queries"`
}

type Query struct {
	File           string         `json:"file"`
	Number         int            `json:"number"`
	Line           int            `json:"line"`
	Column         int            `json:"column"`
	Classification Classification `json:"classification"`
	Findings       []Finding      `json:"findings"`
	Patterns       []Pattern      `json:"patterns"`
}

type Finding struct {
	Code        string   `json:"code"`
	Severity    Severity `json:"severity"`
	Line        int      `json:"line"`
	Column      int      `json:"column"`
	Evidence    string   `json:"evidence"`
	Remediation string   `json:"remediation"`
}

type Pattern struct {
	Kind     string `json:"kind"`
	Label    string `json:"label"`
	Property string `json:"property,omitempty"`
	Operator string `json:"operator,omitempty"`
}

type Summary struct {
	Files                      int   `json:"files"`
	Queries                    int   `json:"queries"`
	Compatible                 int   `json:"compatible"`
	CompatibleWithManualChange int   `json:"compatibleWithManualChange"`
	Unsupported                int   `json:"unsupported"`
	Unknown                    int   `json:"unknown"`
	Warnings                   int   `json:"warnings"`
	Findings                   int   `json:"findings"`
	Score                      Score `json:"compatibilityScore"`
}

type Score struct {
	KnownQueries      int  `json:"knownQueries"`
	CompatibleQueries int  `json:"compatibleQueries"`
	Percent           *int `json:"percent"`
	Conclusive        bool `json:"conclusive"`
}

type tokenKind uint8

const (
	tokenIdentifier tokenKind = iota
	tokenEscapedIdentifier
	tokenString
	tokenNumber
	tokenParameter
	tokenPunctuation
	tokenOperator
	tokenSemicolon
)

type token struct {
	kind   tokenKind
	text   string
	line   int
	column int
	offset int
}

type statement struct {
	tokens []token
}

var (
	errLimit = errors.New("Cypher analysis limit exceeded")
	errInput = errors.New("Cypher input rejected")
)

func AnalyzeFiles(ctx context.Context, paths []string, options Options) (Report, error) {
	if len(paths) == 0 {
		return Report{}, errors.New("at least one local Cypher file is required")
	}
	if len(paths) > MaxFiles {
		return Report{}, fmt.Errorf("%w: at most %d files are accepted", errLimit, MaxFiles)
	}
	analysisCtx, cancel := context.WithTimeout(ctx, AnalysisTimeout)
	defer cancel()
	report := Report{
		SchemaVersion: SchemaVersion,
		Command:       "check-cypher",
		Target:        Target{AGE: TargetAGEVersion},
		Limits: Limits{
			MaxFiles: MaxFiles, MaxFileBytes: MaxFileBytes,
			MaxTotalBytes: MaxTotalBytes, MaxQueries: MaxQueries,
			MaxTokens: MaxTokens, MaxDepth: MaxDepth,
			MaxOutputBytes: MaxOutputBytes,
		},
		Files: []FileSummary{}, Queries: []Query{},
	}
	ordered := displayInputs(paths)
	slices.SortFunc(ordered, func(left, right displayInput) int {
		if comparison := strings.Compare(left.display, right.display); comparison != 0 {
			return comparison
		}
		return strings.Compare(left.path, right.path)
	})
	var totalBytes int64
	totalFindings := 0
	for _, input := range ordered {
		if err := analysisCtx.Err(); err != nil {
			return Report{}, err
		}
		data, size, err := readRegularFile(analysisCtx, input.path)
		if err != nil {
			return Report{}, fmt.Errorf("%s: %w", input.display, err)
		}
		totalBytes += size
		if totalBytes > MaxTotalBytes {
			return Report{}, fmt.Errorf("%w: total input exceeds %d bytes", errLimit, MaxTotalBytes)
		}
		statements, lexicalFinding, err := lex(analysisCtx, data)
		if err != nil {
			return Report{}, err
		}
		if len(report.Queries)+len(statements) > MaxQueries {
			return Report{}, fmt.Errorf("%w: at most %d queries are accepted", errLimit, MaxQueries)
		}
		fileSummary := FileSummary{Path: input.display, Bytes: size}
		for statementIndex, value := range statements {
			if err := analysisCtx.Err(); err != nil {
				return Report{}, err
			}
			if len(value.tokens) > MaxTokens {
				return Report{}, fmt.Errorf(
					"%w: query exceeds %d tokens",
					errLimit,
					MaxTokens,
				)
			}
			query := analyzeStatement(value, input.display, fileSummary.Queries+1, options)
			if lexicalFinding != nil && statementIndex == len(statements)-1 {
				query.Findings = slices.DeleteFunc(
					query.Findings,
					func(finding Finding) bool {
						return finding.Severity == SeverityInfo
					},
				)
				query.Findings = append(query.Findings, *lexicalFinding)
				query.Classification = Unknown
				query.Patterns = []Pattern{}
			}
			if len(query.Patterns) > 128 {
				return Report{}, fmt.Errorf(
					"%w: query exceeds 128 structural patterns",
					errLimit,
				)
			}
			totalFindings += len(query.Findings)
			if totalFindings > MaxFindings {
				return Report{}, ReportLimitError()
			}
			canonicalizeQuery(&query)
			fileSummary.Queries++
			report.Queries = append(report.Queries, query)
		}
		if lexicalFinding != nil && len(statements) == 0 {
			query := Query{
				File: input.display, Number: 1,
				Line: lexicalFinding.Line, Column: lexicalFinding.Column,
				Classification: Unknown,
				Findings:       []Finding{*lexicalFinding}, Patterns: []Pattern{},
			}
			fileSummary.Queries = 1
			report.Queries = append(report.Queries, query)
			totalFindings++
		}
		report.Files = append(report.Files, fileSummary)
	}
	if len(report.Queries) == 0 {
		return Report{}, fmt.Errorf("%w: no Cypher queries were found", errInput)
	}
	if err := summarize(&report); err != nil {
		return Report{}, err
	}
	return report, nil
}

func readRegularFile(ctx context.Context, path string) ([]byte, int64, error) {
	if strings.TrimSpace(path) == "" || path == "-" {
		return nil, 0, fmt.Errorf("%w: standard input and empty paths are not accepted", errInput)
	}
	before, err := boundedFileCall(ctx, nil, func() (os.FileInfo, error) {
		return os.Lstat(path)
	})
	if err != nil {
		return nil, 0, fmt.Errorf("open local input: %w", classifyFileError(err))
	}
	if !before.Mode().IsRegular() {
		return nil, 0, fmt.Errorf("%w: input is not a regular file", errInput)
	}
	if before.Size() > MaxFileBytes {
		return nil, 0, fmt.Errorf("%w: file exceeds %d bytes", errLimit, MaxFileBytes)
	}
	file, err := boundedFileCall(ctx, func(file *os.File) {
		if file != nil {
			_ = file.Close()
		}
	}, func() (*os.File, error) {
		return openRegularNoFollow(path)
	})
	if err != nil {
		return nil, 0, fmt.Errorf("open local input: %w", classifyFileError(err))
	}
	defer closeFileBounded(file)
	after, err := boundedFileCall(ctx, nil, file.Stat)
	if err != nil {
		if errors.Is(err, context.Canceled) ||
			errors.Is(err, context.DeadlineExceeded) {
			return nil, 0, err
		}
		return nil, 0, errors.New("inspect local input")
	}
	if !after.Mode().IsRegular() || !os.SameFile(before, after) {
		return nil, 0, fmt.Errorf("%w: input identity changed while opening", errInput)
	}
	data, err := boundedFileCall(ctx, nil, func() ([]byte, error) {
		reader := &contextReader{
			ctx: ctx, reader: io.LimitReader(file, MaxFileBytes+1),
		}
		return io.ReadAll(reader)
	})
	if err != nil {
		if errors.Is(err, context.Canceled) ||
			errors.Is(err, context.DeadlineExceeded) {
			return nil, 0, err
		}
		return nil, 0, errors.New("read local input")
	}
	if len(data) > MaxFileBytes {
		return nil, 0, fmt.Errorf("%w: file exceeds %d bytes", errLimit, MaxFileBytes)
	}
	if !utf8.Valid(data) {
		return nil, 0, fmt.Errorf("%w: input is not valid UTF-8", errInput)
	}
	return data, int64(len(data)), nil
}

var fileOperationSlots = make(chan struct{}, MaxFiles)

type fileCallResult[T any] struct {
	value T
	err   error
}

func boundedFileCall[T any](
	ctx context.Context,
	cleanup func(T),
	operation func() (T, error),
) (T, error) {
	var zero T
	operationCtx, cancel := context.WithTimeout(ctx, FileOperationTimeout)
	defer cancel()
	select {
	case fileOperationSlots <- struct{}{}:
	case <-operationCtx.Done():
		return zero, operationCtx.Err()
	}
	result := make(chan fileCallResult[T])
	go func() {
		defer func() { <-fileOperationSlots }()
		value, err := operation()
		select {
		case result <- fileCallResult[T]{value: value, err: err}:
		case <-operationCtx.Done():
			if cleanup != nil {
				cleanup(value)
			}
		}
	}()
	select {
	case completed := <-result:
		return completed.value, completed.err
	case <-operationCtx.Done():
		return zero, operationCtx.Err()
	}
}

func closeFileBounded(file *os.File) {
	_, _ = boundedFileCall(
		context.Background(),
		nil,
		func() (struct{}, error) {
			return struct{}{}, file.Close()
		},
	)
}

func WriteOutput(ctx context.Context, writer io.Writer, output []byte) error {
	if len(output) > MaxOutputBytes {
		return fmt.Errorf("%w: output exceeds %d bytes", errLimit, MaxOutputBytes)
	}
	written, err := boundedFileCall(ctx, nil, func() (int, error) {
		return writer.Write(output)
	})
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		return errors.New("write Cypher report")
	}
	if written != len(output) {
		return io.ErrShortWrite
	}
	return nil
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (reader *contextReader) Read(value []byte) (int, error) {
	if err := reader.ctx.Err(); err != nil {
		return 0, err
	}
	return reader.reader.Read(value)
}

func classifyFileError(err error) error {
	switch {
	case errors.Is(err, context.Canceled):
		return context.Canceled
	case errors.Is(err, context.DeadlineExceeded):
		return context.DeadlineExceeded
	case errors.Is(err, os.ErrNotExist):
		return os.ErrNotExist
	case errors.Is(err, os.ErrPermission):
		return os.ErrPermission
	default:
		return errors.New("file operation failed")
	}
}

type displayInput struct {
	path    string
	display string
}

func displayInputs(paths []string) []displayInput {
	basenameCounts := make(map[string]int)
	inputs := make([]displayInput, len(paths))
	for index, path := range paths {
		display := safeBasename(path)
		inputs[index] = displayInput{path: path, display: display}
		basenameCounts[display]++
	}
	slices.SortFunc(inputs, func(left, right displayInput) int {
		if comparison := strings.Compare(left.display, right.display); comparison != 0 {
			return comparison
		}
		return strings.Compare(filepath.Clean(left.path), filepath.Clean(right.path))
	})

	reserved := make(map[string]bool, len(inputs)*2)
	for _, input := range inputs {
		if basenameCounts[input.display] == 1 {
			reserved[input.display] = true
			continue
		}
		reserved[duplicateDisplayCandidate(input.display, input.path)] = true
	}
	used := make(map[string]bool, len(inputs))
	for _, input := range inputs {
		if basenameCounts[input.display] == 1 {
			used[input.display] = true
		}
	}
	for index := range inputs {
		if basenameCounts[inputs[index].display] == 1 {
			continue
		}
		candidate := duplicateDisplayCandidate(
			inputs[index].display,
			inputs[index].path,
		)
		display := candidate
		if used[display] {
			for suffix := 2; ; suffix++ {
				display = fmt.Sprintf("%s-%d", candidate, suffix)
				if !used[display] && !reserved[display] {
					break
				}
			}
		}
		inputs[index].display = display
		used[display] = true
	}
	return inputs
}

func duplicateDisplayCandidate(base, path string) string {
	digest := sha256.Sum256([]byte(filepath.Clean(path)))
	return fmt.Sprintf("%s~%x", base, digest)
}

func safeBasename(path string) string {
	value := filepath.Base(filepath.Clean(path))
	value = strings.Map(func(character rune) rune {
		if unicode.IsControl(character) {
			return '�'
		}
		return character
	}, value)
	runes := []rune(value)
	if len(runes) <= MaxPathRunes {
		return value
	}
	const marker = "…"
	base := []rune(value)
	allowed := MaxPathRunes - len([]rune(marker))
	if len(base) > allowed {
		base = base[len(base)-allowed:]
	}
	return marker + string(base)
}

func canonicalizeQuery(query *Query) {
	slices.SortFunc(query.Findings, func(left, right Finding) int {
		if left.Line != right.Line {
			return left.Line - right.Line
		}
		if left.Column != right.Column {
			return left.Column - right.Column
		}
		return strings.Compare(left.Code, right.Code)
	})
	slices.SortFunc(query.Patterns, func(left, right Pattern) int {
		leftKey := left.Kind + "\x00" + left.Label + "\x00" + left.Property + "\x00" + left.Operator
		rightKey := right.Kind + "\x00" + right.Label + "\x00" + right.Property + "\x00" + right.Operator
		return strings.Compare(leftKey, rightKey)
	})
	query.Patterns = slices.Compact(query.Patterns)
}

func summarize(report *Report) error {
	for _, query := range report.Queries {
		report.Summary.Queries++
		report.Summary.Findings += len(query.Findings)
		for _, finding := range query.Findings {
			if finding.Severity == SeverityWarning {
				report.Summary.Warnings++
			}
		}
		switch query.Classification {
		case Compatible:
			report.Summary.Compatible++
		case CompatibleWithManualChange:
			report.Summary.CompatibleWithManualChange++
		case Unsupported:
			report.Summary.Unsupported++
		case Unknown:
			report.Summary.Unknown++
		default:
			return errors.New("invalid internal Cypher classification")
		}
	}
	report.Summary.Files = len(report.Files)
	known := report.Summary.Compatible + report.Summary.CompatibleWithManualChange + report.Summary.Unsupported
	compatible := report.Summary.Compatible + report.Summary.CompatibleWithManualChange
	report.Summary.Score.KnownQueries = known
	report.Summary.Score.CompatibleQueries = compatible
	report.Summary.Score.Conclusive = report.Summary.Unknown == 0 && known > 0
	if report.Summary.Score.Conclusive {
		percent := compatible * 100 / known
		report.Summary.Score.Percent = &percent
	}
	if report.Summary.Findings > MaxFindings {
		return ReportLimitError()
	}
	return nil
}

func ReportLimitError() error {
	return fmt.Errorf("%w: report exceeds %d findings", errLimit, MaxFindings)
}

func Render(report Report, format Format) ([]byte, error) {
	var output []byte
	var err error
	switch format {
	case FormatJSON:
		output, err = json.MarshalIndent(report, "", "  ")
		if err == nil {
			output = append(output, '\n')
		}
	case FormatMarkdown:
		output = renderMarkdown(report)
	default:
		return nil, fmt.Errorf("unsupported output format %q; use json or markdown", format)
	}
	if err != nil {
		return nil, errors.New("encode Cypher compatibility report")
	}
	if len(output) > MaxOutputBytes {
		return nil, fmt.Errorf("%w: output exceeds %d bytes", errLimit, MaxOutputBytes)
	}
	return output, nil
}

func renderMarkdown(report Report) []byte {
	var output bytes.Buffer
	fmt.Fprintf(&output, "# Cypher Compatibility Report\n\n")
	fmt.Fprintf(&output, "- Apache AGE target: `%s`\n", report.Target.AGE)
	fmt.Fprintf(&output, "- Files: %d\n- Queries: %d\n\n", report.Summary.Files, report.Summary.Queries)
	fmt.Fprintln(&output, "## Queries")
	fmt.Fprintln(&output)
	fmt.Fprintln(&output, "| File | Query | Location | Classification |")
	fmt.Fprintln(&output, "|---|---:|---:|---|")
	for _, query := range report.Queries {
		fmt.Fprintf(&output, "| `%s` | %d | %d:%d | %s |\n",
			markdownCell(query.File), query.Number, query.Line, query.Column,
			query.Classification)
	}
	fmt.Fprintln(&output)
	fmt.Fprintln(&output, "## Findings")
	fmt.Fprintln(&output)
	fmt.Fprintln(&output, "| File | Query | Location | Severity | Rule | Evidence | Remediation |")
	fmt.Fprintln(&output, "|---|---:|---:|---|---|---|---|")
	for _, query := range report.Queries {
		for _, finding := range query.Findings {
			fmt.Fprintf(&output, "| `%s` | %d | %d:%d | %s | `%s` | %s | %s |\n",
				markdownCell(query.File), query.Number, finding.Line,
				finding.Column, finding.Severity, finding.Code,
				markdownCell(finding.Evidence),
				markdownCell(finding.Remediation))
		}
	}
	score := "unknown"
	if report.Summary.Score.Percent != nil {
		score = fmt.Sprintf("%d%%", *report.Summary.Score.Percent)
	}
	fmt.Fprintln(&output)
	fmt.Fprintln(&output, "## Summary")
	fmt.Fprintln(&output)
	fmt.Fprintf(&output,
		"- Compatible: %d\n- Compatible with manual change: %d\n- Unsupported: %d\n- Unknown: %d\n- Warnings: %d\n- Compatibility score: %s\n- Conclusive: %t\n",
		report.Summary.Compatible,
		report.Summary.CompatibleWithManualChange, report.Summary.Unsupported,
		report.Summary.Unknown, report.Summary.Warnings, score,
		report.Summary.Score.Conclusive)
	return output.Bytes()
}

func markdownCell(value string) string {
	value = strings.ReplaceAll(value, "\\", "\\\\")
	value = strings.ReplaceAll(value, "|", "\\|")
	value = strings.ReplaceAll(value, "`", "&#96;")
	return strings.ReplaceAll(value, "\n", " ")
}

func StrictFailure(report Report) bool {
	return report.Summary.Unsupported > 0 || report.Summary.Unknown > 0
}
