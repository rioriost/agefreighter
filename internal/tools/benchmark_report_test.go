package tools

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func benchmarkJSON(
	workload BenchmarkWorkload,
	strategy BenchmarkStrategy,
	rows int,
	propertyBytes int,
	elapsedNanos int64,
	walBytes int64,
) string {
	return fmt.Sprintf(
		`{"workload":%q,"strategy":%q,"rows":%d,"propertyBytes":%d,"elapsedNanos":%d,"rowsPerSecond":%.17g,"walBytes":%d}`,
		workload,
		strategy,
		rows,
		propertyBytes,
		elapsedNanos,
		float64(rows)/(float64(elapsedNanos)/float64(time.Second)),
		walBytes,
	)
}

func TestNormalizeBenchmarkReportStreamsGroupingAndMedians(t *testing.T) {
	odd := []string{
		benchmarkJSON(BenchmarkVertices, BenchmarkDirect, 10, 4, 2_000_000_000, 30),
		benchmarkJSON(BenchmarkVertices, BenchmarkDirect, 10, 4, 1_000_000_000, 10),
		benchmarkJSON(BenchmarkVertices, BenchmarkDirect, 10, 4, 4_000_000_000, 20),
	}
	evenOne := benchmarkJSON(BenchmarkEdges, BenchmarkStaged, 8, 2, 4_000_000_000, 5)
	evenTwo := benchmarkJSON(BenchmarkEdges, BenchmarkStaged, 8, 2, 2_000_000_000, 8)

	report, err := NormalizeBenchmarkReport([]io.Reader{
		strings.NewReader(strings.Join(odd, "\n")),
		strings.NewReader("[" + evenOne + "," + evenTwo + "]"),
	})
	if err != nil {
		t.Fatalf("NormalizeBenchmarkReport() error = %v", err)
	}
	if report.SchemaVersion != BenchmarkReportSchemaVersion {
		t.Fatalf("SchemaVersion = %d", report.SchemaVersion)
	}
	if report.GeneratedAt != nil {
		t.Fatalf("GeneratedAt = %v, want nil", report.GeneratedAt)
	}
	if len(report.Groups) != 2 {
		t.Fatalf("groups = %#v", report.Groups)
	}

	edges := report.Groups[0]
	if edges.Workload != BenchmarkEdges || edges.SampleCount != 2 {
		t.Fatalf("edge group = %#v", edges)
	}
	if edges.ElapsedNanos != (BenchmarkIntegerSummary{
		Median: 3_000_000_000,
		Min:    2_000_000_000,
		Max:    4_000_000_000,
	}) {
		t.Fatalf("edge elapsed = %#v", edges.ElapsedNanos)
	}
	if edges.RowsPerSecond != (BenchmarkFloatSummary{Median: 3, Min: 2, Max: 4}) {
		t.Fatalf("edge throughput = %#v", edges.RowsPerSecond)
	}
	if edges.WALBytes != (BenchmarkIntegerSummary{Median: 6, Min: 5, Max: 8}) {
		t.Fatalf("edge WAL = %#v", edges.WALBytes)
	}

	vertices := report.Groups[1]
	if vertices.SampleCount != 3 {
		t.Fatalf("vertex sample count = %d", vertices.SampleCount)
	}
	if vertices.ElapsedNanos != (BenchmarkIntegerSummary{
		Median: 2_000_000_000,
		Min:    1_000_000_000,
		Max:    4_000_000_000,
	}) {
		t.Fatalf("vertex elapsed = %#v", vertices.ElapsedNanos)
	}
	if vertices.RowsPerSecond != (BenchmarkFloatSummary{Median: 5, Min: 2.5, Max: 10}) {
		t.Fatalf("vertex throughput = %#v", vertices.RowsPerSecond)
	}
	if vertices.WALBytes != (BenchmarkIntegerSummary{Median: 20, Min: 10, Max: 30}) {
		t.Fatalf("vertex WAL = %#v", vertices.WALBytes)
	}
}

func TestNormalizeBenchmarkReportDeterministicSortAndTimestamp(t *testing.T) {
	inputs := []string{
		benchmarkJSON(BenchmarkVertices, BenchmarkStaged, 20, 1, 1_000_000_000, 1),
		benchmarkJSON(BenchmarkVertices, BenchmarkDirect, 10, 2, 1_000_000_000, 1),
		benchmarkJSON(BenchmarkVertices, BenchmarkDirect, 10, 1, 1_000_000_000, 1),
		benchmarkJSON(BenchmarkEdges, BenchmarkRelational, 30, 3, 1_000_000_000, 1),
	}
	generatedAt := time.Date(2026, 8, 27, 10, 4, 22, 123, time.FixedZone("JST", 9*60*60))
	report, err := NormalizeBenchmarkReportWithOptions(
		[]io.Reader{strings.NewReader(strings.Join(inputs, "\n"))},
		BenchmarkReportOptions{GeneratedAt: &generatedAt},
	)
	if err != nil {
		t.Fatalf("NormalizeBenchmarkReportWithOptions() error = %v", err)
	}
	if report.GeneratedAt == nil ||
		report.GeneratedAt.Format(time.RFC3339Nano) != "2026-08-27T01:04:22.000000123Z" {
		t.Fatalf("GeneratedAt = %v", report.GeneratedAt)
	}
	got := make([]string, len(report.Groups))
	for index, group := range report.Groups {
		got[index] = fmt.Sprintf(
			"%s/%d/%d/%s",
			group.Workload,
			group.Rows,
			group.PropertyBytes,
			group.Strategy,
		)
	}
	want := []string{
		"edges/30/3/plain-relational",
		"vertices/10/1/direct-text",
		"vertices/10/2/direct-text",
		"vertices/20/1/staged-binary",
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("group order = %v, want %v", got, want)
	}
}

func TestNormalizeBenchmarkReportValidation(t *testing.T) {
	valid := benchmarkJSON(BenchmarkVertices, BenchmarkDirect, 10, 1, 1_000_000_000, 1)
	tests := []struct {
		name  string
		input string
	}{
		{name: "empty", input: ""},
		{name: "whitespace", input: " \n\t"},
		{name: "empty array", input: "[]"},
		{name: "unknown field", input: strings.Replace(valid, `"walBytes":1`, `"walBytes":1,"dsn":"secret"`, 1)},
		{name: "missing field", input: strings.Replace(valid, `,"walBytes":1`, "", 1)},
		{name: "null field", input: strings.Replace(valid, `"walBytes":1`, `"walBytes":null`, 1)},
		{name: "duplicate field", input: strings.Replace(valid, `"rows":10`, `"rows":10,"rows":20`, 1)},
		{name: "incorrect field case", input: strings.Replace(valid, `"walBytes":1`, `"WalBytes":1`, 1)},
		{name: "malformed", input: valid[:len(valid)-1]},
		{name: "trailing after array", input: "[" + valid + "] true"},
		{name: "malformed trailing after array", input: "[" + valid + "] {"},
		{name: "mixed stream forms", input: valid + " [" + valid + "]"},
		{name: "null", input: "null"},
		{name: "bad workload", input: strings.Replace(valid, `"vertices"`, `"other"`, 1)},
		{name: "bad strategy", input: strings.Replace(valid, `"direct-text"`, `"other"`, 1)},
		{name: "zero rows", input: strings.Replace(valid, `"rows":10`, `"rows":0`, 1)},
		{name: "negative properties", input: strings.Replace(valid, `"propertyBytes":1`, `"propertyBytes":-1`, 1)},
		{name: "zero elapsed", input: strings.Replace(valid, `"elapsedNanos":1000000000`, `"elapsedNanos":0`, 1)},
		{name: "zero throughput", input: strings.Replace(valid, `"rowsPerSecond":10`, `"rowsPerSecond":0`, 1)},
		{name: "negative throughput", input: strings.Replace(valid, `"rowsPerSecond":10`, `"rowsPerSecond":-1`, 1)},
		{name: "underflow throughput", input: strings.Replace(valid, `"rowsPerSecond":10`, `"rowsPerSecond":1e-1000`, 1)},
		{name: "overflow throughput", input: strings.Replace(valid, `"rowsPerSecond":10`, `"rowsPerSecond":1e9999`, 1)},
		{name: "NaN throughput", input: strings.Replace(valid, `"rowsPerSecond":10`, `"rowsPerSecond":NaN`, 1)},
		{name: "negative WAL", input: strings.Replace(valid, `"walBytes":1`, `"walBytes":-1`, 1)},
		{name: "inconsistent throughput", input: strings.Replace(valid, `"rowsPerSecond":10`, `"rowsPerSecond":10.000001`, 1)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := NormalizeBenchmarkReport(
				[]io.Reader{strings.NewReader(test.input)},
			); err == nil {
				t.Fatal("NormalizeBenchmarkReport() succeeded")
			}
		})
	}

	if _, err := NormalizeBenchmarkReport(nil); err == nil {
		t.Fatal("NormalizeBenchmarkReport(nil) succeeded")
	}
	if _, err := NormalizeBenchmarkReport([]io.Reader{nil}); err == nil {
		t.Fatal("NormalizeBenchmarkReport() accepted nil reader")
	}
	if _, err := NormalizeBenchmarkReport([]io.Reader{
		strings.NewReader(valid),
		strings.NewReader(""),
	}); err == nil {
		t.Fatal("NormalizeBenchmarkReport() accepted an empty stream")
	}
	if _, err := NormalizeBenchmarkReport([]io.Reader{
		io.MultiReader(strings.NewReader(" "), failingReader{}),
	}); err == nil {
		t.Fatal("NormalizeBenchmarkReport() ignored read error")
	}

	slow := `{"workload":"vertices","strategy":"direct-text","rows":1,` +
		`"propertyBytes":0,"elapsedNanos":9223372036854775807,` +
		`"rowsPerSecond":1e-300,"walBytes":0}`
	if _, err := NormalizeBenchmarkReport([]io.Reader{strings.NewReader(slow)}); err == nil {
		t.Fatal("NormalizeBenchmarkReport() accepted materially incorrect low throughput")
	}

	tight := strings.Replace(valid, `"rowsPerSecond":10`, `"rowsPerSecond":10.000000009`, 1)
	if _, err := NormalizeBenchmarkReport([]io.Reader{strings.NewReader(tight)}); err != nil {
		t.Fatalf("tight floating tolerance rejected: %v", err)
	}
}

func TestNormalizeBenchmarkReportBounds(t *testing.T) {
	oversized := io.MultiReader(
		strings.NewReader(benchmarkJSON(
			BenchmarkVertices,
			BenchmarkDirect,
			1,
			1,
			1_000_000_000,
			1,
		)),
		io.LimitReader(spaceReader{}, BenchmarkReportMaxInputBytes),
	)
	if _, err := NormalizeBenchmarkReport([]io.Reader{oversized}); err == nil ||
		!strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("oversized input error = %v", err)
	}

	record := benchmarkJSON(
		BenchmarkVertices,
		BenchmarkDirect,
		1,
		1,
		1_000_000_000,
		1,
	)
	if _, err := NormalizeBenchmarkReport([]io.Reader{
		&repeatingRecordReader{record: []byte(record), remaining: BenchmarkReportMaxRecords + 1},
	}); err == nil || !strings.Contains(err.Error(), "records") {
		t.Fatalf("record limit error = %v", err)
	}
}

type spaceReader struct{}

func (spaceReader) Read(buffer []byte) (int, error) {
	for index := range buffer {
		buffer[index] = ' '
	}
	return len(buffer), nil
}

type repeatingRecordReader struct {
	record    []byte
	remaining int
	offset    int
}

func (reader *repeatingRecordReader) Read(buffer []byte) (int, error) {
	if reader.remaining == 0 {
		return 0, io.EOF
	}
	written := 0
	for written < len(buffer) && reader.remaining > 0 {
		if reader.offset == len(reader.record) {
			buffer[written] = '\n'
			written++
			reader.offset = 0
			reader.remaining--
			continue
		}
		count := copy(buffer[written:], reader.record[reader.offset:])
		written += count
		reader.offset += count
	}
	return written, nil
}

func TestWriteBenchmarkReportCanonicalJSON(t *testing.T) {
	report := mustReport(t)
	report.Groups[0], report.Groups[1] = report.Groups[1], report.Groups[0]

	var first, second bytes.Buffer
	if err := WriteBenchmarkReport(&first, report, BenchmarkReportJSON); err != nil {
		t.Fatalf("WriteBenchmarkReport() error = %v", err)
	}
	if err := WriteBenchmarkReport(&second, report, BenchmarkReportJSON); err != nil {
		t.Fatalf("WriteBenchmarkReport() second error = %v", err)
	}
	if first.String() != second.String() {
		t.Fatal("JSON output is nondeterministic")
	}
	want := `{
  "schemaVersion": 1,
  "groups": [
    {
      "workload": "edges",
      "rows": 8,
      "propertyBytes": 2,
      "strategy": "staged-binary",
      "sampleCount": 1,
      "elapsedNanos": {
        "median": 2000000000,
        "min": 2000000000,
        "max": 2000000000
      },
      "rowsPerSecond": {
        "median": 4,
        "min": 4,
        "max": 4
      },
      "walBytes": {
        "median": 8,
        "min": 8,
        "max": 8
      }
    },
    {
      "workload": "vertices",
      "rows": 10,
      "propertyBytes": 4,
      "strategy": "direct-text",
      "sampleCount": 1,
      "elapsedNanos": {
        "median": 1000000000,
        "min": 1000000000,
        "max": 1000000000
      },
      "rowsPerSecond": {
        "median": 10,
        "min": 10,
        "max": 10
      },
      "walBytes": {
        "median": 10,
        "min": 10,
        "max": 10
      }
    }
  ]
}
`
	if first.String() != want {
		t.Fatalf("JSON output:\n%s\nwant:\n%s", first.String(), want)
	}
}

func TestWriteBenchmarkReportMarkdown(t *testing.T) {
	report := mustReport(t)
	generatedAt := time.Date(2026, 1, 2, 3, 4, 5, 600, time.UTC)
	report.GeneratedAt = &generatedAt
	var output bytes.Buffer
	if err := WriteBenchmarkReport(&output, report, BenchmarkReportMarkdown); err != nil {
		t.Fatalf("WriteBenchmarkReport() error = %v", err)
	}
	text := output.String()
	for _, want := range []string{
		"# Benchmark Report",
		"Schema version: 1",
		"Generated at: 2026-01-02T03:04:05.0000006Z",
		"| edges | 8 | 2 | staged-binary | 1 |",
		"| vertices | 10 | 4 | direct-text | 1 |",
		"| 10.000 | 10.000 | 10.000 |",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("Markdown missing %q:\n%s", want, text)
		}
	}
	if got := escapeMarkdownTableText("a|b\\c\r\nd"); got != `a\|b\\c  d` {
		t.Fatalf("escapeMarkdownTableText() = %q", got)
	}
	if got := formatBenchmarkThroughput(1.23456); got != "1.235" {
		t.Fatalf("formatBenchmarkThroughput() = %q", got)
	}
}

func TestWriteBenchmarkReportErrors(t *testing.T) {
	valid := mustReport(t)
	tests := []struct {
		name   string
		change func(*BenchmarkReport)
	}{
		{name: "schema", change: func(report *BenchmarkReport) { report.SchemaVersion = 2 }},
		{name: "no groups", change: func(report *BenchmarkReport) { report.Groups = nil }},
		{name: "zero timestamp", change: func(report *BenchmarkReport) {
			value := time.Time{}
			report.GeneratedAt = &value
		}},
		{name: "samples", change: func(report *BenchmarkReport) { report.Groups[0].SampleCount = 0 }},
		{name: "workload", change: func(report *BenchmarkReport) { report.Groups[0].Workload = "bad" }},
		{name: "strategy", change: func(report *BenchmarkReport) { report.Groups[0].Strategy = "bad" }},
		{name: "rows", change: func(report *BenchmarkReport) { report.Groups[0].Rows = 0 }},
		{name: "properties", change: func(report *BenchmarkReport) { report.Groups[0].PropertyBytes = -1 }},
		{name: "elapsed nonpositive", change: func(report *BenchmarkReport) { report.Groups[0].ElapsedNanos.Min = 0 }},
		{name: "elapsed order", change: func(report *BenchmarkReport) { report.Groups[0].ElapsedNanos.Median = math.MaxInt64 }},
		{name: "throughput nonfinite", change: func(report *BenchmarkReport) { report.Groups[0].RowsPerSecond.Max = math.Inf(1) }},
		{name: "throughput order", change: func(report *BenchmarkReport) { report.Groups[0].RowsPerSecond.Median = 0.5 }},
		{name: "WAL negative", change: func(report *BenchmarkReport) { report.Groups[0].WALBytes.Min = -1 }},
		{name: "WAL order", change: func(report *BenchmarkReport) { report.Groups[0].WALBytes.Median = math.MaxInt64 }},
		{name: "duplicate", change: func(report *BenchmarkReport) { report.Groups = append(report.Groups, report.Groups[0]) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			report := valid
			report.Groups = append([]BenchmarkReportGroup(nil), valid.Groups...)
			test.change(&report)
			if err := WriteBenchmarkReport(io.Discard, report, BenchmarkReportJSON); err == nil {
				t.Fatal("WriteBenchmarkReport() succeeded")
			}
		})
	}

	if err := WriteBenchmarkReport(nil, valid, BenchmarkReportJSON); err == nil {
		t.Fatal("WriteBenchmarkReport() accepted nil writer")
	}
	if err := WriteBenchmarkReport(io.Discard, valid, "csv"); err == nil {
		t.Fatal("WriteBenchmarkReport() accepted unknown format")
	}
	for _, format := range []BenchmarkReportFormat{BenchmarkReportJSON, BenchmarkReportMarkdown} {
		if err := WriteBenchmarkReport(failingWriter{}, valid, format); err == nil {
			t.Fatalf("WriteBenchmarkReport() ignored %s write error", format)
		}
		if err := WriteBenchmarkReport(shortWriter{}, valid, format); !errors.Is(err, io.ErrShortWrite) {
			t.Fatalf("WriteBenchmarkReport() short %s write error = %v", format, err)
		}
	}
}

func TestBenchmarkReportCommand(t *testing.T) {
	root := t.TempDir()
	first := filepath.Join(root, "first.json")
	second := filepath.Join(root, "second.json")
	if err := os.WriteFile(first, []byte(benchmarkJSON(
		BenchmarkVertices, BenchmarkDirect, 10, 4, 1_000_000_000, 10,
	)), 0o600); err != nil {
		t.Fatalf("write first input: %v", err)
	}
	if err := os.WriteFile(second, []byte(benchmarkJSON(
		BenchmarkVertices, BenchmarkDirect, 10, 4, 2_000_000_000, 20,
	)), 0o600); err != nil {
		t.Fatalf("write second input: %v", err)
	}

	var output bytes.Buffer
	command := NewBenchmarkReportCommand()
	command.SetOut(&output)
	command.SetArgs([]string{first, second})
	if err := command.Execute(); err != nil {
		t.Fatalf("benchmark-report command: %v", err)
	}
	if !strings.Contains(output.String(), `"sampleCount": 2`) {
		t.Fatalf("benchmark-report output = %q", output.String())
	}

	budgetInput := filepath.Join(root, "budget.jsonl")
	if err := os.WriteFile(
		budgetInput,
		[]byte(
			benchmarkJSON(BenchmarkVertices, BenchmarkStaged, 10, 4, 2_000_000_000, 10)+"\n"+
				benchmarkJSON(BenchmarkVertices, BenchmarkRelational, 10, 4, 1_000_000_000, 10),
		),
		0o600,
	); err != nil {
		t.Fatalf("write budget input: %v", err)
	}
	command = NewBenchmarkReportCommand()
	command.SetOut(io.Discard)
	command.SetArgs([]string{"--minimum-staged-ratio", "0.5", budgetInput})
	if err := command.Execute(); err != nil {
		t.Fatalf("passing benchmark budget: %v", err)
	}
	command = NewBenchmarkReportCommand()
	command.SetOut(io.Discard)
	command.SetArgs([]string{"--minimum-staged-ratio", "0.6", budgetInput})
	if err := command.Execute(); err == nil ||
		!strings.Contains(err.Error(), "benchmark budget") {
		t.Fatalf("failing benchmark budget error = %v", err)
	}

	for _, test := range []struct {
		name   string
		args   []string
		stdin  string
		writer io.Writer
	}{
		{name: "missing file", args: []string{filepath.Join(root, "missing")}},
		{name: "duplicate stdin", args: []string{"-", "-"}, stdin: benchmarkJSON(
			BenchmarkVertices, BenchmarkDirect, 1, 0, 1_000_000_000, 0,
		)},
		{name: "unknown format", args: []string{"--format", "csv", first}},
		{name: "output error", args: []string{first}, writer: failingWriter{}},
		{name: "invalid input", stdin: "{}"},
	} {
		t.Run(test.name, func(t *testing.T) {
			command := NewBenchmarkReportCommand()
			command.SetIn(strings.NewReader(test.stdin))
			if test.writer != nil {
				command.SetOut(test.writer)
			} else {
				command.SetOut(io.Discard)
			}
			command.SetArgs(test.args)
			if err := command.Execute(); err == nil {
				t.Fatal("benchmark-report command succeeded")
			}
		})
	}

	tooMany := make([]string, BenchmarkReportMaxInputFiles+1)
	for index := range tooMany {
		tooMany[index] = "not-opened"
	}
	command = NewBenchmarkReportCommand()
	command.SetOut(io.Discard)
	command.SetArgs(tooMany)
	if err := command.Execute(); err == nil || !strings.Contains(err.Error(), "at most") {
		t.Fatalf("input file limit error = %v", err)
	}

	command = NewBenchmarkReportCommand()
	command.SetIn(strings.NewReader(""))
	command.SetOut(io.Discard)
	command.SetArgs([]string{"--format", "csv"})
	if err := command.Execute(); err == nil ||
		!strings.Contains(err.Error(), "unsupported benchmark report format") {
		t.Fatalf("format validation order error = %v", err)
	}
}

func TestCloseBenchmarkInputsReportsCloseFailure(t *testing.T) {
	file, err := os.CreateTemp(t.TempDir(), "closed")
	if err != nil {
		t.Fatalf("create temporary file: %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("close temporary file: %v", err)
	}
	if err := closeBenchmarkInputs([]*os.File{file}); err == nil {
		t.Fatal("closeBenchmarkInputs() ignored close error")
	}
}

func mustReport(t *testing.T) BenchmarkReport {
	t.Helper()
	report, err := NormalizeBenchmarkReport([]io.Reader{
		strings.NewReader("[" +
			benchmarkJSON(BenchmarkVertices, BenchmarkDirect, 10, 4, 1_000_000_000, 10) +
			"," +
			benchmarkJSON(BenchmarkEdges, BenchmarkStaged, 8, 2, 2_000_000_000, 8) +
			"]"),
	})
	if err != nil {
		t.Fatalf("NormalizeBenchmarkReport() error = %v", err)
	}
	return report
}

type failingWriter struct{}

func (failingWriter) Write([]byte) (int, error) {
	return 0, errors.New("write failed")
}

type shortWriter struct{}

func (shortWriter) Write(buffer []byte) (int, error) {
	return len(buffer) - 1, nil
}

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) {
	return 0, errors.New("read failed")
}
