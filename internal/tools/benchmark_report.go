package tools

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"
)

const (
	BenchmarkReportSchemaVersion = 1
	BenchmarkReportMaxInputBytes = 16 << 20
	BenchmarkReportMaxRecords    = 100_000
	BenchmarkReportMaxInputFiles = 64
)

type BenchmarkReportFormat string

const (
	BenchmarkReportJSON     BenchmarkReportFormat = "json"
	BenchmarkReportMarkdown BenchmarkReportFormat = "markdown"
)

type BenchmarkReportOptions struct {
	GeneratedAt *time.Time
}

type BenchmarkIntegerSummary struct {
	Median int64 `json:"median"`
	Min    int64 `json:"min"`
	Max    int64 `json:"max"`
}

type BenchmarkFloatSummary struct {
	Median float64 `json:"median"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
}

type BenchmarkReportGroup struct {
	Workload      BenchmarkWorkload       `json:"workload"`
	Rows          int                     `json:"rows"`
	PropertyBytes int                     `json:"propertyBytes"`
	Strategy      BenchmarkStrategy       `json:"strategy"`
	SampleCount   int                     `json:"sampleCount"`
	ElapsedNanos  BenchmarkIntegerSummary `json:"elapsedNanos"`
	RowsPerSecond BenchmarkFloatSummary   `json:"rowsPerSecond"`
	WALBytes      BenchmarkIntegerSummary `json:"walBytes"`
}

type BenchmarkReport struct {
	SchemaVersion int                    `json:"schemaVersion"`
	GeneratedAt   *time.Time             `json:"generatedAt,omitempty"`
	Groups        []BenchmarkReportGroup `json:"groups"`
}

// NormalizeBenchmarkReport parses and aggregates benchmark streams without
// adding time-dependent metadata.
func NormalizeBenchmarkReport(readers []io.Reader) (BenchmarkReport, error) {
	return NormalizeBenchmarkReportWithOptions(readers, BenchmarkReportOptions{})
}

// NormalizeBenchmarkReportWithOptions parses and aggregates benchmark streams.
// For even-sized integer samples, the median is the midpoint rounded down.
func NormalizeBenchmarkReportWithOptions(
	readers []io.Reader,
	options BenchmarkReportOptions,
) (BenchmarkReport, error) {
	if len(readers) == 0 {
		return BenchmarkReport{}, errors.New("at least one benchmark input is required")
	}

	var raw []BulkBenchmarkResult
	var bytesRead int64
	for index, reader := range readers {
		if reader == nil {
			return BenchmarkReport{}, fmt.Errorf("benchmark input %d is nil", index+1)
		}
		counted := &benchmarkReportCountingReader{
			reader: reader,
			total:  &bytesRead,
		}
		records, err := parseBenchmarkStream(counted, len(raw))
		if bytesRead > BenchmarkReportMaxInputBytes {
			return BenchmarkReport{}, fmt.Errorf(
				"benchmark input exceeds %d bytes",
				BenchmarkReportMaxInputBytes,
			)
		}
		if err != nil {
			return BenchmarkReport{}, fmt.Errorf("benchmark input %d: %w", index+1, err)
		}
		raw = append(raw, records...)
	}
	if len(raw) == 0 {
		return BenchmarkReport{}, errors.New("benchmark input contains no records")
	}

	report := BenchmarkReport{
		SchemaVersion: BenchmarkReportSchemaVersion,
		Groups:        aggregateBenchmarkRecords(raw),
	}
	if options.GeneratedAt != nil {
		generatedAt := options.GeneratedAt.UTC()
		report.GeneratedAt = &generatedAt
	}
	return report, nil
}

func parseBenchmarkStream(
	reader io.Reader,
	recordsSeen int,
) ([]BulkBenchmarkResult, error) {
	buffered := bufio.NewReader(reader)
	first, err := firstNonSpaceByte(buffered)
	if errors.Is(err, io.EOF) {
		return nil, errors.New("input is empty")
	}
	if err != nil {
		return nil, err
	}

	decoder := json.NewDecoder(buffered)
	records := make([]BulkBenchmarkResult, 0)
	add := func(record BulkBenchmarkResult) error {
		recordNumber := recordsSeen + len(records) + 1
		if recordNumber > BenchmarkReportMaxRecords {
			return fmt.Errorf(
				"benchmark input exceeds %d records",
				BenchmarkReportMaxRecords,
			)
		}
		if err := validateRawBenchmarkRecord(record); err != nil {
			return fmt.Errorf("record %d: %w", recordNumber, err)
		}
		records = append(records, record)
		return nil
	}

	if first == '[' {
		if _, err := decoder.Token(); err != nil {
			return nil, fmt.Errorf("decode array start: %w", err)
		}
		for decoder.More() {
			record, err := decodeBenchmarkRecord(decoder)
			if err != nil {
				return nil, fmt.Errorf("decode record %d: %w", recordsSeen+len(records)+1, err)
			}
			if err := add(record); err != nil {
				return nil, err
			}
		}
		token, err := decoder.Token()
		if err != nil {
			return nil, fmt.Errorf("decode array end: %w", err)
		}
		if token != json.Delim(']') {
			return nil, errors.New("benchmark array is not terminated")
		}
		if err := requireJSONEOF(decoder); err != nil {
			return nil, err
		}
		return records, nil
	}

	for {
		record, err := decodeBenchmarkRecord(decoder)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("decode record %d: %w", recordsSeen+len(records)+1, err)
		}
		if err := add(record); err != nil {
			return nil, err
		}
	}
	return records, nil
}

var benchmarkRecordFields = []string{
	"workload",
	"strategy",
	"rows",
	"propertyBytes",
	"elapsedNanos",
	"rowsPerSecond",
	"walBytes",
}

func decodeBenchmarkRecord(decoder *json.Decoder) (BulkBenchmarkResult, error) {
	start, err := decoder.Token()
	if err != nil {
		return BulkBenchmarkResult{}, err
	}
	if start != json.Delim('{') {
		return BulkBenchmarkResult{}, errors.New("benchmark record must be an object")
	}
	fields := make(map[string]json.RawMessage, len(benchmarkRecordFields))
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return BulkBenchmarkResult{}, err
		}
		name, ok := token.(string)
		if !ok {
			return BulkBenchmarkResult{}, errors.New("benchmark field name must be a string")
		}
		if !slices.Contains(benchmarkRecordFields, name) {
			return BulkBenchmarkResult{}, fmt.Errorf("unknown field %q", name)
		}
		if _, exists := fields[name]; exists {
			return BulkBenchmarkResult{}, fmt.Errorf("duplicate field %q", name)
		}
		var raw json.RawMessage
		if err := decoder.Decode(&raw); err != nil {
			return BulkBenchmarkResult{}, fmt.Errorf("decode field %q: %w", name, err)
		}
		if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return BulkBenchmarkResult{}, fmt.Errorf("field %q cannot be null", name)
		}
		fields[name] = raw
	}
	end, err := decoder.Token()
	if err != nil {
		return BulkBenchmarkResult{}, err
	}
	if end != json.Delim('}') {
		return BulkBenchmarkResult{}, errors.New("benchmark record is not terminated")
	}
	for _, name := range benchmarkRecordFields {
		if _, exists := fields[name]; !exists {
			return BulkBenchmarkResult{}, fmt.Errorf("missing required field %q", name)
		}
	}

	var record BulkBenchmarkResult
	destinations := map[string]any{
		"workload":      &record.Workload,
		"strategy":      &record.Strategy,
		"rows":          &record.Rows,
		"propertyBytes": &record.PropertyBytes,
		"elapsedNanos":  &record.ElapsedNanos,
		"rowsPerSecond": &record.RowsPerSecond,
		"walBytes":      &record.WALBytes,
	}
	for _, name := range benchmarkRecordFields {
		if err := json.Unmarshal(fields[name], destinations[name]); err != nil {
			return BulkBenchmarkResult{}, fmt.Errorf("decode field %q: %w", name, err)
		}
	}
	return record, nil
}

func firstNonSpaceByte(reader *bufio.Reader) (byte, error) {
	for {
		value, err := reader.ReadByte()
		if err != nil {
			return 0, err
		}
		switch value {
		case ' ', '\t', '\r', '\n':
			continue
		default:
			if err := reader.UnreadByte(); err != nil {
				return 0, fmt.Errorf("inspect benchmark input: %w", err)
			}
			return value, nil
		}
	}
}

func requireJSONEOF(decoder *json.Decoder) error {
	var trailing json.RawMessage
	err := decoder.Decode(&trailing)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("invalid trailing content: %w", err)
	}
	return errors.New("unexpected content after benchmark array")
}

func validateRawBenchmarkRecord(record BulkBenchmarkResult) error {
	switch record.Workload {
	case BenchmarkVertices, BenchmarkEdges:
	default:
		return fmt.Errorf("unsupported benchmark workload %q", record.Workload)
	}
	switch record.Strategy {
	case BenchmarkDirect, BenchmarkStaged, BenchmarkRelational:
	default:
		return fmt.Errorf("unsupported benchmark strategy %q", record.Strategy)
	}
	if record.Rows <= 0 {
		return errors.New("rows must be positive")
	}
	if record.PropertyBytes < 0 {
		return errors.New("propertyBytes cannot be negative")
	}
	if record.ElapsedNanos <= 0 {
		return errors.New("elapsedNanos must be positive")
	}
	if math.IsNaN(record.RowsPerSecond) ||
		math.IsInf(record.RowsPerSecond, 0) ||
		record.RowsPerSecond <= 0 {
		return errors.New("rowsPerSecond must be finite and positive")
	}
	if record.WALBytes < 0 {
		return errors.New("walBytes cannot be negative")
	}

	expected := float64(record.Rows) /
		(float64(record.ElapsedNanos) / float64(time.Second))
	tolerance := math.Abs(expected) * 1e-9
	if math.Abs(record.RowsPerSecond-expected) > tolerance {
		return fmt.Errorf(
			"rowsPerSecond %.17g is inconsistent with rows and elapsedNanos (expected %.17g)",
			record.RowsPerSecond,
			expected,
		)
	}
	return nil
}

type benchmarkReportKey struct {
	workload      BenchmarkWorkload
	rows          int
	propertyBytes int
	strategy      BenchmarkStrategy
}

func aggregateBenchmarkRecords(records []BulkBenchmarkResult) []BenchmarkReportGroup {
	grouped := make(map[benchmarkReportKey][]BulkBenchmarkResult)
	for _, record := range records {
		key := benchmarkReportKey{
			workload:      record.Workload,
			rows:          record.Rows,
			propertyBytes: record.PropertyBytes,
			strategy:      record.Strategy,
		}
		grouped[key] = append(grouped[key], record)
	}

	groups := make([]BenchmarkReportGroup, 0, len(grouped))
	for key, samples := range grouped {
		elapsed := make([]int64, len(samples))
		throughput := make([]float64, len(samples))
		wal := make([]int64, len(samples))
		for index, sample := range samples {
			elapsed[index] = sample.ElapsedNanos
			throughput[index] = sample.RowsPerSecond
			wal[index] = sample.WALBytes
		}
		slices.Sort(elapsed)
		slices.Sort(throughput)
		slices.Sort(wal)
		groups = append(groups, BenchmarkReportGroup{
			Workload:      key.workload,
			Rows:          key.rows,
			PropertyBytes: key.propertyBytes,
			Strategy:      key.strategy,
			SampleCount:   len(samples),
			ElapsedNanos:  summarizeIntegers(elapsed),
			RowsPerSecond: summarizeFloats(throughput),
			WALBytes:      summarizeIntegers(wal),
		})
	}
	sortBenchmarkReportGroups(groups)
	return groups
}

func summarizeIntegers(sorted []int64) BenchmarkIntegerSummary {
	middle := len(sorted) / 2
	median := sorted[middle]
	if len(sorted)%2 == 0 {
		lower, upper := sorted[middle-1], sorted[middle]
		median = lower + (upper-lower)/2
	}
	return BenchmarkIntegerSummary{
		Median: median,
		Min:    sorted[0],
		Max:    sorted[len(sorted)-1],
	}
}

func summarizeFloats(sorted []float64) BenchmarkFloatSummary {
	middle := len(sorted) / 2
	median := sorted[middle]
	if len(sorted)%2 == 0 {
		lower, upper := sorted[middle-1], sorted[middle]
		median = lower + (upper-lower)/2
	}
	return BenchmarkFloatSummary{
		Median: median,
		Min:    sorted[0],
		Max:    sorted[len(sorted)-1],
	}
}

func sortBenchmarkReportGroups(groups []BenchmarkReportGroup) {
	slices.SortFunc(groups, func(left, right BenchmarkReportGroup) int {
		if comparison := strings.Compare(string(left.Workload), string(right.Workload)); comparison != 0 {
			return comparison
		}
		if left.Rows != right.Rows {
			if left.Rows < right.Rows {
				return -1
			}
			return 1
		}
		if left.PropertyBytes != right.PropertyBytes {
			if left.PropertyBytes < right.PropertyBytes {
				return -1
			}
			return 1
		}
		return strings.Compare(string(left.Strategy), string(right.Strategy))
	})
}

// WriteBenchmarkReport validates and writes a report in canonical group order.
func WriteBenchmarkReport(
	writer io.Writer,
	report BenchmarkReport,
	format BenchmarkReportFormat,
) error {
	if writer == nil {
		return errors.New("benchmark report writer is nil")
	}
	canonical, err := canonicalBenchmarkReport(report)
	if err != nil {
		return err
	}

	switch format {
	case BenchmarkReportJSON:
		output, err := json.MarshalIndent(canonical, "", "  ")
		if err != nil {
			return fmt.Errorf("encode benchmark report JSON: %w", err)
		}
		output = append(output, '\n')
		if err := writeBenchmarkReportOutput(writer, output); err != nil {
			return fmt.Errorf("write benchmark report JSON: %w", err)
		}
		return nil
	case BenchmarkReportMarkdown:
		if err := writeBenchmarkReportOutput(
			writer,
			[]byte(formatBenchmarkReportMarkdown(canonical)),
		); err != nil {
			return fmt.Errorf("write benchmark report Markdown: %w", err)
		}
		return nil
	default:
		return fmt.Errorf("unsupported benchmark report format %q", format)
	}
}

func writeBenchmarkReportOutput(writer io.Writer, output []byte) error {
	written, err := writer.Write(output)
	if err != nil {
		return err
	}
	if written != len(output) {
		return io.ErrShortWrite
	}
	return nil
}

func canonicalBenchmarkReport(report BenchmarkReport) (BenchmarkReport, error) {
	if report.SchemaVersion != BenchmarkReportSchemaVersion {
		return BenchmarkReport{}, fmt.Errorf(
			"unsupported benchmark report schema version %d",
			report.SchemaVersion,
		)
	}
	if len(report.Groups) == 0 {
		return BenchmarkReport{}, errors.New("benchmark report contains no groups")
	}

	canonical := report
	canonical.Groups = slices.Clone(report.Groups)
	sortBenchmarkReportGroups(canonical.Groups)
	if report.GeneratedAt != nil {
		if report.GeneratedAt.IsZero() {
			return BenchmarkReport{}, errors.New("generatedAt cannot be zero")
		}
		generatedAt := report.GeneratedAt.UTC()
		canonical.GeneratedAt = &generatedAt
	}

	seen := make(map[benchmarkReportKey]struct{}, len(canonical.Groups))
	for index, group := range canonical.Groups {
		record := BulkBenchmarkResult{
			Workload:      group.Workload,
			Strategy:      group.Strategy,
			Rows:          group.Rows,
			PropertyBytes: group.PropertyBytes,
			ElapsedNanos:  group.ElapsedNanos.Median,
			RowsPerSecond: group.RowsPerSecond.Median,
			WALBytes:      group.WALBytes.Median,
		}
		if group.SampleCount <= 0 {
			return BenchmarkReport{}, fmt.Errorf("group %d: sampleCount must be positive", index+1)
		}
		if err := validateReportGroup(record, group); err != nil {
			return BenchmarkReport{}, fmt.Errorf("group %d: %w", index+1, err)
		}
		key := benchmarkReportKey{
			workload:      group.Workload,
			rows:          group.Rows,
			propertyBytes: group.PropertyBytes,
			strategy:      group.Strategy,
		}
		if _, exists := seen[key]; exists {
			return BenchmarkReport{}, fmt.Errorf("group %d duplicates a benchmark group", index+1)
		}
		seen[key] = struct{}{}
	}
	return canonical, nil
}

func validateReportGroup(record BulkBenchmarkResult, group BenchmarkReportGroup) error {
	switch record.Workload {
	case BenchmarkVertices, BenchmarkEdges:
	default:
		return fmt.Errorf("unsupported benchmark workload %q", record.Workload)
	}
	switch record.Strategy {
	case BenchmarkDirect, BenchmarkStaged, BenchmarkRelational:
	default:
		return fmt.Errorf("unsupported benchmark strategy %q", record.Strategy)
	}
	if record.Rows <= 0 {
		return errors.New("rows must be positive")
	}
	if record.PropertyBytes < 0 {
		return errors.New("propertyBytes cannot be negative")
	}
	if err := validateIntegerSummary("elapsedNanos", group.ElapsedNanos, true); err != nil {
		return err
	}
	if err := validateFloatSummary("rowsPerSecond", group.RowsPerSecond); err != nil {
		return err
	}
	if err := validateIntegerSummary("walBytes", group.WALBytes, false); err != nil {
		return err
	}
	return nil
}

func validateIntegerSummary(
	name string,
	summary BenchmarkIntegerSummary,
	positive bool,
) error {
	if positive && summary.Min <= 0 {
		return fmt.Errorf("%s values must be positive", name)
	}
	if !positive && summary.Min < 0 {
		return fmt.Errorf("%s values cannot be negative", name)
	}
	if summary.Min > summary.Median || summary.Median > summary.Max {
		return fmt.Errorf("%s values must satisfy min <= median <= max", name)
	}
	return nil
}

func validateFloatSummary(name string, summary BenchmarkFloatSummary) error {
	values := []float64{summary.Min, summary.Median, summary.Max}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) || value <= 0 {
			return fmt.Errorf("%s values must be finite and positive", name)
		}
	}
	if summary.Min > summary.Median || summary.Median > summary.Max {
		return fmt.Errorf("%s values must satisfy min <= median <= max", name)
	}
	return nil
}

func formatBenchmarkReportMarkdown(report BenchmarkReport) string {
	var output strings.Builder
	output.WriteString("# Benchmark Report\n\n")
	fmt.Fprintf(&output, "Schema version: %d\n", report.SchemaVersion)
	if report.GeneratedAt != nil {
		fmt.Fprintf(
			&output,
			"\nGenerated at: %s\n",
			report.GeneratedAt.Format(time.RFC3339Nano),
		)
	}
	output.WriteString(
		"\n| Workload | Rows | Property bytes | Strategy | Samples | " +
			"Elapsed ns median | Elapsed ns min | Elapsed ns max | " +
			"Rows/s median | Rows/s min | Rows/s max | " +
			"WAL bytes median | WAL bytes min | WAL bytes max |\n",
	)
	output.WriteString(
		"|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
	)
	for _, group := range report.Groups {
		fmt.Fprintf(
			&output,
			"| %s | %d | %d | %s | %d | %d | %d | %d | %s | %s | %s | %d | %d | %d |\n",
			escapeMarkdownTableText(string(group.Workload)),
			group.Rows,
			group.PropertyBytes,
			escapeMarkdownTableText(string(group.Strategy)),
			group.SampleCount,
			group.ElapsedNanos.Median,
			group.ElapsedNanos.Min,
			group.ElapsedNanos.Max,
			formatBenchmarkThroughput(group.RowsPerSecond.Median),
			formatBenchmarkThroughput(group.RowsPerSecond.Min),
			formatBenchmarkThroughput(group.RowsPerSecond.Max),
			group.WALBytes.Median,
			group.WALBytes.Min,
			group.WALBytes.Max,
		)
	}
	return output.String()
}

func formatBenchmarkThroughput(value float64) string {
	return strconv.FormatFloat(value, 'f', 3, 64)
}

func escapeMarkdownTableText(value string) string {
	value = strings.ReplaceAll(value, `\`, `\\`)
	value = strings.ReplaceAll(value, "|", `\|`)
	value = strings.ReplaceAll(value, "\r", " ")
	return strings.ReplaceAll(value, "\n", " ")
}

type benchmarkReportCountingReader struct {
	reader io.Reader
	total  *int64
}

func (reader *benchmarkReportCountingReader) Read(buffer []byte) (int, error) {
	remaining := int64(BenchmarkReportMaxInputBytes) + 1 - *reader.total
	if remaining <= 0 {
		return 0, errors.New("benchmark input size limit exceeded")
	}
	if int64(len(buffer)) > remaining {
		buffer = buffer[:remaining]
	}
	count, err := reader.reader.Read(buffer)
	*reader.total += int64(count)
	if *reader.total > BenchmarkReportMaxInputBytes {
		return count, errors.New("benchmark input size limit exceeded")
	}
	return count, err
}
