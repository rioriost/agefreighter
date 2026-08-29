package csv

import (
	"bufio"
	"compress/gzip"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
)

func TestParserOptimizedAndCallbackBranches(t *testing.T) {
	if _, err := NewParser(strings.NewReader(""), ParserOptions{
		Delimiter: ',', Quote: '\'', Escape: '\'',
		MaxRecordBytes: 10, OptimizeRFC4180: true,
	}); err == nil {
		t.Fatal("NewParser() accepted incompatible optimized quoting")
	}

	parser, err := NewParser(strings.NewReader("a\n"), ParserOptions{
		Delimiter: ',', Quote: '"', Escape: '"', MaxRecordBytes: 10,
		OnInputBytes: func(int64) error { return errors.New("budget failed") },
	})
	if err != nil {
		t.Fatal(err)
	}
	fields, _, err := parser.ReadRecord(t.Context())
	if err == nil || fields != nil || !strings.Contains(err.Error(), "budget failed") {
		t.Fatalf("ReadRecord() = %#v, %v", fields, err)
	}

	optimizedCases := []struct {
		name    string
		input   string
		maxSize int64
		max     int
		ctx     context.Context
	}{
		{"canceled", "a\n", 10, 10, canceledContext()},
		{"record bytes", "abcd\n", 3, 10, t.Context()},
		{"fields", "a,b,c\n", 100, 2, t.Context()},
		{"invalid UTF-8", string([]byte{'a', ',', 0xff, '\n'}), 100, 10, t.Context()},
		{"parse error", "\"unterminated", 100, 10, t.Context()},
	}
	for _, test := range optimizedCases {
		t.Run(test.name, func(t *testing.T) {
			parser, err := NewParser(strings.NewReader(test.input), ParserOptions{
				Delimiter: ',', Quote: '"', Escape: '"',
				MaxRecordBytes: test.maxSize, MaxFields: test.max,
				OptimizeRFC4180: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := parser.ReadRecord(test.ctx); err == nil {
				t.Fatal("ReadRecord() succeeded")
			}
		})
	}

	parser, err = NewParser(strings.NewReader("a\n"), ParserOptions{
		Delimiter: ',', Quote: '"', Escape: '"',
		MaxRecordBytes: 10, OptimizeRFC4180: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := parser.ReadRecord(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, _, err := parser.ReadRecord(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("final ReadRecord() error = %v", err)
	}
}

func TestParserPendingAndQuotedEOFBranches(t *testing.T) {
	parser, err := NewParser(strings.NewReader(`"a"`), ParserOptions{
		Delimiter: ',', Quote: '"', Escape: '"', MaxRecordBytes: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	fields, _, err := parser.ReadRecord(t.Context())
	if err != nil || strings.Join(fields, "|") != "a" {
		t.Fatalf("ReadRecord() = %#v, %v", fields, err)
	}
	if _, _, err := parser.ReadRecord(t.Context()); !errors.Is(err, io.EOF) {
		t.Fatalf("final ReadRecord() error = %v", err)
	}

	parser, err = NewParser(strings.NewReader("ab"), ParserOptions{
		Delimiter: ',', Quote: '"', Escape: '"', MaxRecordBytes: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	character, _, err := parser.readRune()
	if err != nil {
		t.Fatal(err)
	}
	if err := parser.unreadRune(character); err != nil {
		t.Fatal(err)
	}
	if _, _, err := parser.readRawRune(); err != nil {
		t.Fatal(err)
	}
	parser.canUnread = true
	parser.pending = true
	if err := parser.unreadRune('b'); err == nil {
		t.Fatal("unreadRune() accepted a second pending character")
	}
}

func TestParserRemainingErrorBranches(t *testing.T) {
	tests := []struct {
		name   string
		input  io.Reader
		quote  rune
		escape rune
		max    int
	}{
		{"EOF field limit", strings.NewReader("a,"), '"', '"', 1},
		{"quoted lookahead read", &errorAfterReader{data: []byte(`"a"`)}, '"', '"', 10},
		{"escape read", &errorAfterReader{data: []byte(`'a\`)}, '\'', '\\', 10},
		{"quoted CR read", &errorAfterReader{data: []byte("'a\r")}, '\'', '\\', 10},
		{"closed quote delimiter limit", strings.NewReader(`"a","b",`), '"', '"', 1},
		{"closed quote CR read", &errorAfterReader{data: []byte(`"a"` + "\r")}, '"', '"', 10},
		{"closed quote CR field limit", strings.NewReader("x,\"a\"\r"), '"', '"', 1},
		{"closed quote LF field limit", strings.NewReader("x,\"a\"\n"), '"', '"', 1},
		{"delimiter field limit", strings.NewReader("a,b,"), '"', '"', 1},
		{"CR field limit", strings.NewReader("a,b\r"), '"', '"', 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			parser, err := NewParser(test.input, ParserOptions{
				Delimiter: ',', Quote: test.quote, Escape: test.escape,
				MaxRecordBytes: 100, MaxFields: test.max,
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := parser.ReadRecord(t.Context()); err == nil {
				t.Fatal("ReadRecord() succeeded")
			}
		})
	}

	parser, err := NewParser(
		&errorAfterReader{data: []byte{0xc2}},
		ParserOptions{
			Delimiter: ',', Quote: '"', Escape: '"', MaxRecordBytes: 100,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := parser.ReadRecord(t.Context()); err == nil {
		t.Fatal("ReadRecord() accepted a truncated multibyte rune")
	}
}

func TestCSVIteratorOptionAndRuntimeBranches(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "source.csv", "id\np1\n")
	valid := IteratorOptions{Namespace: "crm", Source: singleVertexSource(path)}
	tests := []struct {
		name    string
		ctx     context.Context
		options IteratorOptions
	}{
		{"nil context", nil, valid},
		{"canceled context", canceledContext(), valid},
		{"empty namespace", t.Context(), func() IteratorOptions {
			value := valid
			value.Namespace = " "
			return value
		}()},
		{"negative reject limit", t.Context(), func() IteratorOptions {
			value := valid
			value.RejectLimit = -1
			return value
		}()},
		{"missing malformed handler", t.Context(), func() IteratorOptions {
			value := valid
			value.RejectLimit = 1
			return value
		}()},
		{"negative record limit", t.Context(), func() IteratorOptions {
			value := valid
			value.MaxRecordBytes = -1
			return value
		}()},
		{"negative field limit", t.Context(), func() IteratorOptions {
			value := valid
			value.MaxFields = -1
			return value
		}()},
		{"negative property limit", t.Context(), func() IteratorOptions {
			value := valid
			value.MaxProperties = -1
			return value
		}()},
		{"invalid resume", t.Context(), func() IteratorOptions {
			value := valid
			value.AfterToken = "invalid"
			return value
		}()},
		{"resume mapping out of range", t.Context(), func() IteratorOptions {
			value := valid
			value.AfterToken = formatResumeToken(resumeToken{
				Mapping: 10, Fingerprint: strings.Repeat("a", sha256HexLength),
			})
			return value
		}()},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := NewIterator(test.ctx, test.options); err == nil {
				t.Fatal("NewIterator() accepted invalid options")
			}
		})
	}

	iterator := &Iterator{closed: true}
	if _, err := iterator.Next(t.Context()); err == nil {
		t.Fatal("Next() accepted a closed iterator")
	}
	iterator = &Iterator{}
	if _, err := iterator.Next(canceledContext()); !errors.Is(err, context.Canceled) {
		t.Fatalf("Next() error = %v", err)
	}

	reader := &profileCountingReader{
		input: strings.NewReader("abc"),
		observe: func(int64) error {
			return errors.New("observe failed")
		},
	}
	output := make([]byte, 3)
	if count, err := reader.Read(output); count != 3 || err == nil {
		t.Fatalf("profileCountingReader.Read() = %d, %v", count, err)
	}
}

func TestCSVIteratorBudgetAndCompileBranches(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "source.csv", "id\np1\n")
	t.Run("row budget", func(t *testing.T) {
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{Rows: 1},
		)
		iterator, err := NewIterator(t.Context(), IteratorOptions{
			Namespace: "crm", Source: singleVertexSource(path), ProfileBudget: budget,
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := iterator.Close(); err != nil {
				t.Errorf("Close() error = %v", err)
			}
		})
		if _, err := iterator.Next(t.Context()); err != nil {
			t.Fatal(err)
		}
		if _, err := iterator.Next(t.Context()); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("second Next() error = %v", err)
		}
	})

	t.Run("page budget", func(t *testing.T) {
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{Pages: 1},
		)
		_ = budget.Charge(sourcecontract.ProfileBudgetUsage{Pages: 1})
		iterator, err := NewIterator(t.Context(), IteratorOptions{
			Namespace: "crm", Source: singleVertexSource(path), ProfileBudget: budget,
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := iterator.Close(); err != nil {
				t.Errorf("Close() error = %v", err)
			}
		})
		if _, err := iterator.Next(t.Context()); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("Next() error = %v", err)
		}
	})

	t.Run("raw byte budget", func(t *testing.T) {
		budget := sourcecontract.NewProfileBudget(
			sourcecontract.ProfileBudgetLimits{RawInputBytes: 1},
		)
		iterator, err := NewIterator(t.Context(), IteratorOptions{
			Namespace: "crm", Source: singleVertexSource(path), ProfileBudget: budget,
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := iterator.Next(t.Context()); !errors.Is(err, sourcecontract.ErrProfileBudget) {
			t.Fatalf("Next() error = %v", err)
		}
	})

	t.Run("invalid parser format", func(t *testing.T) {
		source := singleVertexSource(path)
		source.Defaults.Delimiter = ",,"
		iterator, err := NewIterator(t.Context(), IteratorOptions{
			Namespace: "crm", Source: source,
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := iterator.Next(t.Context()); err == nil {
			t.Fatal("Next() accepted an invalid delimiter")
		}
	})

	t.Run("header read failure", func(t *testing.T) {
		bad := writeTestFile(t, t.TempDir(), "bad.csv", "\"unterminated")
		iterator, err := NewIterator(t.Context(), IteratorOptions{
			Namespace: "crm", Source: singleVertexSource(bad),
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := iterator.Next(t.Context()); err == nil ||
			!strings.Contains(err.Error(), "read CSV header") {
			t.Fatalf("Next() error = %v", err)
		}
	})

	header := false
	mapping := fileMapping{
		kind: vertexMapping, path: path, idColumn: "0",
		properties: map[string]string{"bad": "x"},
		format: config.DelimitedOptions{
			Delimiter: ",", Quote: `"`, Escape: `"`,
			Header: &header, Encoding: "utf-8",
		},
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	parser, err := NewParser(bufio.NewReader(file), ParserOptions{
		Delimiter: ',', Quote: '"', Escape: '"', MaxRecordBytes: 100,
	})
	if err != nil {
		t.Fatal(err)
	}
	current := &openMapping{file: file, parser: parser, mapping: mapping}
	if err := current.compile(t.Context(), 10, 10); err == nil {
		t.Fatal("compile() accepted an invalid headerless property column")
	}

	mapping.properties = map[string]string{"a": "0", "b": "0"}
	current = &openMapping{file: file, parser: parser, mapping: mapping}
	if err := current.compile(t.Context(), 10, 1); err == nil {
		t.Fatal("compile() accepted too many properties")
	}
	mapping.properties = map[string]string{"a": "0"}
	current = &openMapping{file: file, parser: parser, mapping: mapping}
	if err := current.compile(canceledContext(), 10, 10); !errors.Is(
		err, context.Canceled,
	) {
		t.Fatalf("compile(canceled) error = %v", err)
	}

	mapping.properties = map[string]string{string([]byte{0xff}): "0"}
	current = &openMapping{file: file, parser: parser, mapping: mapping}
	if err := current.compile(t.Context(), 10, 10); err == nil {
		t.Fatal("compile() accepted an invalid property name")
	}
}

func TestCSVMappingAndEncodingBranches(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "source.csv", "id\np1\n")
	options := IteratorOptions{
		Namespace: "crm", MaxProperties: 1,
		Source: config.CSVSource{Vertices: []config.CSVVertex{{
			Label: "Person", Path: path, IDColumn: "id",
			Properties: map[string]string{"a": "id", "b": "id"},
		}}},
	}
	if _, err := buildMappings(t.Context(), options); err == nil {
		t.Fatal("buildMappings() accepted too many vertex properties")
	}
	if _, err := buildMappings(canceledContext(), IteratorOptions{
		Namespace: "crm", MaxProperties: 10,
		Source: singleVertexSource(path),
	}); !errors.Is(err, context.Canceled) {
		t.Fatalf("buildMappings(canceled) error = %v", err)
	}
	invalidEncoding := singleVertexSource(path)
	invalidEncoding.Defaults.Encoding = "utf-16"
	if _, err := buildMappings(t.Context(), IteratorOptions{
		Namespace: "crm", MaxProperties: 10, Source: invalidEncoding,
	}); err == nil {
		t.Fatal("buildMappings() accepted invalid vertex encoding")
	}

	header := true
	nullValue := ""
	edgeSource := config.CSVSource{
		Defaults: config.DelimitedOptions{
			Delimiter: ",", Quote: `"`, Escape: `"`,
			Header: &header, Encoding: "utf-8", NullValue: &nullValue,
		},
		Edges: []config.CSVEdge{{
			Label: "KNOWS", Path: path,
			Start:      config.EndpointMapping{Label: "Person", Field: "id"},
			End:        config.EndpointMapping{Label: "Person", Field: "id"},
			Properties: map[string]string{"a": "id", "b": "id"},
		}},
	}
	if _, err := buildMappings(t.Context(), IteratorOptions{
		Namespace: "crm", Source: edgeSource, MaxProperties: 1,
	}); err == nil {
		t.Fatal("buildMappings() accepted too many edge properties")
	}
	edgeSource.Edges[0].Properties = nil
	edgeSource.Defaults.Encoding = "utf-16"
	if _, err := buildMappings(t.Context(), IteratorOptions{
		Namespace: "crm", Source: edgeSource, MaxProperties: 10,
	}); err == nil {
		t.Fatal("buildMappings() accepted invalid edge encoding")
	}
	edgeSource.Defaults.Encoding = "utf-8"
	if _, err := buildMappings(canceledContext(), IteratorOptions{
		Namespace: "crm", Source: edgeSource, MaxProperties: 10,
	}); !errors.Is(err, context.Canceled) {
		t.Fatalf("edge buildMappings(canceled) error = %v", err)
	}

	properties := []compiledProperty{{
		name: "value", encodedName: []byte(`"value"`), index: 0,
	}}
	if _, err := encodeCSVProperties(
		canceledContext(), properties, []string{"x"}, "",
	); !errors.Is(err, context.Canceled) {
		t.Fatalf("encodeCSVProperties(canceled) error = %v", err)
	}
	if _, err := encodeCSVProperties(
		t.Context(), properties, []string{string([]byte{0xff})}, "",
	); err == nil {
		t.Fatal("encodeCSVProperties() accepted invalid UTF-8")
	}

	fallback := model.SourcePosition{Resource: "fallback"}
	if got := positionFromError(errors.New("plain"), fallback); got != fallback {
		t.Fatalf("positionFromError() = %#v", got)
	}

	closed, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := closed.Close(); err != nil {
		t.Fatal(err)
	}
	current := &openMapping{
		file: closed,
		mapping: fileMapping{
			path: path, fingerprintInput: []byte("mapping"),
		},
	}
	if err := current.verifyFingerprint(t.Context()); err == nil {
		t.Fatal("verifyFingerprint() accepted a closed file")
	}
	open, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer open.Close()
	current = &openMapping{
		file: open,
		mapping: fileMapping{
			path: path, fingerprintInput: []byte("mapping"),
		},
	}
	if err := current.verifyFingerprint(canceledContext()); !errors.Is(
		err, context.Canceled,
	) {
		t.Fatalf("verifyFingerprint(canceled) error = %v", err)
	}

	iterator := &Iterator{}
	if err := iterator.closeCurrent(); err != nil {
		t.Fatalf("closeCurrent(nil) error = %v", err)
	}
}

func TestCSVIteratorDirectErrorPaths(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "source.csv", "id,name\np1,Ada\n")
	missing := &Iterator{
		options: IteratorOptions{MaxRecordBytes: 100, MaxFields: 10, MaxProperties: 10},
		mappings: []fileMapping{{
			path: filepath.Join(t.TempDir(), "missing.csv"),
		}},
		manifestSet: true,
	}
	if err := missing.openCurrent(t.Context()); err == nil {
		t.Fatal("openCurrent() accepted a missing source")
	}

	canceled := &Iterator{
		options: IteratorOptions{MaxRecordBytes: 100, MaxFields: 10, MaxProperties: 10},
		mappings: []fileMapping{{
			path: path, fingerprintInput: []byte("mapping"),
		}},
		manifestSet: true,
	}
	if err := canceled.openCurrent(canceledContext()); !errors.Is(
		err, context.Canceled,
	) {
		t.Fatalf("openCurrent(canceled) error = %v", err)
	}

	gzipPath := filepath.Join(t.TempDir(), "source.csv.gz")
	file, err := os.Create(gzipPath)
	if err != nil {
		t.Fatal(err)
	}
	gzipWriter := gzip.NewWriter(file)
	if _, err := gzipWriter.Write([]byte("id\np1\n")); err != nil {
		t.Fatal(err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	for _, format := range []config.DelimitedOptions{
		{Delimiter: ",,", Quote: `"`, Escape: `"`, Encoding: "utf-8"},
		{Delimiter: ",", Quote: "\n", Escape: `"`, Encoding: "utf-8"},
	} {
		iterator := &Iterator{
			options: IteratorOptions{
				MaxRecordBytes: 100, MaxFields: 10, MaxProperties: 10,
				ProfileBudget: sourcecontract.NewProfileBudget(
					sourcecontract.ProfileBudgetLimits{},
				),
			},
			mappings:    []fileMapping{{path: gzipPath, format: format}},
			manifestSet: true,
		}
		if err := iterator.openCurrent(t.Context()); err == nil {
			t.Fatalf("openCurrent() accepted format %#v", format)
		}
	}

	nullValue := ""
	iterator := &Iterator{
		options: IteratorOptions{PreencodeProperties: true},
		current: &openMapping{
			mapping: fileMapping{
				kind:   vertexMapping,
				format: config.DelimitedOptions{NullValue: &nullValue},
			},
			compiled: compiledMapping{
				id: 0,
				properties: []compiledProperty{{
					name: "name", encodedName: []byte(`"name"`), index: 1,
				}},
			},
		},
	}
	if _, err := iterator.mapRecord(
		canceledContext(), []string{"p1", "Ada"}, model.SourcePosition{},
	); !errors.Is(err, context.Canceled) {
		t.Fatalf("mapRecord(preencoded canceled) error = %v", err)
	}
	iterator.options.PreencodeProperties = false
	if _, err := iterator.mapRecord(
		canceledContext(), []string{"p1", "Ada"}, model.SourcePosition{},
	); !errors.Is(err, context.Canceled) {
		t.Fatalf("mapRecord(canceled) error = %v", err)
	}
	iterator.current.mapping.kind = edgeMapping
	iterator.current.compiled = compiledMapping{start: 0, end: 1, externalID: -1}
	if _, err := iterator.mapRecord(
		t.Context(), []string{"", "p2"}, model.SourcePosition{},
	); err == nil {
		t.Fatal("mapRecord() accepted an empty edge endpoint")
	}

	ctx, cancel := context.WithCancel(context.Background())
	parser, err := NewParser(strings.NewReader("p1,Ada\n"), ParserOptions{
		Delimiter: ',', Quote: '"', Escape: '"', MaxRecordBytes: 100,
		OnInputBytes: func(int64) error {
			cancel()
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	iterator = &Iterator{
		options: IteratorOptions{
			OnMalformed: func(context.Context, MalformedRecord) error {
				return errors.New("should not quarantine")
			},
		},
		mappings: []fileMapping{{kind: vertexMapping}},
		current: &openMapping{
			parser: parser,
			mapping: fileMapping{
				kind: vertexMapping,
			},
			compiled: compiledMapping{
				id:         0,
				properties: []compiledProperty{{name: "name", index: 1}},
				fieldCount: 2, exactFields: true,
			},
		},
	}
	if _, err := iterator.Next(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("Next(cancel during map) error = %v", err)
	}

	parser, err = NewParser(strings.NewReader("short\n"), ParserOptions{
		Delimiter: ',', Quote: '"', Escape: '"', MaxRecordBytes: 100,
	})
	if err != nil {
		t.Fatal(err)
	}
	iterator = &Iterator{
		options: IteratorOptions{
			RejectLimit: 1,
			OnMalformed: func(context.Context, MalformedRecord) error {
				return errors.New("quarantine failed")
			},
		},
		mappings: []fileMapping{{kind: vertexMapping}},
		current: &openMapping{
			parser: parser,
			mapping: fileMapping{
				kind: vertexMapping,
			},
			compiled: compiledMapping{fieldCount: 2, exactFields: true},
		},
	}
	if _, err := iterator.Next(t.Context()); err == nil ||
		!strings.Contains(err.Error(), "quarantine failed") {
		t.Fatalf("Next(short record) error = %v", err)
	}
}

func canceledContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}

var _ io.Reader = (*profileCountingReader)(nil)
