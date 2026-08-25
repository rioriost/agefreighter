package csv

import (
	"bufio"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
)

func TestResumeTokenValidation(t *testing.T) {
	validFingerprint := strings.Repeat("a", sha256HexLength)
	valid := formatResumeToken(resumeToken{
		Mapping:     2,
		Record:      3,
		Rejected:    4,
		Fingerprint: validFingerprint,
	})
	token, err := parseResumeToken(valid)
	if err != nil {
		t.Fatalf("parseResumeToken() error = %v", err)
	}
	if token.Mapping != 2 || token.Record != 3 || token.Rejected != 4 ||
		token.Fingerprint != validFingerprint {
		t.Fatalf("parseResumeToken() = %#v", token)
	}
	for _, value := range []string{
		"",
		"csv:v1:0:0:" + validFingerprint,
		"csv:v2:x:0:0:" + validFingerprint,
		"csv:v2:-1:0:0:" + validFingerprint,
		"csv:v2:0:x:0:" + validFingerprint,
		"csv:v2:0:-1:0:" + validFingerprint,
		"csv:v2:0:0:x:" + validFingerprint,
		"csv:v2:0:0:-1:" + validFingerprint,
		"csv:v2:0:0:0:short",
		"csv:v2:0:0:0:" + strings.Repeat("z", sha256HexLength),
	} {
		if _, err := parseResumeToken(value); err == nil {
			t.Fatalf("parseResumeToken(%q) succeeded", value)
		}
	}
}

func TestFormatMergingAndValidation(t *testing.T) {
	defaultHeader := true
	defaultNull := ""
	overrideHeader := false
	overrideNull := "NULL"
	defaults := config.DelimitedOptions{
		Delimiter: ",",
		Quote:     `"`,
		Escape:    `"`,
		Header:    &defaultHeader,
		Encoding:  "utf-8",
		NullValue: &defaultNull,
	}
	override := &config.DelimitedOptions{
		Delimiter: ";",
		Quote:     "'",
		Escape:    "\\",
		Header:    &overrideHeader,
		Encoding:  "UTF-8",
		NullValue: &overrideNull,
	}
	merged := mergeFormat(defaults, override)
	if merged.Delimiter != ";" || merged.Quote != "'" || merged.Escape != "\\" ||
		*merged.Header || merged.Encoding != "UTF-8" || *merged.NullValue != "NULL" {
		t.Fatalf("mergeFormat() = %#v", merged)
	}
	empty := mergeFormat(config.DelimitedOptions{}, nil)
	if empty.Delimiter != "," || empty.Quote != `"` || empty.Escape != `"` ||
		!*empty.Header || empty.Encoding != "utf-8" || *empty.NullValue != "" {
		t.Fatalf("default mergeFormat() = %#v", empty)
	}
	for _, format := range []config.DelimitedOptions{
		{Delimiter: "", Quote: `"`, Escape: `"`},
		{Delimiter: ",", Quote: "", Escape: `"`},
		{Delimiter: ",", Quote: `"`, Escape: ""},
		{Delimiter: ",,", Quote: `"`, Escape: `"`},
	} {
		if _, _, _, err := formatRunes(format); err == nil {
			t.Fatalf("formatRunes(%#v) succeeded", format)
		}
	}
	if _, err := newFileMapping(
		vertexMapping,
		"source.csv",
		"Person",
		"crm",
		config.DelimitedOptions{Encoding: "utf-16"},
	); err == nil {
		t.Fatal("newFileMapping() accepted unsupported encoding")
	}
}

func TestIteratorOpenAndCompileFailures(t *testing.T) {
	directory := t.TempDir()
	missing := filepath.Join(directory, "missing.csv")
	tests := []struct {
		name   string
		path   string
		source func(string) config.CSVSource
		want   string
	}{
		{
			name:   "missing file",
			path:   missing,
			source: singleVertexSource,
			want:   "open CSV",
		},
		{
			name:   "directory",
			path:   directory,
			source: singleVertexSource,
			want:   "not a regular file",
		},
		{
			name:   "empty header",
			path:   writeTestFile(t, directory, "empty.csv", ""),
			source: singleVertexSource,
			want:   "has no header",
		},
		{
			name:   "empty header name",
			path:   writeTestFile(t, directory, "empty-name.csv", ",name\np1,Ada\n"),
			source: singleVertexSource,
			want:   "empty header",
		},
		{
			name:   "duplicate header",
			path:   writeTestFile(t, directory, "duplicate.csv", "id,id\np1,Ada\n"),
			source: singleVertexSource,
			want:   "duplicate header",
		},
		{
			name:   "missing mapped column",
			path:   writeTestFile(t, directory, "missing-column.csv", "other\np1\n"),
			source: singleVertexSource,
			want:   "has no column",
		},
		{
			name:   "invalid gzip",
			path:   writeTestBytes(t, directory, "invalid.gz", []byte{0x1f, 0x8b, 0, 1}),
			source: singleVertexSource,
			want:   "open gzip CSV source",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			iterator, err := NewIterator(context.Background(), IteratorOptions{
				Namespace: "crm",
				Source:    test.source(test.path),
			})
			if err != nil {
				t.Fatalf("NewIterator() error = %v", err)
			}
			defer iterator.Close()
			if _, err := iterator.Next(context.Background()); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("Next() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestIteratorAdditionalValidation(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "source.csv", "id\np1\n")
	source := singleVertexSource(path)
	if _, err := NewIterator(context.Background(), IteratorOptions{
		Namespace:      "crm",
		Source:         source,
		MaxRecordBytes: -1,
	}); err == nil {
		t.Fatal("NewIterator() accepted negative maximum record bytes")
	}
	if _, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "crm",
		Source:    source,
		MaxFields: -1,
	}); err == nil {
		t.Fatal("NewIterator() accepted negative maximum fields")
	}
	if _, err := NewIterator(context.Background(), IteratorOptions{
		Namespace:     "crm",
		Source:        source,
		MaxProperties: -1,
	}); err == nil {
		t.Fatal("NewIterator() accepted negative maximum properties")
	}
	source.Vertices = nil
	source.Edges = []config.CSVEdge{{
		Label: "KNOWS",
		Path:  path,
		Start: config.EndpointMapping{Label: "Person", Field: "id"},
		End:   config.EndpointMapping{Label: "Person", Field: "id"},
		Properties: map[string]string{
			"a": "id",
			"b": "id",
		},
	}}
	if _, err := NewIterator(context.Background(), IteratorOptions{
		Namespace:     "crm",
		Source:        source,
		MaxProperties: 1,
	}); err == nil || !strings.Contains(err.Error(), "edge mapping") {
		t.Fatalf("edge property limit error = %v", err)
	}
}

func TestIteratorResumeOpenFailures(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "source.csv", "id\np1\n")
	newPrimed := func(t *testing.T) *Iterator {
		t.Helper()
		iterator, err := NewIterator(context.Background(), IteratorOptions{
			Namespace: "crm",
			Source:    singleVertexSource(path),
		})
		if err != nil {
			t.Fatalf("NewIterator() error = %v", err)
		}
		if err := iterator.openCurrent(context.Background()); err != nil {
			t.Fatalf("initial openCurrent() error = %v", err)
		}
		if err := iterator.closeCurrent(); err != nil {
			t.Fatalf("initial closeCurrent() error = %v", err)
		}
		return iterator
	}

	changed := newPrimed(t)
	if err := os.WriteFile(path, []byte("id\np2\n"), 0o600); err != nil {
		t.Fatalf("change source: %v", err)
	}
	if err := changed.openCurrent(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "changed while opening") {
		t.Fatalf("changed openCurrent() error = %v", err)
	}

	if err := os.WriteFile(path, []byte("id\np1\n"), 0o600); err != nil {
		t.Fatalf("restore source: %v", err)
	}
	beyond := newPrimed(t)
	beyond.resume = resumeToken{
		Mapping:     0,
		Record:      2,
		Fingerprint: beyond.manifest,
	}
	beyond.hasResume = true
	if err := beyond.openCurrent(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "beyond end") {
		t.Fatalf("beyond openCurrent() error = %v", err)
	}

	overLimit := newPrimed(t)
	overLimit.resume = resumeToken{
		Mapping:     0,
		Rejected:    1,
		Fingerprint: overLimit.manifest,
	}
	overLimit.hasResume = true
	if err := overLimit.openCurrent(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "exceeds configured reject limit") {
		t.Fatalf("over-limit openCurrent() error = %v", err)
	}

	if err := os.WriteFile(path, []byte("id\nbad\"tail\n"), 0o600); err != nil {
		t.Fatalf("write malformed source: %v", err)
	}
	replay := newPrimed(t)
	replay.resume = resumeToken{
		Mapping:     0,
		Record:      1,
		Fingerprint: replay.manifest,
	}
	replay.hasResume = true
	if err := replay.openCurrent(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "replay CSV checkpoint") {
		t.Fatalf("malformed replay openCurrent() error = %v", err)
	}
}

func TestFingerprintAndCloseFailures(t *testing.T) {
	if _, err := fingerprint(
		context.Background(),
		filepath.Join(t.TempDir(), "missing"),
		nil,
	); err == nil {
		t.Fatal("fingerprint() accepted missing file")
	}
	file, err := os.CreateTemp(t.TempDir(), "closed")
	if err != nil {
		t.Fatalf("CreateTemp() error = %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("initial Close() error = %v", err)
	}
	if _, err := fingerprintFile(context.Background(), file, file.Name(), nil); err == nil {
		t.Fatal("fingerprintFile() accepted closed file")
	}
	current := &openMapping{file: file, mapping: fileMapping{path: file.Name()}}
	if err := current.close(); err == nil {
		t.Fatal("openMapping.close() ignored closed file")
	}
	if err := current.verifyFingerprint(context.Background()); err == nil {
		t.Fatal("verifyFingerprint() accepted closed file")
	}
}

func TestParserHelperFailures(t *testing.T) {
	parser, err := NewParser(strings.NewReader(""), ParserOptions{
		Delimiter:      ',',
		Quote:          '"',
		Escape:         '"',
		MaxRecordBytes: 10,
	})
	if err != nil {
		t.Fatalf("NewParser() error = %v", err)
	}
	if err := parser.unreadRune('x'); err == nil {
		t.Fatal("unreadRune() succeeded without a preceding read")
	}

	reader := &errorAfterReader{data: []byte("a\r")}
	parser, err = NewParser(bufio.NewReader(reader), ParserOptions{
		Delimiter:      ',',
		Quote:          '"',
		Escape:         '"',
		MaxRecordBytes: 10,
	})
	if err != nil {
		t.Fatalf("error reader NewParser() error = %v", err)
	}
	if _, _, err := parser.ReadRecord(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "injected read failure") {
		t.Fatalf("error reader ReadRecord() error = %v", err)
	}
}

type errorAfterReader struct {
	data []byte
}

func (reader *errorAfterReader) Read(output []byte) (int, error) {
	if len(reader.data) != 0 {
		count := copy(output, reader.data)
		reader.data = reader.data[count:]
		return count, nil
	}
	return 0, errors.New("injected read failure")
}

func writeTestBytes(t *testing.T, directory, name string, contents []byte) string {
	t.Helper()
	path := filepath.Join(directory, name)
	if err := os.WriteFile(path, contents, 0o600); err != nil {
		t.Fatalf("write test bytes: %v", err)
	}
	return path
}

var _ io.Reader = (*errorAfterReader)(nil)
