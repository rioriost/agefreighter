package csv

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
)

func TestParserRecordsAndPositions(t *testing.T) {
	parser, err := NewParser(
		strings.NewReader("id,name\r\n1,\"Ada\r\nLovelace\"\r\n2,\"a\"\"b\"\n"),
		ParserOptions{
			Delimiter:      ',',
			Quote:          '"',
			Escape:         '"',
			Resource:       "people.csv",
			MaxRecordBytes: 1 << 20,
		},
	)
	if err != nil {
		t.Fatalf("NewParser() error = %v", err)
	}

	tests := []struct {
		fields []string
		offset int64
		line   int64
	}{
		{fields: []string{"id", "name"}, offset: 0, line: 1},
		{fields: []string{"1", "Ada\nLovelace"}, offset: 9, line: 2},
		{fields: []string{"2", `a"b`}, offset: 28, line: 4},
	}
	for _, test := range tests {
		fields, position, err := parser.ReadRecord(context.Background())
		if err != nil {
			t.Fatalf("ReadRecord() error = %v", err)
		}
		if strings.Join(fields, "|") != strings.Join(test.fields, "|") {
			t.Fatalf("ReadRecord() = %#v, want %#v", fields, test.fields)
		}
		if position.Offset != test.offset || position.Line != test.line ||
			position.Resource != "people.csv" || position.Connector != "csv" {
			t.Fatalf("ReadRecord() position = %#v", position)
		}
	}
	if _, _, err := parser.ReadRecord(context.Background()); !errors.Is(err, io.EOF) {
		t.Fatalf("final ReadRecord() error = %v, want EOF", err)
	}
}

func TestParserCustomDelimiterQuoteAndEscape(t *testing.T) {
	parser, err := NewParser(
		strings.NewReader("1;'a\\'b';'c\\\\d'\n"),
		ParserOptions{
			Delimiter:      ';',
			Quote:          '\'',
			Escape:         '\\',
			MaxRecordBytes: 1 << 20,
		},
	)
	if err != nil {
		t.Fatalf("NewParser() error = %v", err)
	}
	fields, _, err := parser.ReadRecord(context.Background())
	if err != nil {
		t.Fatalf("ReadRecord() error = %v", err)
	}
	want := []string{"1", "a'b", `c\d`}
	if strings.Join(fields, "|") != strings.Join(want, "|") {
		t.Fatalf("ReadRecord() = %#v, want %#v", fields, want)
	}
}

func TestParserFailures(t *testing.T) {
	tests := []struct {
		name   string
		input  string
		quote  rune
		escape rune
	}{
		{name: "unterminated quote", input: `1,"broken`, quote: '"', escape: '"'},
		{name: "quote in field", input: "1,a\"b\n", quote: '"', escape: '"'},
		{name: "after quote", input: "1,\"a\"b\n", quote: '"', escape: '"'},
		{name: "invalid escape", input: "1,'a\\x'\n", quote: '\'', escape: '\\'},
		{name: "unterminated escape", input: "1,'a\\", quote: '\'', escape: '\\'},
		{
			name:   "invalid UTF-8",
			input:  string([]byte{'1', ',', 0xff, '\n'}),
			quote:  '"',
			escape: '"',
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			parser, err := NewParser(
				strings.NewReader(test.input),
				ParserOptions{
					Delimiter:      ',',
					Quote:          test.quote,
					Escape:         test.escape,
					Resource:       "bad.csv",
					MaxRecordBytes: 1 << 20,
				},
			)
			if err != nil {
				t.Fatalf("NewParser() error = %v", err)
			}
			if _, _, err := parser.ReadRecord(context.Background()); err == nil {
				t.Fatal("ReadRecord() succeeded")
			} else {
				var parseErr *ParseError
				if !errors.As(err, &parseErr) {
					t.Fatalf("ReadRecord() error = %T, want ParseError", err)
				}
			}
		})
	}
}

func TestParserRejectsInvalidOptions(t *testing.T) {
	tests := []ParserOptions{
		{},
		{Delimiter: '\n', Quote: '"', Escape: '"', MaxRecordBytes: 1},
		{Delimiter: ',', Quote: -1, Escape: '"', MaxRecordBytes: 1},
		{Delimiter: ',', Quote: '"', Escape: -1, MaxRecordBytes: 1},
		{Delimiter: '"', Quote: '"', Escape: '"', MaxRecordBytes: 1},
		{Delimiter: ',', Quote: '\n', Escape: '"', MaxRecordBytes: 1},
		{Delimiter: ',', Quote: '"', Escape: '\r', MaxRecordBytes: 1},
		{Delimiter: ',', Quote: '"', Escape: '"'},
		{Delimiter: ',', Quote: '"', Escape: '"', MaxRecordBytes: 1, MaxFields: -1},
	}
	for _, options := range tests {
		if _, err := NewParser(strings.NewReader(""), options); err == nil {
			t.Fatalf("NewParser(%#v) succeeded", options)
		}
	}
	if _, err := NewParser(nil, ParserOptions{
		Delimiter:      ',',
		Quote:          '"',
		Escape:         '"',
		MaxRecordBytes: 1,
	}); err == nil {
		t.Fatal("NewParser(nil) succeeded")
	}
}

func TestParserEnforcesRecordLimitAndCancellation(t *testing.T) {
	parser, err := NewParser(strings.NewReader("abcd\n"), ParserOptions{
		Delimiter:      ',',
		Quote:          '"',
		Escape:         '"',
		Resource:       "large.csv",
		MaxRecordBytes: 3,
	})
	if err != nil {
		t.Fatalf("NewParser() error = %v", err)
	}
	if _, _, err := parser.ReadRecord(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "exceeds 3 bytes") {
		t.Fatalf("oversized ReadRecord() error = %v", err)
	}
	parser, err = NewParser(strings.NewReader("ab\r\n"), ParserOptions{
		Delimiter:      ',',
		Quote:          '"',
		Escape:         '"',
		MaxRecordBytes: 3,
	})
	if err != nil {
		t.Fatalf("CRLF NewParser() error = %v", err)
	}
	if _, _, err := parser.ReadRecord(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "exceeds 3 bytes") {
		t.Fatalf("CRLF ReadRecord() error = %v", err)
	}
	parser, err = NewParser(strings.NewReader("a\rb\r"), ParserOptions{
		Delimiter:      ',',
		Quote:          '"',
		Escape:         '"',
		MaxRecordBytes: 2,
	})
	if err != nil {
		t.Fatalf("lone CR NewParser() error = %v", err)
	}
	first, _, err := parser.ReadRecord(context.Background())
	if err != nil || strings.Join(first, "|") != "a" {
		t.Fatalf("first lone CR record = %#v, error = %v", first, err)
	}
	second, _, err := parser.ReadRecord(context.Background())
	if err != nil || strings.Join(second, "|") != "b" {
		t.Fatalf("second lone CR record = %#v, error = %v", second, err)
	}

	parser, err = NewParser(
		strings.NewReader(string([]byte{'a', '\r', 0xff, '\r'})),
		ParserOptions{
			Delimiter:      ',',
			Quote:          '"',
			Escape:         '"',
			MaxRecordBytes: 2,
		},
	)
	if err != nil {
		t.Fatalf("lone CR invalid UTF-8 NewParser() error = %v", err)
	}
	first, _, err = parser.ReadRecord(context.Background())
	if err != nil || strings.Join(first, "|") != "a" {
		t.Fatalf("valid record before invalid UTF-8 = %#v, error = %v", first, err)
	}
	if _, _, err := parser.ReadRecord(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "not valid UTF-8") {
		t.Fatalf("next invalid UTF-8 error = %v", err)
	}

	parser, err = NewParser(strings.NewReader("a\n"), ParserOptions{
		Delimiter:      ',',
		Quote:          '"',
		Escape:         '"',
		MaxRecordBytes: 10,
	})
	if err != nil {
		t.Fatalf("cancelled NewParser() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := parser.ReadRecord(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled ReadRecord() error = %v", err)
	}

	parser, err = NewParser(strings.NewReader("a,b,c\n"), ParserOptions{
		Delimiter:      ',',
		Quote:          '"',
		Escape:         '"',
		MaxRecordBytes: 100,
		MaxFields:      2,
	})
	if err != nil {
		t.Fatalf("field-limited NewParser() error = %v", err)
	}
	if _, _, err := parser.ReadRecord(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "exceeds 2 fields") {
		t.Fatalf("field-limited ReadRecord() error = %v", err)
	}

	parser, err = NewParser(strings.NewReader("x,\"a,b\"\n"), ParserOptions{
		Delimiter:      ',',
		Quote:          '"',
		Escape:         '"',
		MaxRecordBytes: 100,
		MaxFields:      2,
	})
	if err != nil {
		t.Fatalf("quoted delimiter NewParser() error = %v", err)
	}
	fields, _, err := parser.ReadRecord(context.Background())
	if err != nil {
		t.Fatalf("quoted delimiter ReadRecord() error = %v", err)
	}
	if strings.Join(fields, "|") != "x|a,b" {
		t.Fatalf("quoted delimiter fields = %#v", fields)
	}
}

func TestParserAdvancesOffsetPastInvalidUTF8(t *testing.T) {
	parser, err := NewParser(
		strings.NewReader(string([]byte{'a', ',', 0xff, '\n'})),
		ParserOptions{
			Delimiter:      ',',
			Quote:          '"',
			Escape:         '"',
			MaxRecordBytes: 10,
		},
	)
	if err != nil {
		t.Fatalf("NewParser() error = %v", err)
	}
	if _, _, err := parser.ReadRecord(context.Background()); err == nil {
		t.Fatal("ReadRecord() accepted invalid UTF-8")
	}
	if parser.offset != 3 {
		t.Fatalf("parser offset = %d, want 3 consumed bytes", parser.offset)
	}
}
