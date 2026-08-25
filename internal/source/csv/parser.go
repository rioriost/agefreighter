package csv

import (
	"bufio"
	"context"
	stdcsv "encoding/csv"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"

	"github.com/rioriost/agefreighter/pkg/model"
)

type ParserOptions struct {
	Delimiter       rune
	Quote           rune
	Escape          rune
	Resource        string
	MaxRecordBytes  int64
	MaxFields       int
	OptimizeRFC4180 bool
}

type Parser struct {
	reader      *bufio.Reader
	standard    *stdcsv.Reader
	options     ParserOptions
	offset      int64
	line        int64
	recordStart int64
	finished    bool
	pending     bool
	pendingRune rune
	pendingSize int
	canUnread   bool
}

type ParseError struct {
	Position model.SourcePosition
	Err      error
}

func (err *ParseError) Error() string {
	return fmt.Sprintf(
		"%s:%d at byte %d: %v",
		err.Position.Resource,
		err.Position.Line,
		err.Position.Offset,
		err.Err,
	)
}

func (err *ParseError) Unwrap() error {
	return err.Err
}

func NewParser(input io.Reader, options ParserOptions) (*Parser, error) {
	if options.MaxFields == 0 {
		options.MaxFields = 4096
	}
	switch {
	case input == nil:
		return nil, errors.New("CSV input is required")
	case !utf8.ValidRune(options.Delimiter) || options.Delimiter == '\n' ||
		options.Delimiter == '\r':
		return nil, errors.New("CSV delimiter must be a valid non-line-break rune")
	case !utf8.ValidRune(options.Quote):
		return nil, errors.New("CSV quote must be a valid rune")
	case !utf8.ValidRune(options.Escape):
		return nil, errors.New("CSV escape must be a valid rune")
	case options.Quote == '\n' || options.Quote == '\r':
		return nil, errors.New("CSV quote must not be a line break")
	case options.Escape == '\n' || options.Escape == '\r':
		return nil, errors.New("CSV escape must not be a line break")
	case options.Delimiter == options.Quote:
		return nil, errors.New("CSV delimiter and quote must differ")
	case options.MaxRecordBytes <= 0:
		return nil, errors.New("CSV maximum record bytes must be positive")
	case options.MaxFields < 0:
		return nil, errors.New("CSV maximum fields must be positive")
	}
	parser := &Parser{
		reader:  bufio.NewReader(input),
		options: options,
		line:    1,
	}
	if options.OptimizeRFC4180 {
		if options.Quote != '"' || options.Escape != '"' {
			return nil, errors.New(
				"optimized RFC 4180 parsing requires double-quote quoting and escaping",
			)
		}
		parser.standard = stdcsv.NewReader(parser.reader)
		parser.standard.Comma = options.Delimiter
		parser.standard.FieldsPerRecord = -1
	}
	return parser, nil
}

func (parser *Parser) ReadRecord(
	ctx context.Context,
) ([]string, model.SourcePosition, error) {
	if parser.standard != nil {
		return parser.readStandardRecord(ctx)
	}
	position := model.SourcePosition{
		Connector: "csv",
		Resource:  parser.options.Resource,
		Offset:    parser.offset,
		Line:      parser.line,
	}

	if parser.finished {
		return nil, position, io.EOF
	}
	parser.recordStart = position.Offset

	var (
		fields      []string
		field       []byte
		quoted      bool
		closedQuote bool
		haveData    bool
	)
	finishField := func() error {
		if len(fields) >= parser.options.MaxFields {
			return fmt.Errorf("record exceeds %d fields", parser.options.MaxFields)
		}
		fields = append(fields, string(field))
		field = field[:0]
		closedQuote = false
		return nil
	}
	parseError := func(message string) ([]string, model.SourcePosition, error) {
		return nil, position, &ParseError{
			Position: position,
			Err:      errors.New(message),
		}
	}

	for {
		if err := ctx.Err(); err != nil {
			return nil, position, err
		}
		character, _, err := parser.readRune()
		if errors.Is(err, io.EOF) {
			parser.finished = true
			if !haveData && len(fields) == 0 && len(field) == 0 {
				return nil, position, io.EOF
			}
			if quoted {
				return parseError("unterminated quoted field")
			}
			if err := finishField(); err != nil {
				return parseError(err.Error())
			}
			return fields, position, nil
		}
		if err != nil {
			return nil, position, err
		}
		haveData = true

		if quoted {
			switch {
			case character == parser.options.Quote &&
				parser.options.Escape == parser.options.Quote:
				next, _, nextErr := parser.readRune()
				if nextErr == nil && next == parser.options.Quote {
					field = utf8.AppendRune(field, character)
					continue
				}
				if nextErr != nil && !errors.Is(nextErr, io.EOF) {
					return nil, position, nextErr
				}
				if nextErr == nil {
					if err := parser.unreadRune(next); err != nil {
						return nil, position, err
					}
				} else {
					parser.finished = true
				}
				quoted = false
				closedQuote = true
			case character == parser.options.Quote:
				quoted = false
				closedQuote = true
			case character == parser.options.Escape:
				next, _, nextErr := parser.readRune()
				if nextErr != nil {
					if errors.Is(nextErr, io.EOF) {
						return parseError("unterminated escape sequence")
					}
					return nil, position, nextErr
				}
				if next != parser.options.Quote && next != parser.options.Escape {
					return parseError("escape must precede quote or escape")
				}
				field = utf8.AppendRune(field, next)
			case character == '\r':
				if err := parser.consumeOptionalLF(); err != nil {
					return nil, position, err
				}
				field = append(field, '\n')
				parser.line++
			case character == '\n':
				field = utf8.AppendRune(field, character)
				parser.line++
			default:
				field = utf8.AppendRune(field, character)
			}
			continue
		}

		if closedQuote {
			switch character {
			case parser.options.Delimiter:
				if err := finishField(); err != nil {
					return parseError(err.Error())
				}
			case '\r':
				if err := parser.consumeOptionalLF(); err != nil {
					return nil, position, err
				}
				if err := finishField(); err != nil {
					return parseError(err.Error())
				}
				parser.line++
				return fields, position, nil
			case '\n':
				if err := finishField(); err != nil {
					return parseError(err.Error())
				}
				parser.line++
				return fields, position, nil
			default:
				return parseError("unexpected character after closing quote")
			}
			continue
		}

		switch character {
		case parser.options.Delimiter:
			if err := finishField(); err != nil {
				return parseError(err.Error())
			}
		case parser.options.Quote:
			if len(field) != 0 {
				return parseError("quote must begin at the start of a field")
			}
			quoted = true
		case '\r':
			if err := parser.consumeOptionalLF(); err != nil {
				return nil, position, err
			}
			if err := finishField(); err != nil {
				return parseError(err.Error())
			}
			parser.line++
			return fields, position, nil
		case '\n':
			if err := finishField(); err != nil {
				return parseError(err.Error())
			}
			parser.line++
			return fields, position, nil
		default:
			field = utf8.AppendRune(field, character)
		}
	}
}

func (parser *Parser) readStandardRecord(
	ctx context.Context,
) ([]string, model.SourcePosition, error) {
	position := model.SourcePosition{
		Connector: "csv",
		Resource:  parser.options.Resource,
		Offset:    parser.offset,
		Line:      parser.line,
	}
	if err := ctx.Err(); err != nil {
		return nil, position, err
	}
	fields, err := parser.standard.Read()
	parser.offset = parser.standard.InputOffset()
	if len(fields) > 0 {
		line, _ := parser.standard.FieldPos(0)
		position.Line = int64(line)
		lastLine, _ := parser.standard.FieldPos(len(fields) - 1)
		parser.line = int64(lastLine + strings.Count(fields[len(fields)-1], "\n") + 1)
	}
	if parser.offset-position.Offset > parser.options.MaxRecordBytes {
		return nil, position, &ParseError{
			Position: position,
			Err:      fmt.Errorf("record exceeds %d bytes", parser.options.MaxRecordBytes),
		}
	}
	if len(fields) > parser.options.MaxFields {
		return nil, position, &ParseError{
			Position: position,
			Err:      fmt.Errorf("record exceeds %d fields", parser.options.MaxFields),
		}
	}
	for _, field := range fields {
		if !utf8.ValidString(field) {
			return nil, position, &ParseError{
				Position: position,
				Err:      errors.New("input is not valid UTF-8"),
			}
		}
	}
	if err != nil {
		if errors.Is(err, io.EOF) {
			parser.finished = true
			return nil, position, io.EOF
		}
		var parseErr *stdcsv.ParseError
		if errors.As(err, &parseErr) {
			position.Line = int64(parseErr.StartLine)
		}
		return nil, position, &ParseError{Position: position, Err: err}
	}
	return fields, position, nil
}

func (parser *Parser) readRune() (rune, int, error) {
	parser.canUnread = false
	character, size, err := parser.readRawRune()
	if err != nil {
		return 0, 0, err
	}
	characterOffset := parser.offset
	parser.offset += int64(size)
	if character == utf8.RuneError && size == 1 {
		return 0, 0, &ParseError{
			Position: model.SourcePosition{
				Connector: "csv",
				Resource:  parser.options.Resource,
				Offset:    characterOffset,
				Line:      parser.line,
			},
			Err: errors.New("input is not valid UTF-8"),
		}
	}
	if parser.offset-parser.recordStart > parser.options.MaxRecordBytes {
		parser.canUnread = false
		return 0, 0, &ParseError{
			Position: model.SourcePosition{
				Connector: "csv",
				Resource:  parser.options.Resource,
				Offset:    parser.recordStart,
				Line:      parser.line,
			},
			Err: fmt.Errorf(
				"record exceeds %d bytes",
				parser.options.MaxRecordBytes,
			),
		}
	}
	parser.canUnread = true
	return character, size, nil
}

func (parser *Parser) readRawRune() (rune, int, error) {
	var (
		character rune
		size      int
	)
	if parser.pending {
		character = parser.pendingRune
		size = parser.pendingSize
		parser.pending = false
	} else {
		first, err := parser.reader.ReadByte()
		if err != nil {
			return 0, 0, err
		}
		if first < utf8.RuneSelf {
			character = rune(first)
			size = 1
		} else {
			if err := parser.reader.UnreadByte(); err != nil {
				return 0, 0, fmt.Errorf("unread CSV byte: %w", err)
			}
			character, size, err = parser.reader.ReadRune()
			if err != nil {
				return 0, 0, err
			}
		}
	}
	return character, size, nil
}

func (parser *Parser) unreadRune(character rune) error {
	if !parser.canUnread {
		return errors.New("no CSV character is available to unread")
	}
	if parser.pending {
		return errors.New("CSV parser already has a pending character")
	}
	size := utf8.RuneLen(character)
	parser.pending = true
	parser.pendingRune = character
	parser.pendingSize = size
	parser.offset -= int64(size)
	parser.canUnread = false
	return nil
}

func (parser *Parser) consumeOptionalLF() error {
	next, size, err := parser.readRawRune()
	parser.canUnread = false
	if errors.Is(err, io.EOF) {
		parser.finished = true
		return nil
	}
	if err != nil {
		return err
	}
	if next != '\n' {
		parser.pending = true
		parser.pendingRune = next
		parser.pendingSize = size
		return nil
	}
	parser.offset += int64(size)
	if parser.offset-parser.recordStart > parser.options.MaxRecordBytes {
		return &ParseError{
			Position: model.SourcePosition{
				Connector: "csv",
				Resource:  parser.options.Resource,
				Offset:    parser.recordStart,
				Line:      parser.line,
			},
			Err: fmt.Errorf("record exceeds %d bytes", parser.options.MaxRecordBytes),
		}
	}
	return nil
}
