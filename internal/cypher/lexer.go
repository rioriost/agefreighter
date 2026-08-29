package cypher

import (
	"context"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

func lex(ctx context.Context, input []byte) ([]statement, *Finding, error) {
	scanner := lexer{ctx: ctx, input: input, line: 1, column: 1}
	tokens, finding, err := scanner.scan()
	if err != nil {
		return nil, nil, err
	}
	if finding != nil && (len(tokens) == 0 ||
		tokens[len(tokens)-1].kind == tokenSemicolon) {
		tokens = append(tokens, token{
			kind: tokenPunctuation, text: "<invalid>",
			line: finding.Line, column: finding.Column,
		})
	}
	statements := make([]statement, 0)
	start := 0
	for index, value := range tokens {
		if value.kind != tokenSemicolon {
			continue
		}
		if index > start {
			statements = append(statements, statement{
				tokens: append([]token(nil), tokens[start:index]...),
			})
		}
		start = index + 1
	}
	if start < len(tokens) {
		statements = append(statements, statement{
			tokens: append([]token(nil), tokens[start:]...),
		})
	}
	return statements, finding, nil
}

type lexer struct {
	ctx    context.Context
	input  []byte
	offset int
	line   int
	column int
	tokens []token
	depth  []rune
}

func (scanner *lexer) scan() ([]token, *Finding, error) {
	for scanner.offset < len(scanner.input) {
		if scanner.offset&1023 == 0 {
			if err := scanner.ctx.Err(); err != nil {
				return nil, nil, err
			}
		}
		character, width := utf8.DecodeRune(scanner.input[scanner.offset:])
		switch {
		case unicode.IsSpace(character):
			scanner.advance(character, width)
		case character == '/' && scanner.peekByte(1) == '/':
			scanner.skipLineComment()
		case character == '/' && scanner.peekByte(1) == '*':
			if finding := scanner.skipBlockComment(); finding != nil {
				if finding.Code == "AGE16-X004" {
					return nil, nil, fmt.Errorf(
						"%w: block-comment nesting exceeds %d",
						errLimit,
						MaxDepth,
					)
				}
				return scanner.tokens, finding, nil
			}
		case character == '\'' || character == '"':
			if finding := scanner.scanString(character); finding != nil {
				return scanner.tokens, finding, nil
			}
		case character == '`':
			if finding := scanner.scanEscapedIdentifier(); finding != nil {
				return scanner.tokens, finding, nil
			}
		case character == '$':
			scanner.scanParameter()
		case unicode.IsLetter(character) || character == '_':
			scanner.scanIdentifier()
		case character >= '0' && character <= '9':
			scanner.scanNumber()
		default:
			finding := scanner.scanSymbol(character, width)
			if finding != nil {
				if finding.Code == "AGE16-X004" {
					return nil, nil, fmt.Errorf(
						"%w: token nesting exceeds %d",
						errLimit,
						MaxDepth,
					)
				}
				return scanner.tokens, finding, nil
			}
		}
		if len(scanner.tokens) > MaxQueries*MaxTokens {
			return nil, nil, fmt.Errorf("%w: token budget exceeded", errLimit)
		}
	}
	if len(scanner.depth) != 0 {
		last := scanner.tokens[len(scanner.tokens)-1]
		return scanner.tokens, &Finding{
			Code: "AGE16-X002", Severity: SeverityUnknown,
			Line: last.line, Column: last.column,
			Evidence:    "unclosed grouping delimiter",
			Remediation: "close every parenthesis, bracket, and map delimiter before rechecking",
		}, nil
	}
	return scanner.tokens, nil, nil
}

func (scanner *lexer) skipLineComment() {
	for scanner.offset < len(scanner.input) {
		character, width := utf8.DecodeRune(scanner.input[scanner.offset:])
		scanner.advance(character, width)
		if character == '\n' {
			return
		}
	}
}

func (scanner *lexer) skipBlockComment() *Finding {
	line, column := scanner.line, scanner.column
	depth := 0
	for scanner.offset < len(scanner.input) {
		switch {
		case scanner.peekByte(0) == '/' && scanner.peekByte(1) == '*':
			depth++
			scanner.advance('/', 1)
			scanner.advance('*', 1)
			if depth > MaxDepth {
				return limitFinding(line, column, "block-comment nesting")
			}
		case scanner.peekByte(0) == '*' && scanner.peekByte(1) == '/':
			depth--
			scanner.advance('*', 1)
			scanner.advance('/', 1)
			if depth == 0 {
				return nil
			}
		default:
			character, width := utf8.DecodeRune(scanner.input[scanner.offset:])
			scanner.advance(character, width)
		}
	}
	return &Finding{
		Code: "AGE16-X001", Severity: SeverityUnknown,
		Line: line, Column: column, Evidence: "unterminated block comment",
		Remediation: "terminate the block comment before rechecking",
	}
}

func (scanner *lexer) scanString(quote rune) *Finding {
	line, column, offset := scanner.line, scanner.column, scanner.offset
	scanner.advance(quote, 1)
	for scanner.offset < len(scanner.input) {
		character, width := utf8.DecodeRune(scanner.input[scanner.offset:])
		if character == '\\' {
			scanner.advance(character, width)
			if scanner.offset < len(scanner.input) {
				next, nextWidth := utf8.DecodeRune(scanner.input[scanner.offset:])
				scanner.advance(next, nextWidth)
			}
			continue
		}
		if character == quote {
			scanner.advance(character, width)
			if scanner.offset < len(scanner.input) {
				next, nextWidth := utf8.DecodeRune(scanner.input[scanner.offset:])
				if next == quote {
					scanner.advance(next, nextWidth)
					continue
				}
			}
			scanner.add(tokenString, "<string>", line, column, offset)
			return nil
		}
		scanner.advance(character, width)
	}
	return &Finding{
		Code: "AGE16-X001", Severity: SeverityUnknown,
		Line: line, Column: column, Evidence: "unterminated string literal",
		Remediation: "terminate the string literal before rechecking",
	}
}

func (scanner *lexer) scanEscapedIdentifier() *Finding {
	line, column, offset := scanner.line, scanner.column, scanner.offset
	scanner.advance('`', 1)
	var value strings.Builder
	for scanner.offset < len(scanner.input) {
		character, width := utf8.DecodeRune(scanner.input[scanner.offset:])
		if character == '`' {
			scanner.advance(character, width)
			if scanner.offset < len(scanner.input) && scanner.peekByte(0) == '`' {
				value.WriteRune('`')
				scanner.advance('`', 1)
				continue
			}
			scanner.add(tokenEscapedIdentifier, value.String(), line, column, offset)
			return nil
		}
		if unicode.IsControl(character) {
			return &Finding{
				Code: "AGE16-X001", Severity: SeverityUnknown,
				Line: line, Column: column,
				Evidence:    "invalid escaped identifier",
				Remediation: "remove control characters from the escaped identifier",
			}
		}
		value.WriteRune(character)
		scanner.advance(character, width)
	}
	return &Finding{
		Code: "AGE16-X001", Severity: SeverityUnknown,
		Line: line, Column: column, Evidence: "unterminated escaped identifier",
		Remediation: "terminate the escaped identifier before rechecking",
	}
}

func (scanner *lexer) scanParameter() {
	line, column, offset := scanner.line, scanner.column, scanner.offset
	scanner.advance('$', 1)
	start := scanner.offset
	for scanner.offset < len(scanner.input) {
		character, width := utf8.DecodeRune(scanner.input[scanner.offset:])
		if !unicode.IsLetter(character) && !unicode.IsDigit(character) &&
			character != '_' {
			break
		}
		scanner.advance(character, width)
	}
	text := "$<parameter>"
	if scanner.offset == start {
		text = "$<invalid-parameter>"
	}
	scanner.add(tokenParameter, text, line, column, offset)
}

func (scanner *lexer) scanIdentifier() {
	line, column, offset := scanner.line, scanner.column, scanner.offset
	start := scanner.offset
	for scanner.offset < len(scanner.input) {
		character, width := utf8.DecodeRune(scanner.input[scanner.offset:])
		if !unicode.IsLetter(character) && !unicode.IsDigit(character) &&
			character != '_' {
			break
		}
		scanner.advance(character, width)
	}
	scanner.add(
		tokenIdentifier,
		string(scanner.input[start:scanner.offset]),
		line,
		column,
		offset,
	)
}

func (scanner *lexer) scanNumber() {
	line, column, offset := scanner.line, scanner.column, scanner.offset
	valid := true
	if scanner.peekByte(0) == '0' &&
		(scanner.peekByte(1) == 'x' || scanner.peekByte(1) == 'X') {
		scanner.advance('0', 1)
		scanner.advance(rune(scanner.peekByte(0)), 1)
		start := scanner.offset
		for isHexDigit(scanner.peekByte(0)) {
			scanner.advance(rune(scanner.peekByte(0)), 1)
		}
		valid = scanner.offset > start
	} else if scanner.peekByte(0) == '0' &&
		(scanner.peekByte(1) == 'o' || scanner.peekByte(1) == 'O') {
		scanner.advance('0', 1)
		scanner.advance(rune(scanner.peekByte(0)), 1)
		start := scanner.offset
		for scanner.peekByte(0) >= '0' && scanner.peekByte(0) <= '7' {
			scanner.advance(rune(scanner.peekByte(0)), 1)
		}
		valid = scanner.offset > start
	} else {
		for isDecimalDigit(scanner.peekByte(0)) {
			scanner.advance(rune(scanner.peekByte(0)), 1)
		}
		if scanner.peekByte(0) == '.' && scanner.peekByte(1) != '.' {
			scanner.advance('.', 1)
			start := scanner.offset
			for isDecimalDigit(scanner.peekByte(0)) {
				scanner.advance(rune(scanner.peekByte(0)), 1)
			}
			valid = scanner.offset > start
		}
		if scanner.peekByte(0) == 'e' || scanner.peekByte(0) == 'E' {
			scanner.advance(rune(scanner.peekByte(0)), 1)
			if scanner.peekByte(0) == '+' || scanner.peekByte(0) == '-' {
				scanner.advance(rune(scanner.peekByte(0)), 1)
			}
			start := scanner.offset
			for isDecimalDigit(scanner.peekByte(0)) {
				scanner.advance(rune(scanner.peekByte(0)), 1)
			}
			valid = valid && scanner.offset > start
		}
	}
	text := "<number>"
	if !valid {
		text = "<invalid-number>"
	}
	scanner.add(tokenNumber, text, line, column, offset)
}

func isDecimalDigit(value byte) bool {
	return value >= '0' && value <= '9'
}

func isHexDigit(value byte) bool {
	return isDecimalDigit(value) ||
		value >= 'a' && value <= 'f' ||
		value >= 'A' && value <= 'F'
}

func (scanner *lexer) scanSymbol(character rune, width int) *Finding {
	line, column, offset := scanner.line, scanner.column, scanner.offset
	if strings.ContainsRune("()[]{}", character) {
		if strings.ContainsRune("([{", character) {
			scanner.depth = append(scanner.depth, character)
			if len(scanner.depth) > MaxDepth {
				return limitFinding(line, column, "token nesting")
			}
		} else {
			if len(scanner.depth) == 0 ||
				!matchingDelimiter(scanner.depth[len(scanner.depth)-1], character) {
				return &Finding{
					Code: "AGE16-X002", Severity: SeverityUnknown,
					Line: line, Column: column,
					Evidence:    "mismatched grouping delimiter",
					Remediation: "balance grouping delimiters before rechecking",
				}
			}
			scanner.depth = scanner.depth[:len(scanner.depth)-1]
		}
		scanner.advance(character, width)
		scanner.add(tokenPunctuation, string(character), line, column, offset)
		return nil
	}
	if character == ';' {
		scanner.advance(character, width)
		kind := tokenSemicolon
		if len(scanner.depth) != 0 {
			kind = tokenPunctuation
		}
		scanner.add(kind, ";", line, column, offset)
		return nil
	}
	if strings.ContainsRune(",:.", character) {
		scanner.advance(character, width)
		scanner.add(tokenPunctuation, string(character), line, column, offset)
		return nil
	}
	if strings.ContainsRune("=<>+-*/%^|&!", character) {
		scanner.advance(character, width)
		text := string(character)
		if scanner.offset < len(scanner.input) {
			next, nextWidth := utf8.DecodeRune(scanner.input[scanner.offset:])
			candidate := text + string(next)
			if candidate == "<=" || candidate == ">=" || candidate == "<>" ||
				candidate == "!=" || candidate == "=~" || candidate == "->" ||
				candidate == "<-" || candidate == ".." || candidate == "+=" {
				scanner.advance(next, nextWidth)
				text = candidate
			}
		}
		scanner.add(tokenOperator, text, line, column, offset)
		return nil
	}
	scanner.advance(character, width)
	return &Finding{
		Code: "AGE16-X001", Severity: SeverityUnknown,
		Line: line, Column: column, Evidence: "unrecognized token",
		Remediation: "replace the token with documented openCypher syntax",
	}
}

func (scanner *lexer) add(
	kind tokenKind,
	text string,
	line, column, offset int,
) {
	scanner.tokens = append(scanner.tokens, token{
		kind: kind, text: text, line: line, column: column, offset: offset,
	})
}

func (scanner *lexer) advance(character rune, width int) {
	scanner.offset += width
	if character == '\n' {
		scanner.line++
		scanner.column = 1
	} else {
		scanner.column++
	}
}

func (scanner *lexer) peekByte(distance int) byte {
	index := scanner.offset + distance
	if index < 0 || index >= len(scanner.input) {
		return 0
	}
	return scanner.input[index]
}

func matchingDelimiter(open, close rune) bool {
	return open == '(' && close == ')' ||
		open == '[' && close == ']' ||
		open == '{' && close == '}'
}

func limitFinding(line, column int, subject string) *Finding {
	return &Finding{
		Code: "AGE16-X004", Severity: SeverityUnknown,
		Line: line, Column: column, Evidence: subject + " exceeds limit",
		Remediation: "split or simplify the query before rechecking",
	}
}
