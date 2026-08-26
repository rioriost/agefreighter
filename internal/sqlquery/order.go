package sqlquery

import "strings"

// HasTopLevelOrderBy reports whether query contains an ORDER BY clause outside
// comments, quoted values, identifiers, dollar-quoted strings, and subqueries.
func HasTopLevelOrderBy(query string) bool {
	depth := 0
	previous := ""
	for index := 0; index < len(query); {
		switch {
		case query[index] == '\'':
			index = skipQuoted(query, index+1, '\'', true)
		case query[index] == '"':
			index = skipQuoted(query, index+1, '"', false)
		case index+1 < len(query) && query[index:index+2] == "--":
			index = skipLineComment(query, index+2)
		case index+1 < len(query) && query[index:index+2] == "/*":
			index = skipBlockComment(query, index+2)
		case query[index] == '$':
			if delimiter, ok := dollarDelimiter(query[index:]); ok {
				index = skipDollarQuoted(query, index, delimiter)
			} else {
				index++
			}
		case query[index] == '(':
			depth++
			previous = ""
			index++
		case query[index] == ')':
			if depth > 0 {
				depth--
			}
			previous = ""
			index++
		case isIdentifierStart(query[index]):
			start := index
			for index < len(query) && isIdentifierPart(query[index]) {
				index++
			}
			if depth != 0 {
				continue
			}
			current := strings.ToLower(query[start:index])
			if previous == "order" && current == "by" {
				return true
			}
			previous = current
		default:
			index++
		}
	}
	return false
}

func skipQuoted(query string, index int, quote byte, backslashEscapes bool) int {
	for index < len(query) {
		if backslashEscapes && query[index] == '\\' && index+1 < len(query) {
			index += 2
			continue
		}
		if query[index] != quote {
			index++
			continue
		}
		index++
		if index < len(query) && query[index] == quote {
			index++
			continue
		}
		return index
	}
	return index
}

func skipLineComment(query string, index int) int {
	for index < len(query) && query[index] != '\n' {
		index++
	}
	return index
}

func skipBlockComment(query string, index int) int {
	depth := 1
	for index < len(query) && depth > 0 {
		switch {
		case index+1 < len(query) && query[index:index+2] == "/*":
			depth++
			index += 2
		case index+1 < len(query) && query[index:index+2] == "*/":
			depth--
			index += 2
		default:
			index++
		}
	}
	return index
}

func dollarDelimiter(query string) (string, bool) {
	if query[0] != '$' {
		return "", false
	}
	for index := 1; index < len(query); index++ {
		switch {
		case query[index] == '$':
			return query[:index+1], true
		case !isIdentifierPart(query[index]):
			return "", false
		}
	}
	return "", false
}

func skipDollarQuoted(query string, index int, delimiter string) int {
	start := index + len(delimiter)
	end := strings.Index(query[start:], delimiter)
	if end < 0 {
		return len(query)
	}
	return start + end + len(delimiter)
}

func isIdentifierStart(character byte) bool {
	return character == '_' ||
		character >= 'A' && character <= 'Z' ||
		character >= 'a' && character <= 'z'
}

func isIdentifierPart(character byte) bool {
	return isIdentifierStart(character) ||
		character >= '0' && character <= '9'
}
