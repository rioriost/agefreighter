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
			index = skipQuoted(query, index+1, '"', true)
		case query[index] == '`':
			index = skipQuoted(query, index+1, '`', true)
		case index+1 < len(query) && query[index:index+2] == "--":
			index = skipLineComment(query, index+2)
		case index+1 < len(query) && query[index:index+2] == "//":
			index = skipLineComment(query, index+2)
		case index+1 < len(query) && query[index:index+2] == "/*":
			index = skipBlockComment(query, index+2)
		case query[index] == '$':
			if delimiter, ok := dollarDelimiter(query[index:]); ok {
				index = skipDollarQuoted(query, index, delimiter)
			} else {
				index++
			}
		case query[index] == '(' || query[index] == '{' || query[index] == '[':
			depth++
			previous = ""
			index++
		case query[index] == ')' || query[index] == '}' || query[index] == ']':
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

// HasParameter reports whether query references a named parameter outside
// comments and quoted values. The parameter name must not include '$'.
func HasParameter(query, name string) bool {
	for index := 0; index < len(query); {
		switch {
		case query[index] == '\'':
			index = skipQuoted(query, index+1, '\'', true)
		case query[index] == '"':
			index = skipQuoted(query, index+1, '"', true)
		case query[index] == '`':
			index = skipQuoted(query, index+1, '`', true)
		case index+1 < len(query) && query[index:index+2] == "--":
			index = skipLineComment(query, index+2)
		case index+1 < len(query) && query[index:index+2] == "//":
			index = skipLineComment(query, index+2)
		case index+1 < len(query) && query[index:index+2] == "/*":
			index = skipBlockComment(query, index+2)
		case query[index] == '$':
			if delimiter, ok := dollarDelimiter(query[index:]); ok {
				index = skipDollarQuoted(query, index, delimiter)
				continue
			}
			start := index + 1
			end := start
			for end < len(query) && isIdentifierPart(query[end]) {
				end++
			}
			if query[start:end] == name {
				return true
			}
			index = end
		default:
			index++
		}
	}
	return false
}

// HasKeyword reports whether query contains a keyword outside comments and
// quoted values.
func HasKeyword(query, keyword string) bool {
	for index := 0; index < len(query); {
		switch {
		case query[index] == '\'':
			index = skipQuoted(query, index+1, '\'', true)
		case query[index] == '"':
			index = skipQuoted(query, index+1, '"', true)
		case query[index] == '`':
			index = skipQuoted(query, index+1, '`', true)
		case index+1 < len(query) && query[index:index+2] == "--":
			index = skipLineComment(query, index+2)
		case index+1 < len(query) && query[index:index+2] == "//":
			index = skipLineComment(query, index+2)
		case index+1 < len(query) && query[index:index+2] == "/*":
			index = skipBlockComment(query, index+2)
		case query[index] == '$':
			if delimiter, ok := dollarDelimiter(query[index:]); ok {
				index = skipDollarQuoted(query, index, delimiter)
			} else {
				index++
			}
		case isIdentifierStart(query[index]):
			start := index
			for index < len(query) && isIdentifierPart(query[index]) {
				index++
			}
			if strings.EqualFold(query[start:index], keyword) {
				return true
			}
		default:
			index++
		}
	}
	return false
}

// HasFinalTopLevelOrderByField reports whether the final top-level result is
// ordered ascending by field as its first ordering expression.
func HasFinalTopLevelOrderByField(query, field string) bool {
	tokens := topLevelTokens(query)
	orderIndex := -1
	for index := 0; index+1 < len(tokens); index++ {
		if strings.EqualFold(tokens[index], "union") {
			return false
		}
		if strings.EqualFold(tokens[index], "order") &&
			strings.EqualFold(tokens[index+1], "by") {
			orderIndex = index
		}
	}
	if orderIndex < 0 || orderIndex+2 >= len(tokens) {
		return false
	}
	for _, token := range tokens[orderIndex+2:] {
		if strings.EqualFold(token, "union") ||
			strings.EqualFold(token, "return") ||
			strings.EqualFold(token, "with") {
			return false
		}
	}
	if tokens[orderIndex+2] != field {
		return false
	}
	next := orderIndex + 3
	if next >= len(tokens) {
		return true
	}
	if strings.EqualFold(tokens[next], "desc") || tokens[next] == "." {
		return false
	}
	if strings.EqualFold(tokens[next], "asc") {
		next++
		if next >= len(tokens) {
			return true
		}
	}
	return tokens[next] == ","
}

// HasTopLevelOrderByField reports whether the final top-level ORDER BY starts
// with field ascending. Unlike HasFinalTopLevelOrderByField, it permits SQL
// pagination clauses after the ordering expression.
func HasTopLevelOrderByField(query, field string) bool {
	tokens := topLevelTokens(query)
	orderIndex := -1
	for index := 0; index+1 < len(tokens); index++ {
		if strings.EqualFold(tokens[index], "union") {
			return false
		}
		if strings.EqualFold(tokens[index], "order") &&
			strings.EqualFold(tokens[index+1], "by") {
			orderIndex = index
		}
	}
	if orderIndex < 0 || orderIndex+2 >= len(tokens) ||
		tokens[orderIndex+2] != field {
		return false
	}
	next := orderIndex + 3
	if next >= len(tokens) {
		return true
	}
	if strings.EqualFold(tokens[next], "desc") || tokens[next] == "." {
		return false
	}
	if strings.EqualFold(tokens[next], "asc") {
		next++
		if next >= len(tokens) {
			return true
		}
	}
	return tokens[next] == "," ||
		strings.EqualFold(tokens[next], "limit") ||
		strings.EqualFold(tokens[next], "offset") ||
		strings.EqualFold(tokens[next], "fetch")
}

// HasFinalTopLevelLimitParameter reports whether the final top-level clause is
// LIMIT followed by the named parameter. Quoted text, comments, and nested
// subqueries are ignored in the same way as the ordering helpers.
func HasFinalTopLevelLimitParameter(query, name string) bool {
	tokens := topLevelTokens(query)
	if len(tokens) < 3 {
		return false
	}
	last := len(tokens) - 1
	return strings.EqualFold(tokens[last-2], "limit") &&
		tokens[last-1] == "$" && tokens[last] == name
}

func topLevelTokens(query string) []string {
	var tokens []string
	depth := 0
	for index := 0; index < len(query); {
		switch {
		case query[index] == '\'':
			index = skipQuoted(query, index+1, '\'', true)
		case query[index] == '"':
			index = skipQuoted(query, index+1, '"', true)
		case query[index] == '`':
			index = skipQuoted(query, index+1, '`', true)
		case index+1 < len(query) &&
			(query[index:index+2] == "--" || query[index:index+2] == "//"):
			index = skipLineComment(query, index+2)
		case index+1 < len(query) && query[index:index+2] == "/*":
			index = skipBlockComment(query, index+2)
		case query[index] == '$':
			if delimiter, ok := dollarDelimiter(query[index:]); ok {
				index = skipDollarQuoted(query, index, delimiter)
			} else {
				if depth == 0 {
					tokens = append(tokens, "$")
				}
				index++
			}
		case query[index] == '(' || query[index] == '{' || query[index] == '[':
			depth++
			index++
		case query[index] == ')' || query[index] == '}' || query[index] == ']':
			if depth > 0 {
				depth--
			}
			index++
		case isIdentifierStart(query[index]):
			start := index
			for index < len(query) && isIdentifierPart(query[index]) {
				index++
			}
			if depth == 0 {
				tokens = append(tokens, query[start:index])
			}
		case query[index] == ' ' || query[index] == '\t' ||
			query[index] == '\r' || query[index] == '\n':
			index++
		default:
			if depth == 0 {
				tokens = append(tokens, query[index:index+1])
			}
			index++
		}
	}
	return tokens
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
