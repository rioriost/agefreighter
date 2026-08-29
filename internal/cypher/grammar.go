package cypher

import "strings"

type structuralClause struct {
	name         string
	token        token
	start        int
	payloadStart int
}

var structuralClauseWords = map[string]bool{
	"MATCH": true, "OPTIONAL": true, "WHERE": true, "WITH": true,
	"RETURN": true, "UNWIND": true, "CREATE": true, "MERGE": true,
	"ON": true, "SET": true, "REMOVE": true, "DELETE": true,
	"DETACH": true, "FOREACH": true, "UNION": true, "ORDER": true,
	"SKIP": true, "LIMIT": true, "CALL": true,
	"SHOW": true, "USE": true, "PROFILE": true, "EXPLAIN": true,
	"DROP": true,
}

func validateStructuralConsumption(tokens []token) (token, bool) {
	if len(tokens) == 0 {
		return token{line: 1, column: 1}, false
	}
	for _, value := range tokens {
		if value.kind == tokenNumber && value.text == "<invalid-number>" ||
			value.kind == tokenParameter && value.text == "$<invalid-parameter>" {
			return value, false
		}
	}
	clauses := structuralClauses(tokens)
	if len(clauses) == 0 || clauses[0].start != 0 {
		return tokens[0], false
	}
	for index, clause := range clauses {
		end := len(tokens)
		if index+1 < len(clauses) {
			end = clauses[index+1].start
		}
		if index > 0 && !validStructuralTransition(clauses[index-1].name, clause.name) {
			return clause.token, false
		}
		payload := tokens[clause.payloadStart:end]
		if !validateClausePayload(clause.name, payload) {
			if len(payload) > 0 {
				return payload[0], false
			}
			return clause.token, false
		}
	}
	return token{}, true
}

func structuralClauses(tokens []token) []structuralClause {
	clauses := make([]structuralClause, 0, 8)
	depth := 0
	for index := 0; index < len(tokens); index++ {
		value := tokens[index]
		if value.kind == tokenPunctuation {
			switch value.text {
			case "(", "[", "{":
				depth++
			case ")", "]", "}":
				depth--
			}
			continue
		}
		if depth != 0 || value.kind != tokenIdentifier ||
			index > 0 && tokens[index-1].text == "." {
			continue
		}
		word := upper(value)
		if !structuralClauseWords[word] {
			continue
		}
		name, width := word, 1
		switch {
		case word == "OPTIONAL" && tokenWord(tokens, index+1) == "MATCH":
			name, width = "MATCH", 2
		case word == "DETACH" && tokenWord(tokens, index+1) == "DELETE":
			name, width = "DELETE", 2
		case word == "ORDER" && tokenWord(tokens, index+1) == "BY":
			name, width = "ORDER", 2
		case word == "ON" &&
			(tokenWord(tokens, index+1) == "MATCH" ||
				tokenWord(tokens, index+1) == "CREATE") &&
			tokenWord(tokens, index+2) == "SET":
			name, width = "MERGE SET", 3
		case word == "UNION" && tokenWord(tokens, index+1) == "ALL":
			width = 2
		case word == "ON":
			continue
		}
		clauses = append(clauses, structuralClause{
			name: name, token: value, start: index, payloadStart: index + width,
		})
		index += width - 1
	}
	return clauses
}

func tokenWord(tokens []token, index int) string {
	if index < 0 || index >= len(tokens) {
		return ""
	}
	return upper(tokens[index])
}

func validStructuralTransition(left, right string) bool {
	switch left {
	case "MATCH":
		return oneOf(right, "WHERE", "WITH", "RETURN", "UNWIND", "CREATE",
			"MERGE", "SET", "REMOVE", "DELETE", "FOREACH")
	case "WHERE":
		return oneOf(right, "WITH", "RETURN", "CREATE", "MERGE", "SET",
			"REMOVE", "DELETE", "FOREACH")
	case "WITH":
		return oneOf(right, "WHERE", "ORDER", "SKIP", "LIMIT", "MATCH",
			"UNWIND", "CREATE", "MERGE", "SET", "REMOVE", "DELETE",
			"FOREACH", "RETURN", "CALL")
	case "RETURN":
		return oneOf(right, "ORDER", "SKIP", "LIMIT", "UNION")
	case "ORDER":
		return oneOf(right, "SKIP", "LIMIT")
	case "SKIP":
		return right == "LIMIT"
	case "UNION":
		return oneOf(right, "MATCH", "RETURN", "WITH", "UNWIND", "CREATE",
			"MERGE", "CALL")
	case "MERGE":
		return oneOf(right, "MERGE SET", "WITH", "RETURN", "SET", "REMOVE",
			"DELETE")
	case "MERGE SET":
		return oneOf(right, "MERGE SET", "WITH", "RETURN", "SET", "REMOVE",
			"DELETE")
	case "UNWIND", "CREATE", "SET", "REMOVE", "DELETE", "FOREACH", "CALL":
		return oneOf(right, "MATCH", "WHERE", "WITH", "RETURN", "UNWIND",
			"CREATE", "MERGE", "SET", "REMOVE", "DELETE", "FOREACH")
	case "USE", "PROFILE", "EXPLAIN":
		return oneOf(right, "MATCH", "RETURN", "WITH", "UNWIND", "CREATE",
			"MERGE", "CALL")
	default:
		return false
	}
}

func oneOf(value string, values ...string) bool {
	for _, candidate := range values {
		if value == candidate {
			return true
		}
	}
	return false
}

func validateClausePayload(name string, payload []token) bool {
	if name == "UNION" {
		return len(payload) == 0
	}
	if name == "PROFILE" || name == "EXPLAIN" {
		return len(payload) == 0
	}
	if len(payload) == 0 {
		return false
	}
	switch name {
	case "MATCH", "CREATE", "MERGE":
		return validatePatternPayload(payload)
	case "WHERE", "SKIP", "LIMIT":
		return parseWholeExpression(payload)
	case "RETURN", "WITH":
		return parseProjection(payload, false)
	case "ORDER":
		return parseProjection(payload, true)
	case "UNWIND":
		return parseUnwind(payload)
	case "SET", "MERGE SET":
		return parseExpressionList(payload)
	case "REMOVE", "DELETE":
		return parseExpressionList(payload)
	case "FOREACH":
		return validatePatternPayload(payload)
	case "CALL":
		if payload[0].text == "{" {
			return true
		}
		return parseWholeExpression(payload)
	case "SHOW":
		return len(payload) == 1 && oneOf(
			tokenWord(payload, 0),
			"DATABASES", "INDEXES", "CONSTRAINTS", "FUNCTIONS",
			"PROCEDURES", "TRANSACTIONS",
		)
	case "USE":
		return len(payload) == 1 && identifier(payload[0])
	case "DROP":
		return len(payload) == 2 &&
			oneOf(tokenWord(payload, 0), "INDEX", "CONSTRAINT") &&
			identifier(payload[1])
	default:
		return false
	}
}

func validatePatternPayload(tokens []token) bool {
	_, ok := parsePatternPayload(tokens)
	return ok
}

type patternBinding struct {
	variable string
	labels   []string
	kind     string
}

type patternParser struct {
	tokens   []token
	index    int
	bindings []patternBinding
}

func parsePatternPayload(tokens []token) ([]patternBinding, bool) {
	parser := patternParser{tokens: tokens}
	if !parser.parsePath() {
		return nil, false
	}
	for parser.consume(",") {
		if !parser.parsePath() {
			return nil, false
		}
	}
	if parser.index != len(tokens) {
		return nil, false
	}
	return parser.bindings, true
}

func (parser *patternParser) parsePath() bool {
	if parser.index+2 < len(parser.tokens) &&
		identifier(parser.tokens[parser.index]) &&
		parser.tokens[parser.index+1].text == "=" {
		parser.bindings = append(parser.bindings, patternBinding{
			variable: parser.tokens[parser.index].text,
		})
		parser.index += 2
	}
	if !parser.parseElement("(", ")", "vertex-label") {
		return false
	}
	for parser.index < len(parser.tokens) &&
		(parser.tokens[parser.index].text == "-" ||
			parser.tokens[parser.index].text == "<-") {
		left := parser.tokens[parser.index].text
		parser.index++
		if parser.index < len(parser.tokens) &&
			parser.tokens[parser.index].text == "[" {
			if !parser.parseElement("[", "]", "edge-label") {
				return false
			}
		}
		if left == "<-" {
			if !parser.consume("-") {
				return false
			}
		} else if !parser.consume("-") && !parser.consume("->") {
			return false
		}
		if !parser.parseElement("(", ")", "vertex-label") {
			return false
		}
	}
	return true
}

func (parser *patternParser) parseElement(open, close, kind string) bool {
	if !parser.consume(open) {
		return false
	}
	binding := patternBinding{kind: kind}
	if parser.index < len(parser.tokens) &&
		identifier(parser.tokens[parser.index]) {
		binding.variable = parser.tokens[parser.index].text
		parser.index++
	}
	if kind == "vertex-label" {
		for parser.consume(":") {
			if parser.index >= len(parser.tokens) ||
				!identifier(parser.tokens[parser.index]) {
				return false
			}
			binding.labels = append(binding.labels, parser.tokens[parser.index].text)
			parser.index++
		}
	} else if parser.consume(":") {
		if !parser.parseRelationshipTypes(&binding) {
			return false
		}
	}
	if kind == "edge-label" && parser.consume("*") &&
		!parser.parseRelationshipRange() {
		return false
	}
	if parser.index < len(parser.tokens) &&
		parser.tokens[parser.index].text == "{" {
		expression := expressionParser{
			tokens: parser.tokens,
			index:  parser.index,
		}
		if !expression.parseMap() {
			return false
		}
		parser.index = expression.index
	} else if parser.index < len(parser.tokens) &&
		parser.tokens[parser.index].kind == tokenParameter {
		if parser.tokens[parser.index].text == "$<invalid-parameter>" {
			return false
		}
		parser.index++
	}
	if !parser.consume(close) {
		return false
	}
	parser.bindings = append(parser.bindings, binding)
	return true
}

func (parser *patternParser) parseRelationshipTypes(
	binding *patternBinding,
) bool {
	if parser.index >= len(parser.tokens) ||
		!identifier(parser.tokens[parser.index]) {
		return false
	}
	binding.labels = append(binding.labels, parser.tokens[parser.index].text)
	parser.index++
	for parser.consume("|") {
		parser.consume(":")
		if parser.index >= len(parser.tokens) ||
			!identifier(parser.tokens[parser.index]) {
			return false
		}
		binding.labels = append(binding.labels, parser.tokens[parser.index].text)
		parser.index++
	}
	return true
}

func (parser *patternParser) parseRelationshipRange() bool {
	if parser.index < len(parser.tokens) &&
		parser.tokens[parser.index].kind == tokenNumber {
		if parser.tokens[parser.index].text == "<invalid-number>" {
			return false
		}
		parser.index++
	}
	if parser.consume("..") {
		if parser.index < len(parser.tokens) &&
			parser.tokens[parser.index].kind == tokenNumber {
			if parser.tokens[parser.index].text == "<invalid-number>" {
				return false
			}
			parser.index++
		}
	}
	return true
}

func (parser *patternParser) consume(text string) bool {
	if parser.index < len(parser.tokens) &&
		parser.tokens[parser.index].text == text {
		parser.index++
		return true
	}
	return false
}

func parseUnwind(tokens []token) bool {
	as := topLevelWord(tokens, "AS")
	return as > 0 && as+2 == len(tokens) &&
		parseWholeExpression(tokens[:as]) && identifier(tokens[as+1])
}

func parseProjection(tokens []token, ordering bool) bool {
	cursor := 0
	if !ordering && tokenWord(tokens, 0) == "DISTINCT" {
		cursor++
	}
	for cursor < len(tokens) {
		next := topLevelComma(tokens, cursor)
		end := len(tokens)
		if next >= 0 {
			end = next
		}
		item := tokens[cursor:end]
		if len(item) == 0 {
			return false
		}
		if ordering {
			last := tokenWord(item, len(item)-1)
			if last == "ASC" || last == "DESC" {
				item = item[:len(item)-1]
			}
		} else {
			as := topLevelWord(item, "AS")
			if as >= 0 {
				if as == 0 || as+2 != len(item) || !identifier(item[as+1]) {
					return false
				}
				item = item[:as]
			}
		}
		if len(item) == 1 && item[0].text == "*" {
			// RETURN * is a complete projection.
		} else if !parseWholeExpression(item) {
			return false
		}
		if next < 0 {
			return true
		}
		cursor = next + 1
	}
	return false
}

func parseExpressionList(tokens []token) bool {
	cursor := 0
	for cursor < len(tokens) {
		next := topLevelComma(tokens, cursor)
		end := len(tokens)
		if next >= 0 {
			end = next
		}
		if end == cursor || !parseWholeExpression(tokens[cursor:end]) {
			return false
		}
		if next < 0 {
			return true
		}
		cursor = next + 1
	}
	return false
}

func topLevelComma(tokens []token, start int) int {
	depth := 0
	for index := start; index < len(tokens); index++ {
		switch tokens[index].text {
		case "(", "[", "{":
			depth++
		case ")", "]", "}":
			depth--
		case ",":
			if depth == 0 {
				return index
			}
		}
	}
	return -1
}

func topLevelWord(tokens []token, word string) int {
	depth := 0
	for index, value := range tokens {
		switch value.text {
		case "(", "[", "{":
			depth++
		case ")", "]", "}":
			depth--
		default:
			if depth == 0 && tokenWord(tokens, index) == word {
				return index
			}
		}
	}
	return -1
}

type expressionParser struct {
	tokens []token
	index  int
}

func parseWholeExpression(tokens []token) bool {
	parser := expressionParser{tokens: tokens}
	return parser.parseExpression(0) && parser.index == len(tokens)
}

func (parser *expressionParser) parseExpression(minimum int) bool {
	if !parser.parsePrefix() {
		return false
	}
	for parser.index < len(parser.tokens) {
		precedence, width := binaryOperator(parser.tokens, parser.index)
		if precedence < minimum {
			break
		}
		parser.index += width
		if !parser.parseExpression(precedence + 1) {
			return false
		}
	}
	return true
}

func (parser *expressionParser) parsePrefix() bool {
	if parser.index >= len(parser.tokens) {
		return false
	}
	value := parser.tokens[parser.index]
	if value.kind == tokenOperator && oneOf(value.text, "+", "-") ||
		tokenWord(parser.tokens, parser.index) == "NOT" {
		parser.index++
		return parser.parseExpression(8)
	}
	if value.kind == tokenNumber && value.text == "<invalid-number>" ||
		value.kind == tokenParameter && value.text == "$<invalid-parameter>" {
		return false
	}
	switch {
	case value.kind == tokenString || value.kind == tokenNumber ||
		value.kind == tokenParameter:
		parser.index++
	case identifier(value):
		parser.index++
	case value.text == "(":
		parser.index++
		if !parser.parseExpression(0) || !parser.consume(")") {
			return false
		}
	case value.text == "[":
		if !parser.parseCollection("]") {
			return false
		}
	case value.text == "{":
		if !parser.parseMap() {
			return false
		}
	case value.text == "*":
		parser.index++
	default:
		return false
	}
	for parser.index < len(parser.tokens) {
		switch parser.tokens[parser.index].text {
		case ".":
			if parser.index+1 >= len(parser.tokens) ||
				!identifier(parser.tokens[parser.index+1]) {
				return false
			}
			parser.index += 2
		case "(":
			if !parser.parseCollection(")") {
				return false
			}
		case "[":
			if !parser.parseCollection("]") {
				return false
			}
		default:
			return true
		}
	}
	return true
}

func (parser *expressionParser) parseCollection(close string) bool {
	parser.index++
	if parser.consume(close) {
		return true
	}
	if tokenWord(parser.tokens, parser.index) == "DISTINCT" {
		parser.index++
	}
	for {
		if !parser.parseExpression(0) {
			return false
		}
		if parser.consume(close) {
			return true
		}
		if !parser.consume(",") {
			return false
		}
	}
}

func (parser *expressionParser) parseMap() bool {
	parser.index++
	if parser.consume("}") {
		return true
	}
	for {
		if parser.index >= len(parser.tokens) ||
			!identifier(parser.tokens[parser.index]) {
			return false
		}
		parser.index++
		if !parser.consume(":") || !parser.parseExpression(0) {
			return false
		}
		if parser.consume("}") {
			return true
		}
		if !parser.consume(",") {
			return false
		}
	}
}

func (parser *expressionParser) consume(text string) bool {
	if parser.index < len(parser.tokens) &&
		parser.tokens[parser.index].text == text {
		parser.index++
		return true
	}
	return false
}

func binaryOperator(tokens []token, index int) (int, int) {
	if index >= len(tokens) {
		return -1, 0
	}
	switch strings.ToUpper(tokens[index].text) {
	case "OR":
		return 1, 1
	case "XOR":
		return 2, 1
	case "AND":
		return 3, 1
	case "=", "<>", "!=", "<", "<=", ">", ">=", "=~", "IN", "CONTAINS":
		return 4, 1
	case "STARTS", "ENDS":
		if tokenWord(tokens, index+1) == "WITH" {
			return 4, 2
		}
	case "IS":
		if tokenWord(tokens, index+1) == "NULL" {
			return 4, 2
		}
		if tokenWord(tokens, index+1) == "NOT" &&
			tokenWord(tokens, index+2) == "NULL" {
			return 4, 3
		}
	case "+", "-":
		return 5, 1
	case "*", "/", "%":
		return 6, 1
	case "^":
		return 7, 1
	}
	return -1, 0
}
