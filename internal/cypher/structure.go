package cypher

import (
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"
)

var supportedFirstClauses = map[string]bool{
	"MATCH": true, "OPTIONAL": true, "RETURN": true, "WITH": true,
	"UNWIND": true, "CREATE": true, "MERGE": true, "SET": true,
	"REMOVE": true, "DELETE": true, "DETACH": true, "FOREACH": true,
}

var clauseWords = map[string]bool{
	"MATCH": true, "OPTIONAL": true, "WHERE": true, "WITH": true,
	"RETURN": true, "UNWIND": true, "CREATE": true, "MERGE": true,
	"ON": true, "SET": true, "REMOVE": true, "DELETE": true,
	"DETACH": true, "FOREACH": true, "UNION": true, "ORDER": true,
	"SKIP": true, "LIMIT": true,
}

var uncatalogedClauseWords = map[string]bool{
	"FILTER": true, "FINISH": true, "INSERT": true, "LET": true,
	"NEXT": true, "NODETACH": true, "OFFSET": true, "QUALIFY": true,
}

var unsupportedCommands = map[string]struct {
	code        string
	remediation string
}{
	"CALL": {
		code:        "AGE16-U001",
		remediation: "replace the procedure call with application code or documented AGE/openCypher clauses",
	},
	"SHOW": {
		code:        "AGE16-U002",
		remediation: "query PostgreSQL and AGE catalogs through a separately reviewed administrative operation",
	},
	"USE": {
		code:        "AGE16-U002",
		remediation: "select the AGE graph in the application integration rather than with a Neo4j USE clause",
	},
	"PROFILE": {
		code:        "AGE16-U002",
		remediation: "remove Neo4j planner directives; inspect AGE plans separately with reviewed SQL",
	},
	"EXPLAIN": {
		code:        "AGE16-U002",
		remediation: "remove Neo4j planner directives; inspect AGE plans separately with reviewed SQL",
	},
	"LOAD": {
		code:        "AGE16-U003",
		remediation: "load local data through agefreighter instead of LOAD CSV",
	},
	"CYPHER": {
		code:        "AGE16-U002",
		remediation: "remove Neo4j CYPHER planner and runtime directives",
	},
	"DROP": {
		code:        "AGE16-U004",
		remediation: "manage PostgreSQL or AGE schema through separately reviewed DDL",
	},
}

var unsupportedFunctions = map[string]string{
	"elementid":       "use a stable application property or review AGE id() semantics",
	"randomuuid":      "generate identifiers in the application before executing Cypher",
	"datetime":        "bind a normalized temporal value and review AGE agtype conversion",
	"localdatetime":   "bind a normalized temporal value and review AGE agtype conversion",
	"date":            "bind a normalized temporal value and review AGE agtype conversion",
	"time":            "bind a normalized temporal value and review AGE agtype conversion",
	"localtime":       "bind a normalized temporal value and review AGE agtype conversion",
	"duration":        "represent duration explicitly and review AGE arithmetic",
	"point":           "model coordinates as numeric properties or use a separately reviewed extension",
	"tostringornull":  "replace Neo4j tolerant conversion with explicit null handling",
	"tointegerornull": "replace Neo4j tolerant conversion with explicit null handling",
	"tofloatornull":   "replace Neo4j tolerant conversion with explicit null handling",
	"tobooleanornull": "replace Neo4j tolerant conversion with explicit null handling",
	"normalize":       "normalize text in the application before binding it",
	"isnormalized":    "normalize and validate text in the application",
	"valuetype":       "replace Neo4j runtime type inspection with explicit property validation",
	"distance":        "model and calculate spatial distance outside Cypher or use a reviewed extension",
}

var supportedFunctions = map[string]bool{
	"abs": true, "acos": true, "all": true, "any": true, "asin": true,
	"atan": true, "atan2": true, "avg": true, "ceil": true,
	"coalesce": true, "collect": true, "cos": true, "cot": true,
	"count": true, "degrees": true, "e": true, "endnode": true,
	"exists": true, "exp": true, "floor": true, "head": true, "id": true,
	"haversin": true, "keys": true, "labels": true, "last": true, "left": true,
	"length": true, "log": true, "log10": true, "ltrim": true,
	"max": true, "min": true, "nodes": true, "percentilecont": true,
	"none": true, "percentiledisc": true, "pi": true, "properties": true,
	"radians": true, "rand": true, "range": true, "relationships": true,
	"reduce": true, "replace": true, "reverse": true, "right": true,
	"round": true, "rtrim": true, "sign": true, "sin": true,
	"single": true, "size": true,
	"split": true, "sqrt": true, "startnode": true, "stdev": true,
	"stdevp": true, "substring": true, "sum": true, "tail": true,
	"tan": true, "toboolean": true, "tofloat": true, "tointeger": true,
	"tolower": true, "tostring": true, "toupper": true, "trim": true,
	"type": true,
}

func analyzeStatement(
	value statement,
	file string,
	number int,
	options Options,
) Query {
	first := token{line: 1, column: 1}
	if len(value.tokens) > 0 {
		first = value.tokens[0]
	}
	result := Query{
		File: file, Number: number, Line: first.line, Column: first.column,
		Classification: Compatible, Findings: []Finding{}, Patterns: []Pattern{},
	}
	if len(value.tokens) == 0 {
		result.Classification = Unknown
		result.Findings = append(result.Findings, unknownFinding(
			first, "empty query", "provide a complete openCypher query",
		))
		return result
	}
	if len(value.tokens) > MaxTokens {
		result.Classification = Unknown
		result.Findings = append(result.Findings, Finding{
			Code: "AGE16-X004", Severity: SeverityUnknown,
			Line: first.line, Column: first.column,
			Evidence:    "query token count exceeds limit",
			Remediation: "split or simplify the query before rechecking",
		})
		return result
	}

	firstWord := upper(value.tokens[0])
	if rule, exists := unsupportedCommands[firstWord]; exists {
		result.Findings = append(result.Findings, Finding{
			Code: rule.code, Severity: SeverityError,
			Line: first.line, Column: first.column,
			Evidence:    evidence(value.tokens[0:min(2, len(value.tokens))]),
			Remediation: rule.remediation,
		})
	} else if !supportedFirstClauses[firstWord] {
		result.Findings = append(result.Findings, unknownFinding(
			first,
			"unrecognized query entry clause",
			"start with a documented AGE/openCypher clause",
		))
	}

	validateClauses(value.tokens, &result)
	analyzeSpecialSyntax(value.tokens, options, &result)
	analyzeFunctions(value.tokens, &result)
	if invalid, ok := validateStructuralConsumption(value.tokens); !ok {
		result.Findings = append(result.Findings, unknownFinding(
			invalid,
			"statement is not fully consumed by the bounded structural grammar",
			"correct the clause, expression, operator, or literal structure before rechecking",
		))
	}

	hasUnsupported := false
	hasUnknown := false
	hasWarning := false
	for _, finding := range result.Findings {
		switch finding.Severity {
		case SeverityError:
			hasUnsupported = true
		case SeverityUnknown:
			hasUnknown = true
		case SeverityWarning:
			hasWarning = true
		}
	}
	switch {
	case hasUnknown:
		result.Classification = Unknown
		result.Patterns = []Pattern{}
	case hasUnsupported:
		result.Classification = Unsupported
		result.Patterns = []Pattern{}
	case hasWarning:
		result.Classification = CompatibleWithManualChange
	default:
		result.Classification = Compatible
		result.Findings = append(result.Findings, Finding{
			Code: "AGE16-C001", Severity: SeverityInfo,
			Line: first.line, Column: first.column,
			Evidence:    "recognized bounded openCypher structure",
			Remediation: "test application behavior against the target AGE release before cutover",
		})
	}
	if result.Classification == Compatible ||
		result.Classification == CompatibleWithManualChange {
		result.Patterns = inferPatterns(value.tokens)
	}
	return result
}

func validateClauses(tokens []token, result *Query) {
	depth := 0
	lastClause := -1
	for index, value := range tokens {
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
		if !clauseWords[word] {
			continue
		}
		if lastClause >= 0 && index == lastClause+1 &&
			!validAdjacentClause(upper(tokens[lastClause]), word) &&
			!validMergeSubclause(tokens, lastClause, word) {
			result.Findings = append(result.Findings, unknownFinding(
				value,
				"clause has no structural payload",
				"complete the preceding clause before starting the next clause",
			))
		}

		lastClause = index
	}

	if lastClause == len(tokens)-1 {
		value := tokens[lastClause]
		result.Findings = append(result.Findings, unknownFinding(
			value, "clause has no structural payload",
			"provide the expression or pattern required by the clause",
		))
	}
}

func validMergeSubclause(tokens []token, lastClause int, right string) bool {
	return right == "SET" && lastClause > 0 &&
		(upper(tokens[lastClause]) == "MATCH" ||
			upper(tokens[lastClause]) == "CREATE") &&
		upper(tokens[lastClause-1]) == "ON"
}

func validAdjacentClause(left, right string) bool {
	return left == "OPTIONAL" && right == "MATCH" ||
		left == "DETACH" && right == "DELETE" ||
		left == "ORDER" && right == "BY" ||
		left == "UNION" && right == "ALL" ||
		left == "ON" && (right == "MATCH" || right == "CREATE")
}

func analyzeSpecialSyntax(tokens []token, options Options, result *Query) {
	parameterReported := false
	depth := 0
	for index, value := range tokens {
		if value.kind == tokenPunctuation {
			switch value.text {
			case "(", "[", "{":
				depth++
			case ")", "]", "}":
				depth--
			}
			continue
		}
		word := upper(value)
		if value.kind == tokenParameter && !parameterReported {
			parameterReported = true
			result.Findings = append(result.Findings, Finding{
				Code: "AGE16-W001", Severity: SeverityWarning,
				Line: value.line, Column: value.column,
				Evidence:    "$<parameter>",
				Remediation: "bind parameters through the AGE cypher() parameter map; do not interpolate values",
			})
		}
		if value.kind == tokenParameter &&
			value.text == "$<invalid-parameter>" {
			result.Findings = append(result.Findings, unknownFinding(
				value,
				"parameter name is missing",
				"provide a named Cypher parameter",
			))
		}
		if value.kind != tokenIdentifier {
			continue
		}
		if depth == 0 && uncatalogedClauseWords[word] &&
			(index == 0 || tokens[index-1].text != ".") {
			result.Findings = append(result.Findings, unknownFinding(
				value,
				"clause is outside the AGE 1.6 compatibility catalog",
				"replace it with documented AGE/openCypher clauses or review it manually",
			))
		}
		if rule, exists := unsupportedCommands[word]; exists {
			if word == "CALL" && index+1 < len(tokens) && tokens[index+1].text == "{" {
				result.Findings = append(result.Findings, unknownFinding(
					value,
					"CALL subquery compatibility is not proven for AGE 1.6",
					"review and test the subquery manually or split it into separate application operations",
				))
				continue
			}
			if index == 0 {
				continue
			}
			result.Findings = append(result.Findings, Finding{
				Code: rule.code, Severity: SeverityError,
				Line: value.line, Column: value.column,
				Evidence:    evidence(tokens[index:min(index+2, len(tokens))]),
				Remediation: rule.remediation,
			})
		}
		if word == "EXISTS" && index+1 < len(tokens) && tokens[index+1].text == "{" {
			result.Findings = append(result.Findings, unknownFinding(
				value,
				"nested EXISTS subquery compatibility is not proven for AGE 1.6",
				"rewrite as a proven pattern predicate or review the subquery manually",
			))
		}
		if word == "USING" {
			result.Findings = append(result.Findings, Finding{
				Code: "AGE16-U005", Severity: SeverityError,
				Line: value.line, Column: value.column,
				Evidence:    "USING <planner-hint>",
				Remediation: "remove Neo4j index, scan, or join planner hints",
			})
		}
		if index == 0 && word == "CREATE" && index+1 < len(tokens) {
			next := upper(tokens[index+1])
			if next == "INDEX" || next == "CONSTRAINT" || next == "DATABASE" {
				result.Findings = append(result.Findings, Finding{
					Code: "AGE16-U004", Severity: SeverityError,
					Line: value.line, Column: value.column,
					Evidence:    "CREATE " + next,
					Remediation: "manage PostgreSQL or AGE schema through separately reviewed DDL",
				})
			}
		}
		if word == "INDEX" && index > 0 && upper(tokens[index-1]) == "DROP" {
			result.Findings = append(result.Findings, Finding{
				Code: "AGE16-U004", Severity: SeverityError,
				Line: value.line, Column: value.column,
				Evidence:    "DROP INDEX",
				Remediation: "manage PostgreSQL or AGE schema through separately reviewed DDL",
			})
		}
		if word == "IS" && index+2 < len(tokens) &&
			upper(tokens[index+1]) == "NOT" && upper(tokens[index+2]) == "TYPED" {
			result.Findings = append(result.Findings, unknownFinding(
				value, "type predicate compatibility is not proven",
				"replace the type predicate with explicit, reviewed property validation",
			))
		}
	}
	_ = options
}

func analyzeFunctions(tokens []token, result *Query) {
	for index := 0; index+1 < len(tokens); index++ {
		value := tokens[index]
		if !identifier(value) {
			continue
		}
		if clauseWords[upper(value)] || unsupportedCommandsContains(upper(value)) {
			continue
		}
		end := index
		parts := []string{value.text}
		for end+2 < len(tokens) && tokens[end+1].text == "." &&
			identifier(tokens[end+2]) {
			parts = append(parts, tokens[end+2].text)
			end += 2
		}
		if end+1 >= len(tokens) || tokens[end+1].text != "(" {
			continue
		}
		name := strings.ToLower(strings.Join(parts, "."))
		base := strings.ToLower(parts[len(parts)-1])
		if strings.HasPrefix(name, "apoc.") ||
			strings.HasPrefix(name, "gds.") ||
			strings.HasPrefix(name, "db.") {
			result.Findings = append(result.Findings, Finding{
				Code: "AGE16-U001", Severity: SeverityError,
				Line: value.line, Column: value.column,
				Evidence:    "<neo4j-namespace>.<function>(…)",
				Remediation: "replace Neo4j extension calls with application code or documented AGE functions",
			})
		} else if remediation, exists := unsupportedFunctions[base]; exists {
			result.Findings = append(result.Findings, Finding{
				Code: "AGE16-U006", Severity: SeverityError,
				Line: value.line, Column: value.column,
				Evidence: base + "(…)", Remediation: remediation,
			})
		} else if len(parts) > 1 || !supportedFunctions[base] {
			result.Findings = append(result.Findings, unknownFinding(
				value,
				"function compatibility is not in the AGE 1.6 rule catalog",
				"verify the function against AGE 1.6 documentation or replace it with a cataloged function",
			))
		}
		index = end
	}
}

func inferPatterns(tokens []token) []Pattern {
	variables := map[string][]string{}
	patterns := make([]Pattern, 0)
	clauses := structuralClauses(tokens)
	for index, clause := range clauses {
		end := len(tokens)
		if index+1 < len(clauses) {
			end = clauses[index+1].start
		}
		payload := tokens[clause.payloadStart:end]
		switch clause.name {
		case "MATCH", "CREATE", "MERGE":
			bindings, ok := parsePatternPayload(payload)
			if !ok {
				return nil
			}
			patterns = append(
				patterns,
				applyPatternBindings(variables, bindings)...,
			)
		case "WHERE":
			patterns = append(
				patterns,
				inferPredicatePatterns(payload, variables)...,
			)
		case "WITH", "RETURN":
			variables = projectPatternBindings(payload, variables)
		case "ORDER":
			patterns = append(
				patterns,
				inferOrderingPatterns(payload, variables)...,
			)
		case "UNWIND":
			if as := topLevelWord(payload, "AS"); as >= 0 &&
				as+1 < len(payload) {
				delete(variables, payload[as+1].text)
			}
		case "UNION", "CALL", "FOREACH":
			variables = map[string][]string{}
		}
	}
	return patterns
}

func applyPatternBindings(
	variables map[string][]string,
	bindings []patternBinding,
) []Pattern {
	patterns := make([]Pattern, 0)
	occurrences := make(map[string]int)
	labelsByVariable := make(map[string][]string)
	for _, binding := range bindings {
		labels := make([]string, 0, len(binding.labels))
		for _, rawLabel := range binding.labels {
			label := boundedIdentifier(rawLabel)
			if label == "" {
				continue
			}
			labels = append(labels, label)
			if binding.kind != "" {
				patterns = append(patterns, Pattern{
					Kind: binding.kind, Label: label,
				})
			}
		}
		if binding.variable != "" {
			occurrences[binding.variable]++
			labelsByVariable[binding.variable] = labels
		}
	}
	for variable, count := range occurrences {
		_, alreadyBound := variables[variable]
		if count != 1 || alreadyBound {
			variables[variable] = nil
			continue
		}
		variables[variable] = labelsByVariable[variable]
	}
	return patterns
}

func inferPredicatePatterns(
	tokens []token,
	variables map[string][]string,
) []Pattern {
	patterns := make([]Pattern, 0)
	for index := 0; index+2 < len(tokens); index++ {
		if !identifier(tokens[index]) || tokens[index+1].text != "." ||
			!identifier(tokens[index+2]) {
			continue
		}
		labels := variables[tokens[index].text]
		if len(labels) != 1 {
			continue
		}
		property := boundedIdentifier(tokens[index+2].text)
		if property == "" {
			continue
		}
		operator, width := predicateOperator(tokens, index+3)
		validOperand := operator != "" &&
			index+3+width < len(tokens) &&
			expressionOperand(tokens[index+3+width])
		if operator == "" && index > 0 {
			operator = reversePredicateOperator(tokens[index-1].text)
			validOperand = operator != "" && index > 1 &&
				expressionOperand(tokens[index-2])
		}
		if operator != "" && validOperand {
			patterns = append(patterns, Pattern{
				Kind: "predicate", Label: labels[0], Property: property,
				Operator: operator,
			})
		}
	}
	return patterns
}

func inferOrderingPatterns(
	tokens []token,
	variables map[string][]string,
) []Pattern {
	patterns := make([]Pattern, 0)
	for index := 0; index+2 < len(tokens); index++ {
		if !identifier(tokens[index]) || tokens[index+1].text != "." ||
			!identifier(tokens[index+2]) {
			continue
		}
		labels := variables[tokens[index].text]
		property := boundedIdentifier(tokens[index+2].text)
		if len(labels) == 1 && property != "" {
			patterns = append(patterns, Pattern{
				Kind: "ordering", Label: labels[0], Property: property,
				Operator: "ORDER BY",
			})
		}
	}
	return patterns
}

func projectPatternBindings(
	tokens []token,
	variables map[string][]string,
) map[string][]string {
	projected := make(map[string][]string)
	destinations := make(map[string]bool)
	cursor := 0
	if tokenWord(tokens, 0) == "DISTINCT" {
		cursor++
	}
	for cursor < len(tokens) {
		next := topLevelComma(tokens, cursor)
		end := len(tokens)
		if next >= 0 {
			end = next
		}
		item := tokens[cursor:end]
		as := topLevelWord(item, "AS")
		expression := item
		destination := ""
		if as >= 0 {
			expression = item[:as]
			if as+1 < len(item) {
				destination = item[as+1].text
			}
		} else if len(expression) == 1 && identifier(expression[0]) {
			destination = expression[0].text
		}
		if len(expression) == 1 && expression[0].text == "*" {
			for variable, labels := range variables {
				projected[variable] = append([]string(nil), labels...)
			}
		} else if destination != "" {
			if destinations[destination] {
				delete(projected, destination)
			} else if len(expression) == 1 && identifier(expression[0]) {
				if labels, exists := variables[expression[0].text]; exists {
					projected[destination] = append([]string(nil), labels...)
				}
			} else {
				delete(projected, destination)
			}
			destinations[destination] = true
		}
		if next < 0 {
			break
		}
		cursor = next + 1
	}
	return projected
}

func expressionOperand(value token) bool {
	switch value.kind {
	case tokenIdentifier, tokenEscapedIdentifier, tokenString,
		tokenNumber, tokenParameter:
		return value.text != "$<invalid-parameter>"
	case tokenPunctuation:
		return value.text == "[" || value.text == "{" || value.text == "("
	default:
		return false
	}
}

func reversePredicateOperator(operator string) string {
	switch operator {
	case "=":
		return "="
	case "<":
		return ">"
	case "<=":
		return ">="
	case ">":
		return "<"
	case ">=":
		return "<="
	default:
		return ""
	}
}

func predicateOperator(tokens []token, index int) (string, int) {
	if index >= len(tokens) {
		return "", 0
	}
	switch tokens[index].text {
	case "=":
		return "=", 1
	case "<", "<=", ">", ">=":
		return tokens[index].text, 1
	}
	switch upper(tokens[index]) {
	case "IN":
		return "IN", 1
	case "CONTAINS":
		return "CONTAINS", 1
	case "STARTS":
		if index+1 < len(tokens) && upper(tokens[index+1]) == "WITH" {
			return "STARTS WITH", 2
		}
	case "ENDS":
		if index+1 < len(tokens) && upper(tokens[index+1]) == "WITH" {
			return "ENDS WITH", 2
		}
	}
	return "", 0
}

func unknownFinding(value token, message, remediation string) Finding {
	return Finding{
		Code: "AGE16-X003", Severity: SeverityUnknown,
		Line: value.line, Column: value.column,
		Evidence: boundedEvidence(message), Remediation: remediation,
	}
}

func evidence(tokens []token) string {
	parts := make([]string, 0, len(tokens))
	for _, value := range tokens {
		switch value.kind {
		case tokenString:
			parts = append(parts, "<string>")
		case tokenNumber:
			parts = append(parts, "<number>")
		case tokenParameter:
			parts = append(parts, "$<parameter>")
		case tokenIdentifier, tokenEscapedIdentifier:
			word := upper(value)
			if clauseWords[word] || unsupportedCommandsContains(word) ||
				supportedFunctions[strings.ToLower(value.text)] ||
				unsupportedFunctionsContains(strings.ToLower(value.text)) {
				parts = append(parts, value.text)
			} else {
				parts = append(parts, "<identifier>")
			}
		default:
			parts = append(parts, value.text)
		}
	}
	return boundedEvidence(strings.Join(parts, " "))
}

func boundedEvidence(value string) string {
	value = strings.Map(func(character rune) rune {
		if character == '\r' || character == '\n' || character == '\t' {
			return ' '
		}
		if character < 0x20 || character == 0x7f {
			return -1
		}
		return character
	}, value)
	value = strings.Join(strings.Fields(value), " ")
	runes := []rune(value)
	if len(runes) > MaxEvidenceRunes {
		runes = append(runes[:MaxEvidenceRunes-1], '…')
	}
	return string(runes)
}

func boundedIdentifier(value string) string {
	if value == "" || !utf8.ValidString(value) || utf8.RuneCountInString(value) > 128 {
		return ""
	}
	for _, character := range value {
		if unicodeControl(character) {
			return ""
		}
	}
	return value
}

func unicodeControl(character rune) bool {
	return character < 0x20 || character == 0x7f
}

func unsupportedCommandsContains(value string) bool {
	_, exists := unsupportedCommands[value]
	return exists
}

func unsupportedFunctionsContains(value string) bool {
	_, exists := unsupportedFunctions[value]
	return exists
}

func identifier(value token) bool {
	return value.kind == tokenIdentifier || value.kind == tokenEscapedIdentifier
}

func upper(value token) string {
	if value.kind != tokenIdentifier {
		return ""
	}
	return strings.ToUpper(value.text)
}

func sortedPatterns(patterns []Pattern) []Pattern {
	values := slices.Clone(patterns)
	slices.SortFunc(values, func(left, right Pattern) int {
		return strings.Compare(
			fmt.Sprintf("%s\x00%s\x00%s\x00%s", left.Kind, left.Label, left.Property, left.Operator),
			fmt.Sprintf("%s\x00%s\x00%s\x00%s", right.Kind, right.Label, right.Property, right.Operator),
		)
	})
	return slices.Compact(values)
}
