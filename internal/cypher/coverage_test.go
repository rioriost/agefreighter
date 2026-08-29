package cypher

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestStructuralGrammarTransitionAndPayloadMatrix(t *testing.T) {
	validTransitions := map[string][]string{
		"MATCH":     {"WHERE", "WITH", "RETURN", "UNWIND", "CREATE", "MERGE", "SET", "REMOVE", "DELETE", "FOREACH"},
		"WHERE":     {"WITH", "RETURN", "CREATE", "MERGE", "SET", "REMOVE", "DELETE", "FOREACH"},
		"WITH":      {"WHERE", "ORDER", "SKIP", "LIMIT", "MATCH", "UNWIND", "CREATE", "MERGE", "SET", "REMOVE", "DELETE", "FOREACH", "RETURN", "CALL"},
		"RETURN":    {"ORDER", "SKIP", "LIMIT", "UNION"},
		"ORDER":     {"SKIP", "LIMIT"},
		"SKIP":      {"LIMIT"},
		"UNION":     {"MATCH", "RETURN", "WITH", "UNWIND", "CREATE", "MERGE", "CALL"},
		"MERGE":     {"MERGE SET", "WITH", "RETURN", "SET", "REMOVE", "DELETE"},
		"MERGE SET": {"MERGE SET", "WITH", "RETURN", "SET", "REMOVE", "DELETE"},
		"UNWIND":    {"MATCH", "WHERE", "WITH", "RETURN", "UNWIND", "CREATE", "MERGE", "SET", "REMOVE", "DELETE", "FOREACH"},
		"CREATE":    {"MATCH", "WHERE", "WITH", "RETURN", "UNWIND", "CREATE", "MERGE", "SET", "REMOVE", "DELETE", "FOREACH"},
		"SET":       {"MATCH", "WHERE", "WITH", "RETURN", "UNWIND", "CREATE", "MERGE", "SET", "REMOVE", "DELETE", "FOREACH"},
		"REMOVE":    {"MATCH", "WHERE", "WITH", "RETURN", "UNWIND", "CREATE", "MERGE", "SET", "REMOVE", "DELETE", "FOREACH"},
		"DELETE":    {"MATCH", "WHERE", "WITH", "RETURN", "UNWIND", "CREATE", "MERGE", "SET", "REMOVE", "DELETE", "FOREACH"},
		"FOREACH":   {"MATCH", "WHERE", "WITH", "RETURN", "UNWIND", "CREATE", "MERGE", "SET", "REMOVE", "DELETE", "FOREACH"},
		"CALL":      {"MATCH", "WHERE", "WITH", "RETURN", "UNWIND", "CREATE", "MERGE", "SET", "REMOVE", "DELETE", "FOREACH"},
		"USE":       {"MATCH", "RETURN", "WITH", "UNWIND", "CREATE", "MERGE", "CALL"},
		"PROFILE":   {"MATCH", "RETURN", "WITH", "UNWIND", "CREATE", "MERGE", "CALL"},
		"EXPLAIN":   {"MATCH", "RETURN", "WITH", "UNWIND", "CREATE", "MERGE", "CALL"},
	}
	for left, rights := range validTransitions {
		for _, right := range rights {
			if !validStructuralTransition(left, right) {
				t.Errorf("validStructuralTransition(%q, %q) = false", left, right)
			}
		}
	}
	for _, pair := range [][2]string{
		{"SKIP", "RETURN"}, {"RETURN", "MATCH"}, {"BOGUS", "RETURN"},
	} {
		if validStructuralTransition(pair[0], pair[1]) {
			t.Errorf("validStructuralTransition(%q, %q) = true", pair[0], pair[1])
		}
	}

	tests := []struct {
		name    string
		clause  string
		payload string
		want    bool
	}{
		{"union", "UNION", "", true},
		{"profile", "PROFILE", "", true},
		{"explain", "EXPLAIN", "", true},
		{"match", "MATCH", "(a:Person)-[:KNOWS]->(b)", true},
		{"create", "CREATE", "p = (a {id: 1})-[:R|:S]->(b $props)", true},
		{"where", "WHERE", "NOT (n.age < 18 OR n.name STARTS WITH 'x')", true},
		{"return", "RETURN", "DISTINCT n.name AS name, count(DISTINCT n)", true},
		{"return star", "RETURN", "*", true},
		{"order", "ORDER", "n.name DESC, n.age ASC", true},
		{"unwind", "UNWIND", "range(1, 3) AS value", true},
		{"set", "SET", "n.value = 1, n.other = n.other + 2", true},
		{"remove", "REMOVE", "n.value, n.other", true},
		{"delete", "DELETE", "n, r", true},
		{"foreach", "FOREACH", "(n:Person)", true},
		{"call expression", "CALL", "db.proc(1)", true},
		{"call subquery", "CALL", "{ RETURN 1 }", true},
		{"show", "SHOW", "FUNCTIONS", true},
		{"use", "USE", "`graph-name`", true},
		{"drop", "DROP", "INDEX idx", true},
		{"empty match", "MATCH", "", false},
		{"bad show", "SHOW", "TABLES", false},
		{"bad use", "USE", "a.b", false},
		{"bad drop", "DROP", "DATABASE db", false},
		{"unknown", "OTHER", "value", false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tokens := lexSingleTokens(t, test.payload)
			if got := validateClausePayload(test.clause, tokens); got != test.want {
				t.Fatalf("validateClausePayload(%q, %q) = %t, want %t", test.clause, test.payload, got, test.want)
			}
		})
	}
}

func TestPatternProjectionAndExpressionGrammar(t *testing.T) {
	patterns := []struct {
		input string
		want  bool
	}{
		{"(n)", true},
		{"p = (a:One:Two {id: 1})<-[r:R|S]-(:Three)", true},
		{"(a)-[:R]->(b), (c)<--(d)", true},
		{"(n $props)", true},
		{"(n $)", false},
		{"(a)-[:R*0x]->(b)", false},
		{"(a)-[:R|]->(b)", false},
		{"(a)<-[:R](b)", false},
		{"(a)-[:R]->", false},
	}
	for _, test := range patterns {
		_, got := parsePatternPayload(lexSingleTokens(t, test.input))
		if got != test.want {
			t.Errorf("parsePatternPayload(%q) = %t, want %t", test.input, got, test.want)
		}
	}

	expressions := []struct {
		input string
		want  bool
	}{
		{"-1 + +2 * 3 ^ 2", true},
		{"a OR b XOR c AND d", true},
		{"n = 1 AND n <> 2 AND n != 3", true},
		{"n <= 1 OR n >= 2 OR n =~ 'x'", true},
		{"n IN [1, 2] AND n CONTAINS 'x'", true},
		{"n STARTS WITH 'a' AND n ENDS WITH 'z'", true},
		{"n IS NULL OR n IS NOT NULL", false},
		{"f(DISTINCT n, {a: [1, 2], b: g()})[0].value", true},
		{"[]", true},
		{"{}", true},
		{"(1 + 2)", true},
		{"1 / 2 % 1", true},
		{"1 +", false},
		{"[1,]", false},
		{"{a: 1,}", false},
		{"a.", false},
		{"$", false},
		{"0x", false},
	}
	for _, test := range expressions {
		if got := parseWholeExpression(lexSingleTokens(t, test.input)); got != test.want {
			t.Errorf("parseWholeExpression(%q) = %t, want %t", test.input, got, test.want)
		}
	}

	for _, test := range []struct {
		input    string
		ordering bool
		want     bool
	}{
		{"DISTINCT n AS value, m", false, true},
		{"n DESC, m ASC", true, true},
		{"n AS", false, false},
		{"n,,m", false, false},
		{"", false, false},
	} {
		if got := parseProjection(lexSingleTokens(t, test.input), test.ordering); got != test.want {
			t.Errorf("parseProjection(%q, %t) = %t, want %t", test.input, test.ordering, got, test.want)
		}
	}
	if parseUnwind(lexSingleTokens(t, "1 value")) {
		t.Fatal("parseUnwind accepted missing AS")
	}
	if parseExpressionList(lexSingleTokens(t, "a,,b")) {
		t.Fatal("parseExpressionList accepted empty item")
	}
	for _, tokens := range [][]token{
		{{kind: tokenNumber, text: "<number>"}},
		{{kind: tokenOperator, text: ".."}},
		{{kind: tokenNumber, text: "<number>"}, {kind: tokenOperator, text: ".."}},
		{{kind: tokenOperator, text: ".."}, {kind: tokenNumber, text: "<number>"}},
	} {
		parser := patternParser{tokens: tokens}
		if !parser.parseRelationshipRange() || parser.index != len(tokens) {
			t.Errorf("parseRelationshipRange(%#v) failed at %d", tokens, parser.index)
		}
	}
	for _, tokens := range [][]token{
		{{kind: tokenNumber, text: "<invalid-number>"}},
		{{kind: tokenOperator, text: ".."}, {kind: tokenNumber, text: "<invalid-number>"}},
	} {
		parser := patternParser{tokens: tokens}
		if parser.parseRelationshipRange() {
			t.Errorf("parseRelationshipRange(%#v) accepted invalid number", tokens)
		}
	}
}

func TestLexerLiteralCommentAndDelimiterEdges(t *testing.T) {
	valid := []string{
		"RETURN 0, 12.5, 1e3, 1E-3, 0x1f, 0XAF, 0o17, 0O7",
		"RETURN 'a\\'b', \"a\"\"b\", `a``b`, $value, $_2",
		"// line comment\nRETURN 1 /* outer /* nested */ done */",
		"RETURN {a: [1, (2)]}; RETURN 3",
	}
	for _, input := range valid {
		statements, finding, err := lex(t.Context(), []byte(input))
		if err != nil || finding != nil || len(statements) == 0 {
			t.Errorf("lex(%q) = statements %d, finding %#v, error %v", input, len(statements), finding, err)
		}
	}
	for _, test := range []struct {
		input string
		code  string
	}{
		{"RETURN 'unterminated", "AGE16-X001"},
		{"RETURN `unterminated", "AGE16-X001"},
		{"RETURN `bad\nname`", "AGE16-X001"},
		{"/* unterminated", "AGE16-X001"},
		{"RETURN (]", "AGE16-X002"},
		{"RETURN (1", "AGE16-X002"},
		{"RETURN @", "AGE16-X001"},
	} {
		_, finding, err := lex(t.Context(), []byte(test.input))
		if err != nil || finding == nil || finding.Code != test.code {
			t.Errorf("lex(%q) finding = %#v, error = %v", test.input, finding, err)
		}
	}
	statements, finding, err := lex(t.Context(), []byte("RETURN 0x, 0o, 1e+"))
	if err != nil || finding != nil || len(statements) != 1 {
		t.Fatalf("invalid-number lex = %#v, %#v, %v", statements, finding, err)
	}
	for _, value := range statements[0].tokens {
		if value.kind == tokenNumber && value.text != "<invalid-number>" {
			t.Errorf("invalid numeric token = %#v", value)
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := lex(ctx, []byte("RETURN 1")); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled lex error = %v", err)
	}
}

func TestStructureSpecialSyntaxScopesAndEvidence(t *testing.T) {
	path := writeQuery(t, `
		MATCH (n:Person)
		WHERE n.name CONTAINS $part
		  AND 10 > n.age
		WITH *, n AS person
		RETURN person
		ORDER BY person.name
		;
		MATCH (n:Person) WITH n AS x, n AS x WHERE x.name = 'hidden' RETURN x;
		MATCH (n:Person) CALL { RETURN 1 } RETURN n;
		CREATE INDEX idx;
		DROP INDEX idx;
		MATCH (n) USING INDEX n:Person(name) RETURN n;
		MATCH (n) WHERE n IS NOT TYPED STRING RETURN n;
		MATCH (n) RETURN apoc.text.clean(n.name), datetime(), custom.ns(n)`)
	report, err := AnalyzeFiles(t.Context(), []string{path}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Queries) != 8 {
		t.Fatalf("queries = %d, want 8", len(report.Queries))
	}
	if report.Queries[0].Classification != CompatibleWithManualChange {
		t.Fatalf("predicate query = %#v", report.Queries[0])
	}
	operators := map[string]bool{}
	for _, pattern := range report.Queries[0].Patterns {
		if pattern.Kind == "predicate" {
			operators[pattern.Operator] = true
		}
	}
	for _, operator := range []string{"CONTAINS", "<"} {
		if !operators[operator] {
			t.Errorf("missing predicate operator %q in %#v", operator, report.Queries[0].Patterns)
		}
	}
	for input, want := range map[string]string{
		"STARTS WITH": "STARTS WITH",
		"ENDS WITH":   "ENDS WITH",
		"IN":          "IN",
		"CONTAINS":    "CONTAINS",
	} {
		operator, _ := predicateOperator(lexSingleTokens(t, input), 0)
		if operator != want {
			t.Errorf("predicateOperator(%q) = %q, want %q", input, operator, want)
		}
	}
	for _, pattern := range report.Queries[1].Patterns {
		if pattern.Kind == "predicate" {
			t.Errorf("duplicate projection retained predicate provenance: %#v", report.Queries[1].Patterns)
		}
	}
	for _, index := range []int{2, 3, 4, 5, 6, 7} {
		if report.Queries[index].Classification == Compatible {
			t.Errorf("query %d unexpectedly compatible: %#v", index+1, report.Queries[index])
		}
	}

	evidenceValue := evidence([]token{
		{kind: tokenIdentifier, text: "MATCH"},
		{kind: tokenIdentifier, text: "secretName"},
		{kind: tokenString, text: "secret"},
		{kind: tokenNumber, text: "123"},
		{kind: tokenParameter, text: "$private"},
	})
	if evidenceValue != "MATCH <identifier> <string> <number> $<parameter>" {
		t.Fatalf("evidence = %q", evidenceValue)
	}
	if got := boundedEvidence(strings.Repeat("x", MaxEvidenceRunes+10)); len([]rune(got)) != MaxEvidenceRunes || !strings.HasSuffix(got, "…") {
		t.Fatalf("boundedEvidence() = %q", got)
	}
	for _, value := range []string{"", string([]byte{0xff}), strings.Repeat("x", 129), "bad\x7f"} {
		if boundedIdentifier(value) != "" {
			t.Errorf("boundedIdentifier(%q) was accepted", value)
		}
	}
	got := sortedPatterns([]Pattern{
		{Kind: "z", Label: "b"}, {Kind: "a", Label: "b"},
		{Kind: "a", Label: "b"}, {Kind: "a", Label: "a"},
	})
	if len(got) != 3 || got[0].Label != "a" || got[2].Kind != "z" {
		t.Fatalf("sortedPatterns() = %#v", got)
	}
}

func TestAnalyzerAndOutputErrorBranches(t *testing.T) {
	if _, err := AnalyzeFiles(t.Context(), nil, Options{}); err == nil {
		t.Fatal("AnalyzeFiles accepted no paths")
	}
	paths := make([]string, MaxFiles+1)
	if _, err := AnalyzeFiles(t.Context(), paths, Options{}); !errors.Is(err, errLimit) {
		t.Fatalf("too-many-files error = %v", err)
	}
	empty := filepath.Join(t.TempDir(), "empty.cypher")
	if err := os.WriteFile(empty, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := AnalyzeFiles(t.Context(), []string{empty}, Options{}); !errors.Is(err, errInput) {
		t.Fatalf("empty input error = %v", err)
	}
	invalidUTF8 := filepath.Join(t.TempDir(), "invalid.cypher")
	if err := os.WriteFile(invalidUTF8, []byte{0xff}, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := AnalyzeFiles(t.Context(), []string{invalidUTF8}, Options{}); !errors.Is(err, errInput) {
		t.Fatalf("invalid UTF-8 error = %v", err)
	}
	oversized := filepath.Join(t.TempDir(), "large.cypher")
	if err := os.WriteFile(oversized, make([]byte, MaxFileBytes+1), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := AnalyzeFiles(t.Context(), []string{oversized}, Options{}); !errors.Is(err, errLimit) {
		t.Fatalf("oversized input error = %v", err)
	}

	if err := WriteOutput(t.Context(), io.Discard, make([]byte, MaxOutputBytes+1)); !errors.Is(err, errLimit) {
		t.Fatalf("oversized output error = %v", err)
	}
	injected := errors.New("writer failed")
	if err := WriteOutput(t.Context(), errorWriter{err: injected}, []byte("x")); err == nil || errors.Is(err, injected) {
		t.Fatalf("writer error = %v", err)
	}
	if err := WriteOutput(t.Context(), shortWriter{}, []byte("abc")); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := WriteOutput(ctx, blockingWriter{}, []byte("x")); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled write error = %v", err)
	}

	for input, want := range map[error]error{
		context.Canceled:         context.Canceled,
		context.DeadlineExceeded: context.DeadlineExceeded,
		os.ErrNotExist:           os.ErrNotExist,
		os.ErrPermission:         os.ErrPermission,
		io.EOF:                   errors.New("file operation failed"),
	} {
		if got := classifyFileError(input); got.Error() != want.Error() {
			t.Errorf("classifyFileError(%v) = %v, want %v", input, got, want)
		}
	}
	if got := safeBasename(strings.Repeat("x", MaxPathRunes+20)); len([]rune(got)) != MaxPathRunes || !strings.HasPrefix(got, "…") {
		t.Fatalf("safeBasename() length = %d, value %q", len([]rune(got)), got)
	}
	if _, err := Render(Report{}, Format("xml")); err == nil {
		t.Fatal("Render accepted unsupported format")
	}
	if err := summarize(&Report{Queries: []Query{{Classification: "bad"}}}); err == nil {
		t.Fatal("summarize accepted invalid classification")
	}
	report := Report{Summary: Summary{Findings: MaxFindings + 1}}
	if err := summarize(&report); !errors.Is(err, errLimit) {
		t.Fatalf("summarize findings error = %v", err)
	}
	if !errors.Is(ReportLimitError(), errLimit) {
		t.Fatal("ReportLimitError does not wrap errLimit")
	}
}

func lexSingleTokens(t *testing.T, input string) []token {
	t.Helper()
	if input == "" {
		return nil
	}
	statements, finding, err := lex(t.Context(), []byte(input))
	if err != nil {
		t.Fatalf("lex(%q) error = %v", input, err)
	}
	if finding != nil {
		t.Fatalf("lex(%q) finding = %#v", input, finding)
	}
	if len(statements) != 1 {
		t.Fatalf("lex(%q) statements = %d", input, len(statements))
	}
	return statements[0].tokens
}

type errorWriter struct{ err error }

func (writer errorWriter) Write([]byte) (int, error) { return 0, writer.err }

type shortWriter struct{}

func (shortWriter) Write(value []byte) (int, error) { return len(value) - 1, nil }

type blockingWriter struct{}

func (blockingWriter) Write([]byte) (int, error) {
	time.Sleep(10 * time.Millisecond)
	return 0, context.Canceled
}
