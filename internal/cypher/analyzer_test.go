package cypher

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

func TestAnalyzerIgnoresKeywordsInCommentsStringsAndIdentifiers(t *testing.T) {
	path := writeQuery(t, `
		// CALL apoc.bad()
		MATCH (n:`+"`CALL`"+`) /* SHOW DATABASES */
		WHERE n.note = 'CALL apoc.bad(); $secret'
		RETURN n`)
	report, err := AnalyzeFiles(t.Context(), []string{path}, Options{})
	if err != nil {
		t.Fatalf("AnalyzeFiles() error = %v", err)
	}
	if got := report.Queries[0].Classification; got != Compatible {
		t.Fatalf("classification = %s; findings = %#v", got, report.Queries[0].Findings)
	}
	output, err := Render(report, FormatJSON)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{"apoc.bad", "$secret", "SHOW DATABASES"} {
		if bytes.Contains(output, []byte(secret)) {
			t.Fatalf("report leaked %q: %s", secret, output)
		}
	}
}

func TestAnalyzerClassifiesKnownUnsupportedAndUnknown(t *testing.T) {
	path := writeQuery(t, `
		CALL apoc.create.uuid();
		SHOW DATABASES;
		MATCH (n) RETURN vendorMagic(n);
		WIBBLE (n) RETURN n;`)
	report, err := AnalyzeFiles(t.Context(), []string{path}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	got := []Classification{
		report.Queries[0].Classification,
		report.Queries[1].Classification,
		report.Queries[2].Classification,
		report.Queries[3].Classification,
	}
	want := []Classification{Unsupported, Unsupported, Unknown, Unknown}
	if !slicesEqual(got, want) {
		t.Fatalf("classifications = %v, want %v", got, want)
	}
	if !StrictFailure(report) {
		t.Fatal("strict mode accepted unsupported and unknown queries")
	}
}

func TestAnalyzerExtractsOnlyQualifiedStructuralEvidence(t *testing.T) {
	path := writeQuery(t, `
		MATCH (n:Person), (ambiguous:One), (ambiguous:Two)
		WHERE n.name = $value
		  AND n.age >= 21
		  AND ambiguous.secret = $hidden
		RETURN n ORDER BY n.name`)
	report, err := AnalyzeFiles(t.Context(), []string{path}, Options{})
	if err != nil {
		t.Fatal(err)
	}

	query := report.Queries[0]
	if query.Classification != CompatibleWithManualChange {
		t.Fatalf("classification = %s", query.Classification)
	}
	want := []Pattern{
		{Kind: "ordering", Label: "Person", Property: "name", Operator: "ORDER BY"},
		{Kind: "predicate", Label: "Person", Property: "age", Operator: ">="},
		{Kind: "predicate", Label: "Person", Property: "name", Operator: "="},
		{Kind: "vertex-label", Label: "One"},
		{Kind: "vertex-label", Label: "Person"},
		{Kind: "vertex-label", Label: "Two"},
	}
	if !patternsEqual(query.Patterns, want) {
		t.Fatalf("patterns = %#v, want %#v", query.Patterns, want)
	}
}

func TestAnalyzerPredicateEvidenceRequiresWhereOperand(t *testing.T) {
	path := writeQuery(t, `
		MATCH (n:Person)
		WHERE $minimum <= n.age
		RETURN n.name = $displayOnly, n`)
	report, err := AnalyzeFiles(t.Context(), []string{path}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	var predicates []Pattern
	for _, pattern := range report.Queries[0].Patterns {
		if pattern.Kind == "predicate" {
			predicates = append(predicates, pattern)
		}
	}
	want := []Pattern{{
		Kind: "predicate", Label: "Person", Property: "age", Operator: ">=",
	}}
	if !patternsEqual(predicates, want) {
		t.Fatalf("predicates = %#v, want %#v", predicates, want)
	}
}

func TestAnalyzerHandlesCompoundClausesAndNestedSemicolon(t *testing.T) {
	path := writeQuery(t, `
		MERGE (n:Person {id: 1}) ON MATCH SET n.seen = true RETURN n;
		CALL { MATCH (n) RETURN n; } RETURN 1`)
	report, err := AnalyzeFiles(t.Context(), []string{path}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Queries) != 2 ||
		report.Queries[0].Classification != Compatible ||
		report.Queries[1].Classification != Unknown {
		t.Fatalf("queries = %#v", report.Queries)
	}
}

func TestAnalyzerMalformedAndBoundsFailClosed(t *testing.T) {
	path := writeQuery(t, `MATCH (n) WHERE n.value = 'unterminated`)
	report, err := AnalyzeFiles(t.Context(), []string{path}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Queries[0].Classification != Unknown ||
		report.Summary.Score.Percent != nil ||
		report.Summary.Score.Conclusive {
		t.Fatalf("malformed summary = %#v", report.Summary)
	}

	directory := t.TempDir()
	if _, err := AnalyzeFiles(t.Context(), []string{directory}, Options{}); err == nil || !strings.Contains(err.Error(), "not a regular file") {
		t.Fatalf("directory error = %v", err)
	}

	deep := writeQuery(
		t,
		"RETURN "+strings.Repeat("(", MaxDepth+1)+"1"+
			strings.Repeat(")", MaxDepth+1),
	)
	if _, err := AnalyzeFiles(t.Context(), []string{deep}, Options{}); err == nil || !strings.Contains(err.Error(), "nesting exceeds") {
		t.Fatalf("depth error = %v", err)
	}

	tooMany := writeQuery(t, strings.Repeat("RETURN 1;", MaxQueries+1))
	if _, err := AnalyzeFiles(t.Context(), []string{tooMany}, Options{}); err == nil || !strings.Contains(err.Error(), "queries") {
		t.Fatalf("query count error = %v", err)
	}
}

func TestAnalyzerFailsClosedOnUnconsumedAndMalformedSyntax(t *testing.T) {
	path := writeQuery(t, `
		MATCH (n:Person) WHERE n.name = 1 FLURBLE RETURN n;
		MATCH (n:Person) WHERE n.name = 1+ RETURN n;
		MATCH (n:Person) WHERE n.name = 1e+ RETURN n;
		MATCH (n:Person) ORDER BY n.name RETURN n`)
	report, err := AnalyzeFiles(t.Context(), []string{path}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Queries) != 4 {
		t.Fatalf("queries = %d, want 4", len(report.Queries))
	}
	for index, query := range report.Queries {
		if query.Classification != Unknown {
			t.Errorf("query %d classification = %s, want unknown", index+1, query.Classification)
		}
		if len(query.Patterns) != 0 {
			t.Errorf("query %d emitted patterns: %#v", index+1, query.Patterns)
		}
	}
}

func TestAnalyzerRejectsMalformedPatternContents(t *testing.T) {
	path := writeQuery(t, `
		MATCH (n:Person,) RETURN n;
		MATCH (n:Person),, (m:Other) RETURN n;
		MATCH (n:Person {name: 'x',}) RETURN n;
		MATCH (n:Person {name: 'x',, age: 1}) RETURN n;
		MATCH ()-[r:KNOWS {since: 1,}]->() RETURN r;
		MATCH ()-[r:KNOWS {since: 1,, weight: 2}]->() RETURN r;
		MATCH ()-[r:KNOWS|]->() RETURN r;
		MATCH ()-[r::KNOWS]->() RETURN r;
		MATCH (n:Person):(m:Other) RETURN n`)
	report, err := AnalyzeFiles(t.Context(), []string{path}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Queries) != 9 {
		t.Fatalf("queries = %d, want 9", len(report.Queries))
	}
	for index, query := range report.Queries {
		if query.Classification != Unknown {
			t.Errorf(
				"query %d classification = %s, want unknown",
				index+1,
				query.Classification,
			)
		}
		if len(query.Patterns) != 0 {
			t.Errorf("query %d emitted patterns: %#v", index+1, query.Patterns)
		}
	}
}

func TestAnalyzerAcceptsWellFormedPatternContents(t *testing.T) {
	path := writeQuery(t, `
		MATCH (a:Person {active: true})-
		      [r:KNOWS {since: 2020}]->(b:Person)
		WHERE r.since >= 2020
		RETURN r`)
	report, err := AnalyzeFiles(t.Context(), []string{path}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	query := report.Queries[0]
	if query.Classification != Compatible {
		t.Fatalf(
			"classification = %s; findings = %#v",
			query.Classification,
			query.Findings,
		)
	}
	want := []Pattern{
		{Kind: "edge-label", Label: "KNOWS"},
		{Kind: "predicate", Label: "KNOWS", Property: "since", Operator: ">="},
		{Kind: "vertex-label", Label: "Person"},
	}
	if !patternsEqual(query.Patterns, want) {
		t.Fatalf("patterns = %#v, want %#v", query.Patterns, want)
	}
}

func TestAnalyzerPatternInferenceRespectsBindingScopes(t *testing.T) {
	path := writeQuery(t, `
		MATCH (n:Person)
		WITH {name: 'x'} AS n
		WHERE n.name = 'x'
		RETURN n;
		MATCH (n:Person)
		WITH n.name AS n
		WHERE n.value = 'x'
		RETURN n;
		MATCH (n:Person)
		UNWIND [1] AS n
		WHERE n.name = 'x'
		RETURN n;
		MATCH (n:Person)
		WITH n AS m
		WHERE m.name = 'x'
		RETURN m`)
	report, err := AnalyzeFiles(t.Context(), []string{path}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Queries) != 4 {
		t.Fatalf("queries = %d, want 4", len(report.Queries))
	}
	for index := 0; index < 3; index++ {
		for _, pattern := range report.Queries[index].Patterns {
			if pattern.Kind == "predicate" || pattern.Kind == "ordering" {
				t.Errorf(
					"query %d emitted label-property pattern: %#v",
					index+1,
					pattern,
				)
			}
		}
	}
	want := Pattern{
		Kind: "predicate", Label: "Person", Property: "name", Operator: "=",
	}
	found := false
	for _, pattern := range report.Queries[3].Patterns {
		if pattern == want {
			found = true
		}
	}
	if !found {
		t.Fatalf(
			"direct alias lost proven provenance: %#v",
			report.Queries[3].Patterns,
		)
	}
}

func TestAnalyzerRejectsSymlinkAndCancellation(t *testing.T) {
	target := writeQuery(t, "RETURN 1")
	link := filepath.Join(t.TempDir(), "query.cypher")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	if _, err := AnalyzeFiles(t.Context(), []string{link}, Options{}); err == nil || !strings.Contains(err.Error(), "not a regular file") {
		t.Fatalf("symlink error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := AnalyzeFiles(ctx, []string{target}, Options{}); err != context.Canceled {
		t.Fatalf("cancellation error = %v", err)
	}
}

func TestAnalyzerRedactsAndDisambiguatesPaths(t *testing.T) {
	firstDirectory := filepath.Join(t.TempDir(), "private-one")
	secondDirectory := filepath.Join(t.TempDir(), "private-two")
	if err := os.Mkdir(firstDirectory, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(secondDirectory, 0o700); err != nil {
		t.Fatal(err)
	}
	first := filepath.Join(firstDirectory, "query.cypher")
	second := filepath.Join(secondDirectory, "query.cypher")
	for _, path := range []string{first, second} {
		if err := os.WriteFile(path, []byte("RETURN 1"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	report, err := AnalyzeFiles(t.Context(), []string{second, first}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Files[0].Path == report.Files[1].Path {
		t.Fatalf("duplicate basenames were not disambiguated: %#v", report.Files)
	}
	output, err := Render(report, FormatJSON)
	if err != nil {
		t.Fatal(err)
	}
	for _, directory := range []string{firstDirectory, secondDirectory} {
		if bytes.Contains(output, []byte(directory)) {
			t.Fatalf("report leaked directory %q: %s", directory, output)
		}
	}

	missingDirectory := filepath.Join(t.TempDir(), "secret-directory")
	_, err = AnalyzeFiles(t.Context(), []string{
		filepath.Join(missingDirectory, "missing.cypher"),
	}, Options{})
	if err == nil || strings.Contains(err.Error(), missingDirectory) {
		t.Fatalf("I/O error leaked path: %v", err)
	}
}

func TestDisplayInputsReservesLiteralAndGeneratedIdentifiers(t *testing.T) {
	root := t.TempDir()
	first := filepath.Join(root, "one", "query.cypher")
	second := filepath.Join(root, "two", "query.cypher")
	generated := duplicateDisplayCandidate("query.cypher", first)
	literal := filepath.Join(root, "literal", generated)

	firstOrder := displayInputs([]string{first, second, literal})
	secondOrder := displayInputs([]string{literal, second, first})
	firstIDs := displayIDsByPath(firstOrder)
	secondIDs := displayIDsByPath(secondOrder)
	if len(firstIDs) != 3 || len(secondIDs) != 3 {
		t.Fatalf(
			"display IDs were not globally unique: %#v %#v",
			firstOrder,
			secondOrder,
		)
	}
	for path, identifier := range firstIDs {
		if secondIDs[path] != identifier {
			t.Fatalf(
				"identifier for %q changed with input order: %q != %q",
				path,
				identifier,
				secondIDs[path],
			)
		}
	}
	if firstIDs[literal] != generated {
		t.Fatalf("literal identifier was not reserved: %#v", firstIDs)
	}
	if firstIDs[first] == generated {
		t.Fatalf("generated identifier collided with literal: %#v", firstIDs)
	}
}

func TestCypherGoldenAndSchema(t *testing.T) {
	report, err := AnalyzeFiles(
		t.Context(),
		[]string{"testdata/corpus.cypher"},
		Options{},
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, format := range []Format{FormatJSON, FormatMarkdown} {
		output, err := Render(report, format)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join("testdata", "check-cypher.golden."+string(format))
		if os.Getenv("UPDATE_GOLDEN") == "1" {
			if err := os.WriteFile(path, output, 0o600); err != nil {
				t.Fatal(err)
			}
		}
		want, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(output, want) {
			t.Fatalf("%s golden differs:\n%s", format, output)
		}
	}
	compiled, err := jsonschema.NewCompiler().Compile(
		"../../docs/reference/cypher-compatibility-report.schema.json",
	)
	if err != nil {
		t.Fatalf("compile schema: %v", err)
	}
	output, err := Render(report, FormatJSON)
	if err != nil {
		t.Fatal(err)
	}
	var document any
	if err := json.Unmarshal(output, &document); err != nil {
		t.Fatal(err)
	}
	if err := compiled.Validate(document); err != nil {
		t.Fatalf("validate golden: %v", err)
	}
}

func FuzzCypherLexer(fuzz *testing.F) {
	for _, seed := range []string{
		"MATCH (n:Person) RETURN n",
		"RETURN 'CALL apoc.bad();'",
		"/* nested /* comment */ still */ RETURN $value",
		"MATCH (`n``x`:Label) WHERE `n``x`.p >= 1 RETURN `n``x`",
		"RETURN ١",
	} {
		fuzz.Add(seed)
	}
	fuzz.Fuzz(func(t *testing.T, input string) {
		if len(input) > 64<<10 {
			t.Skip()
		}
		statements, finding, err := lex(t.Context(), []byte(input))
		if err != nil {
			return
		}
		for index, value := range statements {
			query := analyzeStatement(
				value, "fuzz.cypher", index+1,
				Options{},
			)
			canonicalizeQuery(&query)
			if finding != nil {
				query.Classification = Unknown
			}
		}
	})
}

func writeQuery(t *testing.T, value string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "query.cypher")
	if err := os.WriteFile(path, []byte(value), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func slicesEqual(left, right []Classification) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func patternsEqual(left, right []Pattern) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func displayIDsByPath(inputs []displayInput) map[string]string {
	identifiers := make(map[string]string, len(inputs))
	seen := make(map[string]bool, len(inputs))
	for _, input := range inputs {
		if seen[input.display] {
			continue
		}
		seen[input.display] = true
		identifiers[input.path] = input.display
	}
	return identifiers
}
