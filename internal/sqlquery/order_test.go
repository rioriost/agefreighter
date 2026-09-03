package sqlquery

import "testing"

func TestHasTopLevelOrderBy(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		query string
		want  bool
	}{
		{name: "top level", query: "SELECT id FROM people ORDER /* stable */ BY id", want: true},
		{name: "with", query: "WITH p AS (SELECT id FROM people) SELECT * FROM p ORDER BY id", want: true},
		{name: "line comment", query: "SELECT id FROM people -- ORDER BY id", want: false},
		{name: "nested comment", query: "SELECT id /* outer /* ORDER BY */ done */ FROM people", want: false},
		{name: "string", query: "SELECT 'ORDER BY' FROM people", want: false},
		{name: "escape string", query: `SELECT E'payload \' ORDER BY fake' FROM people`, want: false},
		{name: "backslash string", query: `SELECT 'payload \' ORDER BY fake' FROM people`, want: false},
		{name: "identifier", query: `SELECT "ORDER BY" FROM people`, want: false},
		{name: "backtick identifier", query: "MATCH (n) RETURN n.`ORDER BY`", want: false},
		{name: "dollar quote", query: "SELECT $tag$ ORDER BY $tag$ FROM people", want: false},
		{name: "parameter", query: "SELECT $1 FROM people ORDER BY id", want: true},
		{name: "subquery", query: "SELECT * FROM (SELECT id FROM people ORDER BY id) p", want: false},
		{name: "Cypher subquery", query: "CALL { MATCH (n) RETURN n ORDER BY n.id } RETURN n", want: false},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if got := HasTopLevelOrderBy(test.query); got != test.want {
				t.Fatalf("HasTopLevelOrderBy(%q) = %t, want %t", test.query, got, test.want)
			}
		})
	}
}

func TestHasParameter(t *testing.T) {
	t.Parallel()
	if !HasParameter("MATCH (n) WHERE n.id > $afterKey RETURN n", "afterKey") {
		t.Fatal("HasParameter() missed parameter")
	}
	for _, query := range []string{
		"RETURN '$afterKey'",
		"RETURN 1 // $afterKey",
		"RETURN 1 /* $afterKey */",
		"RETURN $tag$ $afterKey $tag$",
		"RETURN $other",
	} {
		if HasParameter(query, "afterKey") {
			t.Fatalf("HasParameter(%q) accepted hidden or different parameter", query)
		}
	}
}

func TestHasKeyword(t *testing.T) {
	t.Parallel()
	if !HasKeyword("MATCH (n) SKIP 10 RETURN n", "skip") {
		t.Fatal("HasKeyword() missed keyword")
	}

	for _, query := range []string{
		"RETURN 'SKIP'",
		"RETURN 1 // SKIP",
		"RETURN 1 /* SKIP */",
		"RETURN skipping",
	} {
		if HasKeyword(query, "skip") {
			t.Fatalf("HasKeyword(%q) accepted hidden or partial keyword", query)
		}
	}

}

func TestHasFinalTopLevelOrderByField(t *testing.T) {
	t.Parallel()
	for _, query := range []string{
		"MATCH (n) RETURN n.k AS key ORDER BY key",
		"MATCH (n) RETURN n.k AS key ORDER /* stable */ BY key ASC",
		"MATCH (n) RETURN n.k AS key, n.name AS name ORDER BY key, name",
	} {
		if !HasFinalTopLevelOrderByField(query, "key") {
			t.Fatalf("HasFinalTopLevelOrderByField(%q) rejected stable ordering", query)
		}
	}
	for _, query := range []string{
		"MATCH (n) RETURN n.k AS key ORDER BY name",
		"MATCH (n) RETURN n.k AS key ORDER BY key DESC",
		"MATCH (n) RETURN n.k AS key ORDER BY n.key",
		"CALL { MATCH (n) RETURN n ORDER BY n.k } RETURN n",
		"RETURN 1 AS key ORDER BY key UNION RETURN 2 AS key",
		"RETURN 1 AS key ORDER BY key WITH key RETURN key",
		"RETURN 1 AS key ORDER BY key + 1",
		"RETURN 1 AS key ORDER BY key LIMIT 10",
		"RETURN 1 AS key UNION RETURN 2 AS key ORDER BY key",
	} {
		if HasFinalTopLevelOrderByField(query, "key") {
			t.Fatalf("HasFinalTopLevelOrderByField(%q) accepted unsafe ordering", query)
		}
	}
}

func TestHasTopLevelOrderByFieldAllowsSQLPagination(t *testing.T) {
	t.Parallel()
	for _, query := range []string{
		"SELECT id FROM people ORDER BY id LIMIT $2",
		"SELECT id FROM people ORDER BY id ASC OFFSET 1",
		"SELECT id FROM people ORDER BY id, created_at FETCH FIRST 10 ROWS ONLY",
	} {
		if !HasTopLevelOrderByField(query, "id") {
			t.Fatalf("HasTopLevelOrderByField(%q) rejected stable ordering", query)
		}
	}

	for _, query := range []string{
		"SELECT id FROM people ORDER BY created_at LIMIT $2",
		"SELECT id FROM people ORDER BY id DESC LIMIT $2",
		"SELECT id FROM people ORDER BY people.id LIMIT $2",
		"SELECT id FROM a UNION SELECT id FROM b ORDER BY id",
	} {
		if HasTopLevelOrderByField(query, "id") {
			t.Fatalf("HasTopLevelOrderByField(%q) accepted unsafe ordering", query)
		}
	}
}

func TestHasFinalTopLevelLimitParameter(t *testing.T) {
	t.Parallel()
	for _, query := range []string{
		"RETURN 1 AS key ORDER BY key LIMIT $pageRows",
		"RETURN 1 AS key ORDER BY key LIMIT /* bounded */ $pageRows // tail",
	} {
		if !HasFinalTopLevelLimitParameter(query, "pageRows") {
			t.Fatalf("HasFinalTopLevelLimitParameter(%q) = false", query)
		}
	}
	for _, query := range []string{
		"RETURN 1 AS key ORDER BY key LIMIT 10",
		"RETURN 1 AS key ORDER BY key LIMIT $other",
		"CALL { RETURN 1 AS key LIMIT $pageRows } RETURN key",
		"RETURN 'LIMIT $pageRows' AS key ORDER BY key",
	} {
		if HasFinalTopLevelLimitParameter(query, "pageRows") {
			t.Fatalf("HasFinalTopLevelLimitParameter(%q) = true", query)
		}
	}
}

func TestOrderScannerRemainingBranches(t *testing.T) {
	for _, query := range []string{
		"RETURN 1 // ORDER BY hidden\nRETURN 1",
		`RETURN "$afterKey"`,
		"RETURN `$afterKey`",
	} {
		_ = HasTopLevelOrderBy(query)
		_ = HasParameter(query, "afterKey")
		_ = HasKeyword(query, "return")
	}
	if !HasKeyword("$tag$hidden$tag$ RETURN 1", "return") {
		t.Fatal("HasKeyword() missed keyword after dollar quote")
	}

	if !HasTopLevelOrderByField("SELECT id FROM people ORDER BY id", "id") ||
		!HasTopLevelOrderByField("SELECT id FROM people ORDER BY id ASC", "id") {
		t.Fatal("HasTopLevelOrderByField() rejected terminal ascending order")
	}
	if HasTopLevelOrderByField("SELECT id FROM people", "id") {
		t.Fatal("HasTopLevelOrderByField() accepted missing order")
	}

	tokens := topLevelTokens(
		`SELECT "$x", ` + "`$y`" + `, $tag$hidden$tag$, $, (nested) // comment
			 FROM t ORDER BY id`,
	)
	if len(tokens) == 0 {
		t.Fatal("topLevelTokens() returned no tokens")
	}
	if got := skipQuoted(`'a''b' tail`, 1, '\'', true); got != 6 {
		t.Fatalf("skipQuoted() = %d", got)
	}
	if got := skipQuoted(`'unterminated`, 1, '\'', true); got != len(`'unterminated`) {
		t.Fatalf("skipQuoted(unterminated) = %d", got)
	}
	if delimiter, ok := dollarDelimiter("plain"); ok || delimiter != "" {
		t.Fatalf("dollarDelimiter(plain) = %q, %t", delimiter, ok)
	}
	if delimiter, ok := dollarDelimiter("$bad!"); ok || delimiter != "" {
		t.Fatalf("dollarDelimiter(invalid) = %q, %t", delimiter, ok)
	}
	if delimiter, ok := dollarDelimiter("$unterminated"); ok || delimiter != "" {
		t.Fatalf("dollarDelimiter(unterminated) = %q, %t", delimiter, ok)
	}
	if got := skipDollarQuoted("$tag$unterminated", 0, "$tag$"); got != len("$tag$unterminated") {
		t.Fatalf("skipDollarQuoted() = %d", got)
	}
}
