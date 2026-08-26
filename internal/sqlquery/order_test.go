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
