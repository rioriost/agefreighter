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
		{name: "dollar quote", query: "SELECT $tag$ ORDER BY $tag$ FROM people", want: false},
		{name: "parameter", query: "SELECT $1 FROM people ORDER BY id", want: true},
		{name: "subquery", query: "SELECT * FROM (SELECT id FROM people ORDER BY id) p", want: false},
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
