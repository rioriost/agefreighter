package postgres

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

// Dedicated local fixture only. Do not reuse a cloud qualification DSN: this
// setup creates its own schema/role and drops only those exact test objects.
func TestCatalogIntegrationMetadataAndReadPrivileges(t *testing.T) {
	dsn := os.Getenv("AGEFREIGHTER_CATALOG_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_CATALOG_TEST_DSN to a disposable local PostgreSQL fixture")
	}
	conn, err := pgx.Connect(t.Context(), dsn)
	if err != nil {
		t.Fatal("connect local catalog fixture")
	}
	t.Cleanup(func() { _ = conn.Close(context.Background()) })
	schema := fmt.Sprintf("af_catalog_%d", time.Now().UnixNano())
	role := schema + "_reader"
	s, r := pgx.Identifier{schema}.Sanitize(), pgx.Identifier{role}.Sanitize()
	if _, err = conn.Exec(t.Context(), "CREATE SCHEMA "+s); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, _ = conn.Exec(ctx, "DROP SCHEMA "+s+" CASCADE; DROP ROLE IF EXISTS "+r)
	})
	setup := `CREATE TABLE ` + s + `.person (id bigint PRIMARY KEY, name text);
CREATE TABLE ` + s + `.orders (id bigint PRIMARY KEY, person_id bigint NOT NULL REFERENCES ` + s + `.person(id), optional_person bigint REFERENCES ` + s + `.person(id));
ALTER TABLE ` + s + `.orders ADD CONSTRAINT pending_fk FOREIGN KEY (person_id) REFERENCES ` + s + `.person(id) NOT VALID;
CREATE TABLE ` + s + `.composite (a bigint, b bigint, PRIMARY KEY (a,b));
CREATE COLLATION ` + s + `.case_insensitive (provider=icu, locale='und-u-ks-level2', deterministic=false);
CREATE TABLE ` + s + `.text_identity (id text COLLATE ` + s + `.case_insensitive PRIMARY KEY);
CREATE TABLE ` + s + `.secured (id bigint PRIMARY KEY);
ALTER TABLE ` + s + `.secured ENABLE ROW LEVEL SECURITY;
CREATE TABLE ` + s + `.child (extra text) INHERITS (` + s + `.person);
CREATE VIEW ` + s + `.view_only AS SELECT 1 / (id - id) AS must_not_execute FROM ` + s + `.orders;
INSERT INTO ` + s + `.person VALUES (1, 'catalog-must-not-read-this-value');
INSERT INTO ` + s + `.orders VALUES (1, 1, NULL);
CREATE ROLE ` + r + ` LOGIN PASSWORD 'local-catalog-fixture-only';
GRANT USAGE ON SCHEMA ` + s + ` TO ` + r + `;
GRANT SELECT ON ` + s + `.person, ` + s + `.orders TO ` + r + `;`
	if _, err = conn.Exec(t.Context(), setup); err != nil {
		t.Fatal(err)
	}
	parsed, err := url.Parse(dsn)
	if err != nil {
		t.Fatal(err)
	}
	parsed.User = url.UserPassword(role, "local-catalog-fixture-only")
	doc, err := ReadCatalog(t.Context(), parsed.String(), []string{schema})
	if err != nil {
		t.Fatal(err)
	}
	if !doc.Complete || len(doc.Tables) != 7 {
		t.Fatalf("bad catalog coverage: %+v", doc)
	}
	byName := map[string]CatalogTable{}
	for _, table := range doc.Tables {
		byName[table.Name] = table
	}
	if !byName["person"].Readable || !byName["orders"].Readable || byName["secured"].Readable || !byName["secured"].RLS || !byName["child"].Inheritance || !byName["person"].Inheritance || byName["view_only"].Kind != "v" {
		t.Fatal("privilege, RLS, view or inheritance flags lost")
	}
	if len(byName["composite"].Constraints[0].Columns) != 2 {
		t.Fatal("composite PK collapsed")
	}
	if byName["text_identity"].Columns[0].Deterministic || byName["text_identity"].Constraints[0].StandardEquality || !byName["person"].Constraints[0].StandardEquality {
		t.Fatal("non-binary identity equality was not distinguished")
	}
	constraints := map[string]CatalogConstraint{}
	for _, c := range byName["orders"].Constraints {
		constraints[c.Name] = c
	}
	if constraints["pending_fk"].Validated || !constraints["orders_person_id_fkey"].Enforced || !constraints["orders_person_id_fkey"].StandardEquality || constraints["orders_person_id_fkey"].ReferencedSchema != schema || constraints["orders_person_id_fkey"].ReferencedColumns[0] != "id" {
		t.Fatal("foreign key evidence lost")
	}
	columns := map[string]CatalogColumn{}
	for _, c := range byName["orders"].Columns {
		columns[c.Name] = c
	}
	if columns["optional_person"].NotNull || !columns["person_id"].NotNull || columns["id"].Type != "int8" {
		t.Fatal("column metadata lost")
	}
	encoded, err := json.Marshal(doc)
	if err != nil || strings.Contains(string(encoded), "fixture-only") || strings.Contains(string(encoded), "catalog-must-not-read-this-value") || len(encoded) > CatalogMaxBytes {
		t.Fatal("unsafe report")
	}
	if _, err = ReadCatalog(t.Context(), parsed.String(), []string{schema + "_missing"}); err == nil {
		t.Fatal("missing scope accepted")
	}
	// A table metadata scan never executes its view query or requires its rows.
	var rows int
	if err = conn.QueryRow(t.Context(), "SELECT count(*) FROM "+s+".orders").Scan(&rows); err != nil || rows != 1 {
		t.Fatal("catalog changed source rows")
	}
}
