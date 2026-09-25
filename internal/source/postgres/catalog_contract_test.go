package postgres

import (
	"encoding/json"
	"strings"
	"testing"
)

const catalogRequestFixture = `{"schemaVersion":1,"host":"source.example","port":5432,"database":"source","username":"reader","schemas":["public"],"sourceCASHA256":""}`
const catalogDSNFixture = "postgresql://reader:private-password@source.example:5432/source?sslmode=verify-full&connect_timeout=15"

func TestCatalogRequestStrictConnectionContract(t *testing.T) {
	r, err := DecodeCatalogRequest([]byte(catalogRequestFixture))
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateCatalogConnection(r, catalogDSNFixture, ""); err != nil {
		t.Fatal(err)
	}
	for _, data := range []string{
		strings.Replace(catalogRequestFixture, `"schemaVersion":1`, `"schemaVersion":1,"schemaVersion":1`, 1),
		strings.Replace(catalogRequestFixture, `"host":"source.example",`, "", 1),
		strings.Replace(catalogRequestFixture, `"port":5432`, `"port":65536`, 1),
		strings.Replace(catalogRequestFixture, `"public"`, `"pg_catalog"`, 1),
		strings.Replace(catalogRequestFixture, `"source.example"`, `"host,evil.example"`, 1),
		strings.Replace(catalogRequestFixture, `"source.example"`, `"/tmp"`, 1),
		strings.Replace(catalogRequestFixture, `"source.example"`, `"source..example"`, 1),
		strings.Replace(catalogRequestFixture, `"schemas"`, `"unknown"`, 1),
		catalogRequestFixture + `{}`, strings.Repeat(" ", CatalogRequestMaxBytes) + catalogRequestFixture,
	} {
		if _, err := DecodeCatalogRequest([]byte(data)); err == nil {
			t.Fatal("invalid request accepted")
		}
	}
	for _, dsn := range []string{
		strings.Replace(catalogDSNFixture, "verify-full", "require", 1),
		strings.Replace(catalogDSNFixture, "source.example", "other.example", 1),
		strings.Replace(catalogDSNFixture, ":5432/", ":5433/", 1),
		strings.Replace(catalogDSNFixture, "/source?", "/other?", 1),
		strings.Replace(catalogDSNFixture, "reader:", "other:", 1),
		strings.Replace(catalogDSNFixture, "private-password", "", 1),
		catalogDSNFixture + "&sslmode=verify-full", catalogDSNFixture + "&connect_timeout=15",
		catalogDSNFixture + "&host=evil", catalogDSNFixture + "&options=-crole=admin",
		catalogDSNFixture + "&sslrootcert=/tmp/unreviewed", catalogDSNFixture + "#fragment",
		catalogDSNFixture + "&x=%ZZ", "host=source.example password=private-password",
	} {
		if err := ValidateCatalogConnection(r, dsn, ""); err == nil || strings.Contains(err.Error(), "private-password") {
			t.Fatal("connection mismatch or leaked secret")
		}
	}
	if err := ValidateCatalogConnection(r, catalogDSNFixture+"&sslrootcert=%2Fprivate%2Fsource-ca.pem", "/private/source-ca.pem"); err != nil {
		t.Fatal(err)
	}
	if err := ValidateCatalogConnection(r, catalogDSNFixture, "/private/source-ca.pem"); err == nil {
		t.Fatal("missing staged CA accepted")
	}
	for _, host := range []string{"127.0.0.1", "::1", "source.internal"} {
		if !catalogHost(host) {
			t.Fatal("valid host rejected")
		}
	}
}

func TestCatalogArtifactRequiresExactScopeAndAllSafetyFields(t *testing.T) {
	doc, err := readCatalog(t.Context(), catalogFixture(), []string{"public"})
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(doc)
	if _, err := DecodeCatalog(data, []string{"public"}); err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeCatalog(data, []string{"other"}); err == nil {
		t.Fatal("wrong scope accepted")
	}
	for _, text := range []string{
		strings.Replace(string(data), `"rls":false,`, "", 1),
		strings.Replace(string(data), `"rls":false`, `"rls":null`, 1),
		strings.Replace(string(data), `"rls":false`, `"rls":true,"rls":false`, 1),
		strings.Replace(string(data), `"complete":true`, `"complete":false`, 1),
		strings.Replace(string(data), `"command":"postgres-catalog"`, `"command":"inventory"`, 1),
		strings.Replace(string(data), `"columns":["id"]`, `"columns":["missing"]`, 1),
		strings.Replace(string(data), `"referencedColumns":[]`, `"referencedColumns":null`, 1),
		string(data) + `{}`, strings.Repeat(" ", CatalogMaxBytes) + string(data),
	} {
		if _, err := DecodeCatalog([]byte(text), []string{"public"}); err == nil {
			t.Fatal("malformed artifact accepted", text[:min(len(text), 120)])
		}
	}
	for _, mutate := range []func(*Catalog){
		func(d *Catalog) { d.Tables = nil },
		func(d *Catalog) { d.Tables = append(d.Tables, d.Tables[0]) },
		func(d *Catalog) { d.Tables[0].Columns = append(d.Tables[0].Columns, d.Tables[0].Columns[0]) },
		func(d *Catalog) { d.Tables[0].Constraints[0].Kind = "f" },
	} {
		var copy Catalog
		_ = json.Unmarshal(data, &copy)
		mutate(&copy)
		invalid, _ := json.Marshal(copy)
		if _, err := DecodeCatalog(invalid, []string{"public"}); err == nil {
			t.Fatal("invalid structure accepted")
		}
	}
}
