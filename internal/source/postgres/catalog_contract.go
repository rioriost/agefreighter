package postgres

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/url"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

const CatalogRequestMaxBytes = 16 << 10

// CatalogRequest is a connection-only review, not a LoadJob. It contains no
// credential, query, target or caller-selected local path.
type CatalogRequest struct {
	SchemaVersion  int      `json:"schemaVersion"`
	Host           string   `json:"host"`
	Port           int      `json:"port"`
	Database       string   `json:"database"`
	Username       string   `json:"username"`
	Schemas        []string `json:"schemas"`
	SourceCASHA256 string   `json:"sourceCASHA256"`
}

func DecodeCatalogRequest(data []byte) (CatalogRequest, error) {
	var request CatalogRequest
	if len(data) > CatalogRequestMaxBytes || exactCatalogJSON(data, &request) != nil {
		return request, errors.New("invalid PostgreSQL catalog request")
	}
	scope, err := catalogSchemas(request.Schemas)
	if err != nil || request.SchemaVersion != 1 || !catalogHost(request.Host) || request.Port < 1 || request.Port > 65535 || !catalogSchemaName.MatchString(request.Database) || !catalogSchemaName.MatchString(request.Username) {
		return request, errors.New("invalid PostgreSQL catalog connection or scope")
	}
	request.Schemas = scope
	if request.SourceCASHA256 != "" && (len(request.SourceCASHA256) != 64 || strings.Trim(request.SourceCASHA256, "0123456789abcdef") != "") {
		return request, errors.New("invalid reviewed catalog CA checksum")
	}
	return request, nil
}

func catalogHost(host string) bool {
	if net.ParseIP(host) != nil {
		return true
	}
	if len(host) == 0 || len(host) > 253 {
		return false
	}
	for _, label := range strings.Split(host, ".") {
		if len(label) == 0 || len(label) > 63 || label[0] == '-' || label[len(label)-1] == '-' {
			return false
		}
		for _, c := range label {
			if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-') {
				return false
			}
		}
	}
	return true
}

// ValidateCatalogConnection binds the protected connection to the public
// review. stagedCA must be empty at submission; only the worker may supply its
// own operation-private CA path. No libpq options or alternative hosts pass.
func ValidateCatalogConnection(request CatalogRequest, dsn, stagedCA string) error {
	fail := errors.New("catalog requires the reviewed source connection with verified TLS")
	if len(dsn) == 0 || len(dsn) > 64<<10 || strings.ContainsRune(dsn, 0) {
		return fail
	}
	u, err := url.Parse(dsn)
	if err != nil || u.Scheme != "postgresql" || u.Opaque != "" || u.Fragment != "" || u.User == nil || !strings.EqualFold(u.Hostname(), request.Host) || u.Port() != strconv.Itoa(request.Port) || u.Path != "/"+request.Database || u.User.Username() != request.Username {
		return fail
	}
	password, ok := u.User.Password()
	if !ok || password == "" || strings.ContainsRune(password, 0) {
		return fail
	}
	query, err := url.ParseQuery(u.RawQuery)
	if err != nil || len(query["sslmode"]) != 1 || query.Get("sslmode") != "verify-full" || len(query["connect_timeout"]) != 1 || query.Get("connect_timeout") != "15" {
		return fail
	}
	want := 2
	if stagedCA != "" {
		want++
		if len(query["sslrootcert"]) != 1 || query.Get("sslrootcert") != stagedCA {
			return fail
		}
	}
	if len(query) != want {
		return fail
	}
	return nil
}

// DecodeCatalog verifies a complete metadata artifact for exactly the reviewed
// scope. Missing safety flags are rejected, not silently defaulted to false.
func DecodeCatalog(data []byte, schemas []string) (Catalog, error) {
	var doc Catalog
	fail := errors.New("invalid or incomplete PostgreSQL catalog report")
	if len(data) > CatalogMaxBytes || exactCatalogJSON(data, &doc) != nil {
		return doc, fail
	}
	scope, err := catalogSchemas(schemas)
	if err != nil || doc.SchemaVersion != 1 || doc.Command != "postgres-catalog" || !doc.Complete || !slices.Equal(doc.Schemas, scope) || doc.Tables == nil || len(doc.Tables) > CatalogMaxTables {
		return doc, fail
	}
	tables := map[string]bool{}
	for _, table := range doc.Tables {
		key := table.Schema + "\x00" + table.Name
		if !slices.Contains(scope, table.Schema) || !catalogIdentifier(table.Name) || tables[key] || !slices.Contains([]string{"r", "p", "v", "m", "f"}, table.Kind) || table.Columns == nil || len(table.Columns) > CatalogMaxColumns || table.Constraints == nil || len(table.Constraints) > CatalogMaxConstraints {
			return doc, fail
		}
		tables[key] = true
		columns := map[string]bool{}
		for _, column := range table.Columns {
			if !catalogIdentifier(column.Name) || !catalogIdentifier(column.TypeSchema) || !catalogIdentifier(column.Type) || columns[column.Name] {
				return doc, fail
			}
			columns[column.Name] = true
		}
		constraints := map[string]bool{}
		for _, c := range table.Constraints {
			if !catalogIdentifier(c.Name) || constraints[c.Name] || c.Kind != "p" && c.Kind != "f" || len(c.Columns) == 0 || len(c.Columns) > CatalogMaxColumns || c.ReferencedColumns == nil {
				return doc, fail
			}
			constraints[c.Name] = true
			seen := map[string]bool{}
			for _, name := range c.Columns {
				if !columns[name] || seen[name] {
					return doc, fail
				}
				seen[name] = true
			}
			if c.Kind == "p" {
				if c.ReferencedSchema != "" || c.ReferencedTable != "" || len(c.ReferencedColumns) != 0 {
					return doc, fail
				}
			} else {
				if !catalogIdentifier(c.ReferencedSchema) || !catalogIdentifier(c.ReferencedTable) || len(c.ReferencedColumns) != len(c.Columns) {
					return doc, fail
				}
				seen = map[string]bool{}
				for _, name := range c.ReferencedColumns {
					if !catalogIdentifier(name) || seen[name] {
						return doc, fail
					}
					seen[name] = true
				}
			}
		}
	}
	return doc, nil
}

func catalogIdentifier(s string) bool {
	return len(s) > 0 && len(s) <= 63 && utf8.ValidString(s) && !strings.ContainsRune(s, 0)
}

// Compare the parsed shape with the fully serialized type: every field is
// required, null arrays and unknown fields cannot masquerade as complete data.
// Token traversal also rejects duplicate keys and excessively nested input.
func exactCatalogJSON(data []byte, destination any) error {
	d := json.NewDecoder(bytes.NewReader(data))
	if err := catalogJSONValue(d, 0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("extra JSON value")
	}
	d = json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(destination); err != nil {
		return err
	}
	encoded, err := json.Marshal(destination)
	if err != nil {
		return err
	}
	var input, output any
	if json.Unmarshal(data, &input) != nil || json.Unmarshal(encoded, &output) != nil || !reflect.DeepEqual(input, output) {
		return errors.New("missing catalog fields")
	}
	return nil
}

func catalogJSONValue(d *json.Decoder, depth int) error {
	if depth > 12 {
		return errors.New("catalog JSON nesting exceeded")
	}
	token, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	if delim != '{' && delim != '[' {
		return errors.New("invalid catalog JSON")
	}
	keys := map[string]bool{}
	for d.More() {
		if delim == '{' {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || keys[name] {
				return errors.New("duplicate catalog field")
			}
			keys[name] = true
		}
		if err := catalogJSONValue(d, depth+1); err != nil {
			return err
		}
	}
	_, err = d.Token()
	return err
}
