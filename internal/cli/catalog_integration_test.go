package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/source/postgres"
)

// Opt-in disposable local PostgreSQL only. TLS must actually be enabled, and
// the fixture CA supplied separately. This creates/drops only a unique schema.
func TestCatalogCLIIntegrationVerifiedTLS(t *testing.T) {
	dsn, caFile := os.Getenv("AGEFREIGHTER_CATALOG_CLI_TEST_DSN"), os.Getenv("AGEFREIGHTER_CATALOG_CLI_TEST_CA")
	if dsn == "" || caFile == "" {
		t.Skip("requires dedicated TLS-enabled local catalog fixture")
	}
	u, err := url.Parse(dsn)
	if err != nil {
		t.Fatal("invalid fixture URI")
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal("fixture port required")
	}
	dir := t.TempDir()
	ca, err := os.ReadFile(caFile)
	if err != nil {
		t.Fatal("fixture CA unavailable")
	}
	staged := filepath.Join(dir, "source-ca.pem")
	if err = os.WriteFile(staged, ca, 0600); err != nil {
		t.Fatal(err)
	}
	query := u.Query()
	query.Set("sslrootcert", staged)
	u.RawQuery = query.Encode()
	conn, err := pgx.Connect(t.Context(), u.String())
	if err != nil {
		t.Fatal("verified TLS fixture connection failed", err)
	}
	t.Cleanup(func() { _ = conn.Close(context.Background()) })
	var ssl bool
	if err = conn.QueryRow(t.Context(), "SELECT ssl FROM pg_stat_ssl WHERE pid=pg_backend_pid()").Scan(&ssl); err != nil || !ssl {
		t.Fatal("fixture connection did not use TLS")
	}
	schema := fmt.Sprintf("af_catalog_cli_%d", time.Now().UnixNano())
	s := pgx.Identifier{schema}.Sanitize()
	if _, err = conn.Exec(t.Context(), "CREATE SCHEMA "+s); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, _ = conn.Exec(ctx, "DROP SCHEMA "+s+" CASCADE")
	})
	if _, err = conn.Exec(t.Context(), "CREATE TABLE "+s+".person(id bigint PRIMARY KEY); CREATE TABLE "+s+".orders(id bigint PRIMARY KEY, person_id bigint NOT NULL REFERENCES "+s+".person(id)); INSERT INTO "+s+".person VALUES(1); INSERT INTO "+s+".orders VALUES(2,1)"); err != nil {
		t.Fatal(err)
	}
	r := postgres.CatalogRequest{SchemaVersion: 1, Host: u.Hostname(), Port: port, Database: strings.TrimPrefix(u.Path, "/"), Username: u.User.Username(), Schemas: []string{schema}}
	digest := sha256.Sum256(ca)
	r.SourceCASHA256 = hex.EncodeToString(digest[:])
	configuration, _ := json.Marshal(r)
	path := filepath.Join(dir, "job.json")
	if err = os.WriteFile(path, configuration, 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("AGEFREIGHTER_SOURCE_DSN", u.String())
	t.Setenv("SSL_CERT_FILE", staged)
	var output, stderr bytes.Buffer
	if err = ExecuteContext(t.Context(), NewAgefreighter(&output, &stderr), []string{"postgres-catalog", path}); err != nil {
		t.Fatal(err)
	}
	doc, err := postgres.DecodeCatalog(output.Bytes(), r.Schemas)
	if err != nil || len(doc.Tables) != 2 || stderr.Len() != 0 {
		t.Fatal("complete catalog CLI artifact missing", err)
	}
	if len(doc.Tables[0].Constraints) != 2 || len(doc.Tables[1].Constraints) != 1 {
		t.Fatal("PK/FK evidence missing")
	}
	// Removing the CA must fail. The CLI cannot downgrade to unverified TLS.
	query.Del("sslrootcert")
	u.RawQuery = query.Encode()
	r.SourceCASHA256 = ""
	configuration, _ = json.Marshal(r)
	if err = os.WriteFile(path, configuration, 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("AGEFREIGHTER_SOURCE_DSN", u.String())
	t.Setenv("SSL_CERT_FILE", "")
	output.Reset()
	stderr.Reset()
	if err = ExecuteContext(t.Context(), NewAgefreighter(&output, &stderr), []string{"postgres-catalog", path}); err == nil || output.Len() != 0 {
		t.Fatal("untrusted server certificate admitted")
	}
	var count int
	if err = conn.QueryRow(t.Context(), "SELECT count(*) FROM "+s+".orders").Scan(&count); err != nil || count != 1 {
		t.Fatal("catalog modified source data")
	}
}
