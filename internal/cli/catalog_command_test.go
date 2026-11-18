package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCatalogCLIRejectsUnsafeReviewBeforeDatabaseAccess(t *testing.T) {
	config := `{"schemaVersion":1,"host":"127.0.0.1","port":5432,"database":"source","username":"reader","schemas":["public"],"sourceCASHA256":""}`
	file := filepath.Join(t.TempDir(), "catalog.json")
	if err := os.WriteFile(file, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}

	valid := "postgresql://reader:private-cli-secret@127.0.0.1:5432/source?sslmode=verify-full&connect_timeout=15"
	for _, kind := range []string{"missing", "directory", "host", "tls", "pg-options", "ca-path", "cancelled"} {
		t.Run(kind, func(t *testing.T) {
			path := file
			dsn := valid
			t.Setenv("SSL_CERT_FILE", "")
			switch kind {
			case "missing":
				path += "missing"
			case "directory":
				path = filepath.Dir(file)
			case "host":
				dsn = strings.Replace(dsn, "127.0.0.1", "127.0.0.2", 1)
			case "tls":
				dsn = strings.Replace(dsn, "verify-full", "disable", 1)
			case "pg-options":
				t.Setenv("PGOPTIONS", "-crole=admin")
			case "ca-path":
				t.Setenv("SSL_CERT_FILE", "/unreviewed.pem")
			}
			t.Setenv("AGEFREIGHTER_SOURCE_DSN", dsn)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if kind == "cancelled" {
				cancel()
			}
			var out, errs bytes.Buffer
			err := ExecuteContext(ctx, NewAgefreighter(&out, &errs), []string{"postgres-catalog", path})
			if err == nil || strings.Contains(err.Error(), "private-cli-secret") || out.Len() != 0 {
				t.Fatal("unsafe or leaking catalog command", err)
			}
		})
	}
}

func TestCatalogCLIRejectsChangedOrMissingReviewedCA(t *testing.T) {
	dir := t.TempDir()
	ca := []byte("reviewed CA")
	digest := sha256.Sum256(ca)
	config := `{"schemaVersion":1,"host":"127.0.0.1","port":5432,"database":"source","username":"reader","schemas":["public"],"sourceCASHA256":"` + hex.EncodeToString(digest[:]) + `"}`
	path := filepath.Join(dir, "catalog.json")
	if err := os.WriteFile(path, []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("AGEFREIGHTER_SOURCE_DSN", "postgres://reader@127.0.0.1:5432/source?sslmode=verify-full&connect_timeout=15")
	caPath := filepath.Join(dir, "source-ca.pem")
	for _, test := range []struct {
		name string
		ca   []byte
		want string
	}{
		{name: "missing", want: "reviewed catalog source CA is unavailable"},
		{name: "changed", ca: []byte("changed CA"), want: "catalog source CA changed after review"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if test.ca != nil {
				if err := os.WriteFile(caPath, test.ca, 0o600); err != nil {
					t.Fatal(err)
				}
			}
			t.Setenv("SSL_CERT_FILE", caPath)
			var output, stderr bytes.Buffer
			err := ExecuteContext(t.Context(), NewAgefreighter(&output, &stderr), []string{"postgres-catalog", path})
			if err == nil || !strings.Contains(err.Error(), test.want) || output.Len() != 0 {
				t.Fatalf("unreviewed CA accepted or wrong error: %v, output %q", err, output.String())
			}
		})
	}
}
