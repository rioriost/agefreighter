package main

import (
	"net/url"
	"strings"
	"testing"
)

func TestReadOnlyConnection(t *testing.T) {
	host := "afpg-test.postgres.database.azure.com"
	dsn := "postgresql://afadmin:" + strings.Repeat("a", 24) + "@" + host + ":5432/agefreighter?sslmode=verify-full"
	got, err := readonlyDSN(dsn, host)
	if err != nil {
		t.Fatal(err)
	}
	u, _ := url.Parse(got)
	if u.Query().Get("default_transaction_read_only") != "on" || u.Query().Get("sslmode") != "verify-full" {
		t.Fatal("missing read-only verified TLS")
	}
	for _, bad := range []string{strings.Replace(dsn, "verify-full", "require", 1), dsn + "&options=x", strings.Replace(dsn, "afadmin", "other", 1), strings.Replace(dsn, "5432", "5433", 1), strings.Replace(dsn, host, "foreign.postgres.database.azure.com", 1)} {
		if _, err := readonlyDSN(bad, host); err == nil {
			t.Fatal("accepted changed target", bad)
		}
	}
}
