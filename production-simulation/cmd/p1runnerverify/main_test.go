package main

import (
	"errors"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFailureReceiptIsPrivateRedactedAndImmutable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "failure.json")
	secret := "postgresql://afadmin:private-password@example.invalid/db"
	if err := retainFailure(path, errors.New(secret)); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), secret) || !strings.Contains(string(data), `"outcome":"fail"`) {
		t.Fatal("invalid redacted failure receipt")
	}
	info, _ := os.Stat(path)
	if info.Mode().Perm() != 0600 {
		t.Fatal("failure receipt must be private")
	}
	if err := retainFailure(path, &qualificationFailure{"target-digest", "source-key-order"}); !errors.Is(err, os.ErrExist) {
		t.Fatal("overwrote retained failure", err)
	}
	retained, _ := os.ReadFile(path)
	if string(retained) != string(data) {
		t.Fatal("prior evidence changed")
	}
}

func TestFailureReceiptDistinguishesOrderingFromMismatch(t *testing.T) {
	for _, code := range []string{"source-key-order", "canonical-mismatch"} {
		path := filepath.Join(t.TempDir(), "failure.json")
		if err := retainFailure(path, &qualificationFailure{"target-digest", code}); err != nil {
			t.Fatal(err)
		}
		data, _ := os.ReadFile(path)
		if !strings.Contains(string(data), `"code":"`+code+`"`) {
			t.Fatal("missing diagnostic code")
		}
	}
}

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

func TestQualificationProfileIsExplicitAndFailClosed(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want string
	}{{[]string{"job", "host"}, "raw-id"}, {[]string{"job", "host", "raw-id"}, "raw-id"}, {[]string{"job", "host", "gremlin-partition64"}, "gremlin-partition64"}} {
		got, err := qualificationProfile(tc.args)
		if err != nil || got != tc.want {
			t.Fatalf("profile %s %v", got, err)
		}
	}
	for _, args := range [][]string{nil, {"job"}, {"job", "host", "auto"}, {"job", "host", "gremlin-partition64", "extra"}} {
		if _, err := qualificationProfile(args); err == nil {
			t.Fatal("invalid profile accepted")
		}
	}
}
