package app

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/config"
)

func TestTargetCredentialRefresher(t *testing.T) {
	initial := "postgresql://loader:old-token@localhost:5432/fixture?sslmode=verify-full"
	connection, err := pgx.ParseConfig(initial)
	if err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(t.TempDir(), "target.dsn")
	refresh := targetCredentialRefresher(config.SecretRef{File: file})
	for _, password := range []string{"renewed-token", "next-token"} {
		dsn := strings.Replace(initial, "old-token", password, 1)
		if err := os.WriteFile(file, []byte(dsn+"\n"), 0600); err != nil {
			t.Fatal(err)
		}
		tlsConfig := connection.TLSConfig
		if err := refresh(t.Context(), connection); err != nil {
			t.Fatal(err)
		}
		if connection.Password != password || connection.TLSConfig != tlsConfig {
			t.Fatal("password was not refreshed or TLS configuration changed")
		}
	}
	if hook := targetCredentialRefresher(config.SecretRef{Env: "TARGET_DSN"}); hook != nil {
		t.Fatal("environment credentials must retain their existing behavior")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := refresh(ctx, connection); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled refresh: %v", err)
	}
	if err := os.Remove(file); err != nil {
		t.Fatal(err)
	}
	if err := refresh(t.Context(), connection); err == nil {
		t.Fatal("missing credential file accepted")
	}
}

func TestTargetCredentialRefresherRejectsIdentityChangesAndMalformedSecrets(t *testing.T) {
	initial := "postgresql://loader:old-token@localhost:5432/fixture?sslmode=verify-full"
	file := filepath.Join(t.TempDir(), "target.dsn")
	refresh := targetCredentialRefresher(config.SecretRef{File: file})
	for _, dsn := range []string{
		strings.Replace(initial, "localhost", "different-host", 1),
		strings.Replace(initial, "5432", "5433", 1),
		strings.Replace(initial, "loader", "other-user", 1),
		strings.Replace(initial, "/fixture", "/different-database", 1),
		"postgresql://loader:DO-NOT-LOG-ME@%invalid",
	} {
		connection, err := pgx.ParseConfig(initial)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(file, []byte(dsn), 0600); err != nil {
			t.Fatal(err)
		}
		err = refresh(t.Context(), connection)
		if err == nil || strings.Contains(err.Error(), "DO-NOT-LOG-ME") ||
			connection.Password != "old-token" {
			t.Fatalf("unsafe rotation outcome: %v", err)
		}
	}
}
