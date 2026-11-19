package app

import (
	"context"
	"errors"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
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

func TestHorizonDBP1ManagedIdentityRenewalIntegration(t *testing.T) {
	if os.Getenv("AGEFREIGHTER_REQUIRE_HORIZON_P1_REFRESH") != "1" {
		t.Skip("requires an independently leased HorizonDB qualification writer and Azure runner identity")
	}
	host := os.Getenv("AGEFREIGHTER_HORIZON_P1_REFRESH_HOST")
	user := os.Getenv("AGEFREIGHTER_HORIZON_P1_REFRESH_USER")
	clientID := os.Getenv("AGEFREIGHTER_HORIZON_P1_REFRESH_CLIENT_ID")
	writer := regexp.MustCompile(`^afhs-20261118-pg17\.rw\.[a-z0-9]+\.japaneast\.horizondb\.azure\.com$`)
	deadline, err := time.Parse(time.RFC3339, os.Getenv("AGEFREIGHTER_HORIZON_P1_REFRESH_DEADLINE_UTC"))
	if !writer.MatchString(host) || user != "afhs-20261118-p1-csv" ||
		!regexp.MustCompile(`^[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}$`).MatchString(clientID) || err != nil ||
		time.Until(deadline) <= 5*time.Minute || time.Until(deadline) > 2*time.Hour {
		t.Fatal("require the admitted qualification writer, runner identity and bounded deadline")
	}
	expectedExpiry, err := time.Parse(time.RFC3339, os.Getenv("AGEFREIGHTER_HORIZON_P1_REFRESH_INITIAL_EXPIRY_UTC"))
	if err != nil || time.Until(expectedExpiry) < 2*time.Minute || !expectedExpiry.Add(4*time.Minute).Before(deadline) {
		t.Fatal("the independently observed first expiry must be within the read-only window")
	}
	ctx, cancel := context.WithDeadline(t.Context(), deadline.Add(-time.Minute))
	defer cancel()
	identity, err := azidentity.NewManagedIdentityCredential(&azidentity.ManagedIdentityCredentialOptions{
		ID: azidentity.ClientID(clientID),
	})
	if err != nil {
		t.Fatal("Azure managed identity unavailable")
	}
	getToken := func() (azcore.AccessToken, error) {
		requestCtx, stop := context.WithTimeout(ctx, 30*time.Second)
		defer stop()
		return identity.GetToken(requestCtx, policy.TokenRequestOptions{
			Scopes: []string{"https://ossrdbms-aad.database.windows.net/.default"},
		})
	}
	initial, err := getToken()
	if err != nil || initial.Token == "" || !initial.ExpiresOn.Equal(expectedExpiry) {
		t.Fatal("managed identity did not return the credential with the independently observed first expiry")
	}
	mounts, mountsErr := os.ReadFile("/proc/mounts")
	swaps, swapsErr := os.ReadFile("/proc/swaps")
	if mountsErr != nil || swapsErr != nil || !strings.Contains("\n"+string(mounts), "\ntmpfs /dev/shm tmpfs ") ||
		strings.Contains(strings.TrimSpace(string(swaps)), "\n") {
		t.Fatal("require a tmpfs credential path with swap disabled")
	}
	dir, err := os.MkdirTemp("/dev/shm", "agefreighter-p1-renewal-")
	if err != nil {
		t.Fatal("could not create a private in-memory credential directory")
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Error(err)
		}
	})
	path := filepath.Join(dir, "target.dsn")
	dsn := func(token azcore.AccessToken) string {
		u := url.URL{Scheme: "postgresql", Host: net.JoinHostPort(host, "5432"), Path: "/postgres",
			User: url.UserPassword(user, token.Token), RawQuery: "sslmode=verify-full&default_transaction_read_only=on"}
		return u.String()
	}
	cfg, err := pgxpool.ParseConfig(dsn(initial))
	if err != nil {
		t.Fatal("could not configure the verified-TLS pool")
	}
	if err := os.WriteFile(path, []byte(dsn(initial)), 0600); err != nil {
		t.Fatal("could not create the private connection file")
	}
	var renewedPassword atomic.Pointer[string]
	refresh := targetCredentialRefresher(config.SecretRef{File: path})
	cfg.MaxConns = 1
	cfg.BeforeConnect = func(ctx context.Context, conn *pgx.ConnConfig) error {
		if err := refresh(ctx, conn); err != nil {
			return err
		}
		if expected := renewedPassword.Load(); expected != nil && conn.Password != *expected {
			return errors.New("pool did not read the replacement password from the connection file")
		}
		return nil
	}
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Fatal("could not open the HorizonDB connection pool")
	}
	defer pool.Close()
	sample := func(stage string) int32 {
		queryCtx, stop := context.WithTimeout(ctx, 30*time.Second)
		defer stop()
		var pid int32
		var major, role, database string
		var encrypted, readOnly bool
		err := pool.QueryRow(queryCtx, `SELECT pg_backend_pid(),
					(current_setting('server_version_num')::integer / 10000)::text, current_user, current_database(),
					(SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()),
					current_setting('transaction_read_only') = 'on'`).Scan(&pid, &major, &role, &database, &encrypted, &readOnly)
		if err != nil || pid <= 0 || !encrypted || !readOnly || role != user || major != "17" || database != "postgres" {
			t.Fatalf("%s session did not establish the expected verified-TLS, read-only writer identity", stage)
		}
		return pid
	}
	firstPID := sample("initial")
	t.Logf("initial PostgreSQL 17 session established; token expires %s UTC", initial.ExpiresOn.UTC().Format(time.RFC3339))

	timer := time.NewTimer(time.Until(initial.ExpiresOn))
	defer timer.Stop()
	select {
	case <-timer.C:
	case <-ctx.Done():
		t.Fatal("lease ended before the initial credential expired")
	}
	var replacement azcore.AccessToken
	for attempt := 0; attempt < 4; attempt++ {
		replacement, err = getToken()
		if err == nil && replacement.Token != "" && replacement.Token != initial.Token &&
			replacement.ExpiresOn.After(initial.ExpiresOn) && time.Until(replacement.ExpiresOn) >= time.Minute {
			break
		}
		if attempt == 3 {
			t.Fatal("managed identity did not return a distinct unexpired token after the original credential expired")
		}
		timer.Reset(15 * time.Second)
		select {
		case <-timer.C:
		case <-ctx.Done():
			t.Fatal("lease ended before a fresh credential was issued")
		}
	}
	next := filepath.Join(dir, "next.dsn")
	if err := os.WriteFile(next, []byte(dsn(replacement)), 0600); err != nil {
		t.Fatal("could not stage the rotated connection file")
	}
	if err := os.Rename(next, path); err != nil {
		t.Fatal("could not atomically rotate the private connection file")
	}
	renewedPassword.Store(&replacement.Token)
	pool.Reset()
	if sample("renewed") == firstPID {
		t.Fatal("expected a new PostgreSQL backend after pool reset")
	}
	t.Logf("fresh pooled connection verified after initial credential expiration; replacement expires %s",
		replacement.ExpiresOn.UTC().Format(time.RFC3339))
}
