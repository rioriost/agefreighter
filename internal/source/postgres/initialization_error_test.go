package postgres

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"net"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgproto3"
)

func TestInitializationDiagnosticsDiscardPrivateCauses(t *testing.T) {
	const private = "SECRET-CANARY postgresql://reader:password@private.example/db"
	for _, tc := range []struct {
		name     string
		err      error
		category string
	}{
		{"unknown", errors.New(private), "unclassified"},
		{"auth", &pgconn.PgError{Code: "28P01", Message: private, Detail: private}, "authentication"},
		{"hba", &pgconn.PgError{Code: "28000", Message: private}, "authentication"},
		{"permission", &pgconn.PgError{Code: "42501", Message: private}, "access-denied"},
		{"database", &pgconn.PgError{Code: "3D000", Message: private}, "database-not-found"},
		{"untrusted-code", &pgconn.PgError{Code: private, Message: private}, "database-error"},
		{"tls", &tls.CertificateVerificationError{Err: errors.New(private)}, "tls-verification"},
		{"ca", x509.UnknownAuthorityError{Cert: &x509.Certificate{}}, "tls-verification"},
		{"hostname", x509.HostnameError{Certificate: &x509.Certificate{}, Host: private}, "tls-verification"},
		{"expired", x509.CertificateInvalidError{Cert: &x509.Certificate{}, Reason: x509.Expired, Detail: private}, "tls-verification"},
		{"dns", &net.DNSError{Name: private, Err: private}, "dns"},
		{"network", &net.OpError{Op: "dial", Err: errors.New(private)}, "network"},
		{"timeout", &net.OpError{Op: "dial", Err: &net.DNSError{IsTimeout: true}}, "dns"},
		{"network-timeout", &net.OpError{Op: "dial", Err: os.ErrDeadlineExceeded}, "network-timeout"},
		{"deadline", fmt.Errorf("%s: %w", private, context.DeadlineExceeded), "deadline-exceeded"},
		{"cancel", fmt.Errorf("%s: %w", private, context.Canceled), "canceled"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := initializationFailure(t.Context(), "snapshot-connect", fmt.Errorf("%s: %w", private, tc.err))
			if got.Error() != "PostgreSQL initialization failed [snapshot-connect/"+tc.category+"]" || errors.Unwrap(got) != nil {
				t.Fatalf("unsafe diagnostic: %v", got)
			}
			stage, category, ok := InitializationDiagnostic(fmt.Errorf("outer: %w", got))
			if !ok || stage != "snapshot-connect" || category != tc.category {
				t.Fatalf("diagnostic labels = %q %q %v", stage, category, ok)
			}
			for _, sentinel := range []error{context.Canceled, context.DeadlineExceeded} {
				if errors.Is(got, sentinel) != errors.Is(tc.err, sentinel) {
					t.Fatal("context identity lost")
				}
			}
		})
	}
	if _, _, ok := InitializationDiagnostic(errors.New(private)); ok {
		t.Fatal("accepted arbitrary error text")
	}
	for _, stage := range []string{"snapshot-begin", "snapshot-export"} {
		got := initializationFailure(t.Context(), stage, &pgconn.PgError{Code: "42501", Message: private})
		if got.Error() != "PostgreSQL initialization failed ["+stage+"/access-denied]" {
			t.Fatalf("snapshot stage lost: %v", got)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if got := initializationFailure(ctx, "snapshot-begin", errors.New(private)); !errors.Is(got, context.Canceled) {
		t.Fatal("context cancellation lost")
	}
	_, err := sourceIdentity("postgresql://%")
	if stage, category, ok := InitializationDiagnostic(err); !ok || stage != "connection-parse" || category != "connection-configuration" {
		t.Fatalf("parse diagnostic: %v", err)
	}
}

// Use a loopback-only synthetic server: no live Azure resource or credential.
// This proves pgx's wrapped server error reaches the safe classification boundary.
func TestSnapshotAuthenticationDiagnosticThroughPGX(t *testing.T) {
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	done := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			done <- err
			return
		}
		defer conn.Close()
		_ = conn.SetDeadline(time.Now().Add(5 * time.Second))
		backend := pgproto3.NewBackend(conn, conn)
		if _, err := backend.ReceiveStartupMessage(); err != nil {
			done <- err
			return
		}
		backend.Send(&pgproto3.ErrorResponse{Severity: "FATAL", Code: "28P01", Message: "SECRET-CANARY-private-server-response"})
		done <- backend.Flush()
	}()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	_, err = NewSnapshotCoordinator(ctx, "postgresql://test:test@"+listener.Addr().String()+"/test?sslmode=disable", 1)
	if err == nil || err.Error() != "PostgreSQL initialization failed [snapshot-connect/authentication]" || errors.Unwrap(err) != nil {
		t.Fatalf("unexpected diagnostic: %v", err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}
