package postgres

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"net"

	"github.com/jackc/pgx/v5/pgconn"
)

// initializationError retains only fixed diagnostic labels, never the original
// driver error, DSN, server response, certificate name or credentials.
type initializationError struct {
	stage    string
	category string
}

func (e *initializationError) Error() string {
	return "PostgreSQL initialization failed [" + e.stage + "/" + e.category + "]"
}

func (e *initializationError) Is(target error) bool {
	return e.category == "canceled" && target == context.Canceled ||
		e.category == "deadline-exceeded" && target == context.DeadlineExceeded
}

// InitializationDiagnostic returns only labels constructed by this package.
// It deliberately does not inspect error strings or expose wrapped SDK errors.
func InitializationDiagnostic(err error) (stage, category string, ok bool) {
	var diagnostic *initializationError
	if errors.As(err, &diagnostic) {
		return diagnostic.stage, diagnostic.category, true
	}
	return "", "", false
}

func initializationFailure(ctx context.Context, stage string, err error) error {
	category := "unclassified"
	var database *pgconn.PgError
	var parse *pgconn.ParseConfigError
	var certificate *tls.CertificateVerificationError
	var unknownAuthority x509.UnknownAuthorityError
	var hostname x509.HostnameError
	var invalidCertificate x509.CertificateInvalidError
	var dns *net.DNSError
	var network net.Error
	switch {
	case ctx != nil && errors.Is(ctx.Err(), context.Canceled), errors.Is(err, context.Canceled):
		category = "canceled"
	case ctx != nil && errors.Is(ctx.Err(), context.DeadlineExceeded), errors.Is(err, context.DeadlineExceeded):
		category = "deadline-exceeded"
	case errors.As(err, &parse):
		category = "connection-configuration"
	case errors.As(err, &database):
		switch database.Code {
		case "28P01", "28000":
			category = "authentication"
		case "42501":
			category = "access-denied"
		case "3D000":
			category = "database-not-found"
		default:
			category = "database-error"
		}
	case errors.As(err, &certificate), errors.As(err, &unknownAuthority), errors.As(err, &hostname), errors.As(err, &invalidCertificate):
		category = "tls-verification"
	case errors.As(err, &dns):
		category = "dns"
	case errors.As(err, &network):
		category = "network"
		if network.Timeout() {
			category = "network-timeout"
		}
	}
	return &initializationError{stage: stage, category: category}
}
