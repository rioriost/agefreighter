package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	sourcecosmos "github.com/rioriost/agefreighter/internal/source/cosmos"
	sourcepostgres "github.com/rioriost/agefreighter/internal/source/postgres"
)

func TestInventoryInitializationDiagnosticIsRedacted(t *testing.T) {
	_, err := sourcepostgres.NewSnapshotCoordinator(t.Context(), "postgresql://%", 1)
	got := inventoryInitializationError(fmt.Errorf("SECRET-CANARY: %w", err))
	if got.Error() != "network inventory initialization failed [postgresql/snapshot-connect/connection-configuration]" || errors.Unwrap(got) != nil {
		t.Fatalf("unsafe diagnostic: %v", got)
	}
	got = inventoryInitializationError(errors.New("SECRET-CANARY"))
	if got.Error() != "network inventory initialization failed" || errors.Unwrap(got) != nil {
		t.Fatalf("unsafe unknown error: %v", got)
	}
}

func TestInventoryResolutionErrorIsAllowlistedAndRedacted(t *testing.T) {
	const secret = "SECRET-CANARY-https://private.example/?token=value"
	for _, tc := range []struct {
		err      error
		category string
	}{
		{errors.New(secret), "unclassified"},
		{fmt.Errorf("%s: %w", secret, context.Canceled), "canceled"},
		{fmt.Errorf("%s: %w", secret, context.DeadlineExceeded), "deadline-exceeded"},
		{fmt.Errorf("%s: %w", secret, sourcecosmos.ErrDiscoveryLimit), "discovery-limit"},
	} {
		got := inventoryResolutionError(tc.err)
		if got.Error() != "network inventory mapping resolution failed ["+tc.category+"]" || errors.Unwrap(got) != nil {
			t.Fatalf("unsafe error: %v", got)
		}
	}
	for status, category := range map[int]string{400: "request-rejected", 401: "access-denied", 403: "access-denied", 404: "not-found", 408: "service-timeout", 429: "throttled", 500: "service-unavailable", 502: "service-unavailable", 503: "service-unavailable", 504: "service-timeout", 418: "unclassified"} {
		response := &azcore.ResponseError{StatusCode: status, ErrorCode: secret, RawResponse: &http.Response{Body: io.NopCloser(strings.NewReader(secret))}}
		got := inventoryResolutionError(fmt.Errorf("%s: %w", secret, response))
		if got.Error() != "network inventory mapping resolution failed ["+category+"]" || errors.Unwrap(got) != nil {
			t.Fatalf("unsafe status %d error: %v", status, got)
		}
		if got := inventoryResolutionError(errors.Join(response, context.DeadlineExceeded)); !strings.HasSuffix(got.Error(), "[deadline-exceeded]") {
			t.Fatal("deadline must take precedence")
		}
	}
}
