package cosmos

import (
	"crypto/sha256"
	"encoding/hex"
	"sync"

	sourcecontract "github.com/rioriost/agefreighter/internal/source"
)

// continuationDigestLength bounds the truncated, non-secret continuation
// diagnostic exposed through Telemetry. It is intentionally short: enough
// to distinguish pages during debugging, never enough to reconstruct or
// meaningfully leak the underlying continuation token.
const continuationDigestLength = 12

// Telemetry is a point-in-time, non-secret snapshot of an Iterator's
// cumulative diagnostics. It never contains full continuation tokens or any
// document content.
type Telemetry = sourcecontract.Telemetry

// telemetryState accumulates Telemetry counters as pages are fetched.
type telemetryState struct {
	mu                    sync.Mutex
	pages                 int64
	requestCharge         float64
	rawInputBytes         int64
	decodedInputBytes     int64
	failedRequestAttempts int64
	continuationDigest    string
}

func (state *telemetryState) recordPage(page Page) {
	state.mu.Lock()
	defer state.mu.Unlock()
	state.pages++
	state.requestCharge += page.RequestCharge
	state.failedRequestAttempts += int64(page.FailedRequestCount)
	for _, item := range page.Items {
		state.rawInputBytes += int64(len(item))
	}
	if page.HasContinuation {
		state.continuationDigest = truncatedDigest(page.ContinuationToken)
	} else {
		state.continuationDigest = ""
	}
}

func (state *telemetryState) recordDecoded(bytes int64) {
	state.mu.Lock()
	state.decodedInputBytes += bytes
	state.mu.Unlock()
}

func (state *telemetryState) recordFailedRequestAttempt() {
	state.mu.Lock()
	state.failedRequestAttempts++
	state.mu.Unlock()
}

func (state *telemetryState) snapshot(throttledRequests int64) Telemetry {
	state.mu.Lock()
	defer state.mu.Unlock()
	return Telemetry{
		Connector:             "cosmos-nosql",
		Pages:                 state.pages,
		RawInputBytes:         state.rawInputBytes,
		DecodedInputBytes:     state.decodedInputBytes,
		RequestCharge:         state.requestCharge,
		FailedRequestAttempts: state.failedRequestAttempts,
		ThrottledRequests:     throttledRequests,
		ContinuationDigest:    state.continuationDigest,
	}
}

// truncatedDigest returns a short, non-reversible diagnostic for a
// continuation token: never the token itself.
func truncatedDigest(token string) string {
	if token == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(token))
	full := hex.EncodeToString(sum[:])
	if len(full) > continuationDigestLength {
		return full[:continuationDigestLength]
	}
	return full
}
