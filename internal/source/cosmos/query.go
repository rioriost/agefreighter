package cosmos

import "context"

// Parameter is a named, natively-typed value bound to a parametrized Cosmos
// query. Value is one of: nil, bool, int64, float64, string, []any, or
// map[string]any (each following the same recursive shape), matching the
// strict JSON values preserved by config.CosmosParamValue.
type Parameter struct {
	Name  string
	Value any
}

// Page is one page of raw JSON documents returned by a Cosmos query,
// together with the request-level diagnostics needed for telemetry. Only
// the current page is ever retained by Iterator.
type Page struct {
	// Items holds the raw JSON bytes of each document in this page.
	Items [][]byte
	// ContinuationToken is the opaque token needed to resume AFTER this
	// page. It is only meaningful when HasContinuation is true; Cosmos may
	// return an empty page that still carries a continuation token, in
	// which case iteration must keep going.
	ContinuationToken string
	HasContinuation   bool
	// RequestCharge is the request-unit cost reported for this page.
	RequestCharge float64
	// FailedRequestCount is the number of failed backend attempts recorded
	// while fetching this page (for example transient retries handled
	// internally by the SDK).
	FailedRequestCount int
}

// QueryOptions bounds a single mapping's query execution.
type QueryOptions struct {
	// PageSizeHint is the maximum number of items requested per page.
	PageSizeHint int32
	// ContinuationToken, when HasContinuationToken is true, is the token
	// used to fetch the FIRST page returned by this pager. This is how
	// Iterator reopens a page after resuming from a checkpoint.
	ContinuationToken    string
	HasContinuationToken bool
	// ContinuationTokenLimitKB bounds the size of continuation tokens the
	// service is asked to return.
	ContinuationTokenLimitKB int32
}

// Pager fetches successive, bounded pages for a single query execution.
// More reports whether NextPage can usefully be called again: it is true
// before the first call to NextPage, and after that reflects whether the
// most recently fetched page carries a continuation token. Implementations
// must return promptly when ctx is cancelled.
type Pager interface {
	More() bool
	NextPage(ctx context.Context) (Page, error)
}

// QueryClient opens bounded, cross-partition query pagers against a Cosmos
// container. Implementations are responsible for authentication, container
// client caching, and any retry/throttle instrumentation. Iterator creates
// exactly one Pager per mapping and reuses one QueryClient (and therefore
// one underlying account client/database) across every mapping in a
// source.
type QueryClient interface {
	NewQueryPager(
		container string,
		query string,
		parameters []Parameter,
		options QueryOptions,
	) (Pager, error)
}

// ThrottleObserver is an optional interface a QueryClient can implement to
// expose a count of HTTP 429 (throttling) responses observed across every
// request it has issued, without recording or logging any request or
// response content. Iterator reports this via Telemetry when available.
type ThrottleObserver interface {
	ThrottledRequests() int64
}

// Closer is an optional interface a QueryClient can implement to release
// underlying connections when Iterator.Close is called. Implementations
// must make Close safe to call more than once.
type Closer interface {
	Close() error
}
