package cosmos

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/runtime"
	"github.com/Azure/azure-sdk-for-go/sdk/data/azcosmos"
)

func TestNewSDKQueryClientRejectsCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := NewSDKQueryClient(ctx, "https://example.documents.azure.com:443/", "graphdb")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("NewSDKQueryClient with a cancelled context = %v, want context.Canceled", err)
	}
}

func TestNewSDKQueryClientConstructsAndCachesContainers(t *testing.T) {
	client, err := NewSDKQueryClient(context.Background(), "https://example.documents.azure.com:443/", "graphdb")
	if err != nil {
		t.Fatalf("NewSDKQueryClient: %v", err)
	}
	defer client.Close()

	first, err := client.container("people")
	if err != nil {
		t.Fatalf("container: %v", err)
	}
	second, err := client.container("people")
	if err != nil {
		t.Fatalf("container: %v", err)
	}
	if first != second {
		t.Error("expected the same container client to be returned (cached), got different instances")
	}
	other, err := client.container("orgs")
	if err != nil {
		t.Fatalf("container: %v", err)
	}
	if other == first {
		t.Error("expected a distinct container client for a distinct container name")
	}
}

func TestSDKQueryClientCloseIsIdempotent(t *testing.T) {
	client, err := NewSDKQueryClient(context.Background(), "https://example.documents.azure.com:443/", "graphdb")
	if err != nil {
		t.Fatalf("NewSDKQueryClient: %v", err)
	}
	if err := client.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := client.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestSDKQueryClientImplementsThrottleObserverAndCloser(t *testing.T) {
	client, err := NewSDKQueryClient(context.Background(), "https://example.documents.azure.com:443/", "graphdb")
	if err != nil {
		t.Fatalf("NewSDKQueryClient: %v", err)
	}
	defer client.Close()
	var (
		_ ThrottleObserver = client
		_ Closer           = client
		_ QueryClient      = client
	)
	if client.ThrottledRequests() != 0 {
		t.Errorf("ThrottledRequests = %d, want 0 before any request", client.ThrottledRequests())
	}
}

func TestSDKQueryClientBuildsPager(t *testing.T) {
	client, err := NewSDKQueryClient(
		context.Background(),
		"https://example.documents.azure.com:443/",
		"graphdb",
	)
	if err != nil {
		t.Fatalf("NewSDKQueryClient: %v", err)
	}
	defer client.Close()

	pager, err := client.NewQueryPager(
		"people",
		"SELECT * FROM c WHERE c.kind = @kind",
		[]Parameter{{Name: "@kind", Value: "person"}},
		QueryOptions{
			PageSizeHint: 10, HasContinuationToken: true,
			ContinuationToken: "resume", ContinuationTokenLimitKB: 8,
		},
	)
	if err != nil {
		t.Fatalf("NewQueryPager: %v", err)
	}
	if !pager.More() {
		t.Fatal("new SDK pager has no first page")
	}
}

func TestSDKPagerMapsPageAndError(t *testing.T) {
	continuation := "next"
	pager := &sdkPager{pager: runtime.NewPager(runtime.PagingHandler[azcosmos.QueryItemsResponse]{
		More: func(page azcosmos.QueryItemsResponse) bool {
			return page.ContinuationToken != nil
		},
		Fetcher: func(context.Context, *azcosmos.QueryItemsResponse) (azcosmos.QueryItemsResponse, error) {
			return azcosmos.QueryItemsResponse{
				Response:          azcosmos.Response{RequestCharge: 3.5},
				ContinuationToken: &continuation,
				Items:             [][]byte{[]byte(`{"id":"p1"}`)},
			}, nil
		},
	})}
	page, err := pager.NextPage(context.Background())
	if err != nil {
		t.Fatalf("NextPage: %v", err)
	}
	if len(page.Items) != 1 || page.RequestCharge != 3.5 ||
		!page.HasContinuation || page.ContinuationToken != continuation {
		t.Fatalf("NextPage = %#v", page)
	}

	injected := errors.New("page failed")
	failing := &sdkPager{pager: runtime.NewPager(runtime.PagingHandler[azcosmos.QueryItemsResponse]{
		More: func(azcosmos.QueryItemsResponse) bool { return false },
		Fetcher: func(context.Context, *azcosmos.QueryItemsResponse) (azcosmos.QueryItemsResponse, error) {
			return azcosmos.QueryItemsResponse{}, injected
		},
	})}
	if _, err := failing.NextPage(context.Background()); !errors.Is(err, injected) {
		t.Fatalf("NextPage error = %v, want %v", err, injected)
	}
}

// fakeTransport is a minimal policy.Transporter double used to drive the
// throttlePolicy through a real azcore pipeline without any network access.
type fakeTransport struct {
	statusCode int
}

func (transport fakeTransport) Do(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: transport.statusCode,
		Body:       http.NoBody,
		Header:     make(http.Header),
		Request:    req,
	}, nil
}

func TestThrottlePolicyCountsOnly429(t *testing.T) {
	counter := new(int64)
	buildPipeline := func(statusCode int) runtime.Pipeline {
		return runtime.NewPipeline("cosmostest", "v1", runtime.PipelineOptions{
			PerRetry: []policy.Policy{&throttlePolicy{counter: counter}},
		}, &policy.ClientOptions{
			Transport: fakeTransport{statusCode: statusCode},
			Retry:     policy.RetryOptions{MaxRetries: -1},
		})
	}

	req, err := runtime.NewRequest(context.Background(), http.MethodGet, "https://example.com")
	if err != nil {
		t.Fatalf("runtime.NewRequest: %v", err)
	}
	if _, err := buildPipeline(http.StatusOK).Do(req); err != nil {
		t.Fatalf("pipeline.Do (200): %v", err)
	}
	if got := *counter; got != 0 {
		t.Fatalf("throttled count after a 200 response = %d, want 0", got)
	}

	req, err = runtime.NewRequest(context.Background(), http.MethodGet, "https://example.com")
	if err != nil {
		t.Fatalf("runtime.NewRequest: %v", err)
	}
	if _, err := buildPipeline(http.StatusTooManyRequests).Do(req); err != nil {
		t.Fatalf("pipeline.Do (429): %v", err)
	}
	if got := *counter; got != 1 {
		t.Fatalf("throttled count after a 429 response = %d, want 1", got)
	}
}
