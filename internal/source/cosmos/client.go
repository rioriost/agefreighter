package cosmos

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/runtime"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/Azure/azure-sdk-for-go/sdk/data/azcosmos"
)

// SDKQueryClient adapts the Azure Cosmos DB Go SDK (azcosmos) to the
// QueryClient abstraction used by Iterator. It authenticates exclusively
// with azidentity.DefaultAzureCredential, reuses a single client/database
// across an entire source, and caches container clients so repeated
// mappings against the same container do not re-resolve them.
type SDKQueryClient struct {
	client     *azcosmos.Client
	database   *azcosmos.DatabaseClient
	mu         sync.Mutex
	containers map[string]*azcosmos.ContainerClient
	throttled  *int64
}

// NewSDKQueryClient builds a production QueryClient for the given account
// endpoint and database. It installs a per-retry azcore policy that counts
// HTTP 429 (throttling) responses without recording or logging any request
// or response headers, exposed later via ThrottledRequests.
func NewSDKQueryClient(ctx context.Context, endpoint, database string) (*SDKQueryClient, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	credential, err := azidentity.NewDefaultAzureCredential(nil)
	if err != nil {
		return nil, fmt.Errorf("create Cosmos default Azure credential: %w", err)
	}
	throttled := new(int64)
	options := &azcosmos.ClientOptions{
		ClientOptions: policy.ClientOptions{
			PerRetryPolicies: []policy.Policy{&throttlePolicy{counter: throttled}},
		},
	}
	client, err := azcosmos.NewClient(endpoint, credential, options)
	if err != nil {
		return nil, fmt.Errorf("create Cosmos client: %w", err)
	}
	databaseClient, err := client.NewDatabase(database)
	if err != nil {
		return nil, fmt.Errorf("open Cosmos database: %w", err)
	}
	return &SDKQueryClient{
		client:     client,
		database:   databaseClient,
		containers: make(map[string]*azcosmos.ContainerClient),
		throttled:  throttled,
	}, nil
}

// ThrottledRequests implements ThrottleObserver.
func (client *SDKQueryClient) ThrottledRequests() int64 {
	return atomic.LoadInt64(client.throttled)
}

// Close implements Closer. It releases the underlying azcosmos.Client and
// is safe to call more than once.
func (client *SDKQueryClient) Close() error {
	client.client.Close()
	return nil
}

func (client *SDKQueryClient) container(name string) (*azcosmos.ContainerClient, error) {
	client.mu.Lock()
	defer client.mu.Unlock()
	if existing, ok := client.containers[name]; ok {
		return existing, nil
	}
	created, err := client.database.NewContainer(name)
	if err != nil {
		return nil, fmt.Errorf("open Cosmos container: %w", err)
	}
	client.containers[name] = created
	return created, nil
}

// NewQueryPager implements QueryClient. Queries always run cross-partition
// via azcosmos.NewPartitionKey().
func (client *SDKQueryClient) NewQueryPager(
	container string,
	query string,
	parameters []Parameter,
	options QueryOptions,
) (Pager, error) {
	containerClient, err := client.container(container)
	if err != nil {
		return nil, err
	}
	azParameters := make([]azcosmos.QueryParameter, len(parameters))
	for index, parameter := range parameters {
		azParameters[index] = azcosmos.QueryParameter{Name: parameter.Name, Value: parameter.Value}
	}
	queryOptions := &azcosmos.QueryOptions{
		PageSizeHint:                       options.PageSizeHint,
		QueryParameters:                    azParameters,
		ResponseContinuationTokenLimitInKB: options.ContinuationTokenLimitKB,
	}
	if options.HasContinuationToken {
		token := options.ContinuationToken
		queryOptions.ContinuationToken = &token
	}
	pager := containerClient.NewQueryItemsPager(query, azcosmos.NewPartitionKey(), queryOptions)
	return &sdkPager{pager: pager}, nil
}

// sdkPager adapts *runtime.Pager[azcosmos.QueryItemsResponse] to Pager.
type sdkPager struct {
	pager *runtime.Pager[azcosmos.QueryItemsResponse]
}

func (p *sdkPager) More() bool {
	return p.pager.More()
}

func (p *sdkPager) NextPage(ctx context.Context) (Page, error) {
	response, err := p.pager.NextPage(ctx)
	if err != nil {
		return Page{}, err
	}
	page := Page{
		Items:              response.Items,
		RequestCharge:      float64(response.RequestCharge),
		FailedRequestCount: response.Diagnostics.FailedRequestCount(),
	}
	if response.ContinuationToken != nil {
		page.ContinuationToken = *response.ContinuationToken
		page.HasContinuation = true
	}
	return page, nil
}

// throttlePolicy is an azcore per-retry policy that counts HTTP 429
// (throttling) responses. It never records or logs request/response
// headers or bodies.
type throttlePolicy struct {
	counter *int64
}

func (observer *throttlePolicy) Do(req *policy.Request) (*http.Response, error) {
	response, err := req.Next()
	if response != nil && response.StatusCode == http.StatusTooManyRequests {
		atomic.AddInt64(observer.counter, 1)
	}
	return response, err
}
