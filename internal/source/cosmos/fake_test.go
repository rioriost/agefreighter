package cosmos

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
)

// fakePage is a single scripted page returned by fakeClient.
type fakePage struct {
	items              [][]byte
	continuationToken  string
	hasContinuation    bool
	requestCharge      float64
	failedRequestCount int
	nextPageErr        error
	newQueryPagerErr   error
}

// fakePager is a one-shot Pager: Iterator only ever calls NextPage once per
// Pager it creates (each page is fetched from a freshly opened pager,
// parameterized by the continuation needed to reopen it later), so a single
// scripted fakePage is all a fakePager ever needs to serve.
type fakePager struct {
	page   fakePage
	served bool
}

func (p *fakePager) More() bool {
	return !p.served
}

func (p *fakePager) NextPage(ctx context.Context) (Page, error) {
	if err := ctx.Err(); err != nil {
		return Page{}, err
	}
	if p.served {
		return Page{}, errors.New("fakePager: page already served")
	}
	p.served = true
	if p.page.nextPageErr != nil {
		return Page{}, p.page.nextPageErr
	}
	return Page{
		Items:              p.page.items,
		ContinuationToken:  p.page.continuationToken,
		HasContinuation:    p.page.hasContinuation,
		RequestCharge:      p.page.requestCharge,
		FailedRequestCount: p.page.failedRequestCount,
	}, nil
}

// fakeCall records the arguments of one NewQueryPager invocation.
type fakeCall struct {
	container  string
	query      string
	parameters []Parameter
	options    QueryOptions
}

// fakeClient is a deterministic, in-memory QueryClient double. Pages are
// scripted per (container, query) key: each call to NewQueryPager for that
// key pops the next scripted page off the queue.
type fakeClient struct {
	mu        sync.Mutex
	calls     []fakeCall
	queue     map[string][]fakePage
	throttled int64

	closeCount int
	closeErr   error
}

func newFakeClient() *fakeClient {
	return &fakeClient{queue: make(map[string][]fakePage)}
}

func fakeKey(container, query string) string {
	return container + "\x00" + query
}

func (c *fakeClient) script(container, query string, pages ...fakePage) {
	c.mu.Lock()
	defer c.mu.Unlock()
	key := fakeKey(container, query)
	c.queue[key] = append(c.queue[key], pages...)
}

func (c *fakeClient) NewQueryPager(
	container, query string,
	parameters []Parameter,
	options QueryOptions,
) (Pager, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls = append(c.calls, fakeCall{
		container: container, query: query, parameters: parameters, options: options,
	})
	key := fakeKey(container, query)
	pages := c.queue[key]
	if len(pages) == 0 {
		return nil, fmt.Errorf("fakeClient: no scripted page for %s/%s", container, query)
	}
	page := pages[0]
	c.queue[key] = pages[1:]
	if page.newQueryPagerErr != nil {
		return nil, page.newQueryPagerErr
	}
	return &fakePager{page: page}, nil
}

// Close implements Closer so tests can verify Iterator.Close invokes it
// exactly once, even if Iterator.Close is itself called more than once.
func (c *fakeClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closeCount++
	return c.closeErr
}

func (c *fakeClient) ThrottledRequests() int64 {
	return atomic.LoadInt64(&c.throttled)
}

func (c *fakeClient) addThrottled(delta int64) {
	atomic.AddInt64(&c.throttled, delta)
}

func (c *fakeClient) callCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.calls)
}

func (c *fakeClient) callAt(index int) fakeCall {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls[index]
}

func jsonItem(t string) []byte {
	return []byte(t)
}
