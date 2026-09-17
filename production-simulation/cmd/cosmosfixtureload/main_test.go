package main

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/data/azcosmos"
)

type testItemPager struct {
	pages []azcosmos.QueryItemsResponse
	err   error
	read  int
}

func (p *testItemPager) More() bool { return p.read < len(p.pages) }
func (p *testItemPager) NextPage(context.Context) (azcosmos.QueryItemsResponse, error) {
	page := p.pages[p.read]
	p.read++
	return page, p.err
}

func TestRequireEmptyPages(t *testing.T) {
	for _, tc := range []struct {
		name    string
		pages   []azcosmos.QueryItemsResponse
		err     error
		wantErr bool
	}{
		{"empty", []azcosmos.QueryItemsResponse{{}}, nil, false},
		{"empty continuations", []azcosmos.QueryItemsResponse{{}, {}, {}}, nil, false},
		{"later document", []azcosmos.QueryItemsResponse{{}, {Items: [][]byte{[]byte("1")}}}, nil, true},
		{"denied", []azcosmos.QueryItemsResponse{{}}, errors.New("403"), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := &testItemPager{pages: tc.pages, err: tc.err}
			err := requireEmptyPages(context.Background(), p)
			if (err != nil) != tc.wantErr {
				t.Fatalf("error=%v", err)
			}
			if err == nil && p.read != len(tc.pages) {
				t.Fatal("did not reach EOF")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	p := &testItemPager{pages: []azcosmos.QueryItemsResponse{{}}}
	if !errors.Is(requireEmptyPages(ctx, p), context.Canceled) || p.read != 0 {
		t.Fatal("cancel did not fail closed")
	}
}

func TestStreamFiles(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "Supplier.jsonl")
	if err := os.WriteFile(path, []byte("{\"id\":\"1\",\"partitionKey\":\"Supplier-0\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	items := make(chan workItem, 1)
	counts := map[string]int{}
	if err := streamFiles(context.Background(), []string{path}, items, counts); err != nil {
		t.Fatal(err)
	}
	close(items)
	item := <-items
	if item.partitionKey != "Supplier-0" || counts["Supplier.jsonl"] != 1 {
		t.Fatalf("item=%#v counts=%#v", item, counts)
	}
}

func TestRetryPolicy(t *testing.T) {
	if !isRetryable(&azcore.ResponseError{StatusCode: http.StatusTooManyRequests}) ||
		!isRetryable(&azcore.ResponseError{StatusCode: http.StatusInternalServerError}) ||
		isRetryable(&azcore.ResponseError{StatusCode: http.StatusForbidden}) ||
		isRetryable(errors.New("local validation")) {
		t.Fatal("unexpected Cosmos retry classification")
	}
	if delay := retryDelay(0, "Supplier-0"); delay < time.Second || delay >= 2*time.Second {
		t.Fatalf("initial retry delay = %s", delay)
	}
	if delay := retryDelay(99, "Supplier-0"); delay < 32*time.Second || delay >= 33*time.Second {
		t.Fatalf("capped retry delay = %s", delay)
	}
}

func TestStreamFilesRejectsMissingPartitionKey(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.jsonl")
	if err := os.WriteFile(path, []byte("{\"id\":\"1\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := streamFiles(context.Background(), []string{path}, make(chan workItem, 1), map[string]int{}); err == nil {
		t.Fatal("streamFiles accepted a document without partitionKey")
	}
}
