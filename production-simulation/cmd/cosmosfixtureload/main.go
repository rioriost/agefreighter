// Command cosmosfixtureload imports the deterministic P1 JSONL fixture into a
// freshly provisioned Cosmos DB for NoSQL container using the VM's managed
// identity. It is intentionally a fixture-preparation tool, not a production
// migration path.
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/Azure/azure-sdk-for-go/sdk/data/azcosmos"
)

type document struct {
	PartitionKey string `json:"partitionKey"`
}

type workItem struct {
	body         []byte
	partitionKey string
}

// The gateway cannot serve cross-partition TOP/aggregate queries directly.
// Reuse the plain projection for both the empty preflight and final count.
const countProjection = "SELECT VALUE 1 FROM c"

type result struct {
	SchemaVersion int               `json:"schemaVersion"`
	FinishedAt    string            `json:"finishedAt"`
	Rows          int64             `json:"rows"`
	RemoteRows    int64             `json:"remoteRows"`
	Files         map[string]int    `json:"files"`
	Hashes        map[string]string `json:"consumedFileSha256"`
	Workers       int               `json:"workers"`
	ElapsedSecs   float64           `json:"elapsedSeconds"`
	RequestCharge float64           `json:"observedRequestCharge,omitempty"`
	Throttled     int64             `json:"throttledResponses,omitempty"`
	ChargeLimit   float64           `json:"requestChargeLimit,omitempty"`
}

func main() {
	if err := run(context.Background(), os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "cosmos fixture load failed:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("cosmosfixtureload", flag.ContinueOnError)
	flags.SetOutput(stderr)
	endpoint := flags.String("endpoint", "", "Cosmos account endpoint")
	database := flags.String("database", "p1", "database name")
	containerName := flags.String("container", "graph", "container name")
	input := flags.String("input", "", "directory containing JSONL files")
	workers := flags.Int("workers", 64, "concurrent upsert workers")
	requireEmpty := flags.Bool("require-empty", false, "refuse writes unless the entire target container is empty")
	maxCharge := flags.Float64("max-request-charge", 0, "stop after this observed RU budget; zero leaves the existing unbounded mode")
	timeout := flags.Duration("timeout", 0, "bound complete preparation, including remote verification")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *endpoint == "" || *input == "" || *database == "" || *containerName == "" {
		return errors.New("endpoint, input, database and container are required")
	}
	if *workers < 1 || *workers > 256 {
		return errors.New("workers must be between 1 and 256")
	}
	if math.IsNaN(*maxCharge) || math.IsInf(*maxCharge, 0) || *maxCharge < 0 || *timeout < 0 {
		return errors.New("request-charge limit and timeout must be finite and nonnegative")
	}
	if *maxCharge > 0 && (*timeout == 0 || !*requireEmpty) {
		return errors.New("RU-bounded preparation requires a timeout and empty-container preflight")
	}
	if *timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *timeout)
		defer cancel()
	}

	files, err := filepath.Glob(filepath.Join(*input, "*.jsonl"))
	if err != nil {
		return fmt.Errorf("list JSONL files: %w", err)
	}
	sort.Strings(files)
	if len(files) != 18 {
		return fmt.Errorf("expected 18 JSONL files, found %d", len(files))
	}

	credential, err := azidentity.NewManagedIdentityCredential(nil)
	if err != nil {
		return fmt.Errorf("create managed identity credential: %w", err)
	}
	charges := &chargeBudget{limit: *maxCharge}
	var options *azcosmos.ClientOptions
	if *maxCharge > 0 {
		options = &azcosmos.ClientOptions{ClientOptions: policy.ClientOptions{
			PerRetryPolicies: []policy.Policy{charges},
		}}
	}
	client, err := azcosmos.NewClient(*endpoint, credential, options)
	if err != nil {
		return fmt.Errorf("create Cosmos client: %w", err)
	}
	defer client.Close()
	databaseClient, err := client.NewDatabase(*database)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	containerClient, err := databaseClient.NewContainer(*containerName)
	if err != nil {
		return fmt.Errorf("open container: %w", err)
	}
	if *requireEmpty {
		pager := containerClient.NewQueryItemsPager(countProjection, azcosmos.NewPartitionKey(), &azcosmos.QueryOptions{PageSizeHint: 1})
		if err := requireEmptyPages(ctx, pager); err != nil {
			return err
		}
	}

	started := time.Now()
	loadCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	items := make(chan workItem, *workers*4)
	errCh := make(chan error, 1)
	var loaded atomic.Int64
	var group sync.WaitGroup
	for range *workers {
		group.Add(1)
		go func() {
			defer group.Done()
			for item := range items {
				itemErr := upsertWithRetry(loadCtx, containerClient, item)
				if itemErr != nil {
					select {
					case errCh <- itemErr:
						cancel()
					default:
					}
					return
				}
				count := loaded.Add(1)
				if count%100000 == 0 {
					fmt.Fprintf(stderr, "loaded %d documents\n", count)
				}
			}
		}()
	}

	fileCounts := make(map[string]int, len(files))
	hashes := make(map[string]string, len(files))
	readErr := streamFiles(loadCtx, files, items, fileCounts, hashes)
	close(items)
	group.Wait()
	select {
	case workerErr := <-errCh:
		return fmt.Errorf("upsert item after %d successful writes: %w", loaded.Load(), workerErr)
	default:
	}
	if readErr != nil {
		return readErr
	}

	remoteRows, err := countRemote(ctx, containerClient)
	if err != nil {
		return err
	}
	if remoteRows != loaded.Load() {
		return fmt.Errorf("container count mismatch: wrote %d, queried %d", loaded.Load(), remoteRows)
	}
	charges.mu.Lock()
	observed, throttled := charges.total, charges.throttled
	charges.mu.Unlock()
	return json.NewEncoder(stdout).Encode(result{
		SchemaVersion: 1,
		FinishedAt:    time.Now().UTC().Format(time.RFC3339),
		Rows:          loaded.Load(),
		RemoteRows:    remoteRows,
		Files:         fileCounts,
		Hashes:        hashes,
		Workers:       *workers,
		ElapsedSecs:   time.Since(started).Seconds(),
		RequestCharge: observed, Throttled: throttled, ChargeLimit: *maxCharge,
	})
}

type chargeBudget struct {
	mu        sync.Mutex
	limit     float64
	total     float64
	throttled int64
}

func (budget *chargeBudget) charge(response *http.Response, required bool) error {
	budget.mu.Lock()
	defer budget.mu.Unlock()
	if response.StatusCode == http.StatusTooManyRequests {
		budget.throttled++
	}
	raw := response.Header.Get("x-ms-request-charge")
	if raw == "" {
		if required && response.StatusCode >= 200 && response.StatusCode < 300 {
			return errors.New("Cosmos data response omitted request charge")
		}
		return nil
	}
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
		return errors.New("Cosmos response contains an invalid request charge")
	}
	budget.total += value
	if budget.total > budget.limit {
		return fmt.Errorf("Cosmos observed request charge %.3f exceeds limit %.3f", budget.total, budget.limit)
	}
	return nil
}

func (budget *chargeBudget) Do(request *policy.Request) (*http.Response, error) {
	budget.mu.Lock()
	exhausted := budget.total > budget.limit
	budget.mu.Unlock()
	if exhausted {
		return nil, errors.New("Cosmos observed request-charge budget exhausted")
	}
	response, err := request.Next()
	if response != nil {
		if chargeErr := budget.charge(response, strings.HasPrefix(request.Raw().URL.Path, "/dbs/")); chargeErr != nil {
			if response.Body != nil {
				response.Body.Close()
			}
			return nil, errors.Join(err, chargeErr)
		}
	}
	return response, err
}

type itemPager interface {
	More() bool
	NextPage(context.Context) (azcosmos.QueryItemsResponse, error)
}

// Empty pages can carry continuations. Do not begin writes before EOF, and
// never treat denied or incomplete reads as proof that a container is empty.
// This is a preflight, not a distributed writer lock; the operator must retain
// exclusive fixture-writer access throughout preparation.
func requireEmptyPages(ctx context.Context, pager itemPager) error {
	for pager.More() {
		if err := ctx.Err(); err != nil {
			return err
		}
		page, err := pager.NextPage(ctx)
		if err != nil {
			return fmt.Errorf("empty-container preflight: %w", err)
		}
		if len(page.Items) != 0 {
			return errors.New("empty-container preflight: existing documents; refusing all writes")
		}
	}
	return ctx.Err()
}

func upsertWithRetry(ctx context.Context, container *azcosmos.ContainerClient, item workItem) error {
	const maxAttempts = 20
	for attempt := 0; attempt < maxAttempts; attempt++ {
		_, err := container.UpsertItem(
			ctx,
			azcosmos.NewPartitionKeyString(item.partitionKey),
			item.body,
			&azcosmos.ItemOptions{EnableContentResponseOnWrite: false},
		)
		if err == nil {
			return nil
		}
		if !isRetryable(err) || attempt == maxAttempts-1 {
			return err
		}
		timer := time.NewTimer(retryDelay(attempt, item.partitionKey))
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return errors.New("Cosmos upsert retry budget exhausted")
}

func isRetryable(err error) bool {
	var responseError *azcore.ResponseError
	if !errors.As(err, &responseError) {
		return false
	}
	switch responseError.StatusCode {
	case http.StatusRequestTimeout, http.StatusTooManyRequests,
		http.StatusInternalServerError, http.StatusBadGateway,
		http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}

func retryDelay(attempt int, key string) time.Duration {
	exponent := attempt
	if exponent > 5 {
		exponent = 5
	}
	base := time.Second * time.Duration(1<<exponent)
	hasher := fnv.New32a()
	_, _ = hasher.Write([]byte(key))
	jitter := time.Duration(hasher.Sum32()%1000) * time.Millisecond
	return base + jitter
}

func streamFiles(ctx context.Context, files []string, items chan<- workItem, counts map[string]int, hashes map[string]string) error {
	for _, path := range files {
		file, err := os.Open(path)
		if err != nil {
			return fmt.Errorf("open %s: %w", filepath.Base(path), err)
		}
		hash := sha256.New()
		reader := bufio.NewReaderSize(io.TeeReader(file, hash), 1<<20)
		for {
			line, readErr := reader.ReadBytes('\n')
			line = bytes.TrimSpace(line)
			if len(line) > 0 {
				var decoded document
				if err := json.Unmarshal(line, &decoded); err != nil {
					file.Close()
					return fmt.Errorf("decode %s: %w", filepath.Base(path), err)
				}
				if decoded.PartitionKey == "" {
					file.Close()
					return fmt.Errorf("%s contains an empty partitionKey", filepath.Base(path))
				}
				body := bytes.Clone(line)
				select {
				case items <- workItem{body: body, partitionKey: decoded.PartitionKey}:
					counts[filepath.Base(path)]++
				case <-ctx.Done():
					file.Close()
					return ctx.Err()
				}
			}
			if readErr != nil {
				file.Close()
				if errors.Is(readErr, io.EOF) {
					hashes[filepath.Base(path)] = hex.EncodeToString(hash.Sum(nil))
					break
				}
				return fmt.Errorf("read %s: %w", filepath.Base(path), readErr)
			}
		}
	}
	return nil
}

func countRemote(ctx context.Context, container *azcosmos.ContainerClient) (int64, error) {
	// The Go SDK uses the Cosmos gateway. The gateway cannot execute a
	// cross-partition aggregate such as COUNT(1), but it can execute a simple
	// cross-partition projection. Drain every continuation page and count the
	// projected rows locally so verification remains exact.
	pager := container.NewQueryItemsPager(
		countProjection,
		azcosmos.NewPartitionKey(),
		&azcosmos.QueryOptions{PageSizeHint: 10000},
	)
	var total int64
	for pager.More() {
		page, err := pager.NextPage(ctx)
		if err != nil {
			return 0, fmt.Errorf("query container count: %w", err)
		}
		for _, item := range page.Items {
			var marker int
			if err := json.Unmarshal(item, &marker); err != nil {
				return 0, fmt.Errorf("decode container count: %w", err)
			}
			if marker != 1 {
				return 0, fmt.Errorf("unexpected container count marker %d", marker)
			}
			total++
		}
	}
	return total, nil
}
