// Command cosmosfixtureload imports the deterministic P1 JSONL fixture into a
// freshly provisioned Cosmos DB for NoSQL container using the VM's managed
// identity. It is intentionally a fixture-preparation tool, not a production
// migration path.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"hash/fnv"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
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

type result struct {
	SchemaVersion int            `json:"schemaVersion"`
	FinishedAt    string         `json:"finishedAt"`
	Rows          int64          `json:"rows"`
	RemoteRows    int64          `json:"remoteRows"`
	Files         map[string]int `json:"files"`
	Workers       int            `json:"workers"`
	ElapsedSecs   float64        `json:"elapsedSeconds"`
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
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *endpoint == "" || *input == "" || *database == "" || *containerName == "" {
		return errors.New("endpoint, input, database and container are required")
	}
	if *workers < 1 || *workers > 256 {
		return errors.New("workers must be between 1 and 256")
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
	client, err := azcosmos.NewClient(*endpoint, credential, nil)
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
	readErr := streamFiles(loadCtx, files, items, fileCounts)
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
	return json.NewEncoder(stdout).Encode(result{
		SchemaVersion: 1,
		FinishedAt:    time.Now().UTC().Format(time.RFC3339),
		Rows:          loaded.Load(),
		RemoteRows:    remoteRows,
		Files:         fileCounts,
		Workers:       *workers,
		ElapsedSecs:   time.Since(started).Seconds(),
	})
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

func streamFiles(ctx context.Context, files []string, items chan<- workItem, counts map[string]int) error {
	for _, path := range files {
		file, err := os.Open(path)
		if err != nil {
			return fmt.Errorf("open %s: %w", filepath.Base(path), err)
		}
		reader := bufio.NewReaderSize(file, 1<<20)
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
		"SELECT VALUE 1 FROM c",
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
