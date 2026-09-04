package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/age"
)

func TestBulkBenchmarkOptions(t *testing.T) {
	valid := BulkBenchmarkOptions{
		DSN:              "postgres://localhost/database",
		Workload:         BenchmarkEdges,
		Strategy:         BenchmarkDirect,
		Rows:             10,
		EndpointVertices: 2,
		PropertyBytes:    1,
		OperationTimeout: time.Second,
	}
	tests := []struct {
		name string
		edit func(*BulkBenchmarkOptions)
	}{
		{name: "DSN", edit: func(options *BulkBenchmarkOptions) { options.DSN = "" }},
		{name: "rows", edit: func(options *BulkBenchmarkOptions) { options.Rows = 0 }},
		{name: "properties", edit: func(options *BulkBenchmarkOptions) { options.PropertyBytes = -1 }},
		{name: "timeout", edit: func(options *BulkBenchmarkOptions) { options.OperationTimeout = 0 }},
		{name: "workload", edit: func(options *BulkBenchmarkOptions) { options.Workload = "other" }},
		{name: "endpoints", edit: func(options *BulkBenchmarkOptions) { options.EndpointVertices = 1 }},
		{name: "strategy", edit: func(options *BulkBenchmarkOptions) { options.Strategy = "other" }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			options := valid
			test.edit(&options)
			if err := validateBulkBenchmarkOptions(options); err == nil {
				t.Fatal("validateBulkBenchmarkOptions() succeeded")
			}
		})
	}
	valid.Workload = BenchmarkVertices
	valid.Strategy = BenchmarkStaged
	if err := validateBulkBenchmarkOptions(valid); err != nil {
		t.Fatalf("valid vertex options error = %v", err)
	}
}

func TestBenchmarkDataHelpers(t *testing.T) {
	if got := string(benchmarkProperties(3)); got != `{"payload":"xxx"}` {
		t.Fatalf("benchmarkProperties() = %q", got)
	}
	first := benchmarkObjectName("prefix")
	second := benchmarkObjectName("prefix")
	if first == second {
		t.Fatal("benchmarkObjectName() returned a duplicate")
	}
}

func TestRelationalBenchmarkDDLMatchesAGEIDIntegrity(t *testing.T) {
	tests := []struct {
		workload        BenchmarkWorkload
		primaryKeyCount int
	}{
		{workload: BenchmarkVertices, primaryKeyCount: 1},
		{workload: BenchmarkEdges, primaryKeyCount: 2},
	}
	for _, test := range tests {
		t.Run(string(test.workload), func(t *testing.T) {
			ddl := relationalBenchmarkDDL(test.workload, "target", "vertices")
			if got := strings.Count(ddl, "id bigint PRIMARY KEY"); got != test.primaryKeyCount {
				t.Fatalf("primary-key count = %d; want %d in %q", got, test.primaryKeyCount, ddl)
			}
		})
	}
}

func TestBulkBenchmarkConnectionFailures(t *testing.T) {
	if _, err := RunBulkBenchmark(
		context.Background(),
		BulkBenchmarkOptions{},
	); err == nil {
		t.Fatal("RunBulkBenchmark() accepted invalid options")
	}
	options := BulkBenchmarkOptions{
		DSN:              "://bad",
		Workload:         BenchmarkVertices,
		Strategy:         BenchmarkDirect,
		Rows:             1,
		EndpointVertices: 2,
		PropertyBytes:    1,
		OperationTimeout: time.Second,
	}
	if _, err := RunBulkBenchmark(context.Background(), options); err == nil {
		t.Fatal("RunBulkBenchmark() accepted malformed DSN")
	}
	options.DSN = "postgres://localhost/database?pool_max_conns=invalid"
	if _, err := RunBulkBenchmark(context.Background(), options); err == nil {
		t.Fatal("RunBulkBenchmark() accepted invalid pool configuration")
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	options.DSN = "postgres://127.0.0.1:1/database?sslmode=disable"
	if _, err := RunBulkBenchmark(ctx, options); err == nil {
		t.Fatal("RunBulkBenchmark() connected to unreachable server")
	}

	pool, err := pgxpool.New(
		context.Background(),
		"postgres://127.0.0.1:1/database?sslmode=disable",
	)
	if err != nil {
		t.Fatalf("pgxpool.New() error = %v", err)
	}
	pool.Close()
	if _, err := currentWALLSN(context.Background(), pool); err == nil {
		t.Fatal("currentWALLSN() succeeded on closed pool")
	}
	if _, err := walBytesSince(context.Background(), pool, "0/0"); err == nil {
		t.Fatal("walBytesSince() succeeded on closed pool")
	}
	if _, _, err := benchmarkRelational(
		context.Background(),
		pool,
		options,
	); err == nil {
		t.Fatal("benchmarkRelational() succeeded on closed pool")
	}

	options.DSN = "://bad"
	if _, _, err := benchmarkAGE(
		context.Background(),
		nil,
		options,
	); err == nil {
		t.Fatal("benchmarkAGE() accepted malformed DSN")
	}
}

func TestBulkBenchmarkIntegration(t *testing.T) {
	dsn := os.Getenv(benchmarkDSNEnvironment)
	if dsn == "" {
		t.Skip("set " + benchmarkDSNEnvironment + " to run bulk benchmark integration tests")
	}
	tests := []struct {
		workload BenchmarkWorkload
		strategy BenchmarkStrategy
	}{
		{BenchmarkVertices, BenchmarkDirect},
		{BenchmarkVertices, BenchmarkStaged},
		{BenchmarkVertices, BenchmarkRelational},
		{BenchmarkEdges, BenchmarkDirect},
		{BenchmarkEdges, BenchmarkStaged},
		{BenchmarkEdges, BenchmarkRelational},
	}
	for _, test := range tests {
		t.Run(string(test.workload)+"/"+string(test.strategy), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			result, err := RunBulkBenchmark(ctx, BulkBenchmarkOptions{
				DSN:              dsn,
				Workload:         test.workload,
				Strategy:         test.strategy,
				Rows:             5,
				EndpointVertices: 3,
				PropertyBytes:    8,
				OperationTimeout: 10 * time.Second,
			})
			if err != nil {
				t.Fatalf("RunBulkBenchmark() error = %v", err)
			}
			if result.Rows != 5 || result.Elapsed <= 0 ||
				result.RowsPerSecond <= 0 || result.WALBytes < 0 {
				t.Fatalf("invalid benchmark result: %#v", result)
			}
		})
	}

	var output bytes.Buffer
	command := NewBenchmarkCommand()
	command.SetOut(&output)
	command.SetArgs([]string{
		"--workload", "vertices",
		"--strategy", "direct-text",
		"--rows", "5",
		"--property-bytes", "8",
		"--timeout", "10s",
	})
	if err := command.Execute(); err != nil {
		t.Fatalf("benchmark command error = %v", err)
	}
	var result BulkBenchmarkResult
	if err := json.Unmarshal(output.Bytes(), &result); err != nil {
		t.Fatalf("decode benchmark command result: %v", err)
	}
	if result.Rows != 5 || result.Strategy != BenchmarkDirect {
		t.Fatalf("benchmark command result = %#v", result)
	}

	command = NewBenchmarkCommand()
	command.SetOut(errorWriter{})
	command.SetArgs([]string{
		"--workload", "vertices",
		"--strategy", "direct-text",
		"--rows", "1",
		"--property-bytes", "1",
		"--timeout", "10s",
	})
	if err := command.Execute(); err == nil {
		t.Fatal("benchmark command ignored output failure")
	}

	pool, err := pgxpool.New(context.Background(), dsn)
	if err != nil {
		t.Fatalf("open artifact-check pool: %v", err)
	}
	defer pool.Close()
	var artifacts int
	if err := pool.QueryRow(
		context.Background(),
		`SELECT
			(SELECT count(*) FROM ag_catalog.ag_graph WHERE name LIKE 'af_bench_%')
			+
			(SELECT count(*) FROM pg_catalog.pg_tables
			 WHERE schemaname = 'public' AND tablename LIKE 'af_rel_%')`,
	).Scan(&artifacts); err != nil {
		t.Fatalf("count benchmark artifacts: %v", err)
	}
	if artifacts != 0 {
		t.Fatalf("benchmark left %d database artifacts", artifacts)
	}

	adapter, err := age.Open(context.Background(), dsn, age.PoolOptions{
		MaxConnections:   2,
		ConnectTimeout:   10 * time.Second,
		OperationTimeout: 10 * time.Second,
	})
	if err != nil {
		t.Fatalf("open adapter for idempotent cleanup: %v", err)
	}
	defer adapter.Close()
	if err := dropBenchmarkGraph(
		adapter,
		"af_bench_missing_cleanup",
		10*time.Second,
	); err != nil {
		t.Fatalf("drop missing benchmark graph: %v", err)
	}
}

type errorWriter struct{}

func (errorWriter) Write([]byte) (int, error) {
	return 0, errors.New("injected writer failure")
}
