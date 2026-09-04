package app

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/pggraph"
)

const (
	pgGraphBenchmarkVerticesEnvironment = "AGEFREIGHTER_PGGRAPH_BENCHMARK_VERTICES"
	pgGraphBenchmarkEdgesEnvironment    = "AGEFREIGHTER_PGGRAPH_BENCHMARK_EDGES"
)

// BenchmarkPostgreSQLPropertyGraphCreate measures the complete CSV-to-SQL/PGQ
// create path, including metadata checkpoints, final SQL/PGQ validation, and
// the canonical target digest. The checked-in harness selects deliberate
// workload sizes; a plain `go test` never runs this benchmark.
func BenchmarkPostgreSQLPropertyGraphCreate(b *testing.B) {
	dsn := strings.TrimSpace(os.Getenv("AGEFREIGHTER_PGGRAPH_TEST_DSN"))
	if dsn == "" {
		b.Skip("set AGEFREIGHTER_PGGRAPH_TEST_DSN to run the property graph benchmark")
	}
	vertices := benchmarkRowCount(b, pgGraphBenchmarkVerticesEnvironment, 10_000)
	edges := benchmarkRowCount(b, pgGraphBenchmarkEdgesEnvironment, 25_000)
	if vertices < 2 && edges > 0 {
		b.Fatal("property graph edge benchmark requires at least two vertices")
	}
	b.Setenv("AGEFREIGHTER_PGGRAPH_APP_TEST_DSN", dsn)
	directory := b.TempDir()
	vertexPath := filepath.Join(directory, "vertices.csv")
	edgePath := filepath.Join(directory, "edges.csv")
	writePropertyGraphBenchmarkCSV(b, vertexPath, edgePath, vertices, edges)
	total := vertices + edges

	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		b.StopTimer()
		name := fmt.Sprintf("af_pgq_bench_%d_%d", time.Now().UnixNano(), iteration)
		job := testLoadJob(name, vertexPath, edgePath)
		job.Metadata.Name = "pggraph-benchmark"
		job.Target.Type = config.TargetPostgreSQLPropertyGraph
		job.Target.Schema = name
		job.Target.Connection = config.SecretRef{Env: "AGEFREIGHTER_PGGRAPH_APP_TEST_DSN"}
		job.Target.AppendDuplicate = ""
		job.Runtime.MemoryLimit = 4 << 30
		job.Runtime.BatchRows = 10_000
		job.Runtime.BatchBytes = 64 << 20
		job.Runtime.OperationTimeout = config.Duration(12 * time.Hour)
		jobID, err := newJobID()
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		started := time.Now()
		result, runErr := execute(b.Context(), job, jobID, false)
		elapsed := time.Since(started)
		b.StopTimer()
		cleanupPropertyGraphBenchmark(b, dsn, name, jobID)
		if runErr != nil {
			b.Fatalf("benchmark load: %v", runErr)
		}
		if result.Metrics.RecordsCommitted != uint64(total) {
			b.Fatalf("committed %d rows, want %d", result.Metrics.RecordsCommitted, total)
		}
		b.ReportMetric(float64(total)/elapsed.Seconds(), "rows/s")
		b.ReportMetric(float64(vertices), "vertices")
		b.ReportMetric(float64(edges), "edges")
	}
}

func benchmarkRowCount(b *testing.B, environment string, fallback int) int {
	b.Helper()
	value := strings.TrimSpace(os.Getenv(environment))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		b.Fatalf("%s must be a non-negative integer", environment)
	}
	return parsed
}

func writePropertyGraphBenchmarkCSV(
	b *testing.B,
	vertexPath string,
	edgePath string,
	vertices int,
	edges int,
) {
	b.Helper()
	write := func(path string, emit func(*bufio.Writer) error) {
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err != nil {
			b.Fatal(err)
		}
		writer := bufio.NewWriterSize(file, 1<<20)
		if err := emit(writer); err != nil {
			_ = file.Close()
			b.Fatal(err)
		}
		if err := writer.Flush(); err != nil {
			_ = file.Close()
			b.Fatal(err)
		}
		if err := file.Close(); err != nil {
			b.Fatal(err)
		}
	}
	write(vertexPath, func(writer *bufio.Writer) error {
		if _, err := writer.WriteString("id,name\n"); err != nil {
			return err
		}
		for index := 0; index < vertices; index++ {
			if _, err := fmt.Fprintf(writer, "p%d,Person %d\n", index, index); err != nil {
				return err
			}
		}
		return nil
	})
	write(edgePath, func(writer *bufio.Writer) error {
		if _, err := writer.WriteString("id,start,end\n"); err != nil {
			return err
		}
		for index := 0; index < edges; index++ {
			if _, err := fmt.Fprintf(writer, "e%d,p%d,p%d\n", index,
				index%vertices, (index+1)%vertices); err != nil {
				return err
			}
		}
		return nil
	})
}

func cleanupPropertyGraphBenchmark(
	b *testing.B,
	dsn string,
	schema string,
	jobID string,
) {
	b.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		b.Errorf("open benchmark cleanup connection: %v", err)
		return
	}
	defer pool.Close()
	if _, err := pool.Exec(ctx, "DROP SCHEMA IF EXISTS "+
		pggraph.QuoteIdentifier(schema)+" CASCADE"); err != nil {
		b.Errorf("drop benchmark schema: %v", err)
	}
	if _, err := pool.Exec(ctx,
		`DELETE FROM agefreighter_meta.load_job WHERE job_id = $1::uuid`, jobID); err != nil {
		b.Errorf("delete benchmark metadata: %v", err)
	}
}
