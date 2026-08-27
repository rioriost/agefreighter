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

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"go.yaml.in/yaml/v3"
)

func TestLegacyCountriesFixtureThroughCypherIntegration(t *testing.T) {
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run app integration tests")
	}
	fixture, err := filepath.Abs("../../testdata/legacy-baseline/countries")
	if err != nil {
		t.Fatalf("resolve fixture path: %v", err)
	}
	graph := fmt.Sprintf("countries_e2e_%d", time.Now().UnixNano())
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)
	job := countriesFixtureJob(graph, fixture)
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	jobPath := filepath.Join(t.TempDir(), "countries.yaml")
	if err := os.WriteFile(jobPath, data, 0o600); err != nil {
		t.Fatalf("write job: %v", err)
	}

	result, err := Load(t.Context(), jobPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	registerCleanup(t, dsn, graph, result.JobID)
	if result.Status != meta.JobCommitted || result.Metrics.RecordsCommitted != 20_200 {
		t.Fatalf("Load() = %#v", result)
	}

	pool, err := pgxpool.New(t.Context(), dsn)
	if err != nil {
		t.Fatalf("open Cypher pool: %v", err)
	}
	defer pool.Close()
	connection, err := pool.Acquire(t.Context())
	if err != nil {
		t.Fatalf("acquire Cypher connection: %v", err)
	}
	defer connection.Release()
	if _, err := connection.Exec(t.Context(), "LOAD 'age'"); err != nil {
		t.Fatalf("load AGE extension: %v", err)
	}
	if _, err := connection.Exec(
		t.Context(),
		`SET search_path = ag_catalog, "$user", public`,
	); err != nil {
		t.Fatalf("set AGE search path: %v", err)
	}
	assertCypherCount(t, connection.Conn(), graph, "MATCH (n:Country) RETURN count(n)", 200)
	assertCypherCount(t, connection.Conn(), graph, "MATCH (n:City) RETURN count(n)", 10_000)
	assertCypherCount(
		t,
		connection.Conn(),
		graph,
		"MATCH (:Country)-[r:has]->(:City) RETURN count(r)",
		10_000,
	)
	assertCypherCount(
		t,
		connection.Conn(),
		graph,
		`MATCH (n:Country)
		 WHERE n.name = 'El Salvador' AND n.capital = 'Kristybury'
		 RETURN count(n)`,
		1,
	)
	if _, err := Verify(t.Context(), jobPath, result.JobID); err != nil {
		t.Fatalf("Verify() error = %v", err)
	}
}

func BenchmarkLegacyCountriesLoad(b *testing.B) {
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if dsn == "" {
		b.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run app benchmarks")
	}
	fixture, err := filepath.Abs("../../testdata/legacy-baseline/countries")
	if err != nil {
		b.Fatalf("resolve fixture path: %v", err)
	}
	b.SetBytes(20_200)
	for index := 0; index < b.N; index++ {
		b.StopTimer()
		graph := fmt.Sprintf("countries_bench_%d_%d", os.Getpid(), index)
		b.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)
		job := countriesFixtureJob(graph, fixture)
		data, marshalErr := yaml.Marshal(job)
		if marshalErr != nil {
			b.Fatalf("Marshal() error = %v", marshalErr)
		}
		jobPath := filepath.Join(b.TempDir(), "countries.yaml")
		if writeErr := os.WriteFile(jobPath, data, 0o600); writeErr != nil {
			b.Fatalf("write job: %v", writeErr)
		}
		b.StartTimer()
		result, loadErr := Load(b.Context(), jobPath)
		b.StopTimer()
		if loadErr != nil {
			b.Fatalf("Load() error = %v", loadErr)
		}
		if result.Status != meta.JobCommitted || result.Metrics.RecordsCommitted != 20_200 {
			b.Fatalf("Load() = %#v", result)
		}
		cleanupFixtureLoad(b, dsn, graph, result.JobID)
	}
	b.ReportMetric(float64(20_200*b.N)/b.Elapsed().Seconds(), "rows/s")
}

func BenchmarkGeneratedCSVLoad(b *testing.B) {
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if dsn == "" {
		b.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run app benchmarks")
	}
	rows := 200_000
	if value := os.Getenv("AGEFREIGHTER_BENCH_ROWS"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed <= 0 {
			b.Fatalf("AGEFREIGHTER_BENCH_ROWS = %q", value)
		}
		rows = parsed
	}
	path := filepath.Join(b.TempDir(), "vertices.csv")
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		b.Fatalf("create generated CSV: %v", err)
	}
	writer := bufio.NewWriterSize(file, 1<<20)
	if _, err := writer.WriteString("id,name\n"); err != nil {
		b.Fatalf("write generated CSV header: %v", err)
	}
	for index := 0; index < rows; index++ {
		if _, err := fmt.Fprintf(writer, "p%d,Person %d\n", index, index); err != nil {
			b.Fatalf("write generated CSV row: %v", err)
		}
	}
	if err := writer.Flush(); err != nil {
		b.Fatalf("flush generated CSV: %v", err)
	}
	if err := file.Close(); err != nil {
		b.Fatalf("close generated CSV: %v", err)
	}
	b.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		b.StopTimer()
		graph := fmt.Sprintf("generated_bench_%d_%d", os.Getpid(), index)
		job := generatedCSVJob(graph, path)
		data, marshalErr := yaml.Marshal(job)
		if marshalErr != nil {
			b.Fatalf("Marshal() error = %v", marshalErr)
		}
		jobPath := filepath.Join(b.TempDir(), "generated.yaml")
		if writeErr := os.WriteFile(jobPath, data, 0o600); writeErr != nil {
			b.Fatalf("write job: %v", writeErr)
		}
		b.StartTimer()
		result, loadErr := Load(b.Context(), jobPath)
		b.StopTimer()
		if loadErr != nil {
			b.Fatalf("Load() error = %v", loadErr)
		}
		if result.Status != meta.JobCommitted ||
			result.Metrics.RecordsCommitted != uint64(rows) {
			b.Fatalf("Load() = %#v", result)
		}
		cleanupFixtureLoad(b, dsn, graph, result.JobID)
	}
	b.ReportMetric(float64(rows*b.N)/b.Elapsed().Seconds(), "rows/s")
}

func generatedCSVJob(graph, path string) config.LoadJob {
	header := true
	nullValue := ""
	return config.LoadJob{
		APIVersion: config.APIVersion,
		Kind:       config.KindLoadJob,
		Metadata:   config.Metadata{Name: "generated-csv-benchmark"},
		Source: config.Source{
			Type: config.SourceCSV, Namespace: "generated",
			CSV: &config.CSVSource{
				Defaults: config.DelimitedOptions{
					Delimiter: ",", Quote: `"`, Escape: `"`,
					Header: &header, Encoding: "utf-8", NullValue: &nullValue,
				},
				Vertices: []config.CSVVertex{{
					Label: "Person", Path: path, IDColumn: "id",
					Properties: map[string]string{"name": "name"},
				}},
			},
		},
		Target: config.Target{
			Type: config.TargetApacheAGE, Graph: graph, Mode: config.LoadCreate,
			Connection:   config.SecretRef{Env: "AGEFREIGHTER_APP_TEST_DSN"},
			PropertyMode: config.PropertiesReplace,
		},
		Runtime: config.Runtime{
			MemoryLimit: 64 << 20, BatchRows: 10_000, BatchBytes: 16 << 20,
			MaxSourceConcurrency: 1, MaxTransformConcurrency: 1,
			MaxTargetConnections: 3, OperationTimeout: config.Duration(30 * time.Second),
		},
		Errors: config.ErrorPolicies{
			MalformedRecord: config.MalformedFail,
			MissingEndpoint: config.MissingEndpointError,
		},
	}
}

func countriesFixtureJob(graph, fixture string) config.LoadJob {
	header := true
	nullValue := ""
	defaults := config.DelimitedOptions{
		Delimiter: ",", Quote: `"`, Escape: `"`,
		Header: &header, Encoding: "utf-8", NullValue: &nullValue,
	}
	return config.LoadJob{
		APIVersion: config.APIVersion,
		Kind:       config.KindLoadJob,
		Metadata:   config.Metadata{Name: "countries-e2e"},
		Source: config.Source{
			Type: config.SourceCSV, Namespace: "legacy",
			CSV: &config.CSVSource{
				Defaults: defaults,
				Vertices: []config.CSVVertex{
					{
						Label: "Country", Path: filepath.Join(fixture, "country.csv"),
						IDColumn: "id",
						Properties: map[string]string{
							"name": "Name", "capital": "Capital",
							"population": "Population", "iso": "ISO",
						},
					},
					{
						Label: "City", Path: filepath.Join(fixture, "city.csv"),
						IDColumn: "id",
						Properties: map[string]string{
							"name": "Name", "latitude": "Latitude", "longitude": "Longitude",
						},
					},
				},
				Edges: []config.CSVEdge{{
					Label: "has", Path: filepath.Join(fixture, "has_country_city.csv"),
					ExternalIDColumn: "id",
					Start:            config.EndpointMapping{Label: "Country", Field: "start_id"},
					End:              config.EndpointMapping{Label: "City", Field: "end_id"},
					Properties:       map[string]string{"since": "since"},
				}},
			},
		},
		Target: config.Target{
			Type: config.TargetApacheAGE, Graph: graph, Mode: config.LoadCreate,
			Connection:   config.SecretRef{Env: "AGEFREIGHTER_APP_TEST_DSN"},
			PropertyMode: config.PropertiesReplace,
		},
		Runtime: config.Runtime{
			MemoryLimit: 64 << 20, BatchRows: 25_000, BatchBytes: 16 << 20,
			MaxSourceConcurrency: 1, MaxTransformConcurrency: 1,
			MaxTargetConnections: 3, OperationTimeout: config.Duration(30 * time.Second),
		},
		Errors: config.ErrorPolicies{
			MalformedRecord: config.MalformedFail,
			MissingEndpoint: config.MissingEndpointError,
		},
	}
}

func assertCypherCount(
	t *testing.T,
	connection *pgx.Conn,
	graph string,
	query string,
	want int64,
) {
	t.Helper()
	statement := fmt.Sprintf(
		`SELECT value::text
		 FROM ag_catalog.cypher('%s', $$%s$$)
		 AS result(value ag_catalog.agtype)`,
		graph,
		query,
	)
	var value string
	if err := connection.QueryRow(t.Context(), statement).Scan(&value); err != nil {
		t.Fatalf("Cypher %q error = %v", query, err)
	}
	got, err := strconv.ParseInt(strings.Trim(value, `"`), 10, 64)
	if err != nil {
		t.Fatalf("parse Cypher count %q: %v", value, err)
	}
	if got != want {
		t.Fatalf("Cypher %q count = %d, want %d", query, got, want)
	}
}

func cleanupFixtureLoad(b *testing.B, dsn, graph, jobID string) {
	b.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	adapter, err := age.Open(ctx, dsn, age.PoolOptions{
		MinConnections: 1, MaxConnections: 2,
		ConnectTimeout: time.Second, OperationTimeout: 5 * time.Second,
	})
	if err != nil {
		b.Fatalf("open cleanup adapter: %v", err)
	}
	if err := adapter.InTransaction(ctx, func(tx *age.Transaction) error {
		return tx.DropGraph(ctx, graph, true)
	}); err != nil {
		adapter.Close()
		b.Fatalf("drop benchmark graph: %v", err)
	}
	adapter.Close()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		b.Fatalf("open cleanup pool: %v", err)
	}
	defer pool.Close()
	if err := deleteAppTestJob(ctx, pool, jobID); err != nil {
		b.Fatalf("delete benchmark job: %v", err)
	}
}
