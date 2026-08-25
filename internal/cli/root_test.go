package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"go.yaml.in/yaml/v3"
)

type failingWriter struct{}

func (failingWriter) Write([]byte) (int, error) {
	return 0, errors.New("write failed")
}

func TestRootHelp(t *testing.T) {
	var stdout, stderr bytes.Buffer
	command := NewAgefreighter(&stdout, &stderr)

	if err := Execute(command, []string{"--help"}); err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if got := stdout.String(); !strings.Contains(got, "Validated, resumable graph migration") {
		t.Fatalf("help output = %q, want command summary", got)
	}
	if stderr.Len() != 0 {
		t.Fatalf("help stderr = %q, want empty", stderr.String())
	}
}

func TestToolsHelp(t *testing.T) {
	var stdout, stderr bytes.Buffer
	command := NewTools(&stdout, &stderr)

	if err := Execute(command, []string{"--help"}); err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if got := stdout.String(); !strings.Contains(got, "Diagnostics, fixtures, and benchmarks") {
		t.Fatalf("help output = %q, want tools summary", got)
	}
}

func TestVersionRejectsArguments(t *testing.T) {
	var stdout, stderr bytes.Buffer
	command := NewAgefreighter(&stdout, &stderr)

	err := Execute(command, []string{"version", "unexpected"})

	if err == nil {
		t.Fatal("Execute() error = nil, want argument error")
	}
}

func TestVersionPropagatesOutputError(t *testing.T) {
	var stderr bytes.Buffer
	command := NewAgefreighter(failingWriter{}, &stderr)

	err := Execute(command, []string{"version"})

	if err == nil || err.Error() != "write failed" {
		t.Fatalf("Execute() error = %v, want write failed", err)
	}
}

func TestWriteJSON(t *testing.T) {
	var output bytes.Buffer
	command := NewAgefreighter(&output, failingWriter{})
	if err := writeJSON(command, map[string]string{"value": "<ok>"}); err != nil {
		t.Fatalf("writeJSON() error = %v", err)
	}

	if output.String() != "{\"value\":\"<ok>\"}\n" {
		t.Fatalf("writeJSON() = %q", output.String())
	}
	command.SetOut(failingWriter{})
	if err := writeJSON(command, struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "write command output") {
		t.Fatalf("writeJSON(failure) error = %v", err)
	}
}

func TestLifecycleCommandsReportConfigurationErrors(t *testing.T) {
	tests := [][]string{
		{"load", "missing.yaml"},
		{"resume", "--job", "missing.yaml", "11111111-2222-4333-8444-555555555555"},
		{"status", "--target", "missing.yaml", "11111111-2222-4333-8444-555555555555"},
		{"verify", "--target", "missing.yaml", "11111111-2222-4333-8444-555555555555"},
	}
	for _, args := range tests {
		command := NewAgefreighter(&bytes.Buffer{}, &bytes.Buffer{})
		if err := Execute(command, args); err == nil {
			t.Fatalf("Execute(%v) error = nil", args)
		}
	}
}

func TestLifecycleCommandsIntegration(t *testing.T) {
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run CLI integration tests")
	}
	t.Setenv("AGEFREIGHTER_CLI_TEST_DSN", dsn)
	dir := t.TempDir()
	vertices := filepath.Join(dir, "vertices.csv")
	edges := filepath.Join(dir, "edges.csv")
	if err := os.WriteFile(vertices, []byte("id,name\np1,Alice\np2,Bob\n"), 0o600); err != nil {
		t.Fatalf("write vertices: %v", err)
	}
	if err := os.WriteFile(edges, []byte("id,start,end\ne1,p1,p2\n"), 0o600); err != nil {
		t.Fatalf("write edges: %v", err)
	}
	graph := fmt.Sprintf("cli_lifecycle_%d", time.Now().UnixNano())
	job := cliTestLoadJob(graph, vertices, edges)
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	jobPath := filepath.Join(dir, "job.yaml")
	if err := os.WriteFile(jobPath, data, 0o600); err != nil {
		t.Fatalf("write job: %v", err)
	}

	var output bytes.Buffer
	command := NewAgefreighter(&output, &bytes.Buffer{})
	if err := Execute(command, []string{"load", jobPath}); err != nil {
		t.Fatalf("load error = %v", err)
	}
	var loaded struct {
		JobID  string         `json:"jobId"`
		Status meta.JobStatus `json:"status"`
	}
	if err := json.Unmarshal(output.Bytes(), &loaded); err != nil {
		t.Fatalf("decode load output: %v", err)
	}
	if loaded.JobID == "" || loaded.Status != meta.JobCommitted {
		t.Fatalf("load output = %#v", loaded)
	}
	registerCLICleanup(t, dsn, graph, loaded.JobID)

	for _, name := range []string{"status", "verify"} {
		output.Reset()
		command = NewAgefreighter(&output, &bytes.Buffer{})
		if err := Execute(command, []string{name, "--target", jobPath, loaded.JobID}); err != nil {
			t.Fatalf("%s error = %v", name, err)
		}
		var stored meta.Job
		if err := json.Unmarshal(output.Bytes(), &stored); err != nil {
			t.Fatalf("decode %s output: %v", name, err)
		}
		if stored.ID != loaded.JobID || stored.Status != meta.JobCommitted {
			t.Fatalf("%s output = %#v", name, stored)
		}
	}
	command = NewAgefreighter(&bytes.Buffer{}, &bytes.Buffer{})
	if err := Execute(command, []string{
		"resume", "--job", jobPath, loaded.JobID,
	}); err == nil {
		t.Fatal("resume accepted a committed job")
	}
}

func cliTestLoadJob(graph, vertices, edges string) config.LoadJob {
	header := true
	nullValue := ""
	return config.LoadJob{
		APIVersion: config.APIVersion, Kind: config.KindLoadJob,
		Metadata: config.Metadata{Name: "cli-test"},
		Source: config.Source{
			Type: config.SourceCSV, Namespace: "crm",
			CSV: &config.CSVSource{
				Defaults: config.DelimitedOptions{
					Delimiter: ",", Quote: `"`, Escape: `"`,
					Header: &header, Encoding: "utf-8", NullValue: &nullValue,
				},
				Vertices: []config.CSVVertex{{
					Label: "Person", Path: vertices, IDColumn: "id",
					Properties: map[string]string{"name": "name"},
				}},
				Edges: []config.CSVEdge{{
					Label: "KNOWS", Path: edges, ExternalIDColumn: "id",
					Start: config.EndpointMapping{Label: "Person", Field: "start"},
					End:   config.EndpointMapping{Label: "Person", Field: "end"},
				}},
			},
		},
		Target: config.Target{
			Type: config.TargetApacheAGE, Graph: graph, Mode: config.LoadCreate,
			Connection:   config.SecretRef{Env: "AGEFREIGHTER_CLI_TEST_DSN"},
			PropertyMode: config.PropertiesReplace,
		},
		Runtime: config.Runtime{
			MemoryLimit: 16 << 20, BatchRows: 2, BatchBytes: 1 << 20,
			MaxSourceConcurrency: 1, MaxTransformConcurrency: 1,
			MaxTargetConnections: 2, OperationTimeout: config.Duration(10 * time.Second),
		},
		Errors: config.ErrorPolicies{
			MalformedRecord: config.MalformedFail,
			MissingEndpoint: config.MissingEndpointError,
		},
	}
}

func registerCLICleanup(t *testing.T, dsn, graph, jobID string) {
	t.Helper()
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if adapter, err := age.Open(ctx, dsn, age.PoolOptions{
			MinConnections: 1, MaxConnections: 2,
			ConnectTimeout: time.Second, OperationTimeout: 5 * time.Second,
		}); err == nil {
			_ = adapter.InTransaction(ctx, func(tx *age.Transaction) error {
				return tx.DropGraph(ctx, graph, true)
			})
			adapter.Close()
		}
		if pool, err := pgxpool.New(ctx, dsn); err == nil {
			tx, beginErr := pool.Begin(ctx)
			if beginErr == nil {
				_, _ = tx.Exec(ctx, `
					UPDATE agefreighter_meta.load_job
					SET graph_generation_id = NULL
					WHERE job_id = $1::uuid`, jobID)
				_, _ = tx.Exec(ctx, `
					DELETE FROM agefreighter_meta.graph_generation
					WHERE job_id = $1::uuid`, jobID)
				_, _ = tx.Exec(ctx, `
					DELETE FROM agefreighter_meta.load_job
					WHERE job_id = $1::uuid`, jobID)
				_ = tx.Commit(ctx)
			}
			pool.Close()
		}
	})
}
func TestValidateCommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	command := NewAgefreighter(&stdout, &stderr)

	err := Execute(command, []string{"validate", configFixture(t, "valid/csv.yaml")})

	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if got := stdout.String(); got != "valid: csv-people (csv -> apache-age, mode=create)\n" {
		t.Fatalf("validate output = %q", got)
	}
}

func TestPlanCommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	command := NewAgefreighter(&stdout, &stderr)

	err := Execute(command, []string{"plan", configFixture(t, "valid/csv.yaml")})

	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	var got config.StaticPlan
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("json.Unmarshal() error = %v; output = %q", err, stdout.String())
	}
	if got.Job != "csv-people" || got.Limits.MemoryLimit != "1GiB" {
		t.Fatalf("plan output = %#v, want csv-people plan", got)
	}
	golden, err := os.ReadFile(configFixture(t, "plan-csv.golden.json"))
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if stdout.String() != string(golden) {
		t.Fatalf("plan output differs from golden:\n%s", stdout.String())
	}
}

func TestConfigurationCommandsRejectInvalidJobWithoutSecretDisclosure(t *testing.T) {
	for _, subcommand := range []string{"validate", "plan"} {
		t.Run(subcommand, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			command := NewAgefreighter(&stdout, &stderr)

			err := Execute(command, []string{subcommand, configFixture(t, "invalid/literal-secret.yaml")})

			if err == nil {
				t.Fatal("Execute() error = nil, want invalid job error")
			}
			if strings.Contains(err.Error(), "supersecret") || strings.Contains(err.Error(), "admin") {
				t.Fatalf("Execute() disclosed credential: %v", err)
			}
		})
	}
}

func TestConfigurationCommandsPropagateOutputErrors(t *testing.T) {
	for _, subcommand := range []string{"validate", "plan"} {
		t.Run(subcommand, func(t *testing.T) {
			var stderr bytes.Buffer
			command := NewAgefreighter(failingWriter{}, &stderr)

			err := Execute(command, []string{subcommand, configFixture(t, "valid/csv.yaml")})

			if err == nil || !strings.Contains(err.Error(), "write") {
				t.Fatalf("Execute() error = %v, want output error", err)
			}
		})
	}
}

func TestToolsDoesNotExposeLoadCommands(t *testing.T) {
	var stdout, stderr bytes.Buffer
	command := NewTools(&stdout, &stderr)

	err := Execute(command, []string{"validate", configFixture(t, "valid/csv.yaml")})

	if err == nil || !strings.Contains(err.Error(), `unknown command "validate"`) {
		t.Fatalf("Execute() error = %v, want unknown validate command", err)
	}
}

func configFixture(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join("..", "config", "testdata", name)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("config fixture %q: %v", path, err)
	}
	return path
}
