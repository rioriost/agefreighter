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
	"github.com/rioriost/agefreighter/internal/app"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/report"
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
		{"report", "--target", "missing.yaml", "11111111-2222-4333-8444-555555555555"},
		{"doctor", "--target", "missing.yaml"},
		{"doctor", "history", "--target", "missing.yaml"},
		{"verify", "--target", "missing.yaml", "11111111-2222-4333-8444-555555555555"},
		{"profile", "missing.yaml"},
		{"optimize", "--target", "missing.yaml"},
		{"cleanup", "--target", "missing.yaml", "11111111-2222-4333-8444-555555555555"},
	}
	for _, args := range tests {
		command := NewAgefreighter(&bytes.Buffer{}, &bytes.Buffer{})
		if err := Execute(command, args); err == nil {
			t.Fatalf("Execute(%v) error = nil", args)
		}
	}
}

func TestProfileCommandValidatesFlagsBeforeReadingJob(t *testing.T) {
	tests := [][]string{
		{"profile", "--mode", "arbitrary", "missing.yaml"},
		{"profile", "--sample-size", "0", "missing.yaml"},
		{
			"profile", "--sample-size",
			fmt.Sprint(app.MaxProfileSampleSize + 1), "missing.yaml",
		},
		{"profile", "--format", "yaml", "missing.yaml"},
	}
	for _, args := range tests {
		command := NewAgefreighter(&bytes.Buffer{}, &bytes.Buffer{})
		if err := Execute(command, args); err == nil {
			t.Fatalf("Execute(%v) succeeded", args)
		}
	}
}

func TestProfileCommandEmitsSourceOnlyReport(t *testing.T) {
	directory := t.TempDir()
	vertices := filepath.Join(directory, "vertices.csv")
	edges := filepath.Join(directory, "edges.csv")
	if err := os.WriteFile(vertices, []byte("id,name\np1,Alice\np2,Bob\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(edges, []byte("id,start,end\ne1,p1,p2\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	job := cliTestLoadJob("profile_target_not_opened", vertices, edges)
	job.Target.Connection = config.SecretRef{Env: "PROFILE_CLI_TARGET_MUST_NOT_BE_READ"}
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "job.yaml")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	command := NewAgefreighter(&output, &bytes.Buffer{})
	if err := Execute(command, []string{
		"profile", "--mode", "exact", "--format", "markdown", path,
	}); err != nil {
		t.Fatalf("profile command error = %v", err)
	}
	if !strings.Contains(output.String(), "# agefreighter profile report") ||
		strings.Contains(output.String(), "Alice") ||
		strings.Contains(output.String(), "p1") {
		t.Fatalf("profile output = %s", output.String())
	}
}

func TestVerifyDeepFlagValidation(t *testing.T) {
	const jobID = "11111111-2222-4333-8444-555555555555"
	tests := [][]string{
		{"verify", "--target", "job.yaml", "--level", "arbitrary", jobID},
		{
			"verify", "--target", "job.yaml", "--integrity",
			"--limit", "1001", jobID,
		},
		{"verify", "--target", "job.yaml", "--format", "markdown", jobID},
	}
	for _, args := range tests {
		command := NewAgefreighter(&bytes.Buffer{}, &bytes.Buffer{})
		if err := Execute(command, args); err == nil {
			t.Fatalf("Execute(%v) succeeded", args)
		}
	}
}

func TestDoctorCommandValidatesFlagsBeforeConnecting(t *testing.T) {
	tests := [][]string{
		{"doctor", "--target", "missing.yaml", "--format", "yaml"},
		{"doctor", "--target", "missing.yaml", "--output", ""},
		{"doctor", "history", "--target", "missing.yaml", "--limit", "0"},
		{
			"doctor", "history", "--target", "missing.yaml",
			"--limit", fmt.Sprint(app.MaxDoctorHistory + 1),
		},
		{
			"doctor", "history", "--target", "missing.yaml",
			"--format", "yaml",
		},
	}

	for _, args := range tests {
		command := NewAgefreighter(&bytes.Buffer{}, &bytes.Buffer{})
		if err := Execute(command, args); err == nil {
			t.Fatalf("Execute(%v) error = nil", args)
		}
	}
}

func TestOptimizeCommandValidatesFlagsBeforeConnecting(t *testing.T) {
	tests := [][]string{
		{"optimize", "--target", "missing.yaml", "--format", "yaml"},
		{"optimize", "--target", "missing.yaml", "--output", ""},
		{"optimize", "--target", "missing.yaml", "--apply-analyze"},
		{"optimize", "--target", "missing.yaml", "--queries", "workload.cypher"},
	}
	for _, args := range tests {
		command := NewAgefreighter(&bytes.Buffer{}, &bytes.Buffer{})
		if err := Execute(command, args); err == nil {
			t.Fatalf("Execute(%v) succeeded", args)
		}
	}
}

func TestDoctorPropagatesCancellation(t *testing.T) {
	t.Setenv(
		"AGEFREIGHTER_TARGET_DSN",
		"postgres://localhost/agefreighter_cancel_test",
	)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	command := NewAgefreighter(&bytes.Buffer{}, &bytes.Buffer{})
	err := ExecuteContext(ctx, command, []string{
		"doctor",
		"--target",
		configFixture(t, "valid/csv.yaml"),
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("doctor cancellation error = %v", err)
	}
}

func TestReportCommandValidatesFlagsBeforeConnecting(t *testing.T) {
	const validJobID = "11111111-2222-4333-8444-555555555555"
	tests := [][]string{
		{"report", "--target", "missing.yaml", "--format", "yaml", validJobID},
		{"report", "--target", "missing.yaml", "--limit-batches", "0", validJobID},
		{"report", "--target", "missing.yaml", "--output", "", validJobID},
		{
			"report", "--target", "missing.yaml", "--limit-batches",
			fmt.Sprint(app.MaxReportBatches + 1), validJobID,
		},
	}
	for _, args := range tests {
		command := NewAgefreighter(&bytes.Buffer{}, &bytes.Buffer{})
		if err := Execute(command, args); err == nil {
			t.Fatalf("Execute(%v) error = nil", args)
		}
	}
	command := NewAgefreighter(&bytes.Buffer{}, &bytes.Buffer{})
	err := Execute(command, []string{
		"report", "--target", configFixture(t, "valid/csv.yaml"), "not-a-uuid",
	})
	if err == nil || !strings.Contains(err.Error(), "canonical UUID") {
		t.Fatalf("invalid report UUID error = %v", err)
	}
}

func TestWriteExclusiveReport(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "report.json")
	data := []byte("{\"safe\":true}\n")
	if err := writeExclusiveReport(path, data); err != nil {
		t.Fatalf("writeExclusiveReport() error = %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat() error = %v", err)
	}
	if !bytes.Equal(got, data) || info.Mode().Perm() != 0o600 {
		t.Fatalf("report = %q mode=%o", got, info.Mode().Perm())
	}
	if err := writeExclusiveReport(path, []byte("overwrite")); err == nil {
		t.Fatal("writeExclusiveReport() overwrote an existing file")
	}
	link := filepath.Join(directory, "report-link")
	if err := os.Symlink(path, link); err != nil {
		t.Fatalf("Symlink() error = %v", err)
	}
	if err := writeExclusiveReport(link, data); err == nil {
		t.Fatal("writeExclusiveReport() followed a symlink")
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

	output.Reset()
	command = NewAgefreighter(&output, &bytes.Buffer{})
	if err := Execute(command, []string{
		"verify", "--target", jobPath, "--counts", "--integrity",
		"--limit", "10", loaded.JobID,
	}); err != nil {
		t.Fatalf("deep verify error = %v", err)
	}
	var verificationReport report.Document
	if err := json.Unmarshal(output.Bytes(), &verificationReport); err != nil {
		t.Fatalf("decode deep verify output: %v", err)
	}
	if verificationReport.Command != "verify" ||
		verificationReport.Outcome != report.OutcomePass {
		t.Fatalf("deep verify output = %#v", verificationReport)
	}

	output.Reset()
	command = NewAgefreighter(&output, &bytes.Buffer{})
	if err := Execute(command, []string{
		"report", "--target", jobPath, "--limit-batches", "2",
		"--include-counts", loaded.JobID,
	}); err != nil {
		t.Fatalf("report error = %v", err)
	}
	var migrationReport struct {
		SchemaVersion int              `json:"schemaVersion"`
		Job           *report.Job      `json:"job"`
		Sections      []report.Section `json:"sections"`
	}
	if err := json.Unmarshal(output.Bytes(), &migrationReport); err != nil {
		t.Fatalf("decode report output: %v", err)
	}
	if migrationReport.SchemaVersion != report.SchemaVersion ||
		migrationReport.Job == nil ||
		migrationReport.Job.ID != loaded.JobID ||
		len(migrationReport.Sections) == 0 {
		t.Fatalf("report output = %#v", migrationReport)
	}

	for _, arguments := range [][]string{
		{"optimize", "--target", jobPath},
		{"optimize", "--target", jobPath, "--apply-analyze"},
	} {
		output.Reset()
		command = NewAgefreighter(&output, &bytes.Buffer{})
		if err := Execute(command, arguments); err != nil {
			t.Fatalf("%v error = %v", arguments, err)
		}
		var optimizerReport report.Document
		if err := json.Unmarshal(output.Bytes(), &optimizerReport); err != nil {
			t.Fatalf("decode optimizer output: %v", err)
		}
		if optimizerReport.Command != "optimize" ||
			optimizerReport.Target == nil {
			t.Fatalf("optimizer output = %#v", optimizerReport)
		}
	}

	output.Reset()
	command = NewAgefreighter(&output, &bytes.Buffer{})
	if err := Execute(command, []string{
		"doctor", "--target", jobPath, "--persist",
	}); err != nil {
		t.Fatalf("doctor error = %v", err)
	}
	var doctorReport report.Document
	if err := json.Unmarshal(output.Bytes(), &doctorReport); err != nil {
		t.Fatalf("decode doctor output: %v", err)
	}
	if doctorReport.Command != "doctor" || doctorReport.Target == nil {
		t.Fatalf("doctor output = %#v", doctorReport)
	}

	output.Reset()
	command = NewAgefreighter(&output, &bytes.Buffer{})
	if err := Execute(command, []string{
		"doctor", "history", "--target", jobPath, "--limit", "1",
	}); err != nil {
		t.Fatalf("doctor history error = %v", err)
	}
	var historyReport report.Document
	if err := json.Unmarshal(output.Bytes(), &historyReport); err != nil {
		t.Fatalf("decode doctor history output: %v", err)
	}
	if historyReport.Command != "doctor" || len(historyReport.Sections) == 0 {
		t.Fatalf("doctor history output = %#v", historyReport)
	}
	command = NewAgefreighter(&bytes.Buffer{}, &bytes.Buffer{})
	if err := Execute(command, []string{
		"resume", "--job", jobPath, loaded.JobID,
	}); err == nil {
		t.Fatal("resume accepted a committed job")
	}

	replaceJob := job
	replaceJob.Target.Mode = config.LoadReplace
	replaceData, err := yaml.Marshal(replaceJob)
	if err != nil {
		t.Fatalf("marshal replace job: %v", err)
	}
	replacePath := filepath.Join(dir, "replace.yaml")
	if err := os.WriteFile(replacePath, replaceData, 0o600); err != nil {
		t.Fatalf("write replace job: %v", err)
	}
	output.Reset()
	command = NewAgefreighter(&output, &bytes.Buffer{})
	if err := Execute(command, []string{"load", replacePath}); err != nil {
		t.Fatalf("replace load error = %v", err)
	}
	var replaced struct {
		JobID  string         `json:"jobId"`
		Status meta.JobStatus `json:"status"`
	}
	if err := json.Unmarshal(output.Bytes(), &replaced); err != nil {
		t.Fatalf("decode replace output: %v", err)
	}
	backup, err := age.DeriveGraphName(graph, age.BackupName, replaced.JobID)
	if err != nil {
		t.Fatalf("derive CLI backup: %v", err)
	}
	shadow, err := age.DeriveGraphName(graph, age.ShadowName, replaced.JobID)
	if err != nil {
		t.Fatalf("derive CLI shadow: %v", err)
	}
	registerCLIReplaceCleanup(
		t,
		dsn,
		replaced.JobID,
		graph,
		backup,
		shadow,
	)
	output.Reset()
	command = NewAgefreighter(&output, &bytes.Buffer{})
	if err := Execute(command, []string{
		"cleanup", "--target", replacePath, replaced.JobID,
	}); err != nil {
		t.Fatalf("cleanup command error = %v", err)
	}
	var cleaned meta.Job
	if err := json.Unmarshal(output.Bytes(), &cleaned); err != nil {
		t.Fatalf("decode cleanup output: %v", err)
	}
	if cleaned.ID != replaced.JobID || cleaned.BackupCleanedAt == nil {
		t.Fatalf("cleanup output = %#v", cleaned)
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
					DELETE FROM agefreighter_meta.diagnostic_history
					WHERE target_graph = $1`, graph)
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

func registerCLIReplaceCleanup(
	t *testing.T,
	dsn string,
	jobID string,
	graphs ...string,
) {
	t.Helper()
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if adapter, err := age.Open(ctx, dsn, age.PoolOptions{
			MinConnections: 1, MaxConnections: 2,
			ConnectTimeout: time.Second, OperationTimeout: 5 * time.Second,
		}); err == nil {
			for _, graph := range graphs {
				_ = adapter.InTransaction(ctx, func(tx *age.Transaction) error {
					if _, lookupErr := tx.LookupGraph(ctx, graph); errors.Is(
						lookupErr,
						age.ErrCatalogEntryNotFound,
					) {
						return nil
					} else if lookupErr != nil {
						return lookupErr
					}
					return tx.DropGraph(ctx, graph, true)
				})
			}
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
