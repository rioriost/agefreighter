package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/internal/cli"
	"github.com/rioriost/agefreighter/internal/tools"
)

func TestRunVersion(t *testing.T) {
	var stdout, stderr bytes.Buffer

	exitCode := run([]string{"version"}, &stdout, &stderr)

	if exitCode != 0 {
		t.Fatalf("run() exit code = %d, want 0; stderr = %q", exitCode, stderr.String())
	}
	if got := stdout.String(); !strings.HasPrefix(got, "agefreighter-tools dev") {
		t.Fatalf("run() stdout = %q, want tools development version", got)
	}
	if stderr.Len() != 0 {
		t.Fatalf("run() stderr = %q, want empty", stderr.String())
	}
}

func TestRunRejectsUnknownCommand(t *testing.T) {
	var stdout, stderr bytes.Buffer

	exitCode := run([]string{"unknown"}, &stdout, &stderr)

	if exitCode != 1 {
		t.Fatalf("run() exit code = %d, want 1", exitCode)
	}
	if got := stderr.String(); !strings.Contains(got, `unknown command "unknown"`) {
		t.Fatalf("run() stderr = %q, want unknown command error", got)
	}
}

func TestRunGenerateFixture(t *testing.T) {
	var stdout, stderr bytes.Buffer
	output := filepath.Join(t.TempDir(), "fixture")

	exitCode := run(
		[]string{"generate", "fixture", "--output", output},
		&stdout,
		&stderr,
	)

	if exitCode != 0 {
		t.Fatalf("run() exit code = %d; stderr = %q", exitCode, stderr.String())
	}
	if !strings.Contains(stdout.String(), "generated 4 vertices and 6 edges") {
		t.Fatalf("run() stdout = %q", stdout.String())
	}
}

func TestRunBenchmarkRequiresDSN(t *testing.T) {
	t.Setenv("AGEFREIGHTER_AGE_TEST_DSN", "")
	var stdout, stderr bytes.Buffer

	exitCode := run([]string{"benchmark-age-copy"}, &stdout, &stderr)

	if exitCode != 1 {
		t.Fatalf("run() exit code = %d", exitCode)
	}
	if !strings.Contains(stderr.String(), "AGEFREIGHTER_AGE_TEST_DSN is required") {
		t.Fatalf("run() stderr = %q", stderr.String())
	}
}

func TestRunInspect(t *testing.T) {
	var stdout, stderr bytes.Buffer
	path := filepath.Join("..", "..", "internal", "config", "testdata", "valid", "csv.yaml")

	exitCode := run([]string{"inspect", path}, &stdout, &stderr)

	if exitCode != 0 {
		t.Fatalf("run() exit code = %d; stderr = %q", exitCode, stderr.String())
	}
	var report tools.Inspection
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode inspection: %v; output = %q", err, stdout.String())
	}
	if report.FormatVersion != tools.InspectionFormatVersion ||
		report.Source.Type != "csv" ||
		len(report.Source.VertexMappings) == 0 {
		t.Fatalf("inspection = %#v", report)
	}
}

func TestRunBenchmarkReportFromStandardInput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	command := `{"workload":"vertices","strategy":"direct-text","rows":10,` +
		`"propertyBytes":4,"elapsedNanos":1000000000,` +
		`"rowsPerSecond":10,"walBytes":12}`

	root := cli.NewTools(&stdout, &stderr)
	root.SetIn(strings.NewReader(command))
	root.AddCommand(tools.NewBenchmarkReportCommand())
	if err := cli.Execute(root, []string{"benchmark-report", "--format", "markdown"}); err != nil {
		t.Fatalf("benchmark-report: %v", err)
	}
	if !strings.Contains(stdout.String(), "# Benchmark Report") ||
		!strings.Contains(stdout.String(), "| vertices | 10 | 4 | direct-text | 1 |") {
		t.Fatalf("benchmark report = %q", stdout.String())
	}
}

func TestRunContextCancelsBenchmark(t *testing.T) {
	t.Setenv(
		"AGEFREIGHTER_AGE_TEST_DSN",
		"postgres://127.0.0.1:1/database?sslmode=disable",
	)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var stdout, stderr bytes.Buffer

	exitCode := runContext(
		ctx,
		[]string{
			"benchmark-age-copy",
			"--workload", "vertices",
			"--strategy", "direct-text",
			"--rows", "1",
		},
		&stdout,
		&stderr,
	)

	if exitCode != 1 {
		t.Fatalf("runContext() exit code = %d, want 1", exitCode)
	}
	if got := stderr.String(); !strings.Contains(got, context.Canceled.Error()) {
		t.Fatalf("runContext() stderr = %q, want cancellation error", got)
	}
}

func TestSignalContextStops(t *testing.T) {
	ctx, stop := signalContext(context.Background())
	stop()

	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("signal context remained active after stop")
	}
}

func TestSignalContextCancelsOnSignal(t *testing.T) {
	ctx, stop := signalContext(context.Background())
	defer stop()
	process, err := os.FindProcess(os.Getpid())
	if err != nil {
		t.Fatalf("find test process: %v", err)
	}
	if err := process.Signal(syscall.SIGHUP); err != nil {
		t.Fatalf("signal test process: %v", err)
	}

	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("signal context remained active after SIGHUP")
	}
}

func TestSignalContextForcesExitOnSecondSignal(t *testing.T) {
	exitCodes := make(chan int, 1)
	ctx, stop := signalContextWithExit(context.Background(), func(exitCode int) {
		exitCodes <- exitCode
	})
	defer stop()
	process, err := os.FindProcess(os.Getpid())
	if err != nil {
		t.Fatalf("find test process: %v", err)
	}
	if err := process.Signal(syscall.SIGHUP); err != nil {
		t.Fatalf("send first signal: %v", err)
	}
	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("first signal did not cancel context")
	}
	if err := process.Signal(syscall.SIGHUP); err != nil {
		t.Fatalf("send second signal: %v", err)
	}
	select {
	case exitCode := <-exitCodes:
		if exitCode != 129 {
			t.Fatalf("forced exit code = %d, want 129", exitCode)
		}
	case <-time.After(time.Second):
		t.Fatal("second signal did not force exit")
	}
}

func TestSignalExitCode(t *testing.T) {
	tests := []struct {
		signal os.Signal
		want   int
	}{
		{signal: os.Interrupt, want: 130},
		{signal: syscall.SIGHUP, want: 129},
		{signal: syscall.SIGTERM, want: 143},
		{signal: testSignal("other"), want: 1},
	}
	for _, test := range tests {
		if got := signalExitCode(test.signal); got != test.want {
			t.Fatalf("signalExitCode(%v) = %d, want %d", test.signal, got, test.want)
		}
	}
}

type testSignal string

func (testSignal) Signal() {}

func (signal testSignal) String() string {
	return string(signal)
}

func TestExecuteVersion(t *testing.T) {
	var stdout, stderr bytes.Buffer

	if exitCode := execute([]string{"version"}, &stdout, &stderr); exitCode != 0 {
		t.Fatalf("execute() exit code = %d; stderr = %q", exitCode, stderr.String())
	}
	if got := stdout.String(); !strings.HasPrefix(got, "agefreighter-tools dev") {
		t.Fatalf("execute() stdout = %q", got)
	}
}
