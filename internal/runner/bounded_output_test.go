package runner

import (
	"bytes"
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestBoundedOutputCopyCannotBypassWrite(t *testing.T) {
	for _, tc := range []struct {
		name  string
		limit int
		data  string
	}{
		{"zero", 0, "data"},
		{"exact", 4, "data"},
		{"overflow", 3, "data"},
		{"multi-chunk", 8, strings.Repeat("data", 20000)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			output := &boundedOutput{limit: tc.limit}
			// Hide WriterTo so io.Copy must choose the destination's method
			// set, reproducing the formerly promoted bytes.Buffer.ReadFrom.
			source := struct{ io.Reader }{strings.NewReader(tc.data)}
			n, err := io.Copy(output, source)
			want := tc.data[:min(tc.limit, len(tc.data))]
			if err != nil || n != int64(len(tc.data)) || output.Len() != len(want) || output.String() != want || !bytes.Equal(output.Bytes(), []byte(want)) || output.overflow != (len(tc.data) > tc.limit) {
				t.Fatalf("copy capture: consumed=%d retained=%d overflow=%v err=%v", n, len(output.Bytes()), output.overflow, err)
			}
			if n, err := io.WriteString(output, "ignored"); err != nil || n != len("ignored") || !output.overflow || output.String() != want {
				t.Fatalf("subsequent string write escaped bound: consumed=%d retained=%d err=%v", n, len(output.Bytes()), err)
			}
		})
	}
}

func TestBoundedOutputExecHelper(t *testing.T) {
	if os.Getenv("AGEFREIGHTER_RUNNER_OUTPUT_FIXTURE") != "1" {
		return
	}
	if _, err := io.Copy(os.Stdout, strings.NewReader(strings.Repeat("O", MaxArtifactBytes+1))); err != nil {
		os.Exit(2)
	}
	if _, err := io.Copy(os.Stderr, strings.NewReader(strings.Repeat("E", (64<<10)+1))); err != nil {
		os.Exit(3)
	}
	os.Exit(0)
}

func TestBoundedOutputExecDrainsAndBoundsStdoutAndStderr(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, executable, "-test.run=^TestBoundedOutputExecHelper$")
	cmd.Env = append(os.Environ(), "AGEFREIGHTER_RUNNER_OUTPUT_FIXTURE=1")
	stdout := &boundedOutput{limit: MaxArtifactBytes}
	stderr := &boundedOutput{limit: 64 << 10}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("bounded capture did not drain both child pipes: %v", err)
	}
	for _, tc := range []struct {
		name   string
		output *boundedOutput
		marker byte
	}{
		{"stdout", stdout, 'O'}, {"stderr", stderr, 'E'},
	} {
		if !tc.output.overflow || len(tc.output.Bytes()) != tc.output.limit || !bytes.Equal(tc.output.Bytes(), bytes.Repeat([]byte{tc.marker}, tc.output.limit)) {
			t.Fatalf("%s capture escaped bound: length=%d limit=%d overflow=%v", tc.name, len(tc.output.Bytes()), tc.output.limit, tc.output.overflow)
		}
	}
}

func TestOversizedMigrationOutputFailsClosedWithoutVerificationOrReplay(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("local Unix CLI orchestration fixture")
	}
	m, r, starts := resumeFixture(t)
	dir := filepath.Join(m.Root, r.Workflow, operationID)
	state, err := m.Status(r.Workflow, operationID)
	if err != nil {
		t.Fatal(err)
	}
	state.Phase = "accepted"
	if err := replaceJSON(filepath.Join(dir, "state.json"), state); err != nil {
		t.Fatal(err)
	}
	m.migrationPrepare = func(context.Context, []byte, string) error { return nil }
	m.CLI = filepath.Join(dir, "loader-fixture")
	if err := os.WriteFile(m.CLI, []byte("#!/bin/sh\ncat stdout-fixture\ncat stderr-fixture >&2\n"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := writeNew(filepath.Join(dir, "stdout-fixture"), bytes.Repeat([]byte{'O'}, MaxArtifactBytes+1)); err != nil {
		t.Fatal(err)
	}
	if err := writeNew(filepath.Join(dir, "stderr-fixture"), bytes.Repeat([]byte{'E'}, (64<<10)+1)); err != nil {
		t.Fatal(err)
	}
	if err := m.Work(t.Context(), r.Workflow, operationID); err != nil {
		t.Fatal(err)
	}
	state, err = m.Status(r.Workflow, operationID)
	if err != nil || state.Phase != "failed" || state.ExitCode == nil || *state.ExitCode != 1 || state.ReportBytes != 0 {
		t.Fatalf("oversized output became a successful migration: %+v %v", state, err)
	}
	log, err := os.ReadFile(filepath.Join(dir, "load.stderr.log"))
	if err != nil || len(log) != 64<<10 || !bytes.Equal(log, bytes.Repeat([]byte{'E'}, 64<<10)) {
		t.Fatalf("stderr evidence escaped its bound: length=%d err=%v", len(log), err)
	}
	for _, name := range []string{"load.json", "verify.json", "report.json", "secrets.json"} {
		if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
			t.Fatalf("overflow retained unsafe artifact or credential transport: %s", name)
		}
	}
	failure, err := os.ReadFile(filepath.Join(dir, "migration-error.txt"))
	if err != nil || !strings.Contains(string(failure), "No automatic retry was made.") {
		t.Fatal("overflow lost durable failure evidence")
	}
	if _, err := m.Report(r.Workflow, operationID, 0); err == nil {
		t.Fatal("overflow exported a success-shaped artifact")
	}
	if err := m.Work(t.Context(), r.Workflow, operationID); err == nil || *starts != 1 {
		t.Fatal("overflow authorized migration replay")
	}
}
