package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
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
