package cli

import (
	"bytes"
	"errors"
	"strings"
	"testing"
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
