package cli

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
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
