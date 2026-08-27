package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/observability"
)

func TestMain(testingMain *testing.M) {
	_ = os.Unsetenv(observability.LogFormatEnvironment)
	_ = os.Unsetenv(observability.LogLevelEnvironment)
	_ = os.Setenv(observability.SDKDisabledEnvironment, "true")
	os.Exit(testingMain.Run())
}

func TestRunVersion(t *testing.T) {
	var stdout, stderr bytes.Buffer

	exitCode := run([]string{"version"}, &stdout, &stderr)

	if exitCode != 0 {
		t.Fatalf("run() exit code = %d, want 0; stderr = %q", exitCode, stderr.String())
	}
	if got := stdout.String(); !strings.HasPrefix(got, "agefreighter dev") {
		t.Fatalf("run() stdout = %q, want agefreighter development version", got)
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

func TestRunJSONLogging(t *testing.T) {
	t.Setenv(observability.LogFormatEnvironment, "json")
	var stdout, stderr bytes.Buffer

	exitCode := run([]string{"version"}, &stdout, &stderr)

	if exitCode != 0 {
		t.Fatalf("run() exit code = %d; stderr = %q", exitCode, stderr.String())
	}
	decoder := json.NewDecoder(&stderr)
	for index, eventName := range []string{"command_started", "command_completed"} {
		var event map[string]any
		if err := decoder.Decode(&event); err != nil {
			t.Fatalf("decode event %d: %v", index, err)
		}
		if event["event"] != eventName ||
			event["service"] != "agefreighter" ||
			event["command"] != "version" {
			t.Fatalf("event %d = %#v", index, event)
		}
	}
}

func TestRunRejectsInvalidObservabilityConfiguration(t *testing.T) {
	t.Setenv(observability.LogFormatEnvironment, "yaml")
	var stdout, stderr bytes.Buffer

	if exitCode := run([]string{"version"}, &stdout, &stderr); exitCode != 1 {
		t.Fatalf("run() exit code = %d", exitCode)
	}
	if !strings.Contains(stderr.String(), observability.LogFormatEnvironment) {
		t.Fatalf("run() stderr = %q", stderr.String())
	}
}
