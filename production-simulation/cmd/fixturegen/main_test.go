package main

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunGenerateAndVerify(t *testing.T) {
	output := filepath.Join(t.TempDir(), "fixture")
	stdout := new(bytes.Buffer)
	stderr := new(bytes.Buffer)
	code := run(context.Background(), []string{
		"generate", "--phase", "tiny", "--output", output,
		"--shards", "2", "--workers", "2", "--seed", "7",
	}, stdout, stderr)
	if code != 0 || !strings.Contains(stdout.String(), "vertices=160") {
		t.Fatalf("generate code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
	stdout.Reset()
	stderr.Reset()
	code = run(context.Background(), []string{
		"verify", "--manifest", filepath.Join(output, "manifest.json"),
	}, stdout, stderr)
	if code != 0 || !strings.Contains(stdout.String(), "verified") {
		t.Fatalf("verify code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
}

func TestRunUsageAndErrors(t *testing.T) {
	tests := [][]string{
		nil,
		{"unknown"},
		{"generate", "--phase", "tiny", "--seed", "bad"},
		{"generate", "extra"},
		{"verify"},
		{"verify", "--manifest", "missing.json"},
	}
	for index, args := range tests {
		stdout := new(bytes.Buffer)
		stderr := new(bytes.Buffer)
		if code := run(context.Background(), args, stdout, stderr); code == 0 {
			t.Fatalf("case %d succeeded", index)
		}
	}
	stdout := new(bytes.Buffer)
	if code := run(context.Background(), []string{"help"}, stdout, new(bytes.Buffer)); code != 0 || stdout.Len() == 0 {
		t.Fatalf("help code=%d output=%q", code, stdout)
	}
}
