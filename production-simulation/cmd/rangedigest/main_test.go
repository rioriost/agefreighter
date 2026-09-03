package main

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"

	fixturemodel "github.com/rioriost/agefreighter/production-simulation/internal/fixture"
)

func TestFixtureAndCompareCommands(t *testing.T) {
	t.Parallel()

	root := filepath.Join(t.TempDir(), "fixture")
	_, err := fixturemodel.Generate(context.Background(), fixturemodel.GenerateConfig{
		Phase: fixturemodel.PhaseTiny, Output: root, Shards: 2, Workers: 2, Seed: 7,
	})
	if err != nil {
		t.Fatal(err)
	}
	expected := filepath.Join(t.TempDir(), "expected.json")
	var stdout, stderr bytes.Buffer
	if exitCode := run(context.Background(), []string{
		"fixture", "--manifest", filepath.Join(root, "manifest.json"),
		"--range-rows", "50", "--output", expected,
	}, &stdout, &stderr); exitCode != 0 {
		t.Fatalf("fixture exit=%d stderr=%s", exitCode, stderr.String())
	}
	if stdout.Len() == 0 {
		t.Fatal("fixture command omitted its summary")
	}

	stdout.Reset()
	stderr.Reset()
	if exitCode := run(context.Background(), []string{
		"compare", "--expected", expected, "--actual", expected,
	}, &stdout, &stderr); exitCode == 0 {
		t.Fatal("compare accepted a fixture manifest in the target role")
	}
}

func TestRunRejectsUnknownCommand(t *testing.T) {
	t.Parallel()

	var stdout, stderr bytes.Buffer
	if exitCode := run(context.Background(), []string{"unknown"}, &stdout, &stderr); exitCode != 1 {
		t.Fatalf("exit = %d, stderr=%s", exitCode, stderr.String())
	}
}
