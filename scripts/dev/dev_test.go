package dev_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestDevelopmentScriptsHaveValidShellSyntax(t *testing.T) {
	root := scriptDirectory(t)
	command := exec.Command(
		"sh",
		"-n",
		filepath.Join(root, "dev.sh"),
		filepath.Join(root, "runtime.sh"),
		filepath.Join(root, "services.sh"),
	)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("sh -n error = %v\n%s", err, output)
	}
}

func TestPullUsesPinnedImagesAndPlatform(t *testing.T) {
	root := scriptDirectory(t)
	temp := t.TempDir()
	logPath := filepath.Join(temp, "runtime.log")
	fakePath := filepath.Join(temp, "container")
	fake := `#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_RUNTIME_LOG"
if [ "$1 $2" = "system status" ]; then
	printf 'status running\n'
fi
`
	if err := os.WriteFile(fakePath, []byte(fake), 0o755); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	command := exec.Command(filepath.Join(root, "dev.sh"), "pull")
	command.Env = append(
		os.Environ(),
		"DEV_RUNTIME=apple",
		"DEV_PLATFORM=linux/arm64",
		"CONTAINER_CLI="+fakePath,
		"FAKE_RUNTIME_LOG="+logPath,
	)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("dev.sh pull error = %v\n%s", err, output)
	}
	logData, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	log := string(logData)
	for _, expected := range []string{
		"image pull --platform linux/arm64 apache/age@sha256:fe8b33905a61549a067f8512808b88011bdcaa82ab65d8788f39fb07a04aa5be",
		"image pull --platform linux/arm64 postgres@sha256:ef257d85f76e48da1c64832459b59fcaba1a4dac97bf5d7450c77753542eee94",
		"image pull --platform linux/arm64 neo4j@sha256:89d577f2e49606de76441eca8cf7a0fe88e594cbaac4d2a3d86c6e59676e2b1e",
	} {
		if !strings.Contains(log, expected) {
			t.Errorf("runtime log does not contain %q:\n%s", expected, log)
		}
	}
}

func TestManagedResourcesAreExplicitAndScoped(t *testing.T) {
	root := scriptDirectory(t)
	command := exec.Command(
		"sh",
		"-c",
		`. "$1"; service_containers; service_volumes`,
		"sh",
		filepath.Join(root, "services.sh"),
	)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("list resources error = %v\n%s", err, output)
	}
	lines := strings.Fields(string(output))
	if len(lines) != 6 {
		t.Fatalf("managed resources = %v, want six", lines)
	}
	for _, resource := range lines {
		if !strings.HasPrefix(resource, "agefreighter-") {
			t.Errorf("resource %q is outside agefreighter namespace", resource)
		}
	}

	for _, forbidden := range []string{
		" prune",
		"delete --all",
		"rm --all",
		"volume delete --all",
		"volume rm --all",
	} {
		for _, name := range []string{"dev.sh", "runtime.sh", "services.sh"} {
			script, err := os.ReadFile(filepath.Join(root, name))
			if err != nil {
				t.Fatalf("ReadFile(%s) error = %v", name, err)
			}
			if strings.Contains(string(script), forbidden) {
				t.Errorf("%s contains forbidden broad operation %q", name, forbidden)
			}
		}
	}
}

func TestRejectsInvalidReadinessTimeout(t *testing.T) {
	root := scriptDirectory(t)
	command := exec.Command(filepath.Join(root, "dev.sh"), "status")
	command.Env = append(os.Environ(), "DEV_READY_TIMEOUT=invalid")
	output, err := command.CombinedOutput()
	if err == nil || !strings.Contains(string(output), "positive integer") {
		t.Fatalf("dev.sh status error = %v\n%s", err, output)
	}
}

func scriptDirectory(t *testing.T) string {
	t.Helper()
	directory, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd() error = %v", err)
	}
	return directory
}
