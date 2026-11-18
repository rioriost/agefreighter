package simulation_test

import (
	"os/exec"
	"testing"
)

func TestHorizonDBScaleGuard(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 is required for the Automation guard's isolated contract tests")
	}
	command := exec.CommandContext(t.Context(), python, "-B", "-m", "unittest", "discover",
		"-s", "scripts", "-p", "horizondb_scale_guard_test.py", "-v")
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("scale guard contracts: %v\n%s", err, output)
	}
}
