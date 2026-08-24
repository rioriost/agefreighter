package architecture_test

import (
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type goPackage struct {
	ImportPath string
	Deps       []string
}

func TestDependencyBoundaries(t *testing.T) {
	moduleRoot := findModuleRoot(t)
	command := exec.Command("go", "list", "-json", "./...")
	command.Dir = moduleRoot
	output, err := command.StdoutPipe()
	if err != nil {
		t.Fatalf("StdoutPipe() error = %v", err)
	}
	command.Stderr = os.Stderr
	if err := command.Start(); err != nil {
		t.Fatalf("go list start error = %v", err)
	}

	decoder := json.NewDecoder(output)
	for {
		var pkg goPackage
		if err := decoder.Decode(&pkg); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("decode go list output: %v", err)
		}
		assertPackageBoundaries(t, pkg)
	}
	if err := command.Wait(); err != nil {
		t.Fatalf("go list error = %v", err)
	}
}

func findModuleRoot(t *testing.T) string {
	t.Helper()
	current, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd() error = %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(current, "go.mod")); err == nil {
			return current
		}
		parent := filepath.Dir(current)
		if parent == current {
			t.Fatal("go.mod not found")
		}
		current = parent
	}
}

func assertPackageBoundaries(t *testing.T, pkg goPackage) {
	t.Helper()
	const module = "github.com/rioriost/agefreighter"

	if isPackageOrChild(pkg.ImportPath, module+"/internal/source") &&
		containsPackageOrChild(pkg.Deps, module+"/internal/age") {
		t.Errorf("%s depends on internal/age", pkg.ImportPath)
	}
	if !isPackageOrChild(pkg.ImportPath, module+"/cmd/agefreighter-tools") &&
		!isPackageOrChild(pkg.ImportPath, module+"/internal/tools") &&
		containsPackageOrChild(pkg.Deps, module+"/internal/tools") {
		t.Errorf("%s depends on internal/tools", pkg.ImportPath)
	}
}

func containsPackageOrChild(packages []string, root string) bool {
	return slices.ContainsFunc(packages, func(candidate string) bool {
		return isPackageOrChild(candidate, root)
	})
}

func isPackageOrChild(candidate, root string) bool {
	return candidate == root || strings.HasPrefix(candidate, root+"/")
}
