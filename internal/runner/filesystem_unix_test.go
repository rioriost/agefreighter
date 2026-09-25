//go:build linux || darwin

package runner

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFilesystemCapacityMeasuresOnlyExistingRoots(t *testing.T) {
	dir := t.TempDir()
	total, free, err := csvDiskCapacity(dir)
	if err != nil || total <= 0 || free < 0 || free > total {
		t.Fatalf("invalid local filesystem evidence: %g %g %v", total, free, err)
	}
	if _, _, err := csvDiskCapacity(filepath.Join(dir, "missing")); err == nil {
		t.Fatal("missing filesystem reported usable capacity")
	}
}

func TestAdmissionLockRejectsUnsafeRootsAndSymlinkWithoutChangingTarget(t *testing.T) {
	for _, mode := range []string{"file-root", "public-root", "symlink-lock"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			root := filepath.Join(dir, "workflows")
			if mode == "file-root" {
				if err := writeNew(root, []byte("retained")); err != nil {
					t.Fatal(err)
				}
			} else if err := os.Mkdir(root, 0700); err != nil {
				t.Fatal(err)
			}
			target := filepath.Join(dir, "evidence")
			if err := writeNew(target, []byte("retained")); err != nil {
				t.Fatal(err)
			}
			if mode == "public-root" {
				if err := os.Chmod(root, 0755); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "symlink-lock" {
				if err := os.Symlink(target, filepath.Join(root, ".dispatch.lock")); err != nil {
					t.Fatal(err)
				}
			}
			if unlock, err := (Manager{Root: root}).dispatchLock(); err == nil {
				unlock()
				t.Fatal("unsafe lock acquired")
			}
			data, err := os.ReadFile(target)
			if err != nil || string(data) != "retained" {
				t.Fatal("admission changed unrelated evidence")
			}
		})
	}
}
