package runner

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestReadinessFailsClosedAtEveryEvidenceBoundary(t *testing.T) {
	for _, mode := range []string{"marker-directory", "digest-missing", "digest-malformed", "boot", "version-error", "version-mismatch", "health-error"} {
		t.Run(mode, func(t *testing.T) {
			m, _, starts := testManager(t)
			base := filepath.Dir(m.Root)
			marker := filepath.Join(base, "bootstrap.complete")
			if mode == "marker-directory" {
				if err := os.Mkdir(marker, 0700); err != nil {
					t.Fatal(err)
				}
			} else if err := writeNew(marker, nil); err != nil {
				t.Fatal(err)
			}
			if err := os.Mkdir(filepath.Join(base, "evidence"), 0700); err != nil {
				t.Fatal(err)
			}
			if mode != "digest-missing" {
				digest := strings.Repeat("a", 64)
				if mode == "digest-malformed" {
					digest = strings.Repeat("A", 64)
				}
				if err := writeNew(filepath.Join(base, "evidence", "archive.sha256"), []byte(digest)); err != nil {
					t.Fatal(err)
				}
			}
			switch mode {
			case "boot":
				m.BootID = func() (string, error) { return "", errors.New("boot unavailable") }
			case "version-error":
				m.versionProbe = func(context.Context) (string, error) { return "", errors.New("version unavailable") }
			case "version-mismatch":
				m.versionProbe = func(context.Context) (string, error) { return "other version", nil }
			case "health-error":
				m.healthProbe = func(context.Context) (*GuestHealth, error) { return nil, errors.New("health unavailable") }
			}
			got, err := m.Ready(t.Context())
			if err == nil || got.Ready || *starts != 0 {
				t.Fatalf("unsupported readiness: %+v %v", got, err)
			}
		})
	}
}

func TestInstalledVersionUsesBoundedLocalExecutable(t *testing.T) {
	m, _, _ := testManager(t)
	m.versionProbe = nil
	got, err := m.installedVersion(t.Context())
	if err != nil || !strings.Contains(got, "agefreighter") {
		t.Fatalf("installed executable: %q %v", got, err)
	}
	m.CLI = filepath.Join(t.TempDir(), "missing")
	if _, err := m.installedVersion(t.Context()); err == nil {
		t.Fatal("missing executable accepted")
	}
}

func TestInstalledVersionRejectsOversizedExecutableOutput(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("local shell output fixture")
	}
	m, _, _ := testManager(t)
	m.versionProbe = nil
	m.CLI = filepath.Join(t.TempDir(), "version-fixture")
	if err := os.WriteFile(m.CLI, []byte("#!/bin/sh\nprintf '%s' '"+strings.Repeat("x", 4097)+"'\n"), 0700); err != nil {
		t.Fatal(err)
	}
	if _, err := m.installedVersion(t.Context()); err == nil || !strings.Contains(err.Error(), "bound") {
		t.Fatalf("oversized version accepted: %v", err)
	}
}

func TestPrivilegedLinuxManagerRejectsUnsupportedHostWithoutSystemChanges(t *testing.T) {
	if runtime.GOOS == "linux" && runtime.GOARCH == "amd64" && os.Geteuid() == 0 {
		t.Skip("do not construct a real privileged systemd manager")
	}
	if _, err := LinuxManager(); err == nil {
		t.Fatal("unsupported host admitted")
	}
}

func TestAbsentNativeHealthCannotAdmitMigration(t *testing.T) {
	if runtime.GOOS == "linux" {
		t.Skip("non-Linux unsupported health boundary; never query real systemd")
	}
	m, r, starts := resumeFixture(t)
	m.healthProbe = nil
	if _, err := m.SubmitResume(t.Context(), r); err == nil || *starts != 1 {
		t.Fatal("missing native guest health authorized recovery")
	}
	if _, err := os.Stat(filepath.Join(m.Root, r.Workflow, operationID, "resume.claim")); !os.IsNotExist(err) {
		t.Fatal("missing health reserved a continuation")
	}
}
