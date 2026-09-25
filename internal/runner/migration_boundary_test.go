package runner

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMigrationEvidenceFailuresNeverReplayOrOverwriteRetainedFiles(t *testing.T) {
	for _, mode := range []string{"identity", "load-stderr", "load-json", "migration-error", "report", "lease", "prepare-evidence"} {
		t.Run(mode, func(t *testing.T) {
			m, r, starts := resumeFixture(t)
			root := filepath.Join(m.Root, r.Workflow)
			dir := filepath.Join(root, operationID)
			state, err := m.Status(r.Workflow, operationID)
			if err != nil {
				t.Fatal(err)
			}
			state.Phase = "accepted"
			m.migrationPrepare = func(context.Context, []byte, string) error { return nil }
			if mode == "identity" {
				state.JobID = resumedID
			}
			if err := replaceJSON(filepath.Join(dir, "state.json"), state); err != nil {
				t.Fatal(err)
			}
			retained := ""
			switch mode {
			case "load-stderr":
				retained = filepath.Join(dir, "load.stderr.log")
			case "load-json":
				retained = filepath.Join(dir, "load.json")
			case "migration-error":
				retained = filepath.Join(dir, "migration-error.txt")
				m.migrationPrepare = func(context.Context, []byte, string) error { return errors.New("PRIVATE-TARGET-DETAIL") }
			case "report":
				retained = filepath.Join(dir, "report.json")
			case "lease":
				if err := os.WriteFile(filepath.Join(root, "active"), []byte(resumedID), 0600); err != nil {
					t.Fatal(err)
				}
			case "prepare-evidence":
				retained = filepath.Join(dir, "target-prepared.json")
			}
			if retained != "" {
				if err := writeNew(retained, []byte("retained-evidence")); err != nil {
					t.Fatal(err)
				}
			}
			err = m.Work(t.Context(), r.Workflow, operationID)
			if (err == nil) != (mode == "prepare-evidence") {
				t.Fatalf("unexpected worker result: %v", err)
			}
			if err != nil && strings.Contains(err.Error(), "PRIVATE-TARGET-DETAIL") {
				t.Fatal("private target diagnostic escaped worker")
			}
			if retained != "" {
				data, err := os.ReadFile(retained)
				if err != nil || string(data) != "retained-evidence" {
					t.Fatal("worker overwrote prior evidence")
				}
			}
			if _, err := os.Stat(filepath.Join(dir, "secrets.json")); !os.IsNotExist(err) {
				t.Fatal("failed migration retained protected transport")
			}
			if err := m.Work(t.Context(), r.Workflow, operationID); err == nil || *starts != 1 {
				t.Fatal("failed evidence persistence authorized replay")
			}
			if mode != "report" && mode != "lease" {
				if _, err := os.Stat(filepath.Join(dir, "verify.json")); !os.IsNotExist(err) {
					t.Fatal("verification ran after failed prerequisite")
				}
			}
			if mode == "lease" {
				lease, err := os.ReadFile(filepath.Join(root, "active"))
				if err != nil || string(lease) != resumedID {
					t.Fatal("migration removed another operation's lease")
				}
			}
		})
	}
}

func TestLoaderSuccessWithWrongJobIdentityCannotTriggerVerification(t *testing.T) {
	m, r, starts := resumeFixture(t)
	dir := filepath.Join(m.Root, r.Workflow, operationID)
	state, err := m.Status(r.Workflow, operationID)
	if err != nil {
		t.Fatal(err)
	}
	state.Phase = "accepted"
	if err := replaceJSON(filepath.Join(dir, "state.json"), state); err != nil {
		t.Fatal(err)
	}
	m.migrationPrepare = func(context.Context, []byte, string) error { return nil }
	m.CLI = filepath.Join(dir, "loader-fixture")
	data, err := json.Marshal(map[string]string{"jobId": resumedID, "status": "committed"})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(m.CLI, []byte("#!/bin/sh\nprintf '%s' '"+string(data)+"'\n"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := m.Work(t.Context(), r.Workflow, operationID); err != nil {
		t.Fatal(err)
	}
	state, err = m.Status(r.Workflow, operationID)
	if err != nil || state.Phase != "failed" || state.ExitCode == nil || *state.ExitCode != 1 || state.ReportBytes != 0 || state.JobID != operationID {
		t.Fatalf("wrong committed identity passed: %+v %v", state, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "verify.json")); !os.IsNotExist(err) {
		t.Fatal("verification ran for another job's loader response")
	}
	if err := m.Work(t.Context(), r.Workflow, operationID); err == nil || *starts != 1 {
		t.Fatal("wrong loader identity triggered replay")
	}
}

func TestMigrationConnectionRejectsMissingCredentialAndUnavailableTLSRoots(t *testing.T) {
	dsn := testMigrationDSN()
	dsn = strings.Replace(dsn, strings.Repeat("a", 32), "short", 1)
	if _, err := migrationConnection(dsn); err == nil || err.Error() != "target credential is unavailable" {
		t.Fatalf("short credential admitted: %v", err)
	}
	t.Setenv("PGSSLROOTCERT", filepath.Join(t.TempDir(), "missing-ca.pem"))
	if _, err := migrationConnection(testMigrationDSN()); err == nil || err.Error() != "target TLS validation is unavailable" {
		t.Fatalf("missing TLS root evidence admitted: %v", err)
	}
}
