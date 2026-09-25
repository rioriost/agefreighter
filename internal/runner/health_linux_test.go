package runner

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGuestHealthRequiresCompleteJournalEvidence(t *testing.T) {
	for _, tc := range []struct {
		name   string
		stdout string
		stderr string
		status string
		active bool
		ooms   int
		ok     bool
	}{
		{name: "empty-success", status: "0", ok: true},
		{name: "no-matches", status: "1", ok: true},
		{name: "active-workflow", status: "1", active: true, ok: true},
		{name: "oom-events", stdout: "Out of memory: first\nKilled process 12\nunrelated\n", status: "0", ooms: 2, ok: true},
		{name: "permission-warning", stderr: "Insufficient permissions\n", status: "1"},
		{name: "success-with-warning", stderr: "Journal unavailable\n", status: "0"},
		{name: "command-failure", status: "2"},
		{name: "failed-with-output", stdout: "Out of memory\n", status: "1"},
		{name: "stdout-overflow", stdout: strings.Repeat("x", (64<<10)+1), status: "0"},
		{name: "stderr-overflow", stderr: strings.Repeat("x", 4097), status: "0"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			bin := t.TempDir()
			for name, value := range map[string]string{"stdout": tc.stdout, "stderr": tc.stderr} {
				if err := os.WriteFile(filepath.Join(bin, name), []byte(value), 0600); err != nil {
					t.Fatal(err)
				}
			}
			script := `#!/bin/sh
test "$#" = 6 &&
test "$1" = -k &&
test "$2" = -b &&
test "$3" = --no-pager &&
test "$4" = '--grep=Out of memory|Killed process' &&
test "$5" = -o &&
test "$6" = cat || { printf 'unexpected journal arguments\n' >&2; exit 2; }
/bin/cat "$(dirname "$0")/stdout"
/bin/cat "$(dirname "$0")/stderr" >&2
exit ` + tc.status + "\n"
			if err := os.WriteFile(filepath.Join(bin, "journalctl"), []byte(script), 0700); err != nil {
				t.Fatal(err)
			}
			t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
			if tc.active {
				dir := filepath.Join(root, workflowID)
				if err := os.Mkdir(dir, 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dir, "active"), []byte(operationID), 0600); err != nil {
					t.Fatal(err)
				}
			}
			health, err := (Manager{Root: root}).health(t.Context())
			if !tc.ok {
				if err == nil || err.Error() != "guest OOM evidence is unavailable" || health != nil {
					t.Fatalf("unavailable journal became health evidence: %+v, %v", health, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if health.Idle == tc.active || health.OOMEvents != tc.ooms ||
				health.StorageUsedPercent < 0 || health.StorageUsedPercent > 100 || health.SwapUsedBytes%1024 != 0 {
				t.Fatalf("health evidence = %+v", health)
			}
		})
	}
}

func TestGuestHealthRejectsUnavailableStorageOrCommand(t *testing.T) {
	m := Manager{Root: filepath.Join(t.TempDir(), "missing")}
	if health, err := m.health(t.Context()); err == nil || health != nil {
		t.Fatalf("missing storage accepted: %+v, %v", health, err)
	}
	m.Root = t.TempDir()
	t.Setenv("PATH", t.TempDir())
	if health, err := m.health(t.Context()); err == nil || health != nil {
		t.Fatalf("missing journal command accepted: %+v, %v", health, err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if health, err := m.health(ctx); err == nil || health != nil {
		t.Fatalf("canceled health accepted: %+v, %v", health, err)
	}
}
