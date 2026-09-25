package runner

import (
	"context"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
)

func TestMigrationPreparationUsesVerifiedTLSAndStopsAtFailedPrerequisite(t *testing.T) {
	for _, mode := range []string{"success", "authentication", "version", "old-version", "unencrypted", "extension", "preload", "invalid-configuration", "graph-exists", "existing-graph"} {
		t.Run(mode, func(t *testing.T) {
			_, r, _ := testManager(t)
			job, err := config.Parse(r.Configuration)
			if err != nil {
				t.Fatal(err)
			}
			fixture := newTargetFixture(t, mode, job.Target.Graph, "", false)
			data := r.Configuration
			if mode == "invalid-configuration" {
				data = []byte(`{`)
			}
			err = prepareMigration(t.Context(), data, testMigrationDSN())
			if (err == nil) != (mode == "success") {
				t.Fatalf("preparation result: %v", err)
			}
			if err != nil && (strings.Contains(err.Error(), "PRIVATE-TARGET-DETAIL") || strings.Contains(err.Error(), testMigrationDSN())) {
				t.Fatalf("target diagnostic leaked: %v", err)
			}
			var queries []string
			for len(fixture.queries) != 0 {
				queries = append(queries, <-fixture.queries)
			}
			if mode == "authentication" {
				if len(queries) != 0 {
					t.Fatal("unauthenticated preparation issued SQL")
				}
				return
			}
			if len(queries) == 0 || !strings.Contains(queries[0], "server_version_num") {
				t.Fatalf("preparation skipped version/TLS evidence: %q", queries)
			}
			if mode == "version" || mode == "old-version" || mode == "unencrypted" {
				if len(queries) != 1 {
					t.Fatalf("target preparation continued after failed gate: %q", queries)
				}
			}
			if mode == "success" {
				if len(queries) != 5 || queries[1] != "CREATE EXTENSION IF NOT EXISTS age" || !strings.Contains(queries[4], "SELECT EXISTS") {
					t.Fatalf("unexpected preparation sequence: %q", queries)
				}
			}
		})
	}
	if err := prepareMigration(t.Context(), nil, "invalid"); err == nil {
		t.Fatal("unreviewed preparation connection accepted")
	}
}

func TestResumeInspectionQueriesOnlyOriginalReadOnlyTargetEvidence(t *testing.T) {
	for _, mode := range []string{"success", "released-lease", "authentication", "begin", "job", "generation", "verification", "mapping-mismatch"} {
		t.Run(mode, func(t *testing.T) {
			m, r, starts := resumeFixture(t)
			r.Action, r.Operation, r.Resume = "inspect-resume", operationID, nil
			root := filepath.Join(m.Root, r.Workflow)
			dir := filepath.Join(root, r.Operation)
			state, err := m.Status(r.Workflow, r.Operation)
			if err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile(filepath.Join(dir, "job.json"))
			if err != nil {
				t.Fatal(err)
			}
			job, err := config.Parse(data)
			if err != nil {
				t.Fatal(err)
			}
			if mode == "released-lease" {
				if err := os.Remove(filepath.Join(root, "active")); err != nil {
					t.Fatal(err)
				}
			}
			before, err := os.ReadDir(dir)
			if err != nil {
				t.Fatal(err)
			}
			fixture := newTargetFixture(t, mode, job.Target.Graph, state.ConfigSHA256, false)
			result, err := m.InspectResume(t.Context(), r)
			success := mode == "success" || mode == "released-lease"
			if (err == nil) != success {
				t.Fatalf("inspection result: %+v %v", result, err)
			}
			if success {
				reasons := 2
				if mode == "released-lease" {
					reasons = 1
				}
				if result.CanResume || result.Outcome != "review-required" || result.GenerationID != "17" || result.CommittedRows != "1" || result.JobID != operationID || result.ConfigSHA256 != state.ConfigSHA256 || len(result.Reasons) != reasons {
					t.Fatalf("unbound or authoritative inspection: %+v", result)
				}
			} else if strings.Contains(err.Error(), "PRIVATE-TARGET-DETAIL") || strings.Contains(err.Error(), testMigrationDSN()) {
				t.Fatalf("private target evidence leaked: %v", err)
			}
			if mode != "begin" && mode != "authentication" {
				fixture.assertReadOnly(t)
			}
			after, err := os.ReadDir(dir)
			if err != nil || !reflect.DeepEqual(before, after) || *starts != 1 {
				t.Fatal("read-only inspection changed operation files or launched a worker")
			}
		})
	}
}

func TestFinalResumeGenerationCheckRejectsChangedCommittedIdentity(t *testing.T) {
	for _, mode := range []string{"success", "authentication", "begin", "job", "generation", "identity-mismatch"} {
		t.Run(mode, func(t *testing.T) {
			m, r, starts := resumeFixture(t)
			m.resumeComplete = nil
			state, err := m.Status(r.Workflow, operationID)
			if err != nil {
				t.Fatal(err)
			}
			state.Resume = r.Resume
			data, err := os.ReadFile(filepath.Join(m.Root, r.Workflow, operationID, "job.json"))
			if err != nil {
				t.Fatal(err)
			}
			job, err := config.Parse(data)
			if err != nil {
				t.Fatal(err)
			}
			fixture := newTargetFixture(t, mode, job.Target.Graph, state.ConfigSHA256, true)
			err = m.checkResumedGeneration(t.Context(), state, data, testMigrationDSN())
			if (err == nil) != (mode == "success") {
				t.Fatalf("final generation result: %v", err)
			}
			if err != nil && strings.Contains(err.Error(), "PRIVATE-TARGET-DETAIL") {
				t.Fatalf("private target diagnostic leaked: %v", err)
			}
			if mode != "begin" && mode != "authentication" {
				fixture.assertReadOnly(t)
			}
			if *starts != 1 {
				t.Fatal("final target check launched another worker")
			}
		})
	}
	m, r, _ := resumeFixture(t)
	m.resumeComplete = nil
	if err := m.checkResumedGeneration(t.Context(), State{}, []byte(`{`), testMigrationDSN()); err == nil {
		t.Fatal("final check accepted corrupt configuration")
	}
	data, err := os.ReadFile(filepath.Join(m.Root, r.Workflow, operationID, "job.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := m.checkResumedGeneration(t.Context(), State{}, data, "invalid"); err == nil {
		t.Fatal("final check accepted unreviewed connection")
	}
}

func TestTargetFixturePortCollisionFailsInsteadOfSkipping(t *testing.T) {
	listener, err := net.Listen("tcp4", "127.0.0.1:5432")
	if err != nil {
		t.Fatalf("cannot isolate collision fixture on loopback port 5432: %v", err)
	}
	defer listener.Close()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, executable, "-test.run=^TestMigrationPreparationUsesVerifiedTLSAndStopsAtFailedPrerequisite$/^success$")
	out, err := cmd.CombinedOutput()
	if err == nil || ctx.Err() != nil || !strings.Contains(string(out), "target TLS fixture requires exclusive loopback port 5432") || strings.Contains(string(out), "--- SKIP") {
		t.Fatalf("port collision silently skipped or used another target: err=%v output=%s", err, out)
	}
}
