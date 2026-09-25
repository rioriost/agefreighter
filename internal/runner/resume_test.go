package runner

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/rioriost/agefreighter/internal/config"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const resumedID = "44444444-4444-4444-8444-444444444444"

func resumeFixture(t *testing.T) (Manager, Request, *int) {
	t.Helper()
	m, r, starts := testManager(t)
	job, err := config.Parse(r.Configuration)
	if err != nil {
		t.Fatal(err)
	}
	job.Target.Connection = config.SecretRef{Env: "AGEFREIGHTER_TARGET_DSN"}
	r.Configuration, _ = json.Marshal(job)
	r.Action = "migrate-csv"
	r.Secrets = map[string]string{"AGEFREIGHTER_TARGET_DSN": testMigrationDSN()}
	s, err := m.Submit(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	s.Phase = "failed"
	if err := replaceJSON(filepath.Join(m.Root, r.Workflow, r.Operation, "state.json"), s); err != nil {
		t.Fatal(err)
	}
	m.workerInactive = func(context.Context, string) error { return nil }
	m.resumeProbe = func(_ context.Context, q Request) (ResumeInspection, error) {
		now := time.Now().UTC()
		return ResumeInspection{Version: 1, Workflow: q.Workflow, Operation: q.Operation, JobID: operationID, BootID: bootID, ConfigSHA256: s.ConfigSHA256, Fingerprint: strings.Repeat("a", 64), GenerationID: "17", CommittedRows: "1", CheckpointAt: now.Add(-time.Minute).Format(time.RFC3339Nano), CheckedAt: now.Format(time.RFC3339Nano)}, nil
	}
	m.resumeComplete = func(context.Context, State, []byte, string) error { return nil }
	r.Action = "resume-migration"
	r.Operation = resumedID
	r.Configuration = nil
	r.Resume = &ResumeBinding{PreviousOperation: operationID, JobID: operationID, ConfigSHA256: s.ConfigSHA256, Fingerprint: strings.Repeat("a", 64), GenerationID: "17", CommittedRows: "1"}
	return m, r, starts
}
func TestExplicitResumePreservesJobConfigurationAndEvidence(t *testing.T) {
	m, r, starts := resumeFixture(t)
	root := filepath.Join(m.Root, r.Workflow)
	old := filepath.Join(root, operationID)
	original, _ := os.ReadFile(filepath.Join(old, "job.json"))
	oldState, _ := os.ReadFile(filepath.Join(old, "state.json"))
	prepares := 0
	m.migrationPrepare = func(context.Context, []byte, string) error { prepares++; return errors.New("must never prepare") }
	s, err := m.SubmitResume(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	if s.Operation != resumedID || s.JobID != operationID || s.Resume.GenerationID != "17" || *starts != 2 {
		t.Fatal(s)
	}
	copied, _ := os.ReadFile(filepath.Join(root, resumedID, "job.json"))
	if string(copied) != string(original) {
		t.Fatal("configuration rewritten")
	}
	if _, err := m.SubmitResume(context.Background(), r); err == nil || *starts != 2 {
		t.Fatal("resume replayed")
	}
	other := r
	other.Operation = "55555555-5555-4555-8555-555555555555"
	if _, err := m.SubmitResume(context.Background(), other); err == nil {
		t.Fatal("parallel continuation admitted")
	}
	if err := m.Work(context.Background(), r.Workflow, r.Operation); err != nil {
		t.Fatal(err)
	}
	s, err = m.Status(r.Workflow, r.Operation)
	if err != nil || s.Phase != "finished" || s.JobID != operationID || prepares != 0 || s.Fingerprint != r.Resume.Fingerprint {
		t.Fatalf("resume failed: %+v %v", s, err)
	}
	after, _ := os.ReadFile(filepath.Join(old, "state.json"))
	if string(after) != string(oldState) {
		t.Fatal("previous evidence overwritten")
	}
	claim, _ := os.ReadFile(filepath.Join(old, "resume.claim"))
	if string(claim) != resumedID {
		t.Fatal("continuation identity lost")
	}
	if err := m.Work(context.Background(), r.Workflow, r.Operation); err == nil {
		t.Fatal("worker replayed")
	}
}
func TestResumeGatesPreventContinuationClaim(t *testing.T) {
	for _, kind := range []string{"active-worker", "boot", "checkpoint", "generation", "config", "job", "lease", "health", "configuration-input"} {
		t.Run(kind, func(t *testing.T) {
			m, r, starts := resumeFixture(t)
			switch kind {
			case "active-worker":
				m.workerInactive = func(context.Context, string) error { return errors.New("active") }
			case "boot":
				r.ExpectedBootID = workflowID
			case "checkpoint":
				r.Resume.CommittedRows = "2"
			case "generation":
				r.Resume.GenerationID = "18"
			case "config":
				r.Resume.ConfigSHA256 = strings.Repeat("d", 64)
			case "job":
				r.Resume.JobID = workflowID
			case "lease":
				if err := os.WriteFile(filepath.Join(m.Root, r.Workflow, "active"), []byte(workflowID), 0600); err != nil {
					t.Fatal(err)
				}
			case "health":
				m.healthProbe = func(context.Context) (*GuestHealth, error) {
					return &GuestHealth{Idle: false, StorageUsedPercent: 80}, nil
				}
			case "configuration-input":
				r.Configuration = json.RawMessage(`{}`)
			}
			if _, err := m.SubmitResume(context.Background(), r); err == nil || *starts != 1 {
				t.Fatal("unsafe continuation admitted")
			}
			if _, err := os.Stat(filepath.Join(m.Root, r.Workflow, operationID, "resume.claim")); !os.IsNotExist(err) {
				t.Fatal("failed gate reserved a continuation")
			}
		})
	}
}
func TestResumeLostStartAndPostLoadMismatchNeverPass(t *testing.T) {
	m, r, starts := resumeFixture(t)
	m.Start = func(context.Context, string) error { *starts++; return errors.New("lost") }
	s, err := m.SubmitResume(context.Background(), r)
	if err == nil || s.JobID != operationID {
		t.Fatal("lost reply not retained")
	}
	if _, err := m.SubmitResume(context.Background(), r); err == nil || *starts != 2 {
		t.Fatal("uncertain operation replayed")
	}
	m.resumeComplete = func(context.Context, State, []byte, string) error { return errors.New("generation changed") }
	if err := m.Work(context.Background(), r.Workflow, r.Operation); err != nil {
		t.Fatal(err)
	}
	s, _ = m.Status(r.Workflow, r.Operation)
	if s.Phase != "failed" || s.ReportBytes != 0 {
		t.Fatal("changed generation reported success")
	}
	if _, err := os.Stat(filepath.Join(m.Root, r.Workflow, r.Operation, "verify.json")); !os.IsNotExist(err) {
		t.Fatal("verification executed after failed generation check")
	}
}
func TestDispatchAdmissionLockExcludesConcurrentAdmissionAndDoesNotRemoveLease(t *testing.T) {
	m, r, _ := resumeFixture(t)
	unlock, err := m.dispatchLock()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.SubmitResume(context.Background(), r); err == nil {
		t.Fatal("admission lock ignored")
	}
	unlock()
	if _, err := os.Stat(filepath.Join(m.Root, r.Workflow, "active")); err != nil {
		t.Fatal("lease cleared")
	}
}
