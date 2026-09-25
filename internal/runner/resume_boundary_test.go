package runner

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestResumeAdmissionRetainsFailedContinuationsWithoutReplay(t *testing.T) {
	for _, mode := range []string{"malformed-json", "private-previous", "lineage", "missing-job", "tampered-job", "unreviewed-secrets", "busy-without-lease", "probe", "existing-claim", "existing-continuation", "unsafe-unit", "unit-collision"} {
		t.Run(mode, func(t *testing.T) {
			m, r, starts := resumeFixture(t)
			root := filepath.Join(m.Root, r.Workflow)
			previous := filepath.Join(root, operationID)
			jobPath := filepath.Join(previous, "job.json")
			switch mode {
			case "malformed-json":
				r.Configuration = json.RawMessage(`{`)
			case "private-previous":
				if err := os.Chmod(previous, 0755); err != nil {
					t.Fatal(err)
				}
			case "lineage":
				s, err := m.Status(r.Workflow, operationID)
				if err != nil {
					t.Fatal(err)
				}
				s.Action = "resume-migration"
				if err := replaceJSON(filepath.Join(previous, "state.json"), s); err != nil {
					t.Fatal(err)
				}
			case "missing-job":
				if err := os.Remove(jobPath); err != nil {
					t.Fatal(err)
				}
			case "tampered-job":
				if err := os.WriteFile(jobPath, []byte(`{}`), 0600); err != nil {
					t.Fatal(err)
				}
			case "unreviewed-secrets":
				r.Secrets["LD_PRELOAD"] = "private"
			case "busy-without-lease":
				if err := os.Remove(filepath.Join(root, "active")); err != nil {
					t.Fatal(err)
				}
				m.healthProbe = func(context.Context) (*GuestHealth, error) { return &GuestHealth{Idle: false}, nil }
			case "probe":
				m.resumeProbe = func(context.Context, Request) (ResumeInspection, error) {
					return ResumeInspection{}, errors.New("target unavailable")
				}
			case "existing-claim":
				if err := writeNew(filepath.Join(previous, "resume.claim"), []byte(resumedID)); err != nil {
					t.Fatal(err)
				}
			case "existing-continuation":
				if err := os.Mkdir(filepath.Join(root, resumedID), 0700); err != nil {
					t.Fatal(err)
				}
			case "unsafe-unit":
				m.Tools = "/not safe"
			case "unit-collision":
				if err := writeNew(filepath.Join(m.UnitDirectory, unitName(resumedID)), []byte("retained-unit")); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := m.SubmitResume(t.Context(), r); err == nil || *starts != 1 {
				t.Fatal("unsafe continuation started")
			}
			claim, err := os.ReadFile(filepath.Join(previous, "resume.claim"))
			claimed := mode == "existing-claim" || mode == "existing-continuation" || mode == "unsafe-unit" || mode == "unit-collision"
			if claimed {
				if err != nil || string(claim) != resumedID {
					t.Fatal("partially persisted continuation lost its claim")
				}
				if _, err := m.SubmitResume(t.Context(), r); err == nil || *starts != 1 {
					t.Fatal("partially persisted continuation replayed")
				}
			} else if !os.IsNotExist(err) {
				t.Fatalf("failed admission reserved continuation: %v", err)
			}
		})
	}
}

func TestExplicitResumeFromReleasedLeaseRequiresFreshMatchingEvidence(t *testing.T) {
	m, r, starts := resumeFixture(t)
	root := filepath.Join(m.Root, r.Workflow)
	if err := os.Remove(filepath.Join(root, "active")); err != nil {
		t.Fatal(err)
	}
	if _, err := m.SubmitResume(t.Context(), r); err != nil {
		t.Fatal(err)
	}
	lease, err := os.ReadFile(filepath.Join(root, "active"))
	if err != nil || string(lease) != resumedID || *starts != 2 {
		t.Fatal("continuation did not exclusively acquire released lease")
	}
}

func TestResumeTargetRequiresFreshExactCheckpointEvidence(t *testing.T) {
	m, r, _ := resumeFixture(t)
	probe := m.resumeProbe
	for _, mode := range []string{"checked-malformed", "checkpoint-malformed", "future", "stale", "checkpoint-stale", "workflow", "operation", "boot", "fingerprint"} {
		t.Run(mode, func(t *testing.T) {
			m.resumeProbe = func(ctx context.Context, q Request) (ResumeInspection, error) {
				if q.Action != "inspect-resume" || q.Operation != r.Resume.PreviousOperation || len(q.Secrets) != 1 || q.Secrets["AGEFREIGHTER_TARGET_DSN"] != r.Secrets["AGEFREIGHTER_TARGET_DSN"] {
					t.Fatal("recovery probe was not target-only and predecessor-bound")
				}
				e, err := probe(ctx, q)
				switch mode {
				case "checked-malformed":
					e.CheckedAt = "invalid"
				case "checkpoint-malformed":
					e.CheckpointAt = "invalid"
				case "future":
					e.CheckedAt = time.Now().Add(time.Minute).Format(time.RFC3339Nano)
				case "stale":
					e.CheckedAt = time.Now().Add(-2 * time.Minute).Format(time.RFC3339Nano)
				case "checkpoint-stale":
					e.CheckpointAt = time.Now().Add(-16 * time.Minute).Format(time.RFC3339Nano)
				case "workflow":
					e.Workflow = operationID
				case "operation":
					e.Operation = resumedID
				case "boot":
					e.BootID = operationID
				case "fingerprint":
					e.Fingerprint = strings.Repeat("f", 64)
				}
				return e, err
			}
			if err := m.checkResumeTarget(t.Context(), r); err == nil {
				t.Fatal("changed or expired checkpoint admitted")
			}
		})
	}
}

func TestResumeInspectionLocalFailuresDoNotCreateClaimsOrStartWorkers(t *testing.T) {
	for _, mode := range []string{"malformed-json", "action", "private-directory", "identity", "boot", "missing-configuration", "unsupported-target"} {
		t.Run(mode, func(t *testing.T) {
			m, r, starts := resumeFixture(t)
			r.Action, r.Operation, r.Resume = "inspect-resume", operationID, nil
			dir := filepath.Join(m.Root, r.Workflow, r.Operation)
			switch mode {
			case "malformed-json":
				r.Configuration = json.RawMessage(`{`)
			case "action":
				r.Action = "status"
			case "private-directory":
				if err := os.Chmod(dir, 0755); err != nil {
					t.Fatal(err)
				}
			case "identity":
				s, err := m.Status(r.Workflow, r.Operation)
				if err != nil {
					t.Fatal(err)
				}
				s.JobID = "invalid"
				if err := replaceJSON(filepath.Join(dir, "state.json"), s); err != nil {
					t.Fatal(err)
				}
			case "boot":
				r.ExpectedBootID = resumedID
			case "missing-configuration":
				if err := os.Remove(filepath.Join(dir, "job.json")); err != nil {
					t.Fatal(err)
				}
			case "unsupported-target":
				data := []byte(`{}`)
				if err := os.WriteFile(filepath.Join(dir, "job.json"), data, 0600); err != nil {
					t.Fatal(err)
				}
				s, err := m.Status(r.Workflow, r.Operation)
				if err != nil {
					t.Fatal(err)
				}
				s.ConfigSHA256 = sum(data)
				if err := replaceJSON(filepath.Join(dir, "state.json"), s); err != nil {
					t.Fatal(err)
				}
			}

			if _, err := m.InspectResume(t.Context(), r); err == nil || *starts != 1 {
				t.Fatal("invalid local recovery evidence admitted")
			}
			for _, name := range []string{"worker.claim", "resume.claim"} {
				if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
					t.Fatalf("inspection created %s", name)
				}
			}
		})
	}
}

func TestCancelledInactivityProbeCannotAuthorizeResumeOrInvokeSystemd(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	// exec.CommandContext rejects an already-cancelled context before starting
	// the fixed systemctl executable, including on a privileged Linux host.
	if err := (Manager{}).checkInactive(ctx, operationID); err == nil || err.Error() != "previous worker inactivity is unproven" {
		t.Fatalf("cancelled inactivity evidence admitted: %v", err)
	}
}
