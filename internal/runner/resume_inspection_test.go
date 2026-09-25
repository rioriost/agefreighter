package runner

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
)

func TestResumeInspectionProtocolIsTargetReadOnly(t *testing.T) {
	r := Request{Version: 1, Workflow: workflowID, Operation: operationID, Action: "inspect-resume", ExpectedBootID: bootID, Secrets: map[string]string{"AGEFREIGHTER_TARGET_DSN": testMigrationDSN()}}
	data, _ := json.Marshal(r)
	if _, err := Decode(strings.NewReader(string(data))); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*Request){
		func(r *Request) { r.Configuration = json.RawMessage(`{}`) },
		func(r *Request) { r.ExpectedBootID = "" },
		func(r *Request) { r.Offset = 1 },
		func(r *Request) { r.Secrets = map[string]string{"AGEFREIGHTER_SOURCE_PASSWORD": "secret"} },
		func(r *Request) {
			r.Secrets = map[string]string{"AGEFREIGHTER_TARGET_DSN": testMigrationDSN(), "AGEFREIGHTER_SOURCE_PASSWORD": "secret"}
		},
		func(r *Request) {
			r.Secrets = map[string]string{"AGEFREIGHTER_TARGET_DSN": strings.Replace(testMigrationDSN(), "verify-full", "disable", 1)}
		},
	} {
		bad := r
		change(&bad)
		data, _ = json.Marshal(bad)
		if _, err := Decode(strings.NewReader(string(data))); err == nil {
			t.Fatal("unsafe resume inspection accepted")
		}
	}
	if _, err := Arguments("inspect-resume", "job.json"); err == nil {
		t.Fatal("inspection admitted as an executable worker")
	}
}

func TestResumeInspectionBindsOriginalMappingAndGenerationWithoutAdmission(t *testing.T) {
	now := time.Now().UTC()
	s := State{Workflow: workflowID, Operation: operationID, JobID: operationID, ConfigSHA256: strings.Repeat("a", 64), Phase: "failed"}
	j := config.LoadJob{Source: config.Source{Type: config.SourceCSV}, Target: config.Target{Graph: "retained"}}
	stored := meta.Job{ID: operationID, TargetBackend: meta.TargetBackendApacheAGE, TargetGraph: "retained", SourceType: "csv", LoadMode: "create", GraphGenerationID: 9007199254740993, ConfigFingerprint: strings.Repeat("b", 64), Status: meta.JobFailed, CommittedRows: 5600000, UpdatedAt: now.Add(-time.Minute)}
	g := meta.GraphGeneration{ID: stored.GraphGenerationID, JobID: operationID, GraphName: "retained", State: meta.GenerationLoading}
	v := meta.JobVerification{JobID: operationID, SubmittedConfigFingerprint: s.ConfigSHA256, ResolvedMappingFingerprint: strings.Repeat("c", 64)}
	result, err := resumeInspectionEvidence(s, j, stored, g, v, bootID, now)
	if err != nil {
		t.Fatal(err)
	}
	if result.CanResume || result.Outcome != "review-required" || result.GenerationID != "9007199254740993" || result.CommittedRows != "5600000" || result.Fingerprint != stored.ConfigFingerprint {
		t.Fatalf("unsafe/lossy result: %+v", result)
	}
	for _, field := range []string{"job", "graph", "source", "backend", "mode", "generation", "owner", "generation-state", "submitted", "verification-job", "fingerprint", "rejects", "source-rejects", "future", "negative"} {
		t.Run(field, func(t *testing.T) {
			a, b, c := stored, g, v
			switch field {
			case "job":
				a.ID = workflowID
			case "graph":
				a.TargetGraph = "other"
			case "source":
				a.SourceType = "neo4j"
			case "backend":
				a.TargetBackend = meta.TargetBackendPostgreSQLPropertyGraph
			case "mode":
				a.LoadMode = "replace"
			case "generation":
				b.ID++
			case "owner":
				b.JobID = workflowID
			case "generation-state":
				b.State = meta.GenerationActive
			case "submitted":
				c.SubmittedConfigFingerprint = strings.Repeat("f", 64)
			case "verification-job":
				c.JobID = workflowID
			case "fingerprint":
				a.ConfigFingerprint = "bad"
			case "rejects":
				a.RejectedRows = 1
			case "source-rejects":
				a.SourceRejectedRows = 1
			case "future":
				a.UpdatedAt = now.Add(time.Minute)
			case "negative":
				a.CommittedRows = -1
			}
			if _, err := resumeInspectionEvidence(s, j, a, b, c, bootID, now); err == nil {
				t.Fatal("changed evidence accepted")
			}
		})
	}
	stored.Status = meta.JobRunning
	stored.UpdatedAt = now.Add(-time.Hour)
	result, err = resumeInspectionEvidence(s, j, stored, g, v, bootID, now)
	if err != nil || result.CanResume || len(result.Reasons) != 3 {
		t.Fatalf("stale/running checkpoint silently admitted: %+v %v", result, err)
	}
}

func TestResumeInspectionRejectsTamperedLocalEvidenceWithoutStartOrFiles(t *testing.T) {
	m, r, starts := testManager(t)
	m.healthProbe = func(context.Context) (*GuestHealth, error) {
		return &GuestHealth{Idle: true, StorageUsedPercent: 1}, nil
	}
	j, err := config.Parse(r.Configuration)
	if err != nil {
		t.Fatal(err)
	}
	j.Target.Connection = config.SecretRef{Env: "AGEFREIGHTER_TARGET_DSN"}
	r.Configuration, _ = json.Marshal(j)
	r.Action = "migrate-csv"
	r.Secrets = map[string]string{"AGEFREIGHTER_TARGET_DSN": testMigrationDSN()}
	if _, err := m.Submit(context.Background(), r); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(m.Root, r.Workflow, r.Operation)
	// Tamper only the disposable fixture, not retained qualification evidence.
	if err := os.WriteFile(filepath.Join(dir, "job.json"), []byte(`{}`), 0600); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	r.Action = "inspect-resume"
	r.Configuration = nil
	if _, err := m.InspectResume(context.Background(), r); err == nil {
		t.Fatal("tampered configuration accepted")
	}
	after, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if *starts != 1 || !reflect.DeepEqual(before, after) {
		t.Fatal("inspection changed operation or started a worker")
	}
	active, err := os.ReadFile(filepath.Join(m.Root, r.Workflow, "active"))
	if err != nil || string(active) != r.Operation {
		t.Fatal("inspection changed the lease")
	}
}
