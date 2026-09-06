package runner

import (
	"context"
	"encoding/json"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
)

func testMigrationDSN() string {
	u := &url.URL{Scheme: "postgresql", Host: "afpg-test.postgres.database.azure.com:5432", Path: "/agefreighter", User: url.UserPassword("afadmin", strings.Repeat("a", 32)), RawQuery: "sslmode=verify-full"}
	return u.String()
}
func TestMigrationConnectionIsTLSAndSinglePrivateTargetBound(t *testing.T) {
	dsn := testMigrationDSN()
	if _, err := migrationConnection(dsn); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{strings.Replace(dsn, "verify-full", "require", 1), dsn + "&sslmode=disable", dsn + "&options=x", strings.Replace(dsn, "afpg-test.postgres.database.azure.com", "localhost", 1), strings.Replace(dsn, "afadmin", "other", 1), strings.Replace(dsn, "/agefreighter", "/postgres", 1), strings.Replace(dsn, ":5432", ":5433", 1), dsn + "#fragment"} {
		if _, err := migrationConnection(bad); err == nil {
			t.Fatal("unsafe migration connection accepted")
		}
	}
}
func TestMigrationSubmissionSealsJobIDAndNeverReplays(t *testing.T) {
	m, request, starts := testManager(t)
	m.healthProbe = func(context.Context) (*GuestHealth, error) {
		return &GuestHealth{Idle: true, StorageUsedPercent: 6}, nil
	}
	request.Action = "migrate-csv"
	request.Secrets = map[string]string{"AGEFREIGHTER_TARGET_DSN": testMigrationDSN()}
	var job map[string]any
	if err := json.Unmarshal(request.Configuration, &job); err != nil {
		t.Fatal(err)
	}
	job["target"].(map[string]any)["connection"] = map[string]string{"env": "AGEFREIGHTER_TARGET_DSN"}
	request.Configuration, _ = json.Marshal(job)
	encoded, _ := json.Marshal(request)
	if _, err := Decode(strings.NewReader(string(encoded))); err != nil {
		t.Fatal(err)
	}
	state, err := m.Submit(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if state.JobID != request.Operation || state.Phase != "accepted" || *starts != 1 {
		t.Fatal(state)
	}
	if _, err := m.Submit(context.Background(), request); err == nil || *starts != 1 {
		t.Fatal("migration replayed")
	}
	data, err := os.ReadFile(filepath.Join(m.Root, request.Workflow, request.Operation, "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "postgresql:") || !strings.Contains(string(data), request.Operation) {
		t.Fatal("unsafe or missing durable identity")
	}
	job["target"].(map[string]any)["mode"] = "replace"
	request.Configuration, _ = json.Marshal(job)
	if _, err := ValidateConfiguration(request, filepath.Join(m.Root, request.Workflow)); err == nil {
		t.Fatal("remote replacement accepted")
	}
}

func TestMigrationWorkerFixedSequenceAndFailureNeverAutoResume(t *testing.T) {
	for _, fail := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "load-failed"}[fail], func(t *testing.T) {
			m, req, _ := testManager(t)
			m.healthProbe = func(context.Context) (*GuestHealth, error) {
				return &GuestHealth{Idle: true, StorageUsedPercent: 6}, nil
			}
			prepares := 0
			m.migrationPrepare = func(context.Context, []byte, string) error { prepares++; return nil }
			req.Action = "migrate-csv"
			req.Secrets = map[string]string{"AGEFREIGHTER_TARGET_DSN": testMigrationDSN()}
			var config map[string]any
			_ = json.Unmarshal(req.Configuration, &config)
			config["target"].(map[string]any)["connection"] = map[string]string{"env": "AGEFREIGHTER_TARGET_DSN"}
			req.Configuration, _ = json.Marshal(config)
			if _, err := m.Submit(context.Background(), req); err != nil {
				t.Fatal(err)
			}
			dir := filepath.Join(m.Root, req.Workflow, req.Operation)
			if fail {
				if err := writeNew(filepath.Join(dir, "fail-load"), nil); err != nil {
					t.Fatal(err)
				}
			}
			if err := m.Work(context.Background(), req.Workflow, req.Operation); err != nil {
				t.Fatal(err)
			}
			s, err := m.Status(req.Workflow, req.Operation)
			if err != nil {
				t.Fatal(err)
			}
			if s.JobID != req.Operation || s.ExitCode == nil || (s.Phase == "finished") == fail {
				t.Fatalf("unexpected state: %+v", s)
			}
			if !fail && (s.Fingerprint != strings.Repeat("a", 64) || s.ReportBytes == 0 || s.ReportSHA256 == "") {
				t.Fatal("verification identity not sealed")
			}
			if fail {
				if _, err := os.Stat(filepath.Join(dir, "verify.json")); !os.IsNotExist(err) {
					t.Fatal("verification started after load failure")
				}
			}
			if _, err := os.Stat(filepath.Join(dir, "secrets.json")); !os.IsNotExist(err) {
				t.Fatal("transient credentials retained")
			}
			if err := m.Work(context.Background(), req.Workflow, req.Operation); err == nil || prepares != 1 {
				t.Fatal("worker replayed")
			}
		})
	}
}

func TestMigrationSafetyGatesBlockBeforeLeaseOrStart(t *testing.T) {
	for _, h := range []*GuestHealth{nil, {Idle: false}, {Idle: true, StorageUsedPercent: 80}, {Idle: true, SwapUsedBytes: 1}, {Idle: true, OOMEvents: 1}} {
		m, req, starts := testManager(t)
		m.healthProbe = func(context.Context) (*GuestHealth, error) { return h, nil }
		req.Action = "migrate-csv"
		req.Secrets = map[string]string{"AGEFREIGHTER_TARGET_DSN": testMigrationDSN()}
		var c map[string]any
		_ = json.Unmarshal(req.Configuration, &c)
		c["target"].(map[string]any)["connection"] = map[string]string{"env": "AGEFREIGHTER_TARGET_DSN"}
		req.Configuration, _ = json.Marshal(c)
		if _, err := m.Submit(context.Background(), req); err == nil || *starts != 0 {
			t.Fatal("unsafe guest started migration")
		}
	}
}

func TestNeo4jMigrationRequiresReviewedSourceAndBothProtectedCredentials(t *testing.T) {
	m, request, _ := testManager(t)
	data, err := os.ReadFile("../config/testdata/valid/neo4j-discovery.yaml")
	if err != nil {
		t.Fatal(err)
	}
	job, err := config.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	job.Source.Neo4j.Password = &config.SecretRef{Env: "AGEFREIGHTER_SOURCE_PASSWORD"}
	job.Target.Connection = config.SecretRef{Env: "AGEFREIGHTER_TARGET_DSN"}
	job.Target.Mode = config.LoadCreate
	request.Action = "migrate-source"
	request.Configuration, _ = json.Marshal(job)
	request.Secrets = map[string]string{
		"AGEFREIGHTER_SOURCE_PASSWORD": "source-secret",
		"AGEFREIGHTER_TARGET_DSN":      testMigrationDSN(),
	}
	if _, err := ValidateConfiguration(request, filepath.Join(m.Root, request.Workflow)); err != nil {
		t.Fatal(err)
	}
	delete(request.Secrets, "AGEFREIGHTER_SOURCE_PASSWORD")
	if _, err := ValidateConfiguration(request, filepath.Join(m.Root, request.Workflow)); err == nil {
		t.Fatal("missing Neo4j credential accepted")
	}
	request.Secrets["AGEFREIGHTER_SOURCE_PASSWORD"] = "source-secret"
	request.Secrets["AGEFREIGHTER_SOURCE_DSN"] = "unexpected"
	if _, err := ValidateConfiguration(request, filepath.Join(m.Root, request.Workflow)); err == nil {
		t.Fatal("unexpected credential accepted")
	}
	delete(request.Secrets, "AGEFREIGHTER_SOURCE_DSN")
	job.Source.Type = config.SourceCSV
	request.Configuration, _ = json.Marshal(job)
	if _, err := ValidateConfiguration(request, filepath.Join(m.Root, request.Workflow)); err == nil {
		t.Fatal("non-Neo4j network migration accepted")
	}
}
