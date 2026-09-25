package runner

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const catalogConfig = `{"schemaVersion":1,"host":"source.example","port":5432,"database":"source","username":"reader","schemas":["public"],"sourceCASHA256":""}`
const catalogConnection = "postgresql://reader:catalog-password@source.example:5432/source?sslmode=verify-full&connect_timeout=15"
const catalogReport = `{"schemaVersion":1,"command":"postgres-catalog","complete":true,"schemas":["public"],"tables":[]}`

func catalogManager(t *testing.T) (Manager, Request, *int) {
	m, r, starts := testManager(t)
	r.Action = "postgres-catalog"
	r.Configuration = []byte(catalogConfig)
	r.Secrets = map[string]string{"AGEFREIGHTER_SOURCE_DSN": catalogConnection}
	return m, r, starts
}

func TestCatalogWorkerSealsDistinctReportAndNeverReplays(t *testing.T) {
	m, r, starts := catalogManager(t)
	r.Secrets["AGEFREIGHTER_SOURCE_CA_PEM"] = testSourceCAPEM(t)
	r.Configuration = []byte(strings.Replace(catalogConfig, `"sourceCASHA256":""`, `"sourceCASHA256":"`+sum([]byte(r.Secrets["AGEFREIGHTER_SOURCE_CA_PEM"]))+`"`, 1))
	wire, _ := json.Marshal(r)
	decoded, err := Decode(strings.NewReader(string(wire)))
	if err != nil {
		t.Fatal(err)
	}
	s, err := m.Submit(t.Context(), decoded)
	if err != nil {
		t.Fatal(err)
	}
	if s.Action != "postgres-catalog" || s.ConfigSHA256 == "" || s.BootID != bootID || s.JobID != "" {
		t.Fatal(s)
	}
	if _, err := m.Submit(t.Context(), r); err == nil || *starts != 1 {
		t.Fatal("replayed dispatch")
	}
	if err := m.Work(t.Context(), r.Workflow, r.Operation); err != nil {
		t.Fatal(err)
	}
	s, err = m.Status(r.Workflow, r.Operation)
	if err != nil || s.Phase != "finished" || s.ExitCode == nil || *s.ExitCode != 0 {
		t.Fatal(s, err)
	}
	chunk, err := m.Report(r.Workflow, r.Operation, 0)
	if err != nil {
		t.Fatal(err)
	}
	data, err := base64.StdEncoding.DecodeString(chunk.Data)
	if err != nil || sum(data) != s.ReportSHA256 || int64(len(data)) != s.ReportBytes {
		t.Fatal("invalid seal")
	}
	if err := validateWorkerReport(r.Action, r.Configuration, data); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(m.Root, r.Workflow, r.Operation)
	for _, name := range []string{"secrets.json", "source-ca.pem"} {
		if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
			t.Fatal("retained credential", name)
		}
	}
	if err := m.Work(t.Context(), r.Workflow, r.Operation); err == nil {
		t.Fatal("worker replay")
	}
	if err := os.WriteFile(filepath.Join(dir, "report.json"), []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Report(r.Workflow, r.Operation, 0); err == nil {
		t.Fatal("changed report retrieved")
	}
}

func TestCatalogAdmissionFailsBeforeStarting(t *testing.T) {
	for _, kind := range []string{"host", "tls", "extra-secret", "ca-path", "ca-changed", "ca-missing", "boot", "busy", "storage", "swap", "oom"} {
		t.Run(kind, func(t *testing.T) {
			m, r, starts := catalogManager(t)
			switch kind {
			case "host":
				r.Secrets["AGEFREIGHTER_SOURCE_DSN"] = strings.Replace(catalogConnection, "source.example", "other.example", 1)
			case "tls":
				r.Secrets["AGEFREIGHTER_SOURCE_DSN"] = strings.Replace(catalogConnection, "verify-full", "disable", 1)
			case "extra-secret":
				r.Secrets["AGEFREIGHTER_TARGET_DSN"] = "not permitted"
			case "ca-path":
				r.Secrets["AGEFREIGHTER_SOURCE_DSN"] += "&sslrootcert=/unreviewed"
			case "ca-changed":
				r.Secrets["AGEFREIGHTER_SOURCE_CA_PEM"] = testSourceCAPEM(t)
			case "ca-missing":
				r.Configuration = []byte(strings.Replace(catalogConfig, `"sourceCASHA256":""`, `"sourceCASHA256":"`+strings.Repeat("a", 64)+`"`, 1))
			case "boot":
				r.ExpectedBootID = operationID
			default:
				probe := m.healthProbe
				m.healthProbe = func(ctx context.Context) (*GuestHealth, error) {
					h, e := probe(ctx)
					switch kind {
					case "busy":
						h.Idle = false
					case "storage":
						h.StorageUsedPercent = 80
					case "swap":
						h.SwapUsedBytes = 1
					case "oom":
						h.OOMEvents = 1
					}
					return h, e
				}
			}
			if _, err := m.Submit(t.Context(), r); err == nil || *starts != 0 {
				t.Fatal("unsafe submission")
			}
		})
	}
}

func TestCatalogFailedWorkerRetainsNoCompleteEvidence(t *testing.T) {
	for _, mode := range []string{"bad-scope", "failed"} {
		t.Run(mode, func(t *testing.T) {
			m, r, _ := catalogManager(t)
			if _, err := m.Submit(t.Context(), r); err != nil {
				t.Fatal(err)
			}
			dir := filepath.Join(m.Root, r.Workflow, r.Operation)
			if err := os.WriteFile(filepath.Join(dir, "catalog-test-mode"), []byte(mode), 0600); err != nil {
				t.Fatal(err)
			}
			if err := m.Work(t.Context(), r.Workflow, r.Operation); err != nil {
				t.Fatal(err)
			}
			s, err := m.Status(r.Workflow, r.Operation)
			if err != nil || s.Phase != "failed" {
				t.Fatal(s, err)
			}
			if mode == "bad-scope" && s.ReportBytes != 0 {
				t.Fatal("wrong source report sealed")
			}
			if _, err := os.Stat(filepath.Join(dir, "secrets.json")); !os.IsNotExist(err) {
				t.Fatal("secret retained")
			}
		})
	}
}
