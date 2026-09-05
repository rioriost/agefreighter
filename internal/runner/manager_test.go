package runner

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/app"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/report"
	"github.com/rioriost/agefreighter/internal/version"
)

const workflowID = "11111111-1111-4111-8111-111111111111"
const operationID = "22222222-2222-4222-8222-222222222222"
const bootID = "33333333-3333-4333-8333-333333333333"

// The child executes the real source profiler, not a stub success report. The
// systemd transport is injected separately so these tests run on macOS as well.
func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == "version" {
		_, _ = io.WriteString(os.Stdout, version.Current().String("agefreighter"))
		os.Exit(0)
	}
	if len(os.Args) > 2 && os.Args[1] == "profile" {
		doc, err := app.SourceProfile(context.Background(), os.Args[2], app.ProfileOptions{Mode: app.ProfileSample, SampleSize: 10000})
		if err != nil {
			os.Exit(1)
		}
		if err := json.NewEncoder(os.Stdout).Encode(doc); err != nil {
			os.Exit(1)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestReadinessRequiresBootstrapAndMatchingInstallation(t *testing.T) {
	m, _, _ := testManager(t)
	if _, err := m.Ready(context.Background()); err == nil {
		t.Fatal("unbootstrapped guest ready")
	}
	base := filepath.Dir(m.Root)
	if err := os.Mkdir(filepath.Join(base, "evidence"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := writeNew(filepath.Join(base, "bootstrap.complete"), nil); err != nil {
		t.Fatal(err)
	}
	if err := writeNew(filepath.Join(base, "evidence", "archive.sha256"), []byte(strings.Repeat("a", 64))); err != nil {
		t.Fatal(err)
	}
	r, err := m.Ready(context.Background())
	if err != nil || !r.Ready || r.ArchiveSHA256 != strings.Repeat("a", 64) {
		t.Fatalf("%#v %v", r, err)
	}
}

func testManager(t *testing.T) (Manager, Request, *int) {
	t.Helper()
	dir := t.TempDir()
	root := filepath.Join(dir, "workflows")
	units := filepath.Join(dir, "units")
	if err := os.Mkdir(units, 0700); err != nil {
		t.Fatal(err)
	}
	uploads := filepath.Join(root, workflowID, "uploads")
	if err := os.MkdirAll(uploads, 0700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(uploads, "people.csv")
	if err := os.WriteFile(path, []byte("id,name\na,Ada\nb,Grace\n"), 0600); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("../config/testdata/valid/csv.yaml")
	if err != nil {
		t.Fatal(err)
	}
	job, err := config.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	job.Source.CSV.Vertices = []config.CSVVertex{{Label: "Person", Path: path, IDColumn: "id", Properties: map[string]string{"name": "name"}}}
	job.Source.CSV.Edges = nil
	data, err = json.Marshal(job)
	if err != nil {
		t.Fatal(err)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	starts := new(int)
	manager := Manager{Root: root, UnitDirectory: units, CLI: executable, Tools: "/usr/local/bin/agefreighter-tools", BootID: func() (string, error) { return bootID, nil }, Start: func(context.Context, string) error { *starts++; return nil }}
	return manager, Request{Version: 1, Workflow: workflowID, Operation: operationID, Action: "profile", ExpectedBootID: bootID, Configuration: data, Secrets: map[string]string{"AGEFREIGHTER_SOURCE_PASSWORD": "private-secret"}}, starts
}

func TestDurableAssessmentRunsRealCSVProfileAndRetrievesAllReportChunks(t *testing.T) {
	m, request, starts := testManager(t)
	state, err := m.Submit(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if state.Phase != "accepted" || *starts != 1 {
		t.Fatal(state)
	}
	if _, err := m.Submit(context.Background(), request); err == nil || *starts != 1 {
		t.Fatal("duplicate dispatch executed")
	}
	unit, err := os.ReadFile(filepath.Join(m.UnitDirectory, unitName(operationID)))
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{"MemoryMax=4G", "MemorySwapMax=0", "Restart=no", "RuntimeMaxSec=1800"} {
		if !strings.Contains(string(unit), required) {
			t.Fatal("unit lost guard", required)
		}
	}
	if strings.Contains(string(unit), "[Install]") || strings.Contains(string(unit), "private-secret") {
		t.Fatal("unit enables boot replay or contains credentials")
	}
	if err := m.Work(context.Background(), workflowID, operationID); err != nil {
		t.Fatal(err)
	}
	state, err = m.Status(workflowID, operationID)
	if err != nil {
		t.Fatal(err)
	}
	if state.Phase != "finished" || state.ExitCode == nil || *state.ExitCode != 0 || state.ReportBytes == 0 {
		t.Fatalf("bad result %#v", state)
	}
	var data []byte
	for offset := int64(0); offset < state.ReportBytes; {
		chunk, err := m.Report(workflowID, operationID, offset)
		if err != nil {
			t.Fatal(err)
		}
		wire, err := json.Marshal(chunk)
		if err != nil || len(wire) >= 4096 {
			t.Fatal("chunk exceeds ARM output bound")
		}
		part, err := base64.StdEncoding.DecodeString(chunk.Data)
		if err != nil || len(part) == 0 {
			t.Fatal("invalid chunk")
		}
		data = append(data, part...)
		offset += int64(len(part))
	}
	if sum(data) != state.ReportSHA256 || strings.Contains(string(data), "private-secret") {
		t.Fatal("report integrity/redaction failed")
	}
	doc, err := report.Decode(data)
	if err != nil || doc.Command != "profile" {
		t.Fatalf("not a real profile: %v", err)
	}
	if _, err := os.Stat(filepath.Join(m.Root, workflowID, operationID, "secrets.json")); !os.IsNotExist(err) {
		t.Fatal("transient secrets retained after completion")
	}
	if err := m.Work(context.Background(), workflowID, operationID); err == nil {
		t.Fatal("worker replay accepted")
	}
	if _, err := m.Report(workflowID, operationID, state.ReportBytes); err == nil {
		t.Fatal("out-of-range artifact read")
	}
}

func TestGuestRebootAndConfigurationTamperingNeverResume(t *testing.T) {
	for _, reboot := range []bool{true, false} {
		m, request, _ := testManager(t)
		if _, err := m.Submit(context.Background(), request); err != nil {
			t.Fatal(err)
		}
		if reboot {
			m.BootID = func() (string, error) { return "second-boot", nil }
		} else {
			if err := os.WriteFile(filepath.Join(m.Root, workflowID, operationID, "job.json"), []byte("tampered"), 0600); err != nil {
				t.Fatal(err)
			}
		}
		if err := m.Work(context.Background(), workflowID, operationID); err == nil {
			t.Fatal("replayed or changed configuration accepted")
		}
		if _, err := m.Report(workflowID, operationID, 0); err == nil {
			t.Fatal("unfinished artifact exposed")
		}
	}
}

func TestProtocolRejectsExecutionInjectionAndEscapingCSV(t *testing.T) {
	m, request, _ := testManager(t)
	for _, action := range []string{"load", "resume", "cleanup", "bash"} {
		request.Action = action
		data, _ := json.Marshal(request)
		if _, err := Decode(strings.NewReader(string(data))); err == nil {
			t.Fatal("action accepted", action)
		}
	}
	request.Action = "profile"
	request.Secrets["LD_PRELOAD"] = "/tmp/evil.so"
	if _, err := m.Submit(context.Background(), request); err == nil {
		t.Fatal("environment injection")
	}
	delete(request.Secrets, "LD_PRELOAD")
	var job config.LoadJob
	if err := json.Unmarshal(request.Configuration, &job); err != nil {
		t.Fatal(err)
	}
	job.Source.CSV.Vertices[0].Path = "/etc/passwd"
	request.Configuration, _ = json.Marshal(job)
	if _, err := m.Submit(context.Background(), request); err == nil {
		t.Fatal("file escape")
	}
	if _, err := Decode(io.LimitReader(strings.NewReader(strings.Repeat("x", MaxRequestBytes+2)), MaxRequestBytes+2)); err == nil {
		t.Fatal("oversized request")
	}
	if _, err := Decode(strings.NewReader(`{"version":1,"workflow":"` + workflowID + `","operation":"` + operationID + `","action":"status","unexpected":true}`)); err == nil {
		t.Fatal("unknown field")
	}
}

func TestRedactionAndOutputBounds(t *testing.T) {
	data, err := redactedReport([]byte(`{"nested":[{"value":"before secret after"}],"number":9223372036854775807}`), map[string]string{"password": "secret"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "secret") || !strings.Contains(string(data), "9223372036854775807") {
		t.Fatal(string(data))
	}
	b := boundedOutput{limit: 3}
	if n, err := b.Write([]byte("abcdef")); err != nil || n != 6 || !b.overflow || b.String() != "abc" {
		t.Fatal("unbounded output")
	}
}

func TestChangedBootRejectsSubmitBeforeCreatingOperation(t *testing.T) {
	m, request, starts := testManager(t)
	request.ExpectedBootID = operationID
	if _, err := m.Submit(context.Background(), request); err == nil || *starts != 0 {
		t.Fatal("stale readiness accepted")
	}
	if _, err := os.Stat(filepath.Join(m.Root, workflowID, operationID)); !os.IsNotExist(err) {
		t.Fatal("stale readiness created operation state")
	}
}

func TestUncertainStartRetainsIntentAndBlocksOtherOperations(t *testing.T) {
	m, request, _ := testManager(t)
	starts := 0
	m.Start = func(context.Context, string) error { starts++; return errors.New("lost response") }
	state, err := m.Submit(context.Background(), request)
	if err == nil || state.Phase != "accepted" || starts != 1 {
		t.Fatal("uncertain start lost intent")
	}
	if _, err := m.Status(workflowID, operationID); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Submit(context.Background(), request); err == nil {
		t.Fatal("replayed uncertain start")
	}
	request.Operation = "44444444-4444-4444-8444-444444444444"
	if _, err := m.Submit(context.Background(), request); err == nil || starts != 1 {
		t.Fatal("new operation bypassed unreconciled lease")
	}
}

func TestSymlinkUploadAndChangedReportAreRejected(t *testing.T) {
	t.Run("symlink", func(t *testing.T) {
		m, request, _ := testManager(t)
		var job config.LoadJob
		if err := json.Unmarshal(request.Configuration, &job); err != nil {
			t.Fatal(err)
		}
		link := filepath.Join(m.Root, workflowID, "uploads", "escape.csv")
		if err := os.Symlink(filepath.Join(m.UnitDirectory, "outside.csv"), link); err != nil {
			t.Fatal(err)
		}
		if err := writeNew(filepath.Join(m.UnitDirectory, "outside.csv"), []byte("id,name\na,Ada\n")); err != nil {
			t.Fatal(err)
		}
		job.Source.CSV.Vertices[0].Path = link
		request.Configuration, _ = json.Marshal(job)
		if _, err := m.Submit(context.Background(), request); err != nil {
			t.Fatal(err)
		}
		if err := m.Work(context.Background(), workflowID, operationID); err == nil {
			t.Fatal("symlink escaped workflow")
		}
	})
	t.Run("report", func(t *testing.T) {
		m, request, _ := testManager(t)
		if _, err := m.Submit(context.Background(), request); err != nil {
			t.Fatal(err)
		}
		if err := m.Work(context.Background(), workflowID, operationID); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(m.Root, workflowID, operationID, "report.json"), []byte("changed"), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := m.Report(workflowID, operationID, 0); err == nil {
			t.Fatal("tampered report returned")
		}
	})
}

func TestGeneratedUnitPassesSystemdValidation(t *testing.T) {
	validator, err := exec.LookPath("systemd-analyze")
	if err != nil {
		t.Skip("systemd validator is not installed on this host")
	}
	m, _, _ := testManager(t)
	// This test executable is available in the validation environment; production
	// uses the fixed installed tools path. No service is started by this check.
	m.Tools = m.CLI
	unit, err := m.unit(workflowID, operationID, filepath.Join(m.Root, workflowID))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(m.UnitDirectory, unitName(operationID))
	if err := writeNew(path, []byte(unit)); err != nil {
		t.Fatal(err)
	}
	output, err := exec.Command(validator, "verify", path).CombinedOutput()
	if err != nil {
		t.Fatalf("invalid systemd service: %v: %s", err, output)
	}
}
