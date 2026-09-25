package runner

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestDurableJSONFailureNeverReplacesPriorEvidence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "state.json")
	prior := []byte(`{"phase":"accepted"}`)
	if err := writeNew(path, prior); err != nil {
		t.Fatal(err)
	}
	for _, run := range []func() error{
		func() error { return writeNewJSON(path, make(chan int)) },
		func() error { return replaceJSON(path, make(chan int)) },
		func() error { return replaceJSON(filepath.Join(dir, "missing", "state.json"), true) },
		func() error { return replaceJSON(dir, true) },
		func() error { return replaceBytes(filepath.Join(dir, "missing", "active"), nil) },
		func() error { return replaceBytes(dir, []byte("replacement")) },
		func() error { return syncDirectory(filepath.Join(dir, "missing")) },
	} {
		if err := run(); err == nil {
			t.Fatal("failed persistence reported success")
		}
		data, err := os.ReadFile(path)
		if err != nil || string(data) != string(prior) {
			t.Fatalf("retained evidence changed: %q %v", data, err)
		}
	}
	files, err := os.ReadDir(dir)
	if err != nil || len(files) != 1 || files[0].Name() != "state.json" {
		t.Fatalf("failed replacements left unpublished files: %+v %v", files, err)
	}
	if err := os.WriteFile(path, []byte(strings.Repeat(" ", MaxRequestBytes+1)), 0600); err != nil {
		t.Fatal(err)
	}
	if err := readJSON(path, new(State)); err == nil || !strings.Contains(err.Error(), "bound") {
		t.Fatalf("oversized evidence accepted: %v", err)
	}
}

func TestStatusRejectsMissingCorruptAndMismatchedEvidence(t *testing.T) {
	m, r, _ := testManager(t)
	if _, err := m.Status("../escape", r.Operation); err == nil {
		t.Fatal("invalid identity accepted")
	}
	if err := m.Work(t.Context(), r.Workflow, r.Operation); err == nil {
		t.Fatal("worker without intent accepted")
	}
	if _, err := m.Report(r.Workflow, r.Operation, 0); err == nil {
		t.Fatal("report without retained operation accepted")
	}
	if err := m.Work(t.Context(), r.Workflow, "invalid"); err == nil {
		t.Fatal("worker with invalid identity accepted")
	}
	if _, err := m.Submit(t.Context(), r); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(m.Root, r.Workflow, r.Operation, "state.json")
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, data := range [][]byte{[]byte(`{`), []byte(`{"version":1,"workflow":"wrong","operation":"` + r.Operation + `"}`)} {
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := m.Status(r.Workflow, r.Operation); err == nil {
			t.Fatal("corrupt/mismatched intent accepted")
		}
	}
	if err := os.WriteFile(path, original, 0600); err != nil {
		t.Fatal(err)
	}
	m.BootID = func() (string, error) { return "", errors.New("boot probe failed") }
	if _, err := m.Status(r.Workflow, r.Operation); err == nil {
		t.Fatal("missing boot evidence accepted")
	}
	after, err := os.ReadFile(path)
	if err != nil || string(after) != string(original) {
		t.Fatal("status inspection changed intent")
	}
}

func TestSubmissionFailuresPreserveAdmissionAndNeverStart(t *testing.T) {
	for _, mode := range []string{"action", "identity", "boot-probe", "private-root", "unsafe-unit", "unit-collision"} {
		t.Run(mode, func(t *testing.T) {
			m, r, starts := testManager(t)
			root := filepath.Join(m.Root, r.Workflow)
			switch mode {
			case "action":
				r.Action = "status"
			case "identity":
				r.Operation = "../outside"
			case "boot-probe":
				m.BootID = func() (string, error) { return "", errors.New("private detail") }
			case "private-root":
				if err := os.Chmod(root, 0755); err != nil {
					t.Fatal(err)
				}
			case "unsafe-unit":
				m.Tools = "/not a safe executable"
			case "unit-collision":
				if err := writeNew(filepath.Join(m.UnitDirectory, unitName(r.Operation)), []byte("retained-unit")); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := m.Submit(t.Context(), r); err == nil || *starts != 0 || strings.Contains(err.Error(), "private detail") {
				t.Fatalf("unsafe admission: starts=%d err=%v", *starts, err)
			}
			if mode == "unsafe-unit" || mode == "unit-collision" {
				lease, err := os.ReadFile(filepath.Join(root, "active"))
				if err != nil || string(lease) != r.Operation {
					t.Fatal("sealed admission lost its lease")
				}
				if _, err := m.Submit(t.Context(), r); err == nil || *starts != 0 {
					t.Fatal("partially persisted admission replayed")
				}
			}
		})
	}
}

func TestClaimedAssessmentFailuresRemoveSecretsButRetainClaim(t *testing.T) {
	for _, mode := range []string{"missing-secrets", "invalid-secrets", "invalid-action", "stderr-collision", "report-collision", "lease-changed", "missing-cli"} {
		t.Run(mode, func(t *testing.T) {
			m, r, starts := testManager(t)
			state, err := m.Submit(t.Context(), r)
			if err != nil {
				t.Fatal(err)
			}
			dir := filepath.Join(m.Root, r.Workflow, r.Operation)
			secrets := filepath.Join(dir, "secrets.json")
			switch mode {
			case "missing-secrets":
				if err := os.Remove(secrets); err != nil {
					t.Fatal(err)
				}
			case "invalid-secrets":
				if err := os.WriteFile(secrets, []byte(`{"LD_PRELOAD":"private"}`), 0600); err != nil {
					t.Fatal(err)
				}
			case "invalid-action":
				state.Action = "unreviewed-command"
				if err := replaceJSON(filepath.Join(dir, "state.json"), state); err != nil {
					t.Fatal(err)
				}
			case "stderr-collision", "report-collision":
				name := "stderr.log"
				if mode == "report-collision" {
					name = "report.json"
				}
				if err := writeNew(filepath.Join(dir, name), []byte("retained-evidence")); err != nil {
					t.Fatal(err)
				}
			case "lease-changed":
				if err := os.WriteFile(filepath.Join(m.Root, r.Workflow, "active"), []byte(resumedID), 0600); err != nil {
					t.Fatal(err)
				}
			case "missing-cli":
				m.CLI = filepath.Join(dir, "missing-cli")
			}
			err = m.Work(t.Context(), r.Workflow, r.Operation)
			if (err == nil) != (mode == "missing-cli") {
				t.Fatalf("worker result for %s: %v", mode, err)
			}
			if _, err := os.Stat(secrets); !os.IsNotExist(err) {
				t.Fatal("claimed failure retained secret transport")
			}
			if _, err := os.Stat(filepath.Join(dir, "worker.claim")); err != nil {
				t.Fatal("claimed failure lost no-replay evidence")
			}
			if err := m.Work(t.Context(), r.Workflow, r.Operation); err == nil || *starts != 1 {
				t.Fatal("failed operation replayed")
			}
			if mode == "lease-changed" {
				lease, err := os.ReadFile(filepath.Join(m.Root, r.Workflow, "active"))
				if err != nil || string(lease) != resumedID {
					t.Fatal("worker removed another operation's lease")
				}
			}
			if mode == "missing-cli" {
				got, err := m.Status(r.Workflow, r.Operation)
				if err != nil || got.Phase != "failed" || got.ExitCode == nil || *got.ExitCode != 1 || got.ReportBytes != 0 {
					t.Fatalf("missing executable not durably failed: %+v %v", got, err)
				}
			}
		})
	}
}

func TestReportRequiresMatchingBytesDigestAndRegularFile(t *testing.T) {
	for _, mode := range []string{"same-size-tamper", "symlink", "missing"} {
		t.Run(mode, func(t *testing.T) {
			m, r, _ := testManager(t)
			state, err := m.Submit(t.Context(), r)
			if err != nil {
				t.Fatal(err)
			}
			dir := filepath.Join(m.Root, r.Workflow, r.Operation)
			path := filepath.Join(dir, "report.json")
			data := []byte(`{"evidence":"original"}`)
			state.Phase, state.ReportBytes, state.ReportSHA256 = "failed", int64(len(data)), sum(data)
			if err := replaceJSON(filepath.Join(dir, "state.json"), state); err != nil {
				t.Fatal(err)
			}
			switch mode {
			case "same-size-tamper":
				if err := writeNew(path, []byte(strings.ReplaceAll(string(data), "original", "modified"))); err != nil {
					t.Fatal(err)
				}
			case "symlink":
				target := filepath.Join(dir, "other.json")
				if err := writeNew(target, data); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(target, path); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := m.Report(r.Workflow, r.Operation, 0); err == nil {
				t.Fatal("changed report exported")
			}
		})
	}
	if _, err := redactedReport([]byte(`{`), nil); err == nil {
		t.Fatal("invalid report redacted into valid evidence")
	}
}

func TestReportValidationDoesNotAcceptMalformedCatalogConfiguration(t *testing.T) {
	if err := validateWorkerReport("postgres-catalog", []byte(`{`), []byte(`{}`)); err == nil {
		t.Fatal("catalog report accepted without reviewed scope")
	}
	m, r, _ := testManager(t)
	state, err := m.Submit(t.Context(), r)
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(m.Root, r.Workflow, r.Operation)
	before, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	m.BootID = func() (string, error) { return resumedID, nil }
	got, err := m.Status(r.Workflow, r.Operation)
	if err != nil || got.Phase != "interrupted" {
		t.Fatalf("reboot not reflected: %+v %v", got, err)
	}
	retained, err := os.ReadFile(filepath.Join(dir, "state.json"))
	if err != nil || !reflect.DeepEqual(before, retained) {
		t.Fatal("read-only status rewrote interrupted evidence")
	}
}

func TestDispatchLockContentionBlocksAssessmentAndCSVAdmission(t *testing.T) {
	m, r, starts := testManager(t)
	unlock, err := m.dispatchLock()
	if err != nil {
		t.Fatal(err)
	}
	defer unlock()
	if _, err := m.Submit(t.Context(), r); err == nil || *starts != 0 {
		t.Fatal("assessment ignored exclusive admission lock")
	}
	if _, err := m.SubmitCSV(t.Context(), csvRequest("id\n1\n")); err == nil || *starts != 0 {
		t.Fatal("CSV import ignored exclusive admission lock")
	}
	if _, err := os.Stat(filepath.Join(m.Root, r.Workflow, r.Operation)); !os.IsNotExist(err) {
		t.Fatal("rejected concurrent admission created operation evidence")
	}
}

func TestUncreatableWorkflowCannotStartOrDiscardExistingFile(t *testing.T) {
	for _, csv := range []bool{false, true} {
		t.Run(map[bool]string{false: "catalog", true: "csv"}[csv], func(t *testing.T) {
			m, r, starts := catalogManager(t)
			r.Workflow = resumedID
			root := filepath.Join(m.Root, r.Workflow)
			if err := writeNew(root, []byte("retained-file")); err != nil {
				t.Fatal(err)
			}
			if csv {
				r = csvRequest("id\n1\n")
				r.Workflow = resumedID
				r.Import.URL = strings.ReplaceAll(r.Import.URL, workflowID, resumedID)
				if _, err := m.SubmitCSV(t.Context(), r); err == nil {
					t.Fatal("CSV workflow overwrote file")
				}
			} else if _, err := m.Submit(t.Context(), r); err == nil {
				t.Fatal("assessment workflow overwrote file")
			}
			data, err := os.ReadFile(root)
			if err != nil || string(data) != "retained-file" || *starts != 0 {
				t.Fatal("failed directory creation changed existing evidence")
			}
		})
	}
}

func TestDirectExportRequiresProtectedCapabilityControl(t *testing.T) {
	if _, err := (Manager{}).ExportReport(t.Context(), Request{Action: "export-report"}); err == nil {
		t.Fatal("export without capability admitted")
	}
}

func TestWorkerCannotOverwriteRetainedSourceCertificate(t *testing.T) {
	m, r, starts := catalogManager(t)
	ca := testSourceCAPEM(t)
	r.Secrets["AGEFREIGHTER_SOURCE_CA_PEM"] = ca
	r.Configuration = []byte(strings.Replace(catalogConfig, `"sourceCASHA256":""`, `"sourceCASHA256":"`+sum([]byte(ca))+`"`, 1))
	if _, err := m.Submit(t.Context(), r); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(m.Root, r.Workflow, r.Operation)
	path := filepath.Join(dir, "source-ca.pem")
	if err := writeNew(path, []byte("retained-certificate")); err != nil {
		t.Fatal(err)
	}
	if err := m.Work(t.Context(), r.Workflow, r.Operation); err == nil {
		t.Fatal("worker overwrote existing source CA")
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "retained-certificate" {
		t.Fatal("failed staging modified prior certificate evidence")
	}
	if _, err := os.Stat(filepath.Join(dir, "secrets.json")); !os.IsNotExist(err) {
		t.Fatal("failed staging retained credential transport")
	}
	if err := m.Work(t.Context(), r.Workflow, r.Operation); err == nil || *starts != 1 {
		t.Fatal("failed certificate staging permitted worker replay")
	}
}

func TestUnreadableReportCannotBeExportedEvenWithMatchingManifest(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("privileged users can read mode-000 files")
	}
	m, r, _ := retainedExport(t)
	path := filepath.Join(m.Root, r.Workflow, r.Operation, "report.json")
	if err := os.Chmod(path, 0000); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(path, 0600) })
	if _, err := m.Report(r.Workflow, r.Operation, 0); err == nil || err.Error() != "report artifact unavailable" {
		t.Fatalf("unreadable report accepted: %v", err)
	}
}
