package runner

import (
	"context"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCSVAdmissionFailuresNeverStartOrDiscardSealedIntent(t *testing.T) {
	for _, mode := range []string{"control", "capability", "identity", "boot", "private-root", "active", "unsafe-unit", "unit-collision", "lost-start"} {
		t.Run(mode, func(t *testing.T) {
			m, _, starts := testManager(t)
			m.csvCapacity = func(string) (float64, float64, error) { return 1000, 900, nil }
			r := csvRequest("id\n1\n")
			root := filepath.Join(m.Root, r.Workflow)
			switch mode {
			case "control":
				r.Offset = 1
			case "capability":
				r.Import.Bytes = 0
			case "identity":
				r.Operation = "../escape"
			case "boot":
				m.BootID = func() (string, error) { return "", errors.New("private boot failure") }
			case "private-root":
				if err := os.Chmod(root, 0755); err != nil {
					t.Fatal(err)
				}
			case "active":
				if err := writeNew(filepath.Join(root, "active"), []byte(resumedID)); err != nil {
					t.Fatal(err)
				}
			case "unsafe-unit":
				m.Tools = "/not safe"
			case "unit-collision":
				if err := writeNew(filepath.Join(m.UnitDirectory, unitName(r.Operation)), []byte("retained-unit")); err != nil {
					t.Fatal(err)
				}
			case "lost-start":
				m.Start = func(context.Context, string) error { *starts++; return errors.New("private start failure") }
			}
			state, err := m.SubmitCSV(t.Context(), r)
			expectedStarts := 0
			if mode == "lost-start" {
				expectedStarts = 1
			}
			if err == nil || *starts != expectedStarts || strings.Contains(err.Error(), "private boot failure") || strings.Contains(err.Error(), "private start failure") {
				t.Fatalf("CSV admission result: %+v %v starts=%d", state, err, *starts)
			}
			if mode == "lost-start" || mode == "unsafe-unit" || mode == "unit-collision" {
				retained, err := m.Status(r.Workflow, r.Operation)
				if err != nil || retained.Phase != "accepted" || retained.FileSHA256 != r.Import.SHA256 {
					t.Fatalf("sealed intent lost: %+v %v", retained, err)
				}
				if _, err := m.SubmitCSV(t.Context(), r); err == nil || *starts != expectedStarts {
					t.Fatal("uncertain CSV operation replayed")
				}
			}
		})
	}
}

func TestCSVWorkerManifestAndLeaseFailuresAreDurable(t *testing.T) {
	for _, mode := range []string{"manifest", "capability-missing", "expired", "lease"} {
		t.Run(mode, func(t *testing.T) {
			m, _, _ := testManager(t)
			m.csvCapacity = func(string) (float64, float64, error) { return 1000, 900, nil }
			r := csvRequest("id\n1\n")
			calls := 0
			m.blobTransport = roundTripFunc(func(*http.Request) (*http.Response, error) {
				calls++
				return &http.Response{StatusCode: 200, ContentLength: r.Import.Bytes, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("id\n1\n"))}, nil
			})
			if _, err := m.SubmitCSV(t.Context(), r); err != nil {
				t.Fatal(err)
			}
			root := filepath.Join(m.Root, r.Workflow)
			capability := filepath.Join(root, r.Operation, "capability.json")
			switch mode {
			case "manifest":
				bad := *r.Import
				bad.Bytes++
				if err := replaceJSON(capability, bad); err != nil {
					t.Fatal(err)
				}
			case "capability-missing":
				if err := os.Remove(capability); err != nil {
					t.Fatal(err)
				}
			case "expired":
				bad := *r.Import
				bad.URL = strings.Replace(bad.URL, "sp=r", "sp=rw", 1)
				if err := replaceJSON(capability, bad); err != nil {
					t.Fatal(err)
				}
			case "lease":
				if err := os.WriteFile(filepath.Join(root, "active"), []byte(resumedID), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if err := m.Work(t.Context(), r.Workflow, r.Operation); err == nil || strings.Contains(err.Error(), r.Import.URL) {
				t.Fatalf("unsafe worker error: %v", err)
			}
			state, err := m.Status(r.Workflow, r.Operation)
			expectedPhase, expectedCalls := "failed", 0
			if mode == "lease" {
				expectedPhase, expectedCalls = "finished", 1
				lease, err := os.ReadFile(filepath.Join(root, "active"))
				if err != nil || string(lease) != resumedID {
					t.Fatal("unrelated lease erased")
				}
			}
			if err != nil || state.Phase != expectedPhase || calls != expectedCalls || state.ExitCode == nil || state.FinishedAt == "" {
				t.Fatalf("failure evidence not sealed: %+v %v calls=%d", state, err, calls)
			}
			if _, err := os.Stat(capability); !os.IsNotExist(err) {
				t.Fatal("transient capability retained")
			}
			if err := m.Work(t.Context(), r.Workflow, r.Operation); err == nil || calls != expectedCalls {
				t.Fatal("CSV failure replayed")
			}
		})
	}
}

func TestCSVDownloadPreservesExistingEvidenceAndBoundsTransport(t *testing.T) {
	for _, mode := range []string{"disk", "uploads-file", "public-uploads", "destination", "invalid-url", "transport", "partial", "seal", "publication-race"} {
		t.Run(mode, func(t *testing.T) {
			m, _, _ := testManager(t)
			m.csvCapacity = func(string) (float64, float64, error) { return 1000, 900, nil }
			root := filepath.Join(m.Root, workflowID)
			dir := filepath.Join(root, operationID)
			if err := os.Mkdir(dir, 0700); err != nil {
				t.Fatal(err)
			}
			r := csvRequest("id\n1\n")
			uploads := filepath.Join(root, "uploads")
			calls := 0
			m.blobTransport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
				calls++
				if req.Header.Get("Accept-Encoding") != "identity" || req.Header.Get("x-ms-version") != "2023-11-03" {
					t.Fatal("unbounded transfer headers")
				}
				if mode == "transport" {
					return nil, errors.New("PRIVATE-CAPABILITY")
				}
				if mode == "publication-race" {
					if err := writeNew(filepath.Join(uploads, r.Import.File+".csv"), []byte("prior-evidence")); err != nil {
						t.Fatal(err)
					}
				}
				return &http.Response{StatusCode: 200, ContentLength: r.Import.Bytes, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("id\n1\n"))}, nil
			})
			retained := ""
			switch mode {
			case "disk":
				m.csvCapacity = func(string) (float64, float64, error) { return 1000, 1, nil }
			case "uploads-file":
				root = filepath.Join(dir, "other")
				if err := os.Mkdir(root, 0700); err != nil {
					t.Fatal(err)
				}
				retained = filepath.Join(root, "uploads")
			case "public-uploads":
				if err := os.Chmod(uploads, 0755); err != nil {
					t.Fatal(err)
				}
			case "destination", "publication-race":
				retained = filepath.Join(uploads, r.Import.File+".csv")
			case "invalid-url":
				r.Import.URL = "https://%"
			case "partial":
				retained = filepath.Join(dir, "upload.partial")
			case "seal":
				retained = filepath.Join(uploads, r.Import.File+".csv.seal.json")
			}
			if retained != "" && mode != "publication-race" {
				if err := writeNew(retained, []byte("prior-evidence")); err != nil {
					t.Fatal(err)
				}
			}
			if err := m.downloadCSV(t.Context(), root, dir, *r.Import); err == nil || strings.Contains(err.Error(), "PRIVATE-CAPABILITY") {
				t.Fatalf("unsafe download result: %v", err)
			}
			expectedCalls := 0
			if mode == "transport" || mode == "partial" || mode == "seal" || mode == "publication-race" {
				expectedCalls = 1
			}
			if calls != expectedCalls {
				t.Fatalf("requests=%d, want %d", calls, expectedCalls)
			}
			if retained != "" {
				data, err := os.ReadFile(retained)
				if err != nil || string(data) != "prior-evidence" {
					t.Fatal("prior evidence overwritten")
				}
			}
		})
	}
}
