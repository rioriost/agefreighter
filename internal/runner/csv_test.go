package runner

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

const csvFileID = "44444444-4444-4444-8444-444444444444"

func csvRequest(data string) Request {
	digest := sum([]byte(data))
	capability := strings.Replace(exportCapability("r", time.Now()), "/reports/"+operationID+".json", "/uploads/"+csvFileID+"/"+digest+".csv", 1)
	return Request{Version: 1, Workflow: workflowID, Operation: operationID, Action: "import-csv", ExpectedBootID: bootID, Import: &CSVImport{URL: capability, File: csvFileID, Bytes: int64(len(data)), SHA256: digest}}
}
func TestCSVWorkerSealsFullBytesWithoutReplayOrCapabilityRetention(t *testing.T) {
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skip("Unix guest filesystem capacity enforcement")
	}
	m, _, starts := testManager(t)
	data := "id,name\n1,東京\n"
	r := csvRequest(data)
	calls := 0
	m.blobTransport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		if req.Method != "GET" || req.Header.Get("Authorization") != "" {
			t.Fatal("unsafe request")
		}
		return &http.Response{StatusCode: 200, ContentLength: int64(len(data)), Body: io.NopCloser(strings.NewReader(data)), Header: http.Header{}}, nil
	})
	encoded, _ := json.Marshal(r)
	if _, err := Decode(strings.NewReader(string(encoded))); err != nil {
		t.Fatal(err)
	}
	state, err := m.SubmitCSV(context.Background(), r)
	if err != nil || state.Phase != "accepted" || *starts != 1 {
		t.Fatalf("%#v %v", state, err)
	}
	if _, err = m.SubmitCSV(context.Background(), r); err == nil {
		t.Fatal("replayed submission")
	}
	if err = m.Work(context.Background(), workflowID, operationID); err != nil {
		t.Fatal(err)
	}
	state, err = m.Status(workflowID, operationID)
	if err != nil || state.Phase != "finished" || state.FileSHA256 != sum([]byte(data)) {
		t.Fatalf("%#v %v", state, err)
	}
	target := filepath.Join(m.Root, workflowID, "uploads", csvFileID+".csv")
	if err = verifyCSVSeal(target); err != nil {
		t.Fatal(err)
	}
	if _, err = os.Stat(filepath.Join(m.Root, workflowID, operationID, "capability.json")); !os.IsNotExist(err) {
		t.Fatal("capability retained")
	}
	if err = m.Work(context.Background(), workflowID, operationID); err == nil || calls != 1 {
		t.Fatal("worker replay")
	}
	if err = os.WriteFile(target, []byte("modified"), 0600); err != nil {
		t.Fatal(err)
	}
	if verifyCSVSeal(target) == nil {
		t.Fatal("modified sealed CSV accepted")
	}
}
func TestCSVFailureRetainsPartialButNeverPublishesOrLeaks(t *testing.T) {
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skip("Unix guest filesystem capacity enforcement")
	}
	for _, mode := range []string{"changed", "short", "oversized", "redirect", "denied"} {
		t.Run(mode, func(t *testing.T) {
			m, _, _ := testManager(t)
			r := csvRequest("id\n1\n")
			calls := 0
			m.blobTransport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
				calls++
				data := "id\n2\n"
				status := 200
				if mode == "short" {
					data = "id"
				}
				if mode == "oversized" {
					data = "id\n1\nextra"
				}
				if mode == "redirect" {
					status = 302
				}
				if mode == "denied" {
					status = 403
				}
				return &http.Response{StatusCode: status, ContentLength: r.Import.Bytes, Body: io.NopCloser(strings.NewReader(data)), Header: http.Header{"Location": []string{"https://attacker.invalid/sensitive-capability"}}}, nil
			})
			if _, err := m.SubmitCSV(context.Background(), r); err != nil {
				t.Fatal(err)
			}
			err := m.Work(context.Background(), workflowID, operationID)
			if err == nil || strings.Contains(err.Error(), "sensitive-capability") {
				t.Fatal("unsafe error")
			}
			state, _ := m.Status(workflowID, operationID)
			if state.Phase != "failed" || calls != 1 {
				t.Fatal("failure not retained or request replayed")
			}
			if _, err = os.Stat(filepath.Join(m.Root, workflowID, "uploads", csvFileID+".csv")); !os.IsNotExist(err) {
				t.Fatal("bad bytes published")
			}
		})
	}
}
func TestCSVCapabilitiesRejectForeignPathsPrivilegesAndUnsealedFiles(t *testing.T) {
	r := csvRequest("id\n1\n")
	now := time.Now()
	for _, url := range []string{strings.Replace(r.Import.URL, "sp=r", "sp=rw", 1), strings.Replace(r.Import.URL, "/uploads/", "/reports/", 1), strings.Replace(r.Import.URL, "https:", "http:", 1), r.Import.URL + "&sp=r", strings.Replace(r.Import.URL, "afstorage.blob.core.windows.net", "localhost", 1)} {
		bad := *r.Import
		bad.URL = url
		if validateCSVImport(workflowID, bad, now) == nil {
			t.Fatal("unsafe CSV capability")
		}
	}
	m, _, _ := testManager(t)
	root := filepath.Join(m.Root, workflowID)
	target := filepath.Join(root, "uploads", csvFileID+".csv")
	if err := writeNew(target, []byte("id\n1\n")); err != nil {
		t.Fatal(err)
	}
	if verifyCSVSeal(target) == nil {
		t.Fatal("unsealed file accepted")
	}
}
