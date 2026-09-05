package runner

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func exportCapability(permission string, now time.Time) string {
	q := url.Values{"sv": {"2023-11-03"}, "spr": {"https"}, "sr": {"b"}, "sp": {permission}, "st": {now.Add(-time.Minute).Format(time.RFC3339)}, "se": {now.Add(10 * time.Minute).Format(time.RFC3339)}, "sig": {"sensitive-capability"}, "skoid": {workflowID}, "sktid": {operationID}, "skt": {now.Add(-time.Hour).Format(time.RFC3339)}, "ske": {now.Add(time.Hour).Format(time.RFC3339)}, "sks": {"b"}, "skv": {"2023-11-03"}}
	return "https://afstorage.blob.core.windows.net/af-" + workflowID + "/reports/" + operationID + ".json?" + q.Encode()
}

func TestReportExportCapabilityBoundaries(t *testing.T) {
	now := time.Now().UTC()
	valid := ReportExport{URL: exportCapability("c", now), SHA256: strings.Repeat("a", 64), Bytes: 42}
	if err := validateReportExport(workflowID, operationID, valid, now); err != nil {
		t.Fatal(err)
	}
	cases := []string{
		strings.Replace(valid.URL, "https:", "http:", 1), strings.Replace(valid.URL, "afstorage.blob.core.windows.net", "localhost", 1),
		strings.Replace(valid.URL, "afstorage.blob.core.windows.net", "afstorage.blob.core.windows.net.attacker.invalid", 1),
		strings.Replace(valid.URL, "afstorage.blob.core.windows.net", "afstorage.blob.core.windows.net:443", 1),
		strings.Replace(valid.URL, "/reports/", "/uploads/", 1), strings.Replace(valid.URL, "sp=c", "sp=cw", 1),
		strings.Replace(valid.URL, "sr=b", "sr=c", 1), valid.URL + "&sp=c", valid.URL + "&comp=block", valid.URL + "#fragment",
		strings.Replace(valid.URL, "spr=https", "spr=https%2Chttp", 1), exportCapability("c", now.Add(time.Hour)), exportCapability("c", now.Add(-time.Hour)),
		strings.Replace(valid.URL, "sks=b", "sks=f", 1), strings.Replace(valid.URL, ".json?", "%2Ejson?", 1),
	}
	for _, raw := range cases {
		bad := valid
		bad.URL = raw
		if err := validateReportExport(workflowID, operationID, bad, now); err == nil || strings.Contains(err.Error(), "sensitive-capability") {
			t.Fatal("unsafe URL accepted or exposed")
		}
	}
	for _, size := range []int64{0, MaxArtifactBytes + 1} {
		bad := valid
		bad.Bytes = size
		if validateReportExport(workflowID, operationID, bad, now) == nil {
			t.Fatal("bad size")
		}
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func retainedExport(t *testing.T) (Manager, Request, []byte) {
	t.Helper()
	m, request, _ := testManager(t)
	// Transport tests use a retained report, not a fabricated source qualification.
	_, dir, _ := m.paths(workflowID, operationID)
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	data := []byte(`{"version":"1","command":"profile","integer":9223372036854775807,"padding":"` + strings.Repeat("x", 100000) + `"}`)
	if err := writeNew(filepath.Join(dir, "report.json"), data); err != nil {
		t.Fatal(err)
	}
	exit := 0
	state := State{Version: 1, Workflow: workflowID, Operation: operationID, Action: "profile", Phase: "finished", BootID: bootID, ExitCode: &exit, ReportBytes: int64(len(data)), ReportSHA256: sum(data)}
	if err := writeNewJSON(filepath.Join(dir, "state.json"), state); err != nil {
		t.Fatal(err)
	}
	request = Request{Version: 1, Workflow: workflowID, Operation: operationID, Action: "export-report", Export: &ReportExport{URL: exportCapability("c", time.Now()), SHA256: sum(data), Bytes: int64(len(data))}}
	return m, request, data
}

func TestReportExportPreservesExactBytesWithOneConditionalPUT(t *testing.T) {
	m, request, data := retainedExport(t)
	calls := 0
	m.blobTransport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		calls++
		if r.Method != "PUT" || r.Header.Get("If-None-Match") != "*" || r.Header.Get("x-ms-blob-type") != "BlockBlob" || r.Header.Get("Authorization") != "" {
			t.Fatal("unsafe transfer headers")
		}
		body, err := io.ReadAll(r.Body)
		if err != nil || string(body) != string(data) || r.ContentLength != int64(len(data)) {
			t.Fatal("report bytes changed")
		}
		if _, ok := r.Context().Deadline(); !ok {
			t.Fatal("missing deadline")
		}
		return &http.Response{StatusCode: 201, Body: io.NopCloser(strings.NewReader("")), Header: http.Header{}}, nil
	})
	receipt, err := m.ExportReport(context.Background(), request)
	if err != nil || calls != 1 || !receipt.Exported || receipt.SHA256 != sum(data) {
		t.Fatalf("%#v %v", receipt, err)
	}
	wire, _ := json.Marshal(receipt)
	if len(wire) > 4096 || strings.Contains(string(wire), "sensitive-capability") {
		t.Fatal("unsafe receipt")
	}
	state, _ := m.Status(workflowID, operationID)
	if state.Phase != "finished" {
		t.Fatal("export changed source operation")
	}
}

func TestReportExportNeverRetriesRedirectsOrLeakingErrors(t *testing.T) {
	for _, code := range []int{0, 302, 403, 412, 500} {
		t.Run(http.StatusText(code), func(t *testing.T) {
			m, request, _ := retainedExport(t)
			calls := 0
			m.blobTransport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
				calls++
				if code == 0 {
					return nil, errors.New(r.URL.String())
				}
				return &http.Response{StatusCode: code, Body: io.NopCloser(strings.NewReader(r.URL.String())), Header: http.Header{"Location": []string{"https://attacker.invalid"}}}, nil
			})
			_, err := m.ExportReport(context.Background(), request)
			if err == nil || calls != 1 || strings.Contains(err.Error(), "sensitive-capability") || strings.Contains(err.Error(), "https:") {
				t.Fatalf("unsafe failure: %d %v", calls, err)
			}
		})
	}
}

func TestReportExportBlocksTamperingBeforeNetwork(t *testing.T) {
	for _, kind := range []string{"hash", "length", "file", "symlink", "active", "oversized"} {
		t.Run(kind, func(t *testing.T) {
			m, request, _ := retainedExport(t)
			_, dir, _ := m.paths(workflowID, operationID)
			switch kind {
			case "hash":
				request.Export.SHA256 = strings.Repeat("b", 64)
			case "length":
				request.Export.Bytes++
			case "file":
				_ = os.WriteFile(filepath.Join(dir, "report.json"), []byte("changed"), 0600)
			case "symlink":
				_ = os.Rename(filepath.Join(dir, "report.json"), filepath.Join(dir, "original.json"))
				if err := os.Symlink(filepath.Join(dir, "original.json"), filepath.Join(dir, "report.json")); err != nil {
					t.Fatal(err)
				}
			case "active":
				state, _ := m.Status(workflowID, operationID)
				state.Phase = "running"
				_ = replaceJSON(filepath.Join(dir, "state.json"), state)
			case "oversized":
				request.Export.Bytes = MaxArtifactBytes + 1
			}
			m.blobTransport = roundTripFunc(func(*http.Request) (*http.Response, error) {
				t.Fatal("network called before evidence validation")
				return nil, nil
			})
			if _, err := m.ExportReport(context.Background(), request); err == nil {
				t.Fatal("bad evidence accepted")
			}
		})
	}
}

func TestExportProtocolRejectsOtherInputsAndPreservesCapabilityPrivately(t *testing.T) {
	request := Request{Version: 1, Workflow: workflowID, Operation: operationID, Action: "export-report", Export: &ReportExport{URL: exportCapability("c", time.Now()), SHA256: strings.Repeat("a", 64), Bytes: 42}}
	data, _ := json.Marshal(request)
	if _, err := Decode(strings.NewReader(string(data))); err != nil {
		t.Fatal(err)
	}
	for _, action := range []string{"ready", "profile", "status", "inventory", "report"} {
		bad := request
		bad.Action = action
		data, _ := json.Marshal(bad)
		if _, err := Decode(strings.NewReader(string(data))); err == nil {
			t.Fatal("unexpected capability accepted")
		}
	}
	request.Secrets = map[string]string{"AGEFREIGHTER_SOURCE_PASSWORD": "secret"}
	data, _ = json.Marshal(request)
	if _, err := Decode(strings.NewReader(string(data))); err == nil {
		t.Fatal("source secrets in export accepted")
	}
}
