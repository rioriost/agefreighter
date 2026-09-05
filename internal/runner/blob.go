package runner

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

// Only one immutable report blob is addressable. A SAS is a transient secret,
// never retained in state or included in an error or acknowledgement.
type ReportExport struct {
	URL    string `json:"url"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

type ExportReceipt struct {
	Version   int    `json:"version"`
	Workflow  string `json:"workflow"`
	Operation string `json:"operation"`
	SHA256    string `json:"sha256"`
	Bytes     int64  `json:"bytes"`
	Exported  bool   `json:"exported"`
}

var blobHost = regexp.MustCompile(`^[a-z0-9]{3,24}\.blob\.core\.windows\.net$`)
var sha256Pattern = regexp.MustCompile(`^[a-f0-9]{64}$`)

func validateReportExport(workflow, operation string, target ReportExport, now time.Time) error {
	fail := errors.New("report export requires a short-lived create-only HTTPS blob capability for this report")
	u, err := url.Parse(target.URL)
	if err != nil || len(target.URL) > 4096 || !uuid.MatchString(workflow) || !uuid.MatchString(operation) ||
		!sha256Pattern.MatchString(target.SHA256) || target.Bytes < 1 || target.Bytes > MaxArtifactBytes ||
		u.Scheme != "https" || u.User != nil || u.Fragment != "" || !blobHost.MatchString(u.Host) ||
		u.RawPath != "" || u.Path != "/af-"+workflow+"/reports/"+operation+".json" {
		return fail
	}
	q, err := url.ParseQuery(u.RawQuery)
	if err != nil {
		return fail
	}
	allowed := map[string]bool{"sv": true, "spr": true, "sr": true, "sp": true, "st": true, "se": true, "sig": true, "skoid": true, "sktid": true, "skt": true, "ske": true, "sks": true, "skv": true}
	for key, values := range q {
		if !allowed[key] || len(values) != 1 || values[0] == "" {
			return fail
		}
	}
	// Require a user delegation SAS, not account keys or container-wide rights.
	if q.Get("sr") != "b" || q.Get("sp") != "c" || q.Get("spr") != "https" || q.Get("sig") == "" ||
		!uuid.MatchString(q.Get("skoid")) || !uuid.MatchString(q.Get("sktid")) || q.Get("sks") != "b" || q.Get("sv") == "" || q.Get("skv") == "" {
		return fail
	}
	start, e1 := time.Parse(time.RFC3339, q.Get("st"))
	end, e2 := time.Parse(time.RFC3339, q.Get("se"))
	keyStart, e3 := time.Parse(time.RFC3339, q.Get("skt"))
	keyEnd, e4 := time.Parse(time.RFC3339, q.Get("ske"))
	if e1 != nil || e2 != nil || e3 != nil || e4 != nil || start.After(now) || start.Before(now.Add(-5*time.Minute)) ||
		!end.After(now) || end.After(now.Add(15*time.Minute)) || end.Sub(start) > 20*time.Minute || keyStart.After(start) || keyEnd.Before(end) {
		return fail
	}
	return nil
}

// ExportReport never re-reads the source, replaces an existing blob, follows a
// redirect, or retries an uncertain PUT. Reconciliation reads the destination.
func (m Manager) ExportReport(ctx context.Context, request Request) (ExportReceipt, error) {
	if !safeExportAction(request) {
		return ExportReceipt{}, errors.New("report export capability is missing")
	}
	if err := validateReportExport(request.Workflow, request.Operation, *request.Export, time.Now()); err != nil {
		return ExportReceipt{}, err
	}
	data, state, err := m.reportData(request.Workflow, request.Operation)
	if err != nil {
		return ExportReceipt{}, err
	}
	if state.ReportSHA256 != request.Export.SHA256 || state.ReportBytes != request.Export.Bytes {
		return ExportReceipt{}, errors.New("report export manifest does not match retained evidence")
	}
	ctx, cancel := context.WithTimeout(ctx, 25*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, request.Export.URL, bytes.NewReader(data))
	if err != nil {
		return ExportReceipt{}, errors.New("invalid report export request")
	}
	req.Header.Set("x-ms-version", "2023-11-03")
	req.Header.Set("x-ms-blob-type", "BlockBlob")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("If-None-Match", "*")
	transport := m.blobTransport
	if transport == nil {
		transport = http.DefaultTransport
	}
	client := &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return errors.New("redirect blocked") }}
	response, err := client.Do(req)
	if err != nil {
		return ExportReceipt{}, errors.New("report export result is uncertain; read retained destination before any new export")
	}
	defer response.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 4096))
	if response.StatusCode != http.StatusCreated {
		// Service bodies and request URLs can contain capability credentials.
		return ExportReceipt{}, errors.New("report export was not acknowledged; inspect destination without overwriting it")
	}
	return ExportReceipt{Version: 1, Workflow: request.Workflow, Operation: request.Operation, SHA256: state.ReportSHA256, Bytes: state.ReportBytes, Exported: true}, nil
}

// Avoid allowing a query string into diagnostic errors through any URL parser.
func safeExportAction(request Request) bool {
	return request.Action == "export-report" && len(request.Configuration) == 0 && len(request.Secrets) == 0 && request.ExpectedBootID == "" && request.Offset == 0 && request.Export != nil && !strings.ContainsRune(request.Export.URL, '\x00')
}
