package runner

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"time"
)

const MaxCSVBytes int64 = 2 << 30

type CSVImport struct {
	URL    string `json:"url"`
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}
type csvSeal struct {
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

func verifyCSVSeal(path string) error {
	fail := errors.New("CSV upload lacks a matching full-content verification seal")
	var seal csvSeal
	if readJSON(path+".seal.json", &seal) != nil || !uuid.MatchString(seal.File) || filepath.Base(path) != seal.File+".csv" || seal.Bytes < 1 || seal.Bytes > MaxCSVBytes || !sha256Pattern.MatchString(seal.SHA256) {
		return fail
	}
	f, err := os.Open(path)
	if err != nil {
		return fail
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(f, seal.Bytes+1))
	if err != nil || n != seal.Bytes || hex.EncodeToString(h.Sum(nil)) != seal.SHA256 {
		return fail
	}
	return nil
}

func safeCSVAction(r Request) bool {
	return r.Action == "import-csv" && r.Import != nil && r.Export == nil && len(r.Configuration) == 0 && len(r.Secrets) == 0 && r.Offset == 0 && uuid.MatchString(r.ExpectedBootID)
}
func validateCSVImport(workflow string, source CSVImport, now time.Time) error {
	fail := errors.New("CSV import requires a short-lived read-only capability for this exact workflow file")
	if !uuid.MatchString(source.File) || !sha256Pattern.MatchString(source.SHA256) || source.Bytes < 1 || source.Bytes > MaxCSVBytes {
		return fail
	}
	u, err := url.Parse(source.URL)
	if err != nil || u.RawPath != "" || u.Path != "/af-"+workflow+"/uploads/"+source.File+"/"+source.SHA256+".csv" {
		return fail
	}
	q, err := url.ParseQuery(u.RawQuery)
	if err != nil || len(q["sp"]) != 1 || q.Get("sp") != "r" {
		return fail
	}
	// Reuse the strict delegation/host/lifetime policy without extending report rights.
	q.Set("sp", "c")
	u.RawQuery = q.Encode()
	u.Path = "/af-" + workflow + "/reports/" + source.File + ".json"
	if validateReportExport(workflow, source.File, ReportExport{URL: u.String(), SHA256: source.SHA256, Bytes: 1}, now) != nil {
		return fail
	}
	return nil
}

// SubmitCSV persists an operation before a disabled-at-boot worker is started.
// The SAS stays only in its private transient capability file, never state/units.
func (m Manager) SubmitCSV(ctx context.Context, r Request) (State, error) {
	if !safeCSVAction(r) {
		return State{}, errors.New("invalid CSV import request")
	}
	if err := validateCSVImport(r.Workflow, *r.Import, time.Now()); err != nil {
		return State{}, err
	}
	root, dir, err := m.paths(r.Workflow, r.Operation)
	if err != nil {
		return State{}, err
	}
	boot, err := m.BootID()
	if err != nil || boot != r.ExpectedBootID {
		return State{}, errors.New("check fresh guest boot readiness")
	}
	if err = os.MkdirAll(root, 0700); err != nil {
		return State{}, err
	}
	if err = privateDirectory(root); err != nil {
		return State{}, err
	}
	if err = csvDiskGate(root, r.Import.Bytes); err != nil {
		return State{}, err
	}
	if err = os.Mkdir(dir, 0700); err != nil {
		return State{}, errors.New("CSV operation exists; reconcile without replay")
	}
	if err = writeNew(filepath.Join(root, "active"), []byte(r.Operation)); err != nil {
		return State{}, errors.New("workflow already has retained active work")
	}
	state := State{Version: 1, Workflow: r.Workflow, Operation: r.Operation, Action: r.Action, Phase: "accepted", BootID: boot, FileID: r.Import.File, FileBytes: r.Import.Bytes, FileSHA256: r.Import.SHA256}
	if err = writeNewJSON(filepath.Join(dir, "state.json"), state); err != nil {
		return State{}, err
	}
	if err = writeNewJSON(filepath.Join(dir, "capability.json"), r.Import); err != nil {
		return State{}, err
	}
	unit, err := m.unit(r.Workflow, r.Operation, root)
	if err != nil {
		return State{}, err
	}
	if err = writeNew(filepath.Join(m.UnitDirectory, unitName(r.Operation)), []byte(unit)); err != nil {
		return State{}, err
	}
	if err = m.Start(ctx, unitName(r.Operation)); err != nil {
		return state, errors.New("CSV start acknowledgement is uncertain; reconcile retained status")
	}
	return state, nil
}

func (m Manager) workCSV(ctx context.Context, root, dir string, state State) error {
	state.Phase = "running"
	state.StartedAt = time.Now().UTC().Format(time.RFC3339Nano)
	if err := replaceJSON(filepath.Join(dir, "state.json"), state); err != nil {
		return err
	}
	var source CSVImport
	err := readJSON(filepath.Join(dir, "capability.json"), &source)
	if err == nil && (source.File != state.FileID || source.Bytes != state.FileBytes || source.SHA256 != state.FileSHA256) {
		err = errors.New("CSV manifest changed")
	}
	if err == nil {
		err = validateCSVImport(state.Workflow, source, time.Now())
	}
	if err == nil {
		err = m.downloadCSV(ctx, root, dir, source)
	}
	exit := 0
	state.Phase = "finished"
	if err != nil {
		exit = 1
		state.Phase = "failed"
	}
	state.ExitCode = &exit
	state.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
	if saveErr := replaceJSON(filepath.Join(dir, "state.json"), state); saveErr != nil {
		return saveErr
	}
	// Partial bytes are retained in this failed operation. Only the capability is erased.
	if removeErr := os.Remove(filepath.Join(dir, "capability.json")); removeErr != nil {
		return removeErr
	}
	active, readErr := os.ReadFile(filepath.Join(root, "active"))
	if readErr != nil || string(active) != state.Operation {
		return errors.New("CSV workflow lease changed")
	}
	if removeErr := os.Remove(filepath.Join(root, "active")); removeErr != nil {
		return removeErr
	}
	if err != nil {
		return errors.New("CSV import failed; retain partial evidence and inspect its status")
	}
	return nil
}

func (m Manager) downloadCSV(ctx context.Context, root, dir string, source CSVImport) error {
	if err := csvDiskGate(root, source.Bytes); err != nil {
		return err
	}
	uploads := filepath.Join(root, "uploads")
	if err := os.MkdirAll(uploads, 0700); err != nil {
		return err
	}
	if err := privateDirectory(uploads); err != nil {
		return err
	}
	target := filepath.Join(uploads, source.File+".csv")
	if _, err := os.Lstat(target); err == nil {
		return errors.New("CSV destination already exists; never overwrite sealed or unsealed evidence")
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, source.URL, nil)
	if err != nil {
		return errors.New("invalid CSV download")
	}
	request.Header.Set("x-ms-version", "2023-11-03")
	request.Header.Set("Accept-Encoding", "identity")
	transport := m.blobTransport
	if transport == nil {
		transport = http.DefaultTransport
	}
	client := http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return errors.New("redirect blocked") }}
	response, err := client.Do(request)
	if err != nil {
		return errors.New("CSV download unavailable")
	}
	defer response.Body.Close()
	if response.StatusCode != 200 || response.ContentLength != source.Bytes || response.Header.Get("Content-Encoding") != "" && response.Header.Get("Content-Encoding") != "identity" {
		return errors.New("CSV download manifest mismatch")
	}
	partial := filepath.Join(dir, "upload.partial")
	f, err := os.OpenFile(partial, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(f, h), io.LimitReader(response.Body, source.Bytes+1))
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil || n != source.Bytes || hex.EncodeToString(h.Sum(nil)) != source.SHA256 {
		return errors.New("CSV full length or SHA-256 mismatch")
	}
	// Atomic no-replace publication, with an independent manifest seal required by profiling.
	if err = os.Link(partial, target); err != nil {
		return err
	}
	return writeNewJSON(target+".seal.json", csvSeal{File: source.File, Bytes: source.Bytes, SHA256: source.SHA256})
}
