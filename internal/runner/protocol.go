// Package runner implements the private Linux execution boundary used by the
// guided extension. It does not provision resources or accept arbitrary commands.
package runner

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/rioriost/agefreighter/internal/config"
)

const MaxRequestBytes = 1 << 20
const MaxArtifactBytes = 4 << 20
const ChunkBytes = 1536 // Base64 plus metadata fits the ARM 4 KiB response bound.

var uuid = regexp.MustCompile(`^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$`)

type Request struct {
	Version        int               `json:"version"`
	Workflow       string            `json:"workflow"`
	Operation      string            `json:"operation"`
	Action         string            `json:"action"`
	ExpectedBootID string            `json:"expectedBootId,omitempty"`
	Configuration  json.RawMessage   `json:"configuration,omitempty"`
	Secrets        map[string]string `json:"secrets,omitempty"`
	Offset         int64             `json:"offset,omitempty"`
	Export         *ReportExport     `json:"export,omitempty"`
	Import         *CSVImport        `json:"import,omitempty"`
}

type State struct {
	Version      int    `json:"version"`
	Workflow     string `json:"workflow"`
	Operation    string `json:"operation"`
	Action       string `json:"action"`
	Phase        string `json:"phase"`
	BootID       string `json:"bootId"`
	ConfigSHA256 string `json:"configSha256"`
	StartedAt    string `json:"startedAt,omitempty"`
	FinishedAt   string `json:"finishedAt,omitempty"`
	ExitCode     *int   `json:"exitCode,omitempty"`
	ReportBytes  int64  `json:"reportBytes,omitempty"`
	ReportSHA256 string `json:"reportSha256,omitempty"`
	FileID       string `json:"fileId,omitempty"`
	FileBytes    int64  `json:"fileBytes,omitempty"`
	FileSHA256   string `json:"fileSha256,omitempty"`
}

func Decode(input io.Reader) (Request, error) {
	data, err := io.ReadAll(io.LimitReader(input, MaxRequestBytes+1))
	if err != nil || len(data) > MaxRequestBytes {
		return Request{}, errors.New("runner request exceeds its input bound")
	}
	var request Request
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&request); err != nil {
		return Request{}, errors.New("invalid runner request")
	}
	if d.Decode(new(any)) != io.EOF {
		return Request{}, errors.New("runner request must contain one JSON value")
	}
	if request.Version != 1 || !uuid.MatchString(request.Workflow) || !uuid.MatchString(request.Operation) {
		return Request{}, errors.New("invalid runner protocol version or operation identity")
	}
	switch request.Action {
	case "ready", "profile", "inventory", "status", "report", "export-report", "import-csv":
	default:
		return Request{}, errors.New("runner operation is not allowed")
	}
	if request.Action == "import-csv" {
		if !safeCSVAction(request) {
			return Request{}, errors.New("invalid CSV import control")
		}
		return request, nil
	}
	if request.Import != nil {
		return Request{}, errors.New("unexpected CSV capability")
	}
	if request.Action == "export-report" {
		if !safeExportAction(request) {
			return Request{}, errors.New("invalid report export control")
		}
		return request, nil
	}
	if request.Export != nil {
		return Request{}, errors.New("unexpected report export capability")
	}
	if request.Action == "ready" || request.Action == "status" || request.Action == "report" {
		if len(request.Configuration) > 0 || len(request.Secrets) > 0 || request.ExpectedBootID != "" || request.Offset < 0 || request.Action != "report" && request.Offset != 0 {
			return Request{}, errors.New("read-only request contains unexpected fields")
		}
	} else if len(request.Configuration) == 0 || request.Offset != 0 || !uuid.MatchString(request.ExpectedBootID) {
		return Request{}, errors.New("assessment configuration and checked boot identity are required")
	}
	return request, nil
}

// ValidateConfiguration constrains guest file access and environment injection.
// Database queries are subsequently checked by the connector's read-only parser.
func ValidateConfiguration(request Request, workflowRoot string) ([]byte, error) {
	job, err := config.Parse(request.Configuration)
	if err != nil {
		return nil, errors.New("invalid assessment LoadJob configuration")
	}
	if job.Trial != nil {
		return nil, errors.New("trial writes are not supported by assessment")
	}
	refs := []config.SecretRef{job.Target.Connection}
	if job.Source.PostgreSQL != nil {
		refs = append(refs, job.Source.PostgreSQL.Connection)
	}
	if job.Source.Neo4j != nil && job.Source.Neo4j.Password != nil {
		refs = append(refs, *job.Source.Neo4j.Password)
	}
	for _, ref := range refs {
		if ref.File != "" || !allowedSecret(ref.Env) {
			return nil, errors.New("runner credentials must use approved environment handles")
		}
	}
	for name, value := range request.Secrets {
		if !allowedSecret(name) || len(value) > 64<<10 || strings.ContainsRune(value, 0) {
			return nil, errors.New("invalid runner secret handle or value")
		}
	}
	if request.Action == "inventory" && job.Source.Type != config.SourceNeo4j {
		return nil, errors.New("exact inventory currently requires Neo4j; bounded profiles are not totals")
	}
	if job.Source.CSV != nil {
		for _, v := range job.Source.CSV.Vertices {
			if !insideUploads(workflowRoot, v.Path) {
				return nil, errors.New("CSV files must belong to this workflow's verified upload directory")
			}
		}
		for _, e := range job.Source.CSV.Edges {
			if !insideUploads(workflowRoot, e.Path) {
				return nil, errors.New("CSV files must belong to this workflow's verified upload directory")
			}
		}
	}
	// Profiling never quarantines or writes to the target. Still bind any future
	// accidental use of this field to the operation's private directory.
	job.Errors.QuarantinePath = filepath.Join(workflowRoot, request.Operation, "quarantine.jsonl")
	return json.Marshal(job)
}

func allowedSecret(name string) bool {
	switch name {
	case "AGEFREIGHTER_SOURCE_DSN", "AGEFREIGHTER_SOURCE_PASSWORD", "AGEFREIGHTER_TARGET_DSN":
		return true
	}
	return false
}

func insideUploads(root, path string) bool {
	if !filepath.IsAbs(path) {
		return false
	}
	rel, err := filepath.Rel(filepath.Join(root, "uploads"), path)
	return err == nil && rel != "." && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func validateUploadPaths(configuration []byte, root string) error {
	var job config.LoadJob
	if json.Unmarshal(configuration, &job) != nil {
		return errors.New("invalid retained configuration")
	}
	if job.Source.CSV == nil {
		return nil
	}
	paths := []string{}
	for _, v := range job.Source.CSV.Vertices {
		paths = append(paths, v.Path)
	}
	for _, e := range job.Source.CSV.Edges {
		paths = append(paths, e.Path)
	}
	for _, path := range paths {
		real, err := filepath.EvalSymlinks(path)
		if err != nil || !insideUploads(root, real) {
			return errors.New("CSV upload path is missing or escapes the workflow")
		}
		info, err := os.Stat(real)
		if err != nil || !info.Mode().IsRegular() {
			return errors.New("CSV upload must be a regular file")
		}
		if err := verifyCSVSeal(real); err != nil {
			return err
		}
	}
	return nil
}

func Arguments(action, path string) ([]string, error) {
	switch action {
	case "profile":
		return []string{"profile", path, "--mode", "sample", "--sample-size", "10000", "--format", "json"}, nil
	case "inventory":
		return []string{"inventory", path, "--format", "json"}, nil
	default:
		return nil, errors.New("runner cannot execute this action")
	}
}
