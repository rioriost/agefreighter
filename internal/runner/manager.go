package runner

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/rioriost/agefreighter/internal/report"
)

type Manager struct {
	Root          string
	UnitDirectory string
	CLI           string
	Tools         string
	BootID        func() (string, error)
	Start         func(context.Context, string) error
}

func (m Manager) paths(workflow, operation string) (string, string, error) {
	if !uuid.MatchString(workflow) || !uuid.MatchString(operation) {
		return "", "", errors.New("invalid operation identity")
	}
	root := filepath.Join(m.Root, workflow)
	return root, filepath.Join(root, operation), nil
}

func (m Manager) Submit(ctx context.Context, request Request) (State, error) {
	if _, err := Arguments(request.Action, ""); err != nil {
		return State{}, err
	}
	root, dir, err := m.paths(request.Workflow, request.Operation)
	if err != nil {
		return State{}, err
	}
	configuration, err := ValidateConfiguration(request, root)
	if err != nil {
		return State{}, err
	}
	boot, err := m.BootID()
	if err != nil {
		return State{}, errors.New("cannot read guest boot identity")
	}
	if !uuid.MatchString(request.ExpectedBootID) || boot != request.ExpectedBootID {
		return State{}, errors.New("guest boot identity changed; check readiness again")
	}
	if err := os.MkdirAll(root, 0700); err != nil {
		return State{}, err
	}
	// Private parent plus UUID-only paths prevent other principals choosing files.
	if err := privateDirectory(root); err != nil {
		return State{}, err
	}
	if err := os.Mkdir(dir, 0700); err != nil {
		return State{}, errors.New("operation already exists or cannot be created; query its status, do not replay")
	}
	if err := writeNew(filepath.Join(root, "active"), []byte(request.Operation)); err != nil {
		return State{}, errors.New("workflow has an active or unreconciled operation")
	}
	state := State{Version: 1, Workflow: request.Workflow, Operation: request.Operation, Action: request.Action, Phase: "accepted", BootID: boot, ConfigSHA256: sum(configuration)}
	if err := writeNewJSON(filepath.Join(dir, "state.json"), state); err != nil {
		return State{}, err
	}
	if err := writeNew(filepath.Join(dir, "job.json"), configuration); err != nil {
		return State{}, err
	}
	if err := writeNewJSON(filepath.Join(dir, "secrets.json"), request.Secrets); err != nil {
		return State{}, err
	}
	unit, err := m.unit(request.Workflow, request.Operation, root)
	if err != nil {
		return State{}, err
	}
	name := unitName(request.Operation)
	if err := writeNew(filepath.Join(m.UnitDirectory, name), []byte(unit)); err != nil {
		return State{}, err
	}
	// State and unit exist before starting. A lost start response is not permission
	// to launch again, even if the short ARM dispatch reports failure.
	if err := m.Start(ctx, name); err != nil {
		return state, errors.New("guest start result is uncertain; inspect retained operation")
	}
	return state, nil
}

func (m Manager) Status(workflow, operation string) (State, error) {
	_, dir, err := m.paths(workflow, operation)
	if err != nil {
		return State{}, err
	}
	var state State
	if err := readJSON(filepath.Join(dir, "state.json"), &state); err != nil {
		return State{}, errors.New("operation state is unavailable")
	}
	if state.Version != 1 || state.Workflow != workflow || state.Operation != operation {
		return State{}, errors.New("operation state identity mismatch")
	}
	if state.Phase == "accepted" || state.Phase == "running" {
		boot, err := m.BootID()
		if err != nil {
			return State{}, errors.New("cannot read guest boot identity")
		}
		if boot != state.BootID {
			state.Phase = "interrupted"
		}
	}
	return state, nil
}

// Work is invoked by a persistent, disabled-at-boot systemd service. The exclusive
// worker marker survives crashes: restarting the unit cannot repeat source work.
func (m Manager) Work(ctx context.Context, workflow, operation string) error {
	root, dir, err := m.paths(workflow, operation)
	if err != nil {
		return err
	}
	state, err := m.Status(workflow, operation)
	if err != nil {
		return err
	}
	if state.Phase != "accepted" {
		return errors.New("operation cannot be automatically resumed")
	}
	if err := writeNew(filepath.Join(dir, "worker.claim"), []byte(state.BootID)); err != nil {
		return errors.New("worker already claimed; automatic replay is forbidden")
	}
	configuration, err := os.ReadFile(filepath.Join(dir, "job.json"))
	if err != nil || sum(configuration) != state.ConfigSHA256 {
		return errors.New("assessment configuration changed")
	}
	var secrets map[string]string
	if err := readJSON(filepath.Join(dir, "secrets.json"), &secrets); err != nil {
		return errors.New("source secrets unavailable")
	}
	if _, err := ValidateConfiguration(Request{Workflow: workflow, Operation: operation, Action: state.Action, Configuration: configuration, Secrets: secrets}, root); err != nil {
		return err
	}
	// Resolve real upload paths before invoking a connector. A symlink cannot
	// turn a lexically contained CSV path into access outside this workflow.
	if err := validateUploadPaths(configuration, root); err != nil {
		return err
	}
	state.Phase = "running"
	state.StartedAt = time.Now().UTC().Format(time.RFC3339Nano)
	if err := replaceJSON(filepath.Join(dir, "state.json"), state); err != nil {
		return err
	}
	args, err := Arguments(state.Action, filepath.Join(dir, "job.json"))
	if err != nil {
		return err
	}
	deadline, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(deadline, m.CLI, args...)
	cmd.Dir = dir
	cmd.Env = []string{"PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8", "HOME=" + dir}
	keys := make([]string, 0, len(secrets))
	for key := range secrets {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		cmd.Env = append(cmd.Env, key+"="+secrets[key])
	}
	stdout, stderr := &boundedOutput{limit: MaxArtifactBytes}, &boundedOutput{limit: 64 << 10}
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	runErr := cmd.Run()
	exit := 0
	if runErr != nil {
		exit = 1
		var ee *exec.ExitError
		if errors.As(runErr, &ee) {
			exit = ee.ExitCode()
		}
	}
	state.ExitCode = &exit
	state.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
	state.Phase = "failed"
	// Raw stderr stays on the guest. It is never an ARM or webview response.
	if err := writeNew(filepath.Join(dir, "stderr.log"), stderr.Bytes()); err != nil {
		return err
	}
	if !stdout.overflow {
		if _, err := report.Decode(stdout.Bytes()); err == nil {
			data, err := redactedReport(stdout.Bytes(), secrets)
			if err != nil {
				return err
			}
			// Redacting a short secret may expand the JSON beyond the input
			// bound. Such output cannot become a retrievable terminal artifact.
			if len(data) <= MaxArtifactBytes {
				if err := writeNew(filepath.Join(dir, "report.json"), data); err != nil {
					return err
				}
				state.ReportBytes = int64(len(data))
				state.ReportSHA256 = sum(data)
				if exit == 0 {
					state.Phase = "finished"
				} // Not equivalent to report outcome pass.
			}
		}
	}
	if err := replaceJSON(filepath.Join(dir, "state.json"), state); err != nil {
		return err
	}
	// Erase only this operation's transient secret transport, never its evidence.
	if err := os.Remove(filepath.Join(dir, "secrets.json")); err != nil {
		return err
	}
	active, err := os.ReadFile(filepath.Join(root, "active"))
	if err != nil || string(active) != operation {
		return errors.New("workflow lease changed; operator reconciliation required")
	}
	return os.Remove(filepath.Join(root, "active"))
}

type ArtifactChunk struct {
	Version   int    `json:"version"`
	Operation string `json:"operation"`
	Offset    int64  `json:"offset"`
	Total     int64  `json:"total"`
	SHA256    string `json:"sha256"`
	Data      string `json:"data"`
}

func (m Manager) Report(workflow, operation string, offset int64) (ArtifactChunk, error) {
	state, err := m.Status(workflow, operation)
	if err != nil {
		return ArtifactChunk{}, err
	}
	if state.Phase != "finished" && state.Phase != "failed" || state.ReportBytes <= 0 || state.ReportBytes > MaxArtifactBytes || offset < 0 || offset >= state.ReportBytes {
		return ArtifactChunk{}, errors.New("complete report artifact is unavailable or offset invalid")
	}
	_, dir, _ := m.paths(workflow, operation)
	data, err := os.ReadFile(filepath.Join(dir, "report.json"))
	if err != nil || int64(len(data)) != state.ReportBytes || sum(data) != state.ReportSHA256 {
		return ArtifactChunk{}, errors.New("report artifact changed")
	}
	end := min(offset+ChunkBytes, int64(len(data)))
	return ArtifactChunk{Version: 1, Operation: operation, Offset: offset, Total: int64(len(data)), SHA256: state.ReportSHA256, Data: base64.StdEncoding.EncodeToString(data[offset:end])}, nil
}

func unitName(operation string) string { return "agefreighter-assessment-" + operation + ".service" }

func (m Manager) unit(workflow, operation, root string) (string, error) {
	for _, path := range []string{m.Tools, root} {
		if !regexpSafePath(path) {
			return "", errors.New("unsafe runner installation path")
		}
	}
	return fmt.Sprintf(`[Unit]
Description=AGEFreighter read-only assessment
After=network-online.target
[Service]
Type=exec
ExecStart=%s runner worker --workflow %s --operation %s
WorkingDirectory=%s
Restart=no
RuntimeMaxSec=1800
TimeoutStopSec=30
KillMode=control-group
MemoryMax=4G
MemorySwapMax=0
CPUQuota=200%%
UMask=0077
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=%s
PrivateTmp=yes
StandardOutput=null
StandardError=null
`, m.Tools, workflow, operation, root, root), nil // No [Install]: boot never resumes work.
}

func regexpSafePath(path string) bool {
	return filepath.IsAbs(path) && !strings.ContainsAny(path, " \t\r\n\"'%%\\")
}
func sum(data []byte) string { digest := sha256.Sum256(data); return hex.EncodeToString(digest[:]) }
func privateDirectory(path string) error {
	info, err := os.Lstat(path)
	if err != nil || !info.IsDir() || info.Mode().Perm()&0077 != 0 {
		return errors.New("runner directory must be private and not a symlink")
	}
	return nil
}
func writeNew(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return syncDirectory(filepath.Dir(path))
}
func writeNewJSON(path string, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return writeNew(path, data)
}
func replaceJSON(path string, value any) error {
	f, err := os.CreateTemp(filepath.Dir(path), ".state-*")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	err = json.NewEncoder(f).Encode(value)
	if err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if err := os.Rename(name, path); err != nil {
		return err
	}
	return syncDirectory(filepath.Dir(path))
}
func syncDirectory(path string) error {
	dir, err := os.Open(path)
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}
func readJSON(path string, value any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, MaxRequestBytes+1))
	if err != nil || len(data) > MaxRequestBytes {
		return errors.New("retained JSON exceeds bound")
	}
	return json.Unmarshal(data, value)
}

type boundedOutput struct {
	bytes.Buffer
	limit    int
	overflow bool
}

func (b *boundedOutput) Write(p []byte) (int, error) {
	size := len(p)
	remaining := b.limit - b.Len()
	if size > remaining {
		b.overflow = true
		p = p[:remaining]
	}
	_, _ = b.Buffer.Write(p)
	return size, nil
}

func redactedReport(data []byte, secrets map[string]string) ([]byte, error) {
	var value any
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	var redact func(any) any
	redact = func(value any) any {
		switch v := value.(type) {
		case string:
			for _, secret := range secrets {
				if secret != "" {
					v = strings.ReplaceAll(v, secret, "[REDACTED]")
				}
			}
			return v
		case []any:
			for i := range v {
				v[i] = redact(v[i])
			}
		case map[string]any:
			for key, item := range v {
				v[key] = redact(item)
			}
		}
		return value
	}
	return json.Marshal(redact(value))
}
