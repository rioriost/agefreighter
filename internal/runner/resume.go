package runner

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Each continuation gets a new operation directory, but never a new load job.
type ResumeBinding struct {
	PreviousOperation string `json:"previousOperation"`
	JobID             string `json:"jobId"`
	ConfigSHA256      string `json:"configSha256"`
	Fingerprint       string `json:"fingerprint"`
	GenerationID      string `json:"generationId"`
	CommittedRows     string `json:"committedRows"`
}

func validResumeBinding(b *ResumeBinding) bool {
	if b == nil || !uuid.MatchString(b.PreviousOperation) || !uuid.MatchString(b.JobID) || !validDigest(b.ConfigSHA256) || !validDigest(b.Fingerprint) {
		return false
	}
	generation, e := strconv.ParseInt(b.GenerationID, 10, 64)
	rows, e2 := strconv.ParseInt(b.CommittedRows, 10, 64)
	return e == nil && e2 == nil && generation > 0 && rows >= 0 && strconv.FormatInt(generation, 10) == b.GenerationID && strconv.FormatInt(rows, 10) == b.CommittedRows
}
func (m Manager) checkInactive(ctx context.Context, operation string) error {
	if m.workerInactive != nil {
		return m.workerInactive(ctx, operation)
	}
	// Query fixed properties only. No stop/reset-failed/restart operation.
	cmd := exec.CommandContext(ctx, "/bin/systemctl", "show", unitName(operation), "--property=LoadState,ActiveState,MainPID,ControlPID")
	out := &boundedOutput{limit: 4096}
	cmd.Stdout = out
	if cmd.Run() != nil || out.overflow {
		return errors.New("previous worker inactivity is unproven")
	}
	fields := map[string]string{}
	for _, line := range strings.Split(strings.TrimSpace(out.String()), "\n") {
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			return errors.New("invalid worker state")
		}
		fields[k] = v
	}
	if fields["LoadState"] != "loaded" || (fields["ActiveState"] != "inactive" && fields["ActiveState"] != "failed") || fields["MainPID"] != "0" || fields["ControlPID"] != "0" {
		return errors.New("previous worker is active or unavailable; no resume allowed")
	}
	return nil
}
func (m Manager) checkResumeTarget(ctx context.Context, r Request) error {
	b := r.Resume
	probe := m.InspectResume
	if m.resumeProbe != nil {
		probe = m.resumeProbe
	}
	evidence, err := probe(ctx, Request{Version: 1, Workflow: r.Workflow, Operation: b.PreviousOperation, Action: "inspect-resume", ExpectedBootID: r.ExpectedBootID, Secrets: map[string]string{"AGEFREIGHTER_TARGET_DSN": r.Secrets["AGEFREIGHTER_TARGET_DSN"]}})
	if err != nil {
		return err
	}
	checked, e := time.Parse(time.RFC3339Nano, evidence.CheckedAt)
	checkpoint, e2 := time.Parse(time.RFC3339Nano, evidence.CheckpointAt)
	now := time.Now()
	if evidence.Workflow != r.Workflow || evidence.Operation != b.PreviousOperation || evidence.JobID != b.JobID || evidence.BootID != r.ExpectedBootID || evidence.ConfigSHA256 != b.ConfigSHA256 || evidence.Fingerprint != b.Fingerprint || evidence.GenerationID != b.GenerationID || evidence.CommittedRows != b.CommittedRows || e != nil || e2 != nil || checked.After(now) || now.Sub(checked) > time.Minute || checkpoint.After(now) || now.Sub(checkpoint) > 15*time.Minute {
		return errors.New("reviewed recovery identity/checkpoint changed or expired")
	}
	return nil
}

// SubmitResume seals a single continuation before replacing only its proven
// inactive predecessor's lease. All old files and claims survive. Crash windows
// are fail-closed: a claimed continuation is never silently recreated.
func (m Manager) SubmitResume(ctx context.Context, r Request) (State, error) {
	data, err := json.Marshal(r)
	if err != nil {
		return State{}, err
	}
	if _, err := Decode(bytes.NewReader(data)); err != nil || r.Action != "resume-migration" {
		return State{}, errors.New("invalid explicit resume request")
	}
	unlock, err := m.dispatchLock()
	if err != nil {
		return State{}, err
	}
	defer unlock()
	b := r.Resume
	root, dir, err := m.paths(r.Workflow, r.Operation)
	if err != nil {
		return State{}, err
	}
	_, previous, _ := m.paths(r.Workflow, b.PreviousOperation)
	for _, path := range []string{root, previous} {
		if err := privateDirectory(path); err != nil {
			return State{}, err
		}
	}
	state, err := m.Status(r.Workflow, b.PreviousOperation)
	if err != nil || state.JobID != b.JobID || state.ConfigSHA256 != b.ConfigSHA256 || (state.Phase != "failed" && state.Phase != "interrupted") || (state.Action != "migrate-csv" && state.Action != "migrate-source" && state.Action != "resume-migration") {
		return State{}, errors.New("previous operation is not a matching stopped migration")
	}
	if state.Action == "resume-migration" && (!validResumeBinding(state.Resume) || state.Resume.JobID != state.JobID || state.Resume.ConfigSHA256 != state.ConfigSHA256) {
		return State{}, errors.New("previous continuation lineage is invalid")
	}
	if err := m.checkInactive(ctx, b.PreviousOperation); err != nil {
		return State{}, err
	}
	boot, err := m.BootID()
	if err != nil || boot != r.ExpectedBootID {
		return State{}, errors.New("guest boot changed")
	}
	info, err := os.Lstat(filepath.Join(previous, "job.json"))
	if err != nil || !info.Mode().IsRegular() || info.Size() > MaxRequestBytes {
		return State{}, errors.New("retained job unavailable")
	}
	configuration, err := os.ReadFile(filepath.Join(previous, "job.json"))
	if err != nil || sum(configuration) != b.ConfigSHA256 {
		return State{}, errors.New("retained configuration changed")
	}
	checked := r
	checked.Configuration = configuration
	if _, err := ValidateConfiguration(checked, root); err != nil {
		return State{}, err
	}
	healthProbe := m.health
	if m.healthProbe != nil {
		healthProbe = m.healthProbe
	}
	h, err := healthProbe(ctx)
	if err != nil || h == nil || math.IsNaN(h.StorageUsedPercent) || math.IsInf(h.StorageUsedPercent, 0) || h.StorageUsedPercent < 0 || h.StorageUsedPercent >= 80 || h.SwapUsedBytes != 0 || h.OOMEvents != 0 {
		return State{}, errors.New("unsafe guest storage/swap/OOM evidence")
	}
	leases, err := filepath.Glob(filepath.Join(m.Root, "*", "active"))
	if err != nil {
		return State{}, err
	}
	for _, path := range leases {
		value, err := os.ReadFile(path)
		if err != nil || path != filepath.Join(root, "active") || string(value) != b.PreviousOperation {
			return State{}, errors.New("another active/unreconciled workflow exists")
		}
	}
	if !h.Idle && len(leases) == 0 {
		return State{}, errors.New("guest activity is unaccounted for")
	}
	if err := m.checkResumeTarget(ctx, r); err != nil {
		return State{}, err
	}
	if err := writeNew(filepath.Join(previous, "resume.claim"), []byte(r.Operation)); err != nil {
		return State{}, errors.New("a continuation is already claimed; inspect it, never replay")
	}
	if err := os.Mkdir(dir, 0700); err != nil {
		return State{}, errors.New("continuation already exists; reconcile it")
	}
	next := State{Version: 1, Workflow: r.Workflow, Operation: r.Operation, Action: r.Action, Phase: "accepted", BootID: boot, ConfigSHA256: b.ConfigSHA256, JobID: b.JobID, Fingerprint: b.Fingerprint, Resume: b}
	if err := writeNewJSON(filepath.Join(dir, "state.json"), next); err != nil {
		return State{}, err
	}
	if err := writeNew(filepath.Join(dir, "job.json"), configuration); err != nil {
		return State{}, err
	}
	if err := writeNewJSON(filepath.Join(dir, "secrets.json"), r.Secrets); err != nil {
		return State{}, err
	}
	unit, err := m.unit(r.Workflow, r.Operation, root)
	if err != nil {
		return State{}, err
	}
	if err := writeNew(filepath.Join(m.UnitDirectory, unitName(r.Operation)), []byte(unit)); err != nil {
		return State{}, err
	}
	if len(leases) == 0 {
		err = writeNew(filepath.Join(root, "active"), []byte(r.Operation))
	} else {
		err = replaceBytes(filepath.Join(root, "active"), []byte(r.Operation))
	}
	if err != nil {
		return State{}, err
	}
	if err := m.Start(ctx, unitName(r.Operation)); err != nil {
		return next, errors.New("resume start response uncertain; inspect retained continuation")
	}
	return next, nil
}

func replaceBytes(path string, data []byte) error {
	f, err := os.CreateTemp(filepath.Dir(path), ".lease-*")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	_, err = f.Write(data)
	if err == nil {
		err = f.Sync()
	}
	if e := f.Close(); err == nil {
		err = e
	}
	if err != nil {
		return err
	}
	if err := os.Rename(name, path); err != nil {
		return err
	}
	return syncDirectory(filepath.Dir(path))
}
