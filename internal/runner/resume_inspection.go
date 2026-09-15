package runner

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
)

// ResumeInspection is evidence, never authority to start a worker. In particular,
// a stale running checkpoint or a retained lease must not be silently repaired.
// Decimal strings avoid losing 64-bit identities/counters in the extension.
type ResumeInspection struct {
	Version       int      `json:"version"`
	Workflow      string   `json:"workflow"`
	Operation     string   `json:"operation"`
	JobID         string   `json:"jobId"`
	BootID        string   `json:"bootId"`
	ConfigSHA256  string   `json:"configSha256"`
	Fingerprint   string   `json:"fingerprint"`
	GenerationID  string   `json:"generationId"`
	CommittedRows string   `json:"committedRows"`
	CheckpointAt  string   `json:"checkpointAt"`
	CheckedAt     string   `json:"checkedAt"`
	Outcome       string   `json:"outcome"`
	Reasons       []string `json:"reasons"`
	CanResume     bool     `json:"canResume"`
}

// InspectResume uses only retained configuration and read-only target queries.
// It does not accept source credentials/configuration, create files, clear leases,
// modify metadata, install AGE, or invoke load/resume.
func (m Manager) InspectResume(ctx context.Context, request Request) (ResumeInspection, error) {
	encoded, err := json.Marshal(request)
	if err != nil {
		return ResumeInspection{}, err
	}
	if _, err = Decode(bytes.NewReader(encoded)); err != nil || request.Action != "inspect-resume" {
		return ResumeInspection{}, errors.New("invalid resume inspection request")
	}
	root, dir, err := m.paths(request.Workflow, request.Operation)
	if err != nil {
		return ResumeInspection{}, err
	}
	for _, path := range []string{root, dir} {
		if err := privateDirectory(path); err != nil {
			return ResumeInspection{}, errors.New("private retained migration directory unavailable")
		}
	}
	state, err := m.Status(request.Workflow, request.Operation)
	if err != nil || !uuid.MatchString(state.JobID) || (state.Action != "migrate-csv" && state.Action != "migrate-source" && state.Action != "resume-migration") {
		return ResumeInspection{}, errors.New("retained migration identity unavailable")
	}
	boot, err := m.BootID()
	if err != nil || boot != request.ExpectedBootID {
		return ResumeInspection{}, errors.New("guest boot changed; refresh readiness")
	}
	info, err := os.Lstat(filepath.Join(dir, "job.json"))
	if err != nil || !info.Mode().IsRegular() || info.Size() > MaxRequestBytes {
		return ResumeInspection{}, errors.New("retained configuration unavailable")
	}
	data, err := os.ReadFile(filepath.Join(dir, "job.json"))
	if err != nil || sum(data) != state.ConfigSHA256 {
		return ResumeInspection{}, errors.New("retained configuration changed")
	}
	job, err := config.Parse(data)
	if err != nil || job.Target.Type != config.TargetApacheAGE || job.Target.Mode != config.LoadCreate || job.Target.Connection.Env != "AGEFREIGHTER_TARGET_DSN" {
		return ResumeInspection{}, errors.New("unsupported retained target")
	}
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	c, err := migrationConnection(request.Secrets["AGEFREIGHTER_TARGET_DSN"])
	if err != nil {
		return ResumeInspection{}, err
	}
	conn, err := pgx.ConnectConfig(ctx, c)
	if err != nil {
		return ResumeInspection{}, errors.New("resume inspection target connection unavailable")
	}
	defer conn.Close(context.Background())
	tx, err := conn.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly})
	if err != nil {
		return ResumeInspection{}, errors.New("read-only inspection transaction unavailable")
	}
	defer tx.Rollback(context.Background())
	store, err := meta.New(tx)
	if err != nil {
		return ResumeInspection{}, errors.New("read-only metadata unavailable")
	}
	stored, err := store.GetJob(ctx, state.JobID)
	if err != nil {
		return ResumeInspection{}, errors.New("retained target job unavailable; no new job will be created")
	}
	generation, err := store.GraphGenerationForJob(ctx, state.JobID)
	if err != nil {
		return ResumeInspection{}, errors.New("retained target generation unavailable")
	}
	verification, err := store.GetJobVerification(ctx, state.JobID)
	if err != nil {
		return ResumeInspection{}, errors.New("original submitted mapping evidence unavailable")
	}
	result, err := resumeInspectionEvidence(state, job, stored, generation, verification, boot, time.Now().UTC())
	if err != nil {
		return ResumeInspection{}, err
	}
	if _, err := os.Lstat(filepath.Join(root, "active")); err == nil {
		result.Reasons = append(result.Reasons, "retained workflow lease requires explicit reconciliation")
	} else if !os.IsNotExist(err) {
		return ResumeInspection{}, errors.New("workflow lease evidence unavailable")
	}
	return result, nil
}

func resumeInspectionEvidence(state State, job config.LoadJob, stored meta.Job, generation meta.GraphGeneration, verification meta.JobVerification, boot string, now time.Time) (ResumeInspection, error) {
	if stored.ID != state.JobID || stored.TargetBackend != meta.TargetBackendApacheAGE || stored.TargetGraph != job.Target.Graph || stored.SourceType != string(job.Source.Type) || stored.LoadMode != "create" || stored.GraphGenerationID <= 0 || generation.ID != stored.GraphGenerationID || generation.JobID != stored.ID || generation.GraphName != stored.TargetGraph || generation.State != meta.GenerationLoading || verification.JobID != stored.ID || verification.SubmittedConfigFingerprint != state.ConfigSHA256 || !validDigest(stored.ConfigFingerprint) || !validDigest(verification.ResolvedMappingFingerprint) {
		return ResumeInspection{}, errors.New("original job, graph, generation or submitted configuration does not match")
	}
	if stored.UpdatedAt.IsZero() || stored.UpdatedAt.After(now) || stored.CommittedRows < 0 || stored.RejectedRows != 0 || stored.SourceRejectedRows != 0 {
		return ResumeInspection{}, errors.New("invalid checkpoint or rejected rows; review retained evidence")
	}
	result := ResumeInspection{Version: 1, Workflow: state.Workflow, Operation: state.Operation, JobID: state.JobID, BootID: boot, ConfigSHA256: state.ConfigSHA256, Fingerprint: stored.ConfigFingerprint, GenerationID: strconv.FormatInt(generation.ID, 10), CommittedRows: strconv.FormatInt(stored.CommittedRows, 10), CheckpointAt: stored.UpdatedAt.UTC().Format(time.RFC3339Nano), CheckedAt: now.Format(time.RFC3339Nano), Outcome: "review-required", Reasons: []string{"read-only inspection; a separate explicit resume approval is required"}, CanResume: false}
	if stored.Status != meta.JobFailed || (state.Phase != "failed" && state.Phase != "interrupted") {
		result.Reasons = append(result.Reasons, "worker and target must be proven inactive before any resume")
	}
	if now.Sub(stored.UpdatedAt) > 15*time.Minute {
		result.Reasons = append(result.Reasons, "checkpoint older than 15 minutes; recovery review required")
	}
	return result, nil
}

func validDigest(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, c := range value {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

func (m Manager) checkResumedGeneration(ctx context.Context, state State, data []byte, dsn string) error {
	if m.resumeComplete != nil {
		return m.resumeComplete(ctx, state, data, dsn)
	}
	job, err := config.Parse(data)
	if err != nil {
		return err
	}
	c, err := migrationConnection(dsn)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	conn, err := pgx.ConnectConfig(ctx, c)
	if err != nil {
		return errors.New("final recovery target unavailable")
	}
	defer conn.Close(context.Background())
	tx, err := conn.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly})
	if err != nil {
		return errors.New("final read-only recovery check unavailable")
	}
	defer tx.Rollback(context.Background())
	store, _ := meta.New(tx)
	stored, err := store.GetJob(ctx, state.JobID)
	if err != nil {
		return errors.New("final recovery job unavailable")
	}
	g, err := store.GraphGenerationForJob(ctx, state.JobID)
	if err != nil {
		return errors.New("final recovery generation unavailable")
	}
	if stored.ID != state.JobID || stored.Status != meta.JobCommitted || stored.ConfigFingerprint != state.Resume.Fingerprint || stored.TargetGraph != job.Target.Graph || stored.TargetBackend != meta.TargetBackendApacheAGE || strconv.FormatInt(stored.GraphGenerationID, 10) != state.Resume.GenerationID || g.ID != stored.GraphGenerationID || g.JobID != state.JobID || g.GraphName != job.Target.Graph || g.State != meta.GenerationActive {
		return errors.New("recovery changed the original committed job or graph generation")
	}
	return nil
}
