package runner

import (
	"context"
	"encoding/json"
	"errors"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/report"
)

// The initial migration boundary deliberately supports only a private Flexible
// Server selected by the host controller. No multi-host, connection option, local
// socket, service-file, plaintext, or credential-bearing command line is allowed.
func migrationConnection(dsn string) (*pgx.ConnConfig, error) {
	u, err := url.Parse(dsn)
	if err != nil || u.Scheme != "postgresql" || u.User == nil || u.User.Username() != "afadmin" || u.Path != "/agefreighter" || u.Fragment != "" || u.Port() != "5432" || !regexp.MustCompile(`^[a-z][a-z0-9-]{2,61}[a-z0-9]\.postgres\.database\.azure\.com$`).MatchString(u.Hostname()) {
		return nil, errors.New("migration requires the reviewed Flexible Server TLS connection")
	}
	q, err := url.ParseQuery(u.RawQuery)
	if err != nil || len(q) != 1 || len(q["sslmode"]) != 1 || q.Get("sslmode") != "verify-full" {
		return nil, errors.New("verified TLS without additional connection options is required")
	}
	if p, ok := u.User.Password(); !ok || len(p) < 24 {
		return nil, errors.New("target credential is unavailable")
	}
	c, err := pgx.ParseConfig(dsn)
	if err != nil || c.TLSConfig == nil || c.TLSConfig.InsecureSkipVerify || len(c.Fallbacks) != 0 {
		return nil, errors.New("target TLS validation is unavailable")
	}
	c.ConnectTimeout = 20 * time.Second
	return c, nil
}

func prepareMigration(ctx context.Context, configuration []byte, dsn string) error {
	c, err := migrationConnection(dsn)
	if err != nil {
		return err
	}
	connection, err := pgx.ConnectConfig(ctx, c)
	if err != nil {
		return errors.New("private target TLS connection failed; credentials are not included in this error")
	}
	defer connection.Close(context.Background())
	var version int
	var tls bool
	if err := connection.QueryRow(ctx, "SELECT current_setting('server_version_num')::int, (SELECT ssl FROM pg_stat_ssl WHERE pid=pg_backend_pid())").Scan(&version, &tls); err != nil || version < 180000 || version >= 190000 || !tls {
		return errors.New("PostgreSQL 18 and an encrypted session are required")
	}
	// This is the only direct preparation write. No role, firewall, schema drop,
	// arbitrary SQL, or source mutation is accepted by this worker.
	if _, err := connection.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS age"); err != nil {
		return errors.New("AGE extension preparation failed; review allowlist and preload configuration")
	}
	if _, err := connection.Exec(ctx, "LOAD 'age'"); err != nil {
		return errors.New("AGE session initialization failed")
	}
	var job config.LoadJob
	if json.Unmarshal(configuration, &job) != nil {
		return errors.New("invalid retained target configuration")
	}
	var exists bool
	if err := connection.QueryRow(ctx, "SELECT EXISTS(SELECT 1 FROM ag_catalog.ag_graph WHERE name=$1)", job.Target.Graph).Scan(&exists); err != nil || exists {
		return errors.New("create-mode migration requires a new target graph; existing data will not be replaced")
	}
	return nil
}

// One durable worker performs a fixed sequence. The operation UUID is the load
// UUID and is sealed before target preparation. Failure never starts a new load
// or resumes an old one. The final artifact is complete counts verification,
// not a success inferred from the loader exit code.
func (m Manager) workMigration(ctx context.Context, root, dir string, state State, configuration []byte, secrets map[string]string) error {
	if state.JobID != state.Operation || !uuid.MatchString(state.JobID) {
		return errors.New("migration identity is not retained")
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()
	dsn := secrets["AGEFREIGHTER_TARGET_DSN"]
	prepCtx, prepCancel := context.WithTimeout(ctx, 2*time.Minute)
	prepare := prepareMigration
	if m.migrationPrepare != nil {
		prepare = m.migrationPrepare
	}
	runErr := prepare(prepCtx, configuration, dsn)
	prepCancel()
	if runErr == nil {
		runErr = writeNewJSON(filepath.Join(dir, "target-prepared.json"), map[string]any{"jobId": state.JobID, "preparedAt": time.Now().UTC(), "tls": "verify-full", "postgresqlMajor": 18})
	}
	var final []byte
	exit := 1
	if runErr == nil {
		path := filepath.Join(dir, "job.json")
		for _, step := range []struct {
			name string
			args []string
		}{
			{"load", []string{"load", path, "--job-id", state.JobID}},
			{"verify", []string{"verify", state.JobID, "--target", path, "--counts", "--require-complete", "--format", "json"}},
		} {
			cmd := exec.CommandContext(ctx, m.CLI, step.args...)
			cmd.Dir = dir
			cmd.Env = []string{"PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8", "HOME=" + dir, "AGEFREIGHTER_TARGET_DSN=" + dsn}
			out, log := &boundedOutput{limit: MaxArtifactBytes}, &boundedOutput{limit: 64 << 10}
			cmd.Stdout = out
			cmd.Stderr = log
			runErr = cmd.Run()
			if err := writeNew(filepath.Join(dir, step.name+".stderr.log"), log.Bytes()); err != nil {
				return err
			}
			if out.overflow {
				runErr = errors.New("migration output exceeded its bound")
			} else if err := writeNew(filepath.Join(dir, step.name+".json"), out.Bytes()); err != nil {
				return err
			}
			if step.name == "load" && runErr == nil {
				var result struct {
					JobID  string `json:"jobId"`
					Status string `json:"status"`
				}
				if json.Unmarshal(out.Bytes(), &result) != nil || result.JobID != state.JobID || result.Status != "committed" {
					runErr = errors.New("load did not return the retained committed job identity")
				}
			}
			if step.name == "verify" && !out.overflow {
				doc, err := report.Decode(out.Bytes())
				if err == nil && doc.Command == "verify" && doc.Job != nil && doc.Job.ID == state.JobID {
					state.Fingerprint = doc.Job.ConfigFingerprint
					final, err = redactedReport(out.Bytes(), secrets)
					if err != nil {
						return err
					}
					if runErr == nil && doc.Outcome == report.OutcomePass && len(doc.IncompleteChecks) == 0 && len(doc.Errors) == 0 {
						exit = 0
					}
				}
			}
			if runErr != nil {
				break
			}
		}
	}
	// Errors are intentionally generic in the exported state. Raw CLI evidence
	// stays in this root-owned directory. No passwords or DSNs enter ARM output.
	if runErr != nil {
		if err := writeNew(filepath.Join(dir, "migration-error.txt"), []byte("Fixed migration sequence failed; inspect retained private step evidence. No automatic retry was made.\n")); err != nil {
			return err
		}
	}
	state.Phase = "failed"
	state.ExitCode = &exit
	state.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
	if len(final) > 0 && len(final) <= MaxArtifactBytes {
		if err := writeNew(filepath.Join(dir, "report.json"), final); err != nil {
			return err
		}
		state.ReportBytes = int64(len(final))
		state.ReportSHA256 = sum(final)
		if exit == 0 {
			state.Phase = "finished"
		}
	}
	if err := replaceJSON(filepath.Join(dir, "state.json"), state); err != nil {
		return err
	}
	if err := os.Remove(filepath.Join(dir, "secrets.json")); err != nil {
		return err
	}
	lease, err := os.ReadFile(filepath.Join(root, "active"))
	if err != nil || strings.TrimSpace(string(lease)) != state.Operation {
		return errors.New("workflow lease changed; reconcile retained evidence")
	}
	return os.Remove(filepath.Join(root, "active"))
}
