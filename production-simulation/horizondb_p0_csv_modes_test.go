package simulation_test

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/app"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
	"github.com/rioriost/agefreighter/production-simulation/internal/rangedigest"
)

// Two complete CSV scenario sequences, not a complete twelve-route P0 phase.
func TestHorizonDBP0CSVModeIntegration(t *testing.T) {
	if os.Getenv("AGEFREIGHTER_REQUIRE_HORIZON_P0_CSV_MODES") != "1" {
		t.Skip("explicit P0 CSV mode execution and a fresh independent lease required")
	}
	output := os.Getenv("AGEFREIGHTER_HORIZON_SCALE_OUTPUT")
	run := os.Getenv("AGEFREIGHTER_HORIZON_SCALE_RUN_ID")
	commit := os.Getenv("AGEFREIGHTER_HORIZON_CANDIDATE_COMMIT")
	if !filepath.IsAbs(output) || !regexp.MustCompile(`^r[a-z0-9]{6,30}$`).MatchString(run) ||
		!regexp.MustCompile(`^[a-f0-9]{40}$`).MatchString(commit) {
		t.Fatal("new absolute evidence directory, unique run ID and exact candidate commit required")
	}
	var guard struct {
		VerifiedAbsentProbe bool      `json:"verifiedAbsentProbe"`
		Enrolled            bool      `json:"enrolledScaleTargets"`
		Deadline            time.Time `json:"deadlineUtc"`
	}
	body, err := os.ReadFile(os.Getenv("AGEFREIGHTER_HORIZON_SCALE_GUARD_RECEIPT"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(body, &guard); err != nil {
		t.Fatal(err)
	}
	if !guard.VerifiedAbsentProbe || !guard.Enrolled || time.Until(guard.Deadline) < 90*time.Minute {
		t.Fatal("verified independent lifecycle and at least ninety minutes of enrolled execution required")
	}
	if err := os.Mkdir(output, 0700); err != nil {
		t.Fatal(err)
	}
	paths := rangedigest.Scenario{Base: filepath.Join(output, "base", "manifest.json"),
		Append: filepath.Join(output, "append", "manifest.json"), Upsert: filepath.Join(output, "upsert", "manifest.json")}
	if _, err := fixture.Generate(t.Context(), fixture.GenerateConfig{
		Phase: fixture.PhaseP0, Output: filepath.Dir(paths.Base), Shards: 4, Workers: 2, Seed: 20260829,
	}); err != nil {
		t.Fatal(err)
	}
	for path, kind := range map[string]fixture.DeltaKind{paths.Append: fixture.DeltaAppend, paths.Upsert: fixture.DeltaUpsert} {
		if _, err := fixture.GenerateDelta(t.Context(), paths.Base, filepath.Dir(path), kind); err != nil {
			t.Fatal(err)
		}
	}
	exports, directories := map[string]portable.Manifest{}, map[string]string{}
	for name, path := range map[string]string{"base": paths.Base, "append": paths.Append, "upsert": paths.Upsert} {
		dir := filepath.Join(output, "portable-"+name)
		p, err := portable.Export(t.Context(), path, dir)
		if err != nil {
			t.Fatal(err)
		}
		exports[name], directories[name] = p, dir
	}
	credential, err := azidentity.NewManagedIdentityCredential(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, major := range []int{17, 18} {
		if !t.Run(fmt.Sprintf("pg%d", major), func(t *testing.T) {
			host := os.Getenv(fmt.Sprintf("AGEFREIGHTER_HORIZON_PG%d_HOST", major))
			if !strings.HasPrefix(host, fmt.Sprintf("afhs-20261118-pg%d.", major)) ||
				!strings.HasSuffix(host, ".japaneast.horizondb.azure.com") {
				t.Fatal("enrolled scale HorizonDB writer endpoint required")
			}
			var activeGeneration int64
			for _, name := range []string{"create", "replace", "append", "upsert", "resume"} {
				if !t.Run(name, func(t *testing.T) {
					if time.Until(guard.Deadline) < 20*time.Minute {
						t.Fatal("case admission would exceed the execution lease")
					}
					ctx, cancel := context.WithTimeout(t.Context(), 15*time.Minute)
					defer cancel()
					token, err := credential.GetToken(ctx, policy.TokenRequestOptions{
						Scopes: []string{"https://ossrdbms-aad.database.windows.net/.default"},
					})
					if err != nil {
						t.Fatal(err)
					}
					if time.Until(token.ExpiresOn) < 20*time.Minute {
						t.Fatal("token must outlive the bounded case")
					}
					u := url.URL{Scheme: "postgresql", Host: host + ":5432", Path: "/postgres",
						User: url.UserPassword("afht-20261118-runner", token.Token), RawQuery: "sslmode=verify-full"}
					secret := filepath.Join(t.TempDir(), "target.dsn")
					if err := os.WriteFile(secret, []byte(u.String()), 0600); err != nil {
						t.Fatal(err)
					}
					conn, err := pgx.Connect(ctx, u.String())
					if err != nil {
						t.Fatal(err)
					}
					defer conn.Close(context.Background())
					var number int
					var pg, age string
					if err := conn.QueryRow(ctx, `SELECT current_setting('server_version_num')::integer`).Scan(&number); err != nil {
						t.Fatal(err)
					}
					if number/10000 != major {
						t.Fatal("effective PostgreSQL major differs before extension initialization")
					}
					if _, err := conn.Exec(ctx, `CREATE EXTENSION IF NOT EXISTS age`); err != nil {
						t.Fatal(err)
					}
					var preloaded string
					var pendingRestart bool
					if err := conn.QueryRow(ctx, `SELECT setting,pending_restart FROM pg_settings
						WHERE name='shared_preload_libraries'`).Scan(&preloaded, &pendingRestart); err != nil {
						t.Fatal(err)
					}
					agePreloaded := false
					for _, library := range strings.Split(preloaded, ",") {
						agePreloaded = agePreloaded || strings.TrimSpace(library) == "age"
					}
					if !agePreloaded || pendingRestart {
						t.Fatal("AGE must actually be preloaded without an unsupported pending restart")
					}
					if err := conn.QueryRow(ctx, `SELECT current_setting('server_version_num')::integer,
						version(),extversion FROM pg_extension WHERE extname='age'`).Scan(&number, &pg, &age); err != nil {
						t.Fatal(err)
					}
					if number/10000 != major || !strings.HasPrefix(age, "1.8.") {
						t.Fatal("effective SQL versions differ")
					}
					dataset, expectedPaths := "base", rangedigest.Scenario{Base: paths.Base}
					if name == "append" || name == "upsert" {
						dataset, expectedPaths.Append = name, paths.Append
					}
					if name == "upsert" {
						expectedPaths.Upsert = paths.Upsert
					}
					job := tinyJob(exports[dataset], directories[dataset], "csv", major)
					job.Metadata.Name = fmt.Sprintf("horizon-p0-csv-pg%d-%s-%s", major, name, run)
					job.Target.Graph = fmt.Sprintf("hs_p0_csv_pg%d_%s", major, run)
					job.Target.Connection = config.SecretRef{File: secret}
					job.Target.Mode = config.LoadMode(name)
					if name == "resume" {
						job.Target.Mode = config.LoadCreate
						job.Target.Graph += "_recovery"
					}
					job.Runtime.BatchRows, job.Runtime.BatchBytes = 20_000, 64<<20
					job.Runtime.MaxTargetConnections = 4
					path := filepath.Join(output, job.Metadata.Name+".json")
					writeTinyJSON(t, path, job)
					id := horizonModeJobID(t)
					writeTinyJSON(t, filepath.Join(output, job.Metadata.Name+"-identity.json"), map[string]any{
						"jobId": id, "candidateCommit": commit, "mode": name, "config": path,
					})
					start := time.Now().UTC()
					stopProgress, err := horizonCSVProgress(ctx, cancel, u.String(), id,
						filepath.Join(output, job.Metadata.Name+"-progress.jsonl"))
					if err != nil {
						t.Fatal(err)
					}
					defer func() {
						if err := stopProgress(); err != nil {
							t.Error(err)
						}
					}()
					var result app.LoadResult
					if name == "resume" {
						result = horizonInterruptedCSV(t, ctx, conn, path, id, output, job.Metadata.Name)
					} else {
						result, err = app.LoadWithID(ctx, path, id)
						if err != nil {
							writeTinyJSON(t, filepath.Join(output, job.Metadata.Name+"-load.json"), result)
							t.Fatal(err)
						}
					}
					writeTinyJSON(t, filepath.Join(output, job.Metadata.Name+"-load.json"), result)
					stored, err := app.Status(ctx, path, id)
					wantCommitted := int64(458_000)
					if name == "append" || name == "upsert" {
						wantCommitted = 4_580
					}
					if err != nil || result.Status != meta.JobCommitted || stored.CommittedRows != wantCommitted ||
						result.Metrics.BatchesFailed != 0 || stored.RejectedRows != 0 {
						t.Fatalf("case did not commit exactly the expected rows without rejects: %+v / %v", result, err)
					}
					var generation int64
					if err := conn.QueryRow(ctx, `SELECT graph_generation_id FROM agefreighter_meta.load_job WHERE job_id=$1::uuid`, id).Scan(&generation); err != nil {
						t.Fatal(err)
					}
					if (name == "append" || name == "upsert") && generation != activeGeneration {
						t.Fatal("incremental case replaced the active generation")
					}
					if name == "replace" && generation == activeGeneration {
						t.Fatal("replace failed to create a new generation")
					}
					if name != "resume" {
						activeGeneration = generation
					}
					if _, err := app.Verify(ctx, path, id); err != nil {
						t.Fatal(err)
					}
					expected, err := rangedigest.ScenarioManifest(ctx, expectedPaths, 100000, false)
					if err != nil {
						t.Fatal(err)
					}
					actual, err := rangedigest.ScaleScenarioTargetManifest(ctx, u.String(), expectedPaths, id, 100000, false)
					if err != nil {
						t.Fatal(err)
					}
					comparison, compareErr := rangedigest.Compare(expected, actual)
					writeTinyJSON(t, filepath.Join(output, job.Metadata.Name+"-verification.json"), map[string]any{
						"coverage": "P0 CSV only; no complete twelve-route phase claim", "mode": name,
						"candidateCommit": commit, "jobId": id, "startedAtUtc": start, "completedAtUtc": time.Now().UTC(),
						"expected": expected, "actual": actual, "comparison": comparison, "postgresql": pg, "age": age,
						"committedRows": stored.CommittedRows, "generation": generation,
					})
					if compareErr != nil {
						t.Fatal(compareErr)
					}
				}) {
					t.Fatal("route paused after a failed prerequisite case")
				}
			}
		}) {
			t.Fatal("CSV execution paused after a failed route")
		}
	}
}

func horizonCSVProgress(ctx context.Context, cancelCase context.CancelFunc, dsn, id, path string) (func() error, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return nil, err
	}
	monitorCtx, stop := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() {
		err := func() (err error) {
			defer func() {
				if errors.Is(err, context.Canceled) && monitorCtx.Err() != nil {
					err = nil
				}
				err = errors.Join(err, file.Close())
			}()
			conn, err := pgx.Connect(monitorCtx, dsn)
			if err != nil {
				return err
			}
			defer conn.Close(context.Background())
			ticker := time.NewTicker(15 * time.Second)
			defer ticker.Stop()
			encoder := json.NewEncoder(file)
			lastProgress, previous := time.Now(), int64(-1)
			for {
				var table *string
				if err := conn.QueryRow(monitorCtx, `SELECT to_regclass('agefreighter_meta.load_job')::text`).Scan(&table); err != nil {
					return err
				}
				var rows, batch int64
				state, checkpoint := "waiting-for-job", ""
				if table != nil {
					err := conn.QueryRow(monitorCtx, `SELECT committed_rows,next_batch_id,status,resume_token
						FROM agefreighter_meta.load_job WHERE job_id=$1::uuid`, id).Scan(&rows, &batch, &state, &checkpoint)
					if err != nil && !errors.Is(err, pgx.ErrNoRows) {
						return err
					}
				}
				if rows > previous {
					previous, lastProgress = rows, time.Now()
				}
				if err := encoder.Encode(map[string]any{
					"atUtc": time.Now().UTC(), "jobId": id, "status": state,
					"committedRows": rows, "nextBatchId": batch, "sourceCheckpoint": checkpoint,
				}); err != nil {
					return err
				}
				if time.Since(lastProgress) >= 14*time.Minute {
					return errors.New("committed progress stalled for fourteen minutes")
				}
				select {
				case <-monitorCtx.Done():
					return monitorCtx.Err()
				case <-ticker.C:
				}
			}
		}()
		if err != nil {
			cancelCase()
		}
		done <- err
	}()
	return func() error {
		stop()
		return <-done
	}, nil
}

func horizonModeJobID(t *testing.T) string {
	t.Helper()
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		t.Fatal(err)
	}
	b[6], b[8] = b[6]&15|64, b[8]&63|128
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[:4], b[4:6], b[6:8], b[8:10], b[10:])
}

func horizonInterruptedCSV(t *testing.T, ctx context.Context, conn *pgx.Conn, path, id, output, name string) app.LoadResult {
	t.Helper()
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	log, err := os.OpenFile(filepath.Join(output, name+"-worker.log"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	defer log.Close()
	child := exec.CommandContext(ctx, binary, "-test.run=^TestHorizonDBP0CSVModeWorker$", "-test.v")
	child.Env = append(os.Environ(), "AGEFREIGHTER_HORIZON_MODE_WORKER=1",
		"AGEFREIGHTER_HORIZON_WORKER_CONFIG="+path, "AGEFREIGHTER_HORIZON_WORKER_JOB="+id)
	child.Stdout, child.Stderr = log, log
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- child.Wait() }()
	reaped := false
	defer func() {
		if !reaped {
			_ = child.Process.Kill()
			<-done
		}
	}()
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	var committed, batch int64
	var checkpoint, status string
	for committed < 458_000/4 {
		select {
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case err := <-done:
			reaped = true
			t.Fatalf("worker exited before the required fault: %v", err)
		case <-ticker.C:
			err := conn.QueryRow(ctx, `SELECT committed_rows,next_batch_id,resume_token,status
				FROM agefreighter_meta.load_job WHERE job_id=$1::uuid`, id).Scan(&committed, &batch, &checkpoint, &status)
			if err != nil && !errors.Is(err, pgx.ErrNoRows) {
				t.Fatal(err)
			}
		}
	}
	if status != "running" || committed >= 458_000/2 || batch <= 1 || checkpoint == "" {
		t.Fatal("fault did not occur at an eligible committed checkpoint")
	}
	if err := child.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	var exit *exec.ExitError
	err = <-done
	reaped = true
	if !errors.As(err, &exit) || exit.ExitCode() != -1 {
		t.Fatalf("expected a killed worker, got %v", err)
	}
	writeTinyJSON(t, filepath.Join(output, name+"-fault.json"), map[string]any{
		"fault": "SIGKILL", "jobId": id, "committedBeforeKill": committed, "nextBatchBeforeKill": batch,
		"checkpointBeforeKill": checkpoint, "injectedAtUtc": time.Now().UTC(),
		"processTermination": exit.ProcessState.String(),
	})
	before, err := app.Status(ctx, path, id)
	if err != nil || before.Status == meta.JobCommitted || before.CommittedRows < committed || before.CommittedRows >= 458_000/2 {
		t.Fatalf("durable killed-worker state differs: %+v / %v", before, err)
	}
	writeTinyJSON(t, filepath.Join(output, name+"-resume.json"), map[string]any{
		"jobId": id, "durableCommittedRowsBeforeResume": before.CommittedRows,
		"resumeStartedAtUtc": time.Now().UTC(),
	})
	result, err := app.Resume(ctx, path, id)
	if err != nil || result.JobID != id || uint64(before.CommittedRows)+result.Metrics.RecordsCommitted != 458_000 {
		t.Fatalf("resume did not preserve the exact job/checkpoint: %+v / %v", result, err)
	}
	return result
}

func TestHorizonDBP0CSVModeWorker(t *testing.T) {
	if os.Getenv("AGEFREIGHTER_HORIZON_MODE_WORKER") != "1" {
		t.Skip("only the explicitly admitted process-recovery parent may start this worker")
	}
	path, id := os.Getenv("AGEFREIGHTER_HORIZON_WORKER_CONFIG"), os.Getenv("AGEFREIGHTER_HORIZON_WORKER_JOB")
	job, err := config.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(path) || job.Source.Type != config.SourceCSV || job.Target.Type != config.TargetApacheAGE ||
		job.Target.Mode != config.LoadCreate || !strings.HasPrefix(job.Target.Graph, "hs_p0_csv_") ||
		job.Target.Connection.File == "" || os.Getenv("AGEFREIGHTER_REQUIRE_HORIZON_P0_CSV_MODES") != "1" {
		t.Fatal("worker is outside its admitted CSV create scope")
	}
	if _, err := app.LoadWithID(t.Context(), path, id); err != nil {
		t.Fatal(err)
	}
}
