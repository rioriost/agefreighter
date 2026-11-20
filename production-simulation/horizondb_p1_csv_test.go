//go:build linux || darwin

package simulation_test

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/rioriost/agefreighter/internal/app"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/pipeline"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
	"github.com/rioriost/agefreighter/production-simulation/internal/rangedigest"
)

const (
	p1Gate          = "AGEFREIGHTER_REQUIRE_HORIZON_P1_CSV_PG17"
	p1GatePG18      = "AGEFREIGHTER_REQUIRE_HORIZON_P1_CSV_PG18"
	p1PGGate        = "AGEFREIGHTER_REQUIRE_HORIZON_P1_POSTGRES_PG17"
	p1PGGatePG18    = "AGEFREIGHTER_REQUIRE_HORIZON_P1_POSTGRES_PG18"
	p1Neo4j44Gate   = "AGEFREIGHTER_REQUIRE_HORIZON_P1_NEO4J44_PG17"
	p1ReceiptEnv    = "AGEFREIGHTER_HORIZON_P1_RECEIPT"
	p1Role          = "afhs-20261118-p1-csv"
	p1BaseVertices  = 1_600_000
	p1BaseEdges     = 4_000_000
	p1BaseRows      = p1BaseVertices + p1BaseEdges
	p1DeltaRows     = p1BaseRows / 100
	p1MaxRSSKiB     = 2 << 20
	p1FullCaseTime  = 75 * time.Minute
	p1DeltaCaseTime = 30 * time.Minute
	p1RunTime       = 3*p1FullCaseTime + 2*p1DeltaCaseTime
)

var (
	p1RunPattern    = regexp.MustCompile(`^r[a-z0-9]{6,30}$`)
	p1UUIDPattern   = regexp.MustCompile(`^[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}$`)
	p1CommitPattern = regexp.MustCompile(`^[a-f0-9]{40}$`)
	p1HostPattern   = regexp.MustCompile(`^afhs-20261118-pg(17|18)\.rw\.[a-z0-9]+\.japaneast\.horizondb\.azure\.com$`)
	p1SHA           = regexp.MustCompile(`^[a-f0-9]{64}$`)
	p1SecretPath    = regexp.MustCompile(`^/dev/shm/agefreighter-p1-[0-9]+/target\.dsn$`)
)

type p1Admission struct {
	RunID                    string    `json:"runId"`
	Deadline                 time.Time `json:"deadlineUtc"`
	FixtureDir               string    `json:"fixtureDir"`
	Output                   string    `json:"output"`
	Host                     string    `json:"writerHost"`
	IdentityClientID         string    `json:"identityClientId"`
	Commit                   string    `json:"candidateCommit"`
	ExecutableSHA256         string    `json:"executableSha256"`
	GuardSettingsSHA256      string    `json:"guardSettingsSha256"`
	RenewalEvidenceSHA256    string    `json:"renewalEvidenceSha256"`
	TokenValidThroughLease   bool      `json:"tokenValidThroughLease,omitempty"`
	FixturePreparationSHA256 string    `json:"fixturePreparationSha256"`
	SnapshotID               string    `json:"lifecycleProbeSnapshotId"`
	WriterReadyAt            time.Time `json:"writerReadyAtUtc"`
	Source                   string    `json:"source,omitempty"`
}

func (a p1Admission) validate(now time.Time, needed time.Duration, major int) error {
	if !p1RunPattern.MatchString(a.RunID) || major == 0 || a.writerMajor() != major ||
		(a.Source != "" && a.Source != "postgresql" && (a.Source != "neo4j44" || major != 17)) ||
		!p1UUIDPattern.MatchString(a.IdentityClientID) ||
		!p1CommitPattern.MatchString(a.Commit) ||
		!p1SHA.MatchString(a.ExecutableSHA256) || !p1SHA.MatchString(a.GuardSettingsSHA256) ||
		!p1SHA.MatchString(a.FixturePreparationSHA256) ||
		!regexp.MustCompile(`^/subscriptions/[a-f0-9-]{36}/resourceGroups/rg-agefreighter-horizon-tiny-20261118/providers/Microsoft\.Compute/snapshots/afhs-20261118-[a-f0-9]{16}$`).MatchString(a.SnapshotID) {
		return errors.New("require the independently admitted P1 route, writer major, candidate, identity, guard and snapshot evidence")
	}
	if a.TokenValidThroughLease {
		if a.RenewalEvidenceSHA256 != "" {
			return errors.New("a lease covered by the initial token does not establish credential renewal")
		}
	} else if !p1SHA.MatchString(a.RenewalEvidenceSHA256) {
		return errors.New("require either verified renewal evidence or actual initial-token coverage for the entire lease")
	}
	rel, err := filepath.Rel(a.FixtureDir, a.Output)
	if !filepath.IsAbs(a.Output) || !filepath.IsAbs(a.FixtureDir) || err != nil ||
		rel != ".." && !strings.HasPrefix(rel, "../") ||
		a.Deadline.Sub(now) < needed || a.Deadline.Sub(now) > 8*time.Hour {
		return errors.New("require separate absolute fixture/evidence paths and sufficient time in an at-most-eight-hour lease")
	}
	if a.WriterReadyAt.Before(a.Deadline.Add(-8*time.Hour)) || !a.WriterReadyAt.Before(now) {
		return errors.New("require a writer readiness timestamp in the current P1 lease")
	}
	return nil
}

func readP1Admission(t *testing.T, needed time.Duration, major int, source string) p1Admission {
	t.Helper()
	var a p1Admission
	p1ReadJSON(t, os.Getenv(p1ReceiptEnv), &a)
	if err := a.validate(time.Now(), needed, major); err != nil {
		t.Fatal(err)
	}
	if a.source() != source {
		t.Fatal("configured P1 source differs from the admitted route")
	}
	if runtime.GOOS != "linux" {
		t.Fatal("the P1 pilot requires the isolated Linux runner for RSS and fault evidence")
	}
	executable, err := os.Executable()
	if err != nil || p1FileSHA(t, executable) != a.ExecutableSHA256 {
		t.Fatal("running test binary does not match the admitted executable")
	}
	return a
}

func (a p1Admission) database() string { return "hs_p1_" + a.RunID }

func (a p1Admission) source() string {
	if a.Source == "" {
		return "csv"
	}
	return a.Source
}

func (a p1Admission) writerMajor() int {
	m := p1HostPattern.FindStringSubmatch(a.Host)
	if m == nil {
		return 0
	}
	if m[1] == "17" {
		return 17
	}
	return 18
}

var p1Routes = []struct {
	gate, source string
	major        int
}{
	{p1Gate, "csv", 17}, {p1GatePG18, "csv", 18},
	{p1PGGate, "postgresql", 17}, {p1PGGatePG18, "postgresql", 18},
	{p1Neo4j44Gate, "neo4j44", 17},
}

func p1EnabledRoute(t *testing.T) (int, string) {
	t.Helper()
	major, source := 0, ""
	for _, route := range p1Routes {
		if os.Getenv(route.gate) == "1" {
			if major != 0 {
				t.Fatal("require exactly one P1 source and writer route")
			}
			major, source = route.major, route.source
		}
	}
	return major, source
}

type p1Case struct {
	name, dataset, digest string
	mode                  config.LoadMode
	rows, finalRows       int64
	bound                 time.Duration
	recovery              bool
}

var p1Cases = []p1Case{
	{"create", "base", "base", config.LoadCreate, p1BaseRows, p1BaseRows, p1FullCaseTime, false},
	{"replace", "base", "base", config.LoadReplace, p1BaseRows, p1BaseRows, p1FullCaseTime, false},
	{"append", "append", "append", config.LoadAppend, p1DeltaRows, p1BaseRows + p1DeltaRows, p1DeltaCaseTime, false},
	{"upsert", "upsert", "upsert", config.LoadUpsert, p1DeltaRows, p1BaseRows + p1DeltaRows, p1DeltaCaseTime, false},
	{"resume", "base", "base", config.LoadCreate, p1BaseRows, p1BaseRows, p1FullCaseTime, true},
}

var p1Expected = map[string]struct{ raw, canonical string }{
	"base":   {"f74220f6c58f0c1a62f80a567520ffcde43a2499ba48100667ee7b78ff4e2e2f", "bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70"},
	"append": {"fec86d9153c0091d148040661e67edb8826751e91144699e66050b02b23acb3e", "254d508f3d355a6b6b339ad8a826706665a501b78cc5007f9fa30d6ea845d69b"},
	"upsert": {"b14d9a47aa4029bfd4d17fcb913c283c98524123f77a81c61612cfb1c76a5bb1", "61a7aa7c0834f2556d5c36070c44eb933da08b8a5a6e513729a515528c923ed7"},
}

func TestHorizonDBP1CSVPG17Integration(t *testing.T) {
	runHorizonDBP1(t, 17, "csv")
}

func TestHorizonDBP1CSVPG18Integration(t *testing.T) {
	runHorizonDBP1(t, 18, "csv")
}

func runHorizonDBP1(t *testing.T, targetMajor int, source string) {
	t.Helper()
	if major, enabled := p1EnabledRoute(t); major != targetMajor || enabled != source {
		t.Skipf("explicit P1 %s PG%d admission required", source, targetMajor)
	}
	needed := p1RunTime + time.Hour
	if source == "neo4j44" {
		needed += 40 * time.Minute
	}
	a := readP1Admission(t, needed, targetMajor, source)
	fixtureReceipt := filepath.Join(a.FixtureDir, p1FixtureReceiptName)
	if p1FileSHA(t, fixtureReceipt) != a.FixturePreparationSHA256 {
		t.Fatal("worker fixture preparation receipt SHA-256 differs from admission")
	}
	var prepared p1FixturePreparation
	p1ReadJSON(t, fixtureReceipt, &prepared)
	if err := prepared.validate(a, time.Now()); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithDeadline(t.Context(), a.Deadline.Add(-15*time.Minute))
	defer cancel()
	if err := os.Mkdir(a.Output, 0700); err != nil {
		t.Fatal(err)
	}
	writeTinyJSON(t, filepath.Join(a.Output, "admission.json"), a)
	stopTelemetry := startP1Telemetry(t, ctx, cancel, a)
	var requireMetricsSince time.Time
	defer func() { stopTelemetry(requireMetricsSince) }()
	paths, expected, exports := prepareP1CSV(t, ctx, a.FixtureDir, a.Output)
	var verifySource func()
	switch source {
	case "postgresql":
		verifySource = prepareP1Postgres(t, ctx, a, paths, exports)
	case "neo4j44":
		verifySource = prepareP1Neo4j(t, ctx, a, paths, exports)
	}
	secret := filepath.Join(p1SecretDir(t), "target.dsn")
	stopCredentials := startP1Credentials(t, ctx, cancel, a, secret)
	defer stopCredentials()
	conn := connectP1(t, ctx, secret, a.Host)
	defer conn.Close(context.Background())
	var major int
	var pg, age, database, role, preload string
	var pending, writable, empty bool
	err := conn.QueryRow(ctx, `SELECT current_setting('server_version_num')::integer / 10000,
		version(), (SELECT extversion FROM pg_extension WHERE extname='age'), current_database(), current_user,
		setting, pending_restart, NOT pg_is_in_recovery(),
		to_regnamespace('agefreighter_meta') IS NULL AND NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph)
		FROM pg_settings WHERE name='shared_preload_libraries'`).
		Scan(&major, &pg, &age, &database, &role, &preload, &pending, &writable, &empty)
	if err != nil {
		p1DatabaseFailure(t, "P1 writer SQL admission", err)
	}
	if major != targetMajor || !strings.HasPrefix(age, "1.8.") || database != a.database() ||
		role != p1Role || pending || !writable || !empty ||
		!strings.Contains(","+strings.ReplaceAll(preload, " ", "")+",", ",age,") {
		t.Fatalf("SQL did not prove the fresh PostgreSQL %d/AGE 1.8 writer and dedicated identity", targetMajor)
	}

	startedCases := time.Now()
	var activeGeneration int64
	for _, c := range p1Cases {
		executed := false
		if !t.Run(c.name, func(t *testing.T) {
			executed = true
			if time.Until(a.Deadline) < c.bound+15*time.Minute {
				t.Fatal("case does not fit the remaining admitted lease")
			}
			caseCtx, stop := context.WithTimeout(ctx, c.bound)
			defer stop()
			job := tinyJob(exports[c.dataset], filepath.Join(a.Output, "csv-"+c.dataset), source, targetMajor)
			if source == "postgresql" {
				if err := horizonPostgresQueries(&job, p1PostgresSchema(a, c.dataset)); err != nil {
					t.Fatal(err)
				}
				job.Source.PostgreSQL.Connection = config.SecretRef{File: p1PostgresReader}
			} else if source == "neo4j44" {
				if err := configureP1Neo4jJob(&job, exports[c.dataset], a, c.dataset); err != nil {
					t.Fatal(err)
				}
			}
			job.Metadata.Name = fmt.Sprintf("horizon-p1-%s-pg%d-%s-%s", source, targetMajor, c.name, a.RunID)
			job.Target.Mode, job.Target.Graph = c.mode, p1Graph(a, c)
			job.Target.Connection = config.SecretRef{File: secret}
			job.Runtime.BatchRows, job.Runtime.BatchBytes, job.Runtime.MaxTargetConnections = 20_000, 64<<20, 4
			path := filepath.Join(a.Output, c.name+".json")
			writeTinyJSON(t, path, job)
			id := horizonModeJobID(t)
			writeTinyJSON(t, filepath.Join(a.Output, c.name+"-identity.json"), map[string]any{"jobId": id, "mode": c.name})
			start := time.Now().UTC()
			before := runP1Process(t, caseCtx, conn, a, c, path, id, false, meta.Job{})
			if c.recovery {
				runP1Process(t, caseCtx, conn, a, c, path, id, true, before)
			}
			var result app.LoadResult
			p1ReadJSON(t, filepath.Join(a.Output, c.name+"-load.json"), &result)
			stored, err := app.Verify(caseCtx, path, id)
			if err != nil {
				p1DatabaseFailure(t, "application verification", err)
			}
			if err := validateP1Commit(c, id, activeGeneration, before, stored, result); err != nil {
				t.Fatal(err)
			}
			if !c.recovery {
				activeGeneration = stored.GraphGenerationID
			}
			dsn, err := os.ReadFile(secret)
			if err != nil {
				t.Fatal("could not read current private connection file")
			}
			actual, err := rangedigest.ScaleScenarioTargetManifest(caseCtx, string(dsn), p1Scenario(paths, c.digest), id, 100_000, false)
			if err != nil {
				p1DatabaseFailure(t, "independent canonical verification", err)
			}
			want := expected[c.digest]
			comparison, err := rangedigest.Compare(want, actual)
			writeTinyJSON(t, filepath.Join(a.Output, c.name+"-verification.json"), map[string]any{
				"jobId": id, "mode": c.name, "committedRows": stored.CommittedRows, "generation": stored.GraphGenerationID,
				"startedAtUtc": start, "completedAtUtc": time.Now().UTC(), "postgresql": pg, "age": age,
				"expected": want, "actual": actual, "comparison": comparison,
			})
			if err != nil || actual.RecordCount != c.finalRows {
				t.Fatal("independent P1 canonical comparison failed")
			}
		}) || !executed {
			t.Fatal("P1 pilot requires all five cases, in order, without skips or failed prerequisites")
		}
	}
	for _, name := range p1FixtureOrder {
		p1CSVIntegrity(t, a.Output, name, exports[name])
	}
	if verifySource != nil {
		verifySource()
	}
	requireMetricsSince = startedCases
}

func prepareP1CSV(t *testing.T, ctx context.Context, fixtureDir, output string) (map[string]string, map[string]rangedigest.Manifest, map[string]portable.Manifest) {
	t.Helper()
	paths, expected, _ := verifiedP1Fixtures(t, ctx, fixtureDir)
	exports := map[string]portable.Manifest{}
	for _, name := range p1FixtureOrder {
		export, err := portable.Export(ctx, paths[name], filepath.Join(output, "csv-"+name))
		if err != nil {
			t.Fatal(err)
		}
		exports[name] = export
		p1CSVIntegrity(t, output, name, export)
	}
	return paths, expected, exports
}

func TestHorizonDBP1CSVSourcePreflight(t *testing.T) {
	fixtureDir := os.Getenv("AGEFREIGHTER_HORIZON_P1_FIXTURE_DIR")
	if fixtureDir == "" {
		t.Skip("P1 source-only fixture directory not requested")
	}
	output := os.Getenv("AGEFREIGHTER_HORIZON_P1_PREFLIGHT_OUTPUT")
	if !filepath.IsAbs(fixtureDir) || !filepath.IsAbs(output) {
		t.Fatal("absolute fixture path and new output directory required")
	}
	if err := os.Mkdir(output, 0700); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Minute)
	defer cancel()
	_, expected, exports := prepareP1CSV(t, ctx, fixtureDir, output)
	base := tinyJob(exports["base"], filepath.Join(output, "csv-base"), "csv", 17)
	actual, err := rangedigest.CSVManifest(ctx, *base.Source.CSV, expected["base"].FixtureRoot, 100_000)
	if err != nil {
		t.Fatal(err)
	}
	comparison, err := rangedigest.CompareCSV(expected["base"], actual)
	if err != nil {
		t.Fatal(err)
	}
	writeTinyJSON(t, filepath.Join(output, "source-preflight.json"), map[string]any{
		"coverage": "source-only; no HorizonDB migration", "expected": expected,
		"csvBaseComparison": comparison,
	})
	t.Logf("CSV connector parity: %d rows, root %s", actual.RecordCount, actual.RootSHA256)
}

func p1SecretDir(t *testing.T) string {
	t.Helper()
	mounts, mountsErr := os.ReadFile("/proc/mounts")
	swaps, swapsErr := os.ReadFile("/proc/swaps")
	if mountsErr != nil || swapsErr != nil || !strings.Contains("\n"+string(mounts), "\ntmpfs /dev/shm tmpfs ") ||
		strings.Contains(strings.TrimSpace(string(swaps)), "\n") {
		t.Fatal("require a tmpfs credential path with swap disabled; never put the DSN in an OS-disk snapshot")
	}
	dir, err := os.MkdirTemp("/dev/shm", "agefreighter-p1-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Error(err)
		}
	})
	return dir
}

func p1Scenario(paths map[string]string, name string) rangedigest.Scenario {
	s := rangedigest.Scenario{Base: paths["base"]}
	if name != "base" {
		s.Append = paths["append"]
	}
	if name == "upsert" {
		s.Upsert = paths["upsert"]
	}
	return s
}

func p1Graph(a p1Admission, c p1Case) string {
	graph := a.database()
	if c.recovery {
		graph += "_recovery"
	}
	return graph
}

func validateP1Commit(c p1Case, id string, active int64, before, stored meta.Job, result app.LoadResult) error {
	if result.JobID != id || stored.ID != id || stored.Status != meta.JobCommitted || result.Status != meta.JobCommitted ||
		stored.CommittedRows != c.rows || stored.RejectedRows != 0 || stored.SourceRejectedRows != 0 ||
		result.Metrics.BatchesFailed != 0 || result.Metrics.BatchesCommitted == 0 || stored.GraphGenerationID <= 0 ||
		stored.NextBatchID <= 1 || stored.ResumeToken == "" ||
		before.CommittedRows < 0 || before.CommittedRows > c.rows ||
		result.Metrics.RecordsCommitted != uint64(c.rows-before.CommittedRows) {
		return errors.New("case did not commit exactly the expected rows and checkpoints without rejects")
	}
	if c.recovery {
		if before.ID != id || before.Status == meta.JobCommitted || before.CommittedRows < c.rows/4 ||
			before.CommittedRows >= c.rows/2 || before.NextBatchID <= 1 || before.ResumeToken == "" ||
			stored.NextBatchID <= before.NextBatchID || stored.ResumeToken == before.ResumeToken ||
			stored.GraphGenerationID != before.GraphGenerationID || before.ConfigFingerprint == "" ||
			stored.ConfigFingerprint != before.ConfigFingerprint || stored.GraphGenerationID == active {
			return errors.New("recovery did not preserve the exact job, generation and durable checkpoint")
		}
	} else if c.mode == config.LoadReplace && (active == 0 || stored.GraphGenerationID == active) ||
		(c.mode == config.LoadAppend || c.mode == config.LoadUpsert) && stored.GraphGenerationID != active {
		return errors.New("unexpected graph generation transition")
	}
	return nil
}

func runP1Process(t *testing.T, ctx context.Context, conn *pgx.Conn, a p1Admission, c p1Case, path, id string, resume bool, previous meta.Job) meta.Job {
	t.Helper()
	step := c.name
	if c.recovery && !resume {
		step += "-initial"
	}
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	log, err := os.OpenFile(filepath.Join(a.Output, step+"-progress.jsonl"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := log.Close(); err != nil {
			t.Error(err)
		}
	}()
	start := time.Now()
	child := exec.CommandContext(ctx, binary, "-test.run=^TestHorizonDBP1CSVWorker$", "-test.v")
	child.Env = append(os.Environ(), "AGEFREIGHTER_HORIZON_P1_WORKER="+c.name,
		"AGEFREIGHTER_HORIZON_P1_JOB_ID="+id, fmt.Sprintf("AGEFREIGHTER_HORIZON_P1_WORKER_RESUME=%t", resume))
	child.Stdout, child.Stderr = t.Output(), t.Output()
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- child.Wait() }()
	reaped, killed := false, false
	defer func() {
		if !reaped {
			_ = child.Process.Kill()
			<-done
		}
	}()
	progress := json.NewEncoder(log)
	interval := 15 * time.Second
	if c.recovery && !resume {
		interval = 500 * time.Millisecond
	}
	ticker := time.NewTimer(0)
	defer ticker.Stop()
	var fault meta.Job
	var injectedAt time.Time
	lastSample, lastProgress := time.Time{}, start
	sampledAt, sampledRows, sampledBytes := start, previous.CommittedRows, previous.CommittedBytes
	tableExists := resume
	for {
		select {
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case err := <-done:
			reaped = true
			usage, ok := child.ProcessState.SysUsage().(*syscall.Rusage)
			if !ok || usage.Maxrss <= 0 {
				t.Fatal("loader process RSS is unavailable")
			}
			writeTinyJSON(t, filepath.Join(a.Output, step+"-process.json"), map[string]any{
				"jobId": id, "wallMillis": time.Since(start).Milliseconds(), "maxRssKiB": usage.Maxrss,
				"userMillis": child.ProcessState.UserTime().Milliseconds(), "systemMillis": child.ProcessState.SystemTime().Milliseconds(),
				"blockInputs": usage.Inblock, "blockOutputs": usage.Oublock,
			})
			if usage.Maxrss > p1MaxRSSKiB {
				t.Fatal("loader exceeded 2 GiB peak RSS")
			}
			if c.recovery && !resume {
				wait, ok := child.ProcessState.Sys().(syscall.WaitStatus)
				if !killed || err == nil || !ok || wait.Signal() != syscall.SIGKILL {
					t.Fatal("worker exited without the required SIGKILL")
				}
				stored, err := app.Status(ctx, path, id)
				if err != nil {
					p1DatabaseFailure(t, "durable recovery checkpoint", err)
				}
				if stored.Status != meta.JobRunning || stored.CommittedRows < fault.CommittedRows ||
					stored.CommittedRows >= c.rows/2 || stored.NextBatchID < fault.NextBatchID || stored.ResumeToken == "" {
					t.Fatal("durable killed-worker checkpoint differs")
				}
				writeTinyJSON(t, filepath.Join(a.Output, "resume-fault.json"), map[string]any{
					"jobId": id, "signal": "SIGKILL", "committedBeforeKill": fault.CommittedRows,
					"injectedAtUtc":        injectedAt,
					"durableCommittedRows": stored.CommittedRows, "durableNextBatch": stored.NextBatchID,
					"durableCheckpoint": stored.ResumeToken, "generation": stored.GraphGenerationID,
				})
				return stored
			}
			if err != nil {
				t.Fatalf("P1 worker failed: %v", err)
			}
			return meta.Job{}
		case <-ticker.C:
			if killed {
				continue
			}
			var current meta.Job
			queryCtx, stop := context.WithTimeout(ctx, 30*time.Second)
			var err error
			if !tableExists {
				err = conn.QueryRow(queryCtx, `SELECT to_regclass('agefreighter_meta.load_job') IS NOT NULL`).Scan(&tableExists)
			}
			if err == nil && tableExists {
				err = conn.QueryRow(queryCtx, `SELECT status,committed_rows,committed_bytes,next_batch_id,resume_token,rejected_rows,source_rejected_rows
					FROM agefreighter_meta.load_job WHERE job_id=$1::uuid`, id).
					Scan(&current.Status, &current.CommittedRows, &current.CommittedBytes, &current.NextBatchID, &current.ResumeToken,
						&current.RejectedRows, &current.SourceRejectedRows)
			}
			stop()
			if err != nil && !errors.Is(err, pgx.ErrNoRows) {
				p1DatabaseFailure(t, "P1 progress", err)
			}
			if current.CommittedRows < previous.CommittedRows || current.NextBatchID < previous.NextBatchID ||
				current.CommittedRows > c.rows || current.CommittedBytes < previous.CommittedBytes {
				t.Fatal("P1 checkpoint regressed or exceeded the case bounds")
			}
			now := time.Now()
			if current.CommittedRows > previous.CommittedRows {
				lastProgress = now
			}
			previous = current
			if now.Sub(lastProgress) >= 15*time.Minute {
				t.Fatal("P1 committed checkpoint did not advance for fifteen minutes")
			}
			if now.Sub(lastSample) >= 15*time.Second {
				elapsed := now.Sub(sampledAt).Seconds()
				if err := progress.Encode(map[string]any{
					"atUtc": now.UTC(), "jobId": id, "status": current.Status, "committedRows": current.CommittedRows,
					"nextBatchId": current.NextBatchID, "checkpoint": current.ResumeToken, "committedBytes": current.CommittedBytes,
					"recordsPerSecond": float64(current.CommittedRows-sampledRows) / elapsed,
					"bytesPerSecond":   float64(current.CommittedBytes-sampledBytes) / elapsed,
					"rejectedRows":     current.RejectedRows, "sourceRejectedRows": current.SourceRejectedRows,
				}); err != nil {
					t.Fatal(err)
				}
				lastSample = now
				sampledAt, sampledRows, sampledBytes = now, current.CommittedRows, current.CommittedBytes
			}
			if current.RejectedRows != 0 || current.SourceRejectedRows != 0 {
				t.Fatal("P1 worker reported a rejected record")
			}
			if c.recovery && !resume && current.CommittedRows >= c.rows/4 {
				if current.Status != meta.JobRunning || current.CommittedRows >= c.rows/2 ||
					current.NextBatchID <= 1 || current.ResumeToken == "" {
					t.Fatal("missed the eligible 25-50% durable fault checkpoint")
				}
				if err := child.Process.Kill(); err != nil {
					t.Fatal(err)
				}
				killed, fault, injectedAt = true, current, now.UTC()
			}
			ticker.Reset(interval)
		}
	}
}

func TestHorizonDBP1CSVWorker(t *testing.T) {
	mode := os.Getenv("AGEFREIGHTER_HORIZON_P1_WORKER")
	major, source := p1EnabledRoute(t)
	if mode == "" || major == 0 {
		t.Skip("only the admitted P1 pilot may start a loader worker")
	}
	a := readP1Admission(t, time.Minute, major, source)
	id := os.Getenv("AGEFREIGHTER_HORIZON_P1_JOB_ID")
	resume := os.Getenv("AGEFREIGHTER_HORIZON_P1_WORKER_RESUME")
	for _, c := range p1Cases {
		if c.name != mode {
			continue
		}
		path := filepath.Join(a.Output, c.name+".json")
		job, err := config.Load(path)
		if err != nil {
			t.Fatal(err)
		}
		if !p1UUIDPattern.MatchString(id) || (resume != "true" && resume != "false") || (resume == "true" && !c.recovery) ||
			job.Target.Type != config.TargetApacheAGE ||
			job.Target.Mode != c.mode || job.Target.Graph != p1Graph(a, c) ||
			!p1SecretPath.MatchString(job.Target.Connection.File) {
			t.Fatal("loader worker is outside the admitted P1 case")
		}
		switch source {
		case "csv":
			if job.Source.Type != config.SourceCSV || job.Source.CSV == nil {
				t.Fatal("loader source is not the admitted CSV connector")
			}
			for _, v := range job.Source.CSV.Vertices {
				if filepath.Dir(v.Path) != filepath.Join(a.Output, "csv-"+c.dataset) {
					t.Fatal("vertex CSV is outside the admitted dataset")
				}
			}
			for _, e := range job.Source.CSV.Edges {
				if filepath.Dir(e.Path) != filepath.Join(a.Output, "csv-"+c.dataset) {
					t.Fatal("edge CSV is outside the admitted dataset")
				}
			}
		case "postgresql":
			if err := validateP1PostgresJob(job, p1PostgresSchema(a, c.dataset)); err != nil {
				t.Fatal(err)
			}
			p1PostgresDSN(t, p1PostgresReader, "p1_reader")
		case "neo4j44":
			var sourceManifest portable.Manifest
			p1ReadJSON(t, filepath.Join(a.Output, "csv-"+c.dataset, "manifest.json"), &sourceManifest)
			if err := validateP1Neo4jJob(job, sourceManifest, a, c.dataset); err != nil {
				t.Fatal(err)
			}
		}
		body, err := os.ReadFile(job.Target.Connection.File)
		target, parseErr := url.Parse(string(body))
		if err != nil || parseErr != nil || target.Scheme != "postgresql" || target.Host != a.Host+":5432" ||
			target.Path != "/"+a.database() || target.RawQuery != "sslmode=verify-full" || target.User == nil ||
			target.User.Username() != p1Role {
			t.Fatal("loader worker target is outside the admitted P1 writer identity")
		}
		ctx, cancel := context.WithTimeout(t.Context(), c.bound)
		defer cancel()
		load := app.LoadWithID
		if resume == "true" {
			load = app.Resume
		}
		result, err := load(ctx, path, id)
		if err != nil {
			p1DatabaseFailure(t, "loader "+c.name, err)
		}
		writeTinyJSON(t, filepath.Join(a.Output, c.name+"-load.json"), result)
		return
	}
	t.Fatal("unknown P1 worker mode")
}

func startP1Credentials(t *testing.T, ctx context.Context, cancelPilot context.CancelFunc, a p1Admission, path string) func() {
	t.Helper()
	credential, err := azidentity.NewManagedIdentityCredential(&azidentity.ManagedIdentityCredentialOptions{
		ID: azidentity.ClientID(a.IdentityClientID),
	})
	if err != nil {
		t.Fatal("P1 managed identity unavailable")
	}
	refreshCtx, stop := context.WithCancel(ctx)
	var lastToken string
	var expires time.Time
	refresh := func() error {
		requestCtx, cancel := context.WithTimeout(refreshCtx, 30*time.Second)
		defer cancel()
		token, err := credential.GetToken(requestCtx, policy.TokenRequestOptions{
			Scopes: []string{"https://ossrdbms-aad.database.windows.net/.default"},
		})
		if err != nil || token.Token == "" || !token.ExpiresOn.After(time.Now()) {
			return errors.New("no unexpired credential returned by the enrolled managed identity")
		}
		if token.Token == lastToken {
			return nil
		}
		if !token.ExpiresOn.After(expires) {
			return errors.New("managed identity replacement did not advance token expiry")
		}
		dsn := url.URL{Scheme: "postgresql", Host: a.Host + ":5432", Path: "/" + a.database(),
			User: url.UserPassword(p1Role, token.Token), RawQuery: "sslmode=verify-full"}
		next, err := os.CreateTemp(filepath.Dir(path), "next-")
		if err != nil {
			return err
		}
		_, writeErr := io.WriteString(next, dsn.String())
		if err := errors.Join(writeErr, next.Close()); err != nil {
			return err
		}
		if err := os.Rename(next.Name(), path); err != nil {
			return err
		}
		lastToken, expires = token.Token, token.ExpiresOn
		return nil
	}
	if err := refresh(); err != nil || time.Until(expires) < 5*time.Minute {
		stop()
		t.Fatal("initial P1 managed identity credential must be valid for at least five minutes")
	}
	if a.TokenValidThroughLease && expires.Before(a.Deadline.Add(5*time.Minute)) {
		stop()
		t.Fatal("initial P1 credential does not cover the full admitted lease plus five minutes")
	}
	writeTinyJSON(t, filepath.Join(a.Output, "credential-validity.json"), map[string]any{
		"clientId": a.IdentityClientID, "initialExpiresAtUtc": expires.UTC(),
		"leaseDeadlineUtc": a.Deadline, "initialTokenCoversLease": !expires.Before(a.Deadline.Add(5 * time.Minute)),
		"verifiedRenewalEvidence": a.RenewalEvidenceSHA256 != "",
	})
	done := make(chan error, 1)
	go func() {
		for {
			delay := max(15*time.Second, time.Until(expires.Add(-5*time.Minute)))
			timer := time.NewTimer(min(delay, time.Until(expires)))
			select {
			case <-refreshCtx.Done():
				timer.Stop()
				done <- nil
				return
			case <-timer.C:
			}
			if err := refresh(); err != nil {
				if refreshCtx.Err() != nil {
					done <- nil
					return
				}
				cancelPilot()
				done <- err
				return
			}
		}
	}()
	return func() {
		stop()
		if err := <-done; err != nil {
			t.Error(err)
		}
	}
}

func connectP1(t *testing.T, ctx context.Context, path, host string) *pgx.Conn {
	t.Helper()
	dsn, err := os.ReadFile(path)
	if err != nil {
		t.Fatal("could not read current private connection file")
	}
	connectCtx, stop := context.WithTimeout(ctx, 30*time.Second)
	defer stop()
	conn, err := pgx.Connect(connectCtx, string(dsn))
	if err != nil {
		p1DatabaseFailure(t, "P1 writer connection", err)
	}
	// HorizonDB's backend pg_stat_ssl does not describe the client-facing TLS session.
	wire, ok := conn.PgConn().Conn().(*tls.Conn)
	if ok {
		state := wire.ConnectionState()
		if state.HandshakeComplete && state.ServerName == host && len(state.VerifiedChains) > 0 {
			return conn
		}
	}
	_ = conn.Close(connectCtx)
	t.Fatal("require a verified TLS connection to the admitted HorizonDB hostname")
	return conn
}

func p1DatabaseFailure(t *testing.T, step string, err error) {
	t.Helper()
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		t.Fatalf("%s failed (SQLSTATE %s)", step, pgErr.Code)
	}
	t.Fatalf("%s failed (%T)", step, err)
}

func p1ReadJSON(t *testing.T, path string, value any) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	d := json.NewDecoder(file)
	d.DisallowUnknownFields()
	if err := d.Decode(value); err != nil {
		t.Fatal(err)
	}
	if err := d.Decode(new(any)); !errors.Is(err, io.EOF) {
		t.Fatal("expected exactly one JSON value")
	}
}

func p1FileSHA(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.New()
	_, err = io.Copy(hash, f)
	if err := errors.Join(err, f.Close()); err != nil {
		t.Fatal(err)
	}
	return fmt.Sprintf("%x", hash.Sum(nil))
}

func p1CSVIntegrity(t *testing.T, output, name string, p portable.Manifest) {
	t.Helper()
	for _, table := range p.Tables {
		if p1FileSHA(t, filepath.Join(output, "csv-"+name, table.CSV)) != table.CSVHash {
			t.Fatalf("CSV source changed: %s/%s", name, table.CSV)
		}
	}
}

func TestHorizonDBP1AdmissionScope(t *testing.T) {
	now := time.Now()
	hash := strings.Repeat("a", 64)
	a := p1Admission{
		RunID: "rp1test01", Deadline: now.Add(7 * time.Hour), Output: "/evidence/new", FixtureDir: "/fixture",
		Host: "afhs-20261118-pg17.rw.scope.japaneast.horizondb.azure.com", Commit: hash[:40],
		IdentityClientID: "00000000-0000-0000-0000-000000000001", ExecutableSHA256: hash,
		GuardSettingsSHA256: hash, RenewalEvidenceSHA256: hash, FixturePreparationSHA256: hash,
		SnapshotID:    "/subscriptions/00000000-0000-0000-0000-000000000001/resourceGroups/rg-agefreighter-horizon-tiny-20261118/providers/Microsoft.Compute/snapshots/afhs-20261118-0123456789abcdef",
		WriterReadyAt: now.Add(-15 * time.Minute),
	}
	if err := a.validate(now, p1RunTime+time.Hour, 17); err != nil {
		t.Fatal(err)
	}
	covered := a
	covered.TokenValidThroughLease, covered.RenewalEvidenceSHA256 = true, ""
	if err := covered.validate(now, p1RunTime+time.Hour, 17); err != nil {
		t.Fatal(err)
	}
	pg18 := covered
	pg18.Host = strings.Replace(covered.Host, "-pg17.", "-pg18.", 1)
	if err := pg18.validate(now, p1RunTime+time.Hour, 18); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(pg18.writerResource(), "/afhs-20261118-pg18") {
		t.Fatal("PG18 admission must be monitored on its own writer")
	}
	pg18.Source = "postgresql"
	if err := pg18.validate(now, p1RunTime+time.Hour, 18); err != nil {
		t.Fatal(err)
	}
	if err := pg18.validate(now, p1RunTime+time.Hour, 17); err == nil {
		t.Fatal("accepted a PostgreSQL source PG18 admission for a PG17 writer")
	}
	for _, tt := range []struct {
		name   string
		change func(*p1Admission)
	}{
		{"reader", func(a *p1Admission) { a.Host = strings.Replace(a.Host, ".rw.", ".ro.", 1) }},
		{"other major", func(a *p1Admission) { a.Host = strings.Replace(a.Host, "-pg17.", "-pg18.", 1) }},
		{"other region", func(a *p1Admission) { a.Host = strings.Replace(a.Host, ".japaneast.", ".japanwest.", 1) }},
		{"not enough time for sequence", func(a *p1Admission) { a.Deadline = now.Add(p1RunTime) }},
		{"unbounded deadline", func(a *p1Admission) { a.Deadline = now.Add(9 * time.Hour) }},
		{"mutable fixture path", func(a *p1Admission) { a.Output = "/fixture/output" }},
		{"no renewal proof", func(a *p1Admission) { a.RenewalEvidenceSHA256 = "" }},
		{"initial token confused with renewal proof", func(a *p1Admission) { a.TokenValidThroughLease = true }},
		{"no fixture receipt", func(a *p1Admission) { a.FixturePreparationSHA256 = "" }},
		{"no measured snapshot", func(a *p1Admission) { a.SnapshotID = strings.Replace(a.SnapshotID, "-tiny-", "-scale-", 1) }},
		{"stale writer readiness", func(a *p1Admission) { a.WriterReadyAt = now.Add(-2 * time.Hour) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			bad := a
			tt.change(&bad)
			if err := bad.validate(now, p1RunTime+time.Hour, 17); err == nil {
				t.Fatal("accepted out-of-scope P1 admission")
			}
		})
	}
}

func TestHorizonDBP1RecoveryRequiresExactDurableContinuation(t *testing.T) {
	c := p1Cases[len(p1Cases)-1]
	before := meta.Job{ID: "same-job", GraphGenerationID: 2, Status: meta.JobRunning,
		ConfigFingerprint: "same-config", CommittedRows: 1_400_000, NextBatchID: 71, ResumeToken: "before"}
	after := before
	after.Status, after.CommittedRows, after.NextBatchID, after.ResumeToken = meta.JobCommitted, 5_600_000, 281, "after"
	result := app.LoadResult{JobID: before.ID, Status: meta.JobCommitted, Metrics: pipeline.MetricsSnapshot{
		RecordsCommitted: 4_200_000, BatchesCommitted: 210,
	}}
	if err := validateP1Commit(c, before.ID, 1, before, after, result); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name   string
		change func(before, after *meta.Job, result *app.LoadResult)
	}{
		{"wrong job", func(_, _ *meta.Job, r *app.LoadResult) { r.JobID = "another-job" }},
		{"wrong generation", func(_, a *meta.Job, _ *app.LoadResult) { a.GraphGenerationID++ }},
		{"wrong configuration", func(_, a *meta.Job, _ *app.LoadResult) { a.ConfigFingerprint = "another" }},
		{"fault too early", func(b, _ *meta.Job, r *app.LoadResult) { b.CommittedRows--; r.Metrics.RecordsCommitted++ }},
		{"fault too late", func(b, _ *meta.Job, r *app.LoadResult) {
			b.CommittedRows = 2_800_000
			r.Metrics.RecordsCommitted = 2_800_000
		}},
		{"checkpoint did not advance", func(b, a *meta.Job, _ *app.LoadResult) { a.ResumeToken = b.ResumeToken }},
		{"replayed durable rows", func(_, _ *meta.Job, r *app.LoadResult) { r.Metrics.RecordsCommitted++ }},
		{"source rejection", func(_, a *meta.Job, _ *app.LoadResult) { a.SourceRejectedRows++ }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b, a, r := before, after, result
			tt.change(&b, &a, &r)
			if err := validateP1Commit(c, before.ID, 1, b, a, r); err == nil {
				t.Fatal("accepted invalid P1 recovery")
			}
		})
	}
}
