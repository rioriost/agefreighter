package simulation_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
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

// This is only the two CSV clean-create cases, not a complete P0 route/phase.
func TestHorizonDBP0CSVCreateIntegration(t *testing.T) {
	if os.Getenv("AGEFREIGHTER_REQUIRE_HORIZON_P0_CSV_CREATE") != "1" {
		t.Skip("explicit P0 CSV-create execution and an enrolled independent guard required")
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
		DeadlineUTC         time.Time `json:"deadlineUtc"`
		EnrolledReused      bool      `json:"enrolledReusedClusters"`
	}
	body, err := os.ReadFile(os.Getenv("AGEFREIGHTER_HORIZON_SCALE_GUARD_RECEIPT"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(body, &guard); err != nil {
		t.Fatal(err)
	}
	if !guard.VerifiedAbsentProbe || !guard.EnrolledReused || time.Until(guard.DeadlineUTC) < 40*time.Minute {
		t.Fatal("verified independent deletion probe and sufficient enrolled lease required")
	}
	if err := os.Mkdir(output, 0700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(output, "fixture", "manifest.json")
	f, err := fixture.Generate(t.Context(), fixture.GenerateConfig{
		Phase: fixture.PhaseP0, Output: filepath.Dir(path), Shards: 4, Workers: 2, Seed: 20260829,
	})
	if err != nil {
		t.Fatal(err)
	}
	if f.Plan.VertexTotal != 100_000 || f.Plan.EdgeTotal != 358_000 {
		t.Fatal("P0 envelope mismatch")
	}
	dir := filepath.Join(output, "portable")
	p, err := portable.Export(t.Context(), path, dir)
	if err != nil {
		t.Fatal(err)
	}
	want, err := rangedigest.FixtureManifest(t.Context(), path, 100000)
	if err != nil {
		t.Fatal(err)
	}
	credential, err := azidentity.NewManagedIdentityCredential(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, major := range []int{17, 18} {
		t.Run(fmt.Sprintf("pg%d", major), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 15*time.Minute)
			defer cancel()
			host := os.Getenv(fmt.Sprintf("AGEFREIGHTER_HORIZON_PG%d_HOST", major))
			if !strings.HasPrefix(host, fmt.Sprintf("afht-20261118-pg%d.", major)) ||
				!strings.HasSuffix(host, ".japaneast.horizondb.azure.com") {
				t.Fatal("enrolled reused HorizonDB endpoint required")
			}
			token, err := credential.GetToken(ctx, policy.TokenRequestOptions{
				Scopes: []string{"https://ossrdbms-aad.database.windows.net/.default"},
			})
			if err != nil {
				t.Fatal(err)
			}
			if time.Until(token.ExpiresOn) < 20*time.Minute {
				t.Fatal("token validity must exceed this strictly bounded P0 case")
			}
			user := os.Getenv("AGEFREIGHTER_HORIZON_USER")
			if user != "afht-20261118-runner" {
				t.Fatal("enrolled runner managed identity required")
			}
			u := url.URL{Scheme: "postgresql", Host: host + ":5432", Path: "/postgres",
				User: url.UserPassword(user, token.Token), RawQuery: "sslmode=verify-full"}
			secret := filepath.Join(t.TempDir(), "target.dsn")
			if err := os.WriteFile(secret, []byte(u.String()), 0600); err != nil {
				t.Fatal(err)
			}
			connection, err := pgx.Connect(ctx, u.String())
			if err != nil {
				t.Fatal(err)
			}
			defer connection.Close(context.Background())
			var pg, age string
			var number int
			if err := connection.QueryRow(ctx, `SELECT current_setting('server_version_num')::integer,
				version(),extversion FROM pg_extension WHERE extname='age'`).Scan(&number, &pg, &age); err != nil {
				t.Fatal(err)
			}
			if number/10000 != major || !strings.HasPrefix(age, "1.8.") {
				t.Fatal("effective PostgreSQL/AGE versions differ")
			}
			job := tinyJob(p, dir, "csv", major)
			job.Metadata.Name = fmt.Sprintf("horizon-p0-csv-pg%d-%s", major, run)
			job.Target.Graph = fmt.Sprintf("hs_p0_csv_pg%d_%s", major, run)
			job.Target.Connection = config.SecretRef{File: secret}
			job.Runtime.BatchRows, job.Runtime.BatchBytes = 20_000, 64<<20
			job.Runtime.MaxTargetConnections = 4
			configPath := filepath.Join(output, job.Metadata.Name+".json")
			writeTinyJSON(t, configPath, job)
			start := time.Now().UTC()
			result, loadErr := app.Load(ctx, configPath)
			writeTinyJSON(t, filepath.Join(output, job.Metadata.Name+"-load.json"), result)
			if loadErr != nil {
				t.Fatal(loadErr)
			}
			if result.Status != meta.JobCommitted || result.Metrics.RecordsCommitted != 458_000 ||
				result.Metrics.BatchesFailed != 0 {
				t.Fatalf("P0 clean-create result differs: %+v", result)
			}
			if _, err := app.Verify(ctx, configPath, result.JobID); err != nil {
				t.Fatal(err)
			}
			actual, err := rangedigest.ScaleTargetManifest(ctx, u.String(), path, result.JobID, 100000, false)
			if err != nil {
				t.Fatal(err)
			}
			comparison, compareErr := rangedigest.Compare(want, actual)
			writeTinyJSON(t, filepath.Join(output, job.Metadata.Name+"-verification.json"), map[string]any{
				"coverage":        "P0 CSV clean-create only; remaining modes/sources are not qualified",
				"candidateCommit": commit, "startedAtUtc": start, "completedAtUtc": time.Now().UTC(),
				"expected": want, "actual": actual, "comparison": comparison,
				"postgresql": pg, "age": age, "jobId": result.JobID,
			})
			if compareErr != nil {
				t.Fatal(compareErr)
			}
			if _, err := fixture.Verify(path); err != nil {
				t.Fatal(err)
			}
			t.Logf("P0 CSV clean-create: %d records; root %s", actual.RecordCount, actual.RootSHA256)
		})
	}
}
