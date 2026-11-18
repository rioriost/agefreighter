package simulation_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
	"github.com/rioriost/agefreighter/production-simulation/internal/rangedigest"
)

func TestHorizonDBP0PostgresSourceIntegration(t *testing.T) {
	if os.Getenv("AGEFREIGHTER_REQUIRE_HORIZON_P0_POSTGRES_SOURCE") != "1" {
		t.Skip("explicit P0 source-only fixture preflight required")
	}
	output := os.Getenv("AGEFREIGHTER_HORIZON_POSTGRES_SOURCE_OUTPUT")
	run := os.Getenv("AGEFREIGHTER_HORIZON_POSTGRES_SOURCE_RUN_ID")
	if !filepath.IsAbs(output) || !regexp.MustCompile(`^r[a-z0-9]{6,30}$`).MatchString(run) {
		t.Fatal("new absolute source evidence directory and unique run ID required")
	}
	if err := os.Mkdir(output, 0700); err != nil {
		t.Fatal(err)
	}
	paths, manifests, directories := horizonP0Fixtures(t, output)
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Minute)
	defer cancel()
	if err := seedHorizonP0Postgres(ctx, paths, manifests, directories, run, output); err != nil {
		t.Fatal(err)
	}
	t.Log("P0 PostgreSQL source COPY counts and immutable CSV hashes verified; no HorizonDB migration")
}

func validateHorizonP0PostgresSource(dsn string) error {
	u, err := url.Parse(dsn)
	if err != nil || u.Scheme != "postgresql" || u.Host != "127.0.0.1:5432" || u.Path != "/source" {
		return errors.New("dedicated local PostgreSQL source database required")
	}
	parsed, err := pgx.ParseConfig(dsn)
	if err != nil || parsed.Host != "127.0.0.1" || parsed.Port != 5432 || parsed.Database != "source" {
		return errors.New("effective PostgreSQL source endpoint differs from the dedicated fixture")
	}
	for _, fallback := range parsed.Fallbacks {
		if fallback.Host != "127.0.0.1" || fallback.Port != 5432 {
			return errors.New("PostgreSQL source fallback escapes the dedicated fixture")
		}
	}
	return nil
}

func horizonP0PostgresQueries(job *config.LoadJob, schema string) error {
	if job.Source.Type != config.SourcePostgreSQL || job.Source.PostgreSQL == nil ||
		!regexp.MustCompile(`^hs_p0_pg_r[a-z0-9]{6,30}_(base|append|upsert)$`).MatchString(schema) {
		return errors.New("P0 PostgreSQL source and run-scoped immutable schema required")
	}
	pg := job.Source.PostgreSQL
	pg.FetchRows = 20_000
	for i := range pg.Vertices {
		q := &pg.Vertices[i]
		q.Query = "SELECT * FROM " + pgx.Identifier{schema, q.Label}.Sanitize() + " ORDER BY source_key"
	}
	for i := range pg.Edges {
		q := &pg.Edges[i]
		q.Query = "SELECT * FROM " + pgx.Identifier{schema, q.Label}.Sanitize() + " ORDER BY source_key"
	}
	return nil
}

func seedHorizonP0Postgres(ctx context.Context, paths rangedigest.Scenario, manifests map[string]portable.Manifest,
	directories map[string]string, run, output string,
) error {
	dsn := os.Getenv("AGEFREIGHTER_POSTGRES_TEST_DSN")
	if err := validateHorizonP0PostgresSource(dsn); err != nil {
		return err
	}
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return err
	}
	defer conn.Close(context.Background())
	var number int
	var version string
	if err := conn.QueryRow(ctx, "SELECT current_setting('server_version_num')::integer,version()").Scan(&number, &version); err != nil {
		return err
	}
	if number/10000 != 17 {
		return errors.New("PostgreSQL 17 fixture source required")
	}
	datasets := map[string]any{}
	receipt := map[string]any{"sourceVersion": version, "datasets": datasets}
	for _, name := range []string{"base", "append", "upsert"} {
		path := map[string]string{"base": paths.Base, "append": paths.Append, "upsert": paths.Upsert}[name]
		f, err := fixture.Verify(path)
		if err != nil {
			return err
		}
		manifest, ok := manifests[name]
		if !ok || f.Plan.Phase != fixture.PhaseP0 || manifest.FixtureRoot != f.RootSHA256 {
			return errors.New("verified P0 portable provenance required before source writes")
		}
		schema := "hs_p0_pg_" + run + "_" + name
		job := tinyJob(manifest, directories[name], "postgresql", 17)
		if err := horizonP0PostgresQueries(&job, schema); err != nil {
			return err
		}
		if _, err := conn.Exec(ctx, "CREATE SCHEMA "+pgx.Identifier{schema}.Sanitize()); err != nil {
			return err
		}
		var count int64
		for _, table := range manifest.Tables {
			var columns, names []string
			for _, name := range table.Columns {
				sqlType := "text"
				switch table.Types[name] {
				case "int64":
					sqlType = "bigint"
				case "float64":
					sqlType = "double precision"
				case "boolean":
					sqlType = "boolean"
				case "string[]", "int64[]":
					sqlType = "jsonb"
				}
				if name == "source_key" {
					sqlType += " PRIMARY KEY"
				}
				column := pgx.Identifier{name}.Sanitize()
				columns, names = append(columns, column+" "+sqlType), append(names, column)
			}
			id := pgx.Identifier{schema, table.Name}.Sanitize()
			if _, err := conn.Exec(ctx, "CREATE TABLE "+id+" ("+strings.Join(columns, ",")+")"); err != nil {
				return err
			}
			file, err := os.Open(filepath.Join(directories[name], table.CSV))
			if err != nil {
				return err
			}
			hash := sha256.New()
			copied, copyErr := conn.PgConn().CopyFrom(ctx, io.TeeReader(file, hash),
				"COPY "+id+" ("+strings.Join(names, ",")+`) FROM STDIN WITH (FORMAT CSV,HEADER true,NULL '\N')`)
			if err := errors.Join(copyErr, file.Close()); err != nil {
				return err
			}
			if copied.RowsAffected() != table.Rows || hex.EncodeToString(hash.Sum(nil)) != table.CSVHash {
				return errors.New("source COPY row count or immutable CSV hash differs")
			}
			var actual int64
			if err := conn.QueryRow(ctx, "SELECT count(*) FROM "+id).Scan(&actual); err != nil {
				return err
			}
			if actual != table.Rows {
				return errors.New("seeded source table count differs")
			}
			count += actual
		}
		if count != f.Plan.VertexTotal+f.Plan.EdgeTotal {
			return errors.New("seeded P0 dataset count differs")
		}
		if _, err := fixture.Verify(path); err != nil {
			return err
		}
		datasets[name] = map[string]any{"schema": schema, "records": count, "fixtureRootSha256": f.RootSHA256}
	}
	body, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(output, "postgres-source-seed.json"), append(body, '\n'), 0600)
}

func TestHorizonP0PostgresConfigScope(t *testing.T) {
	if err := validateHorizonP0PostgresSource("postgresql://postgres:fixture@127.0.0.1:5432/source?sslmode=disable"); err != nil {
		t.Fatal(err)
	}
	for _, dsn := range []string{
		"postgresql://database.example/source", "postgresql://127.0.0.1:5432/production",
		"postgresql://127.0.0.1:5433/source", "postgresql://127.0.0.1:5432/source-other",
		"postgresql://127.0.0.1:5432/source?host=database.example",
		"postgresql://127.0.0.1:5432/source?port=5433",
		"postgresql://127.0.0.1:5432/source?dbname=production",
		"postgresql://127.0.0.1:5432/source?host=127.0.0.1,database.example",
	} {
		if validateHorizonP0PostgresSource(dsn) == nil {
			t.Fatal("accepted an unrelated source endpoint")
		}
	}
	for _, dataset := range []string{"base", "append", "upsert"} {
		job := tinyJob(portable.Manifest{Tables: []portable.Table{
			{Name: "Supplier", Kind: "node"}, {Name: "Product", Kind: "node"},
			{Name: "SUPPLIES", Kind: "edge", StartLabel: "Supplier", EndLabel: "Product"},
		}}, t.TempDir(), "postgresql", 17)
		schema := "hs_p0_pg_r20261118p0pg03_" + dataset
		if err := horizonP0PostgresQueries(&job, schema); err != nil {
			t.Fatal(err)
		}
		pg := job.Source.PostgreSQL
		if pg.Vertices[0].Query != fmt.Sprintf(`SELECT * FROM "%s"."Supplier" ORDER BY source_key`, schema) ||
			pg.Edges[0].Query != fmt.Sprintf(`SELECT * FROM "%s"."SUPPLIES" ORDER BY source_key`, schema) {
			t.Fatal("source query escaped its immutable dataset")
		}
		path := filepath.Join(t.TempDir(), "job.json")
		writeTinyJSON(t, path, job)
		if _, err := config.Load(path); err != nil {
			t.Fatal(err)
		}
		for _, bad := range []string{"tiny", schema + "-other", `hs_p0_pg_r20261118p0pg03_base";DROP SCHEMA tiny;--`} {
			if horizonP0PostgresQueries(&job, bad) == nil {
				t.Fatal("accepted an unrelated schema")
			}
		}
	}
}
