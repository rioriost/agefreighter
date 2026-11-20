//go:build linux || darwin

package simulation_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
)

const (
	p1PostgresAdmin  = "/dev/shm/agefreighter-p1-postgres/admin.dsn"
	p1PostgresReader = "/dev/shm/agefreighter-p1-postgres/reader.dsn"
)

func TestHorizonDBP1PostgresPG17Integration(t *testing.T) {
	runHorizonDBP1(t, 17, "postgresql")
}

func TestHorizonDBP1PostgresPG18Integration(t *testing.T) {
	runHorizonDBP1(t, 18, "postgresql")
}

func p1PostgresSchema(a p1Admission, dataset string) string {
	return "hs_p1_pg_" + a.RunID + "_" + dataset
}

func validateP1PostgresJob(job config.LoadJob, schema string) error {
	pg := job.Source.PostgreSQL
	if job.Source.Type != config.SourcePostgreSQL || pg == nil ||
		pg.ReadMode != config.PostgreSQLReadCopy || pg.Connection != (config.SecretRef{File: p1PostgresReader}) {
		return errors.New("require the admitted local read-only PostgreSQL COPY connector")
	}
	for _, v := range pg.Vertices {
		if v.Query != horizonPostgresQuery(schema, v.Label) {
			return errors.New("vertex query escaped its immutable PostgreSQL dataset")
		}
	}
	for _, e := range pg.Edges {
		if e.Query != horizonPostgresQuery(schema, e.Label) {
			return errors.New("edge query escaped its immutable PostgreSQL dataset")
		}
	}
	return nil
}

func p1PostgresDSN(t *testing.T, path, user string) string {
	t.Helper()
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Mode().Perm() != 0600 {
		t.Fatal("require a private PostgreSQL source DSN file")
	}
	raw, err := os.ReadFile(path)
	if err != nil || validateHorizonP0PostgresSource(string(raw)) != nil {
		t.Fatal("require the dedicated loopback PostgreSQL source database")
	}
	dsn := string(raw)
	parsed, err := pgx.ParseConfig(dsn)
	if err != nil || parsed.User != user || parsed.Password == "" || parsed.TLSConfig != nil || len(parsed.Fallbacks) != 0 {
		t.Fatal("require the dedicated authenticated loopback source role")
	}
	return dsn
}

func prepareP1Postgres(t *testing.T, ctx context.Context, a p1Admission, paths map[string]string, exports map[string]portable.Manifest) func() {
	t.Helper()
	adminDSN := p1PostgresDSN(t, p1PostgresAdmin, "postgres")
	readerDSN := p1PostgresDSN(t, p1PostgresReader, "p1_reader")
	directories := make(map[string]string, len(p1FixtureOrder))
	for _, name := range p1FixtureOrder {
		directories[name] = filepath.Join(a.Output, "csv-"+name)
	}
	seedCtx, stopSeed := context.WithTimeout(ctx, 25*time.Minute)
	defer stopSeed()
	if err := seedHorizonPostgres(seedCtx, p1Scenario(paths, "upsert"), exports, directories,
		a.RunID, a.Output, adminDSN, fixture.PhaseP1); err != nil {
		p1DatabaseFailure(t, "PostgreSQL 17.6 fixture COPY and counts", err)
	}
	admin, err := pgx.Connect(seedCtx, adminDSN)
	if err != nil {
		p1DatabaseFailure(t, "PostgreSQL fixture privilege setup", err)
	}
	defer admin.Close(context.Background())
	for _, name := range p1FixtureOrder {
		schema := pgx.Identifier{p1PostgresSchema(a, name)}.Sanitize()
		if _, err := admin.Exec(seedCtx, "GRANT USAGE ON SCHEMA "+schema+" TO p1_reader"); err != nil {
			p1DatabaseFailure(t, "PostgreSQL fixture schema grant", err)
		}
		if _, err := admin.Exec(seedCtx, "GRANT SELECT ON ALL TABLES IN SCHEMA "+schema+" TO p1_reader"); err != nil {
			p1DatabaseFailure(t, "PostgreSQL fixture SELECT grant", err)
		}
	}
	before, err := hashP1Postgres(seedCtx, readerDSN, a, exports)
	if err != nil {
		p1DatabaseFailure(t, "PostgreSQL source digest before migration", err)
	}
	writeTinyJSON(t, filepath.Join(a.Output, "postgres-source-before.json"), before)
	return func() {
		verifyCtx, stop := context.WithTimeout(ctx, 20*time.Minute)
		defer stop()
		after, err := hashP1Postgres(verifyCtx, readerDSN, a, exports)
		if err != nil {
			p1DatabaseFailure(t, "PostgreSQL source digest after migration", err)
		}
		writeTinyJSON(t, filepath.Join(a.Output, "postgres-source-after.json"), after)
		if !reflect.DeepEqual(before, after) {
			t.Fatal("PostgreSQL source row counts or ordered COPY digests changed")
		}
	}
}

type p1PostgresTable struct {
	Dataset string `json:"dataset"`
	Table   string `json:"table"`
	Rows    int64  `json:"rows"`
	SHA256  string `json:"sha256"`
}

func hashP1Postgres(ctx context.Context, dsn string, a p1Admission, exports map[string]portable.Manifest) ([]p1PostgresTable, error) {
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return nil, err
	}
	defer conn.Close(context.Background())
	var restricted bool
	if err := conn.QueryRow(ctx, `SELECT current_setting('server_version_num')::integer = 170006 AND
		current_user = 'p1_reader' AND NOT (rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls)
		FROM pg_roles WHERE rolname = current_user`).Scan(&restricted); err != nil {
		return nil, err
	}
	if !restricted {
		return nil, errors.New("expected the PostgreSQL 17.6 read-only fixture role")
	}
	records := []p1PostgresTable{}
	for _, dataset := range p1FixtureOrder {
		schema := p1PostgresSchema(a, dataset)
		for _, table := range exports[dataset].Tables {
			id := pgx.Identifier{schema, table.Name}.Sanitize()
			var canWrite bool
			if err := conn.QueryRow(ctx, `SELECT has_table_privilege($1,'INSERT,UPDATE,DELETE,TRUNCATE,TRIGGER')`, id).Scan(&canWrite); err != nil {
				return nil, err
			}
			if canWrite {
				return nil, errors.New("fixture reader has table write privileges")
			}
			hash := sha256.New()
			tag, err := conn.PgConn().CopyTo(ctx, hash, "COPY (SELECT * FROM "+id+" ORDER BY source_key) TO STDOUT WITH (FORMAT binary)")
			if err != nil {
				return nil, err
			}
			if tag.RowsAffected() != table.Rows {
				return nil, fmt.Errorf("source %s/%s row count differs", dataset, table.Name)
			}
			records = append(records, p1PostgresTable{dataset, table.Name, tag.RowsAffected(), hex.EncodeToString(hash.Sum(nil))})
		}
	}
	return records, nil
}

func TestHorizonDBP1PostgresWorkerScope(t *testing.T) {
	a := p1Admission{RunID: "rp1test01"}
	job := tinyJob(portable.Manifest{Tables: []portable.Table{{Kind: "node", Name: "Sample"}}}, t.TempDir(), "postgresql", 17)
	job.Source.PostgreSQL.Connection = config.SecretRef{File: p1PostgresReader}
	schema := p1PostgresSchema(a, "append")
	if err := horizonPostgresQueries(&job, schema); err != nil {
		t.Fatal(err)
	}
	if err := validateP1PostgresJob(job, schema); err != nil {
		t.Fatal(err)
	}
	pg := job.Source.PostgreSQL
	pg.Vertices[0].Query = strings.ReplaceAll(pg.Vertices[0].Query, "_append", "_upsert")
	if validateP1PostgresJob(job, schema) == nil {
		t.Fatal("accepted a query against a different immutable dataset")
	}
}
