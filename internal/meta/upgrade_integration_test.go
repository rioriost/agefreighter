package meta

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

func TestMetadataV14V17V18UpgradeToV19Integration(t *testing.T) {
	dsn := strings.TrimSpace(os.Getenv(metadataTestDSNEnvironment))
	if dsn == "" {
		t.Skip("set " + metadataTestDSNEnvironment + " to run metadata upgrade integration tests")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()

	adminConfig, err := pgx.ParseConfig(dsn)
	if err != nil {
		t.Fatalf("parse integration DSN: %v", err)
	}
	adminConfig.Database = "postgres"
	admin, err := pgx.ConnectConfig(ctx, adminConfig)
	if err != nil {
		t.Fatalf("connect database administrator: %v", err)
	}
	t.Cleanup(func() {
		_ = admin.Close(context.Background())
	})

	databaseName := fmt.Sprintf("af_meta_upgrade_%d", time.Now().UnixNano())
	identifier := pgx.Identifier{databaseName}.Sanitize()
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+identifier); err != nil {
		t.Skipf("metadata upgrade test requires CREATE DATABASE: %v", err)
	}
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		_, _ = admin.Exec(
			cleanupCtx,
			"DROP DATABASE IF EXISTS "+identifier+" WITH (FORCE)",
		)
	})

	testConfig, err := pgx.ParseConfig(dsn)
	if err != nil {
		t.Fatalf("parse test database DSN: %v", err)
	}
	testConfig.Database = databaseName
	connection, err := pgx.ConnectConfig(ctx, testConfig)
	if err != nil {
		t.Fatalf("connect upgrade database: %v", err)
	}
	defer connection.Close(context.Background())

	if _, err := connection.Exec(ctx, `
		CREATE SCHEMA agefreighter_meta;
		CREATE TABLE agefreighter_meta.schema_migration (
			version integer PRIMARY KEY CHECK (version > 0),
			applied_at timestamp with time zone NOT NULL DEFAULT clock_timestamp()
		)`); err != nil {
		t.Fatalf("initialize migration catalog: %v", err)
	}
	for version := 1; version <= MinimumReadCompatibleSchemaVersion; version++ {
		tx, err := connection.Begin(ctx)
		if err != nil {
			t.Fatalf("begin fixture migration v%d: %v", version, err)
		}
		for _, statement := range migrations[version-1] {
			if _, err := tx.Exec(ctx, statement); err != nil {
				_ = tx.Rollback(ctx)
				t.Fatalf("apply fixture migration v%d: %v", version, err)
			}
		}
		if _, err := tx.Exec(
			ctx,
			`INSERT INTO agefreighter_meta.schema_migration (version) VALUES ($1)`,
			version,
		); err != nil {
			_ = tx.Rollback(ctx)
			t.Fatalf("record fixture migration v%d: %v", version, err)
		}
		if err := tx.Commit(ctx); err != nil {
			t.Fatalf("commit fixture migration v%d: %v", version, err)
		}
	}

	store, err := New(connection)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	before, err := store.InspectSchema(ctx)
	if err != nil {
		t.Fatalf("InspectSchema(v14) error = %v", err)
	}
	if before.State != SchemaPending ||
		before.InstalledVersion != MinimumReadCompatibleSchemaVersion ||
		before.RequireReadCompatible() != nil {
		t.Fatalf("v14 inspection = %#v", before)
	}
	again, err := store.InspectSchema(ctx)
	if err != nil || again != before {
		t.Fatalf("read-only inspection changed metadata: before=%#v after=%#v error=%v", before, again, err)
	}

	if err := store.migrate(ctx, 17); err != nil {
		t.Fatalf("upgrade v14 to v17 fixture: %v", err)
	}
	v17, err := store.InspectSchema(ctx)
	if err != nil {
		t.Fatalf("InspectSchema(v17) error = %v", err)
	}
	if v17.State != SchemaPending || v17.InstalledVersion != 17 {
		t.Fatalf("v17 inspection = %#v", v17)
	}
	const legacyJobID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
	if _, err := connection.Exec(ctx, `
		INSERT INTO agefreighter_meta.load_job (
			job_id, name, source_type, load_mode, target_graph,
			config_fingerprint, status
		) VALUES ($1::uuid, 'legacy-age', 'neo4j', 'create', 'supply_graph',
			$2, 'failed')`, legacyJobID, strings.Repeat("a", 64)); err != nil {
		t.Fatalf("insert v17 AGE job: %v", err)
	}
	legacyBeforeMigration, err := store.GetJob(ctx, legacyJobID)
	if err != nil || legacyBeforeMigration.TargetBackend != TargetBackendApacheAGE ||
		legacyBeforeMigration.TargetSchema != "" {
		t.Fatalf("read-compatible v17 target identity = %#v, %v", legacyBeforeMigration, err)
	}

	if err := store.migrate(ctx, 18); err != nil {
		t.Fatalf("upgrade v17 to v18: %v", err)
	}
	v18, err := store.InspectSchema(ctx)
	if err != nil {
		t.Fatalf("InspectSchema(v18) error = %v", err)
	}
	if v18.State != SchemaPending || v18.InstalledVersion != 18 {
		t.Fatalf("v18 inspection = %#v", v18)
	}
	legacy, err := store.GetJob(ctx, legacyJobID)
	if err != nil {
		t.Fatalf("GetJob(v17 legacy) error = %v", err)
	}
	if legacy.TargetBackend != TargetBackendApacheAGE || legacy.TargetSchema != "" {
		t.Fatalf("v17 target identity backfill = %#v", legacy)
	}
	if err := store.StartJob(ctx, legacyJobID); err != nil {
		t.Fatalf("resume upgraded v17 AGE job: %v", err)
	}
	legacy, err = store.GetJob(ctx, legacyJobID)
	if err != nil || legacy.Status != JobRunning ||
		legacy.ConfigFingerprint != strings.Repeat("a", 64) {
		t.Fatalf("resumed v17 AGE job = %#v, %v", legacy, err)
	}
	propertyGraphJob := Job{
		ID:                "bbbbbbbb-bbbb-4ccc-8ddd-eeeeeeeeeeee",
		Name:              "pg19-property-graph",
		SourceType:        "neo4j",
		LoadMode:          "create",
		TargetBackend:     TargetBackendPostgreSQLPropertyGraph,
		TargetSchema:      "Graph Data",
		TargetGraph:       "supply_graph",
		ConfigFingerprint: strings.Repeat("b", 64),
	}
	if err := store.CreateJob(ctx, propertyGraphJob); err != nil {
		t.Fatalf("create v18 property-graph job: %v", err)
	}
	storedPropertyGraph, err := store.GetJob(ctx, propertyGraphJob.ID)
	if err != nil ||
		storedPropertyGraph.TargetBackend != propertyGraphJob.TargetBackend ||
		storedPropertyGraph.TargetSchema != propertyGraphJob.TargetSchema {
		t.Fatalf("v18 property-graph target identity = %#v, %v", storedPropertyGraph, err)
	}
	if err := store.Migrate(ctx); err != nil {
		t.Fatalf("upgrade v18 to v19: %v", err)
	}
	after, err := store.InspectSchema(ctx)
	if err != nil {
		t.Fatalf("InspectSchema(v19) error = %v", err)
	}
	if after.State != SchemaCurrent || after.InstalledVersion != SupportedSchemaVersion {
		t.Fatalf("upgraded inspection = %#v", after)
	}
	mapping := PropertyGraphGeneration{
		JobID: propertyGraphJob.ID, Schema: propertyGraphJob.TargetSchema,
		Graph:                 propertyGraphJob.TargetGraph,
		DefinitionFingerprint: strings.Repeat("c", 64),
		State:                 PropertyGraphLoading,
		Labels: []PropertyGraphLabel{
			{Name: "Person", Kind: VertexLabel, Table: "v_person"},
			{
				Name: "KNOWS", Kind: EdgeLabel, Table: "e_knows",
				StartLabel: "Person", EndLabel: "Person",
			},
		},
	}
	if err := store.RegisterPropertyGraph(ctx, mapping); err != nil {
		t.Fatalf("register v19 property graph mapping: %v", err)
	}
	storedMapping, err := store.GetPropertyGraph(ctx, propertyGraphJob.ID)
	if err != nil || storedMapping.JobID != mapping.JobID ||
		storedMapping.DefinitionFingerprint != mapping.DefinitionFingerprint ||
		len(storedMapping.Labels) != len(mapping.Labels) {
		t.Fatalf("v19 property graph mapping = %#v, %v", storedMapping, err)
	}
	if err := store.migrate(ctx, MinimumReadCompatibleSchemaVersion); err == nil ||
		!strings.Contains(err.Error(), "newer than supported version 14") {
		t.Fatalf("2.0-compatible writer accepted v19 metadata: %v", err)
	}
}
