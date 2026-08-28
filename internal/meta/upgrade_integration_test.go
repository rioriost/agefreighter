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

func TestMetadataV14UpgradeToV17Integration(t *testing.T) {
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

	if err := store.Migrate(ctx); err != nil {
		t.Fatalf("upgrade v14 to v17: %v", err)
	}
	after, err := store.InspectSchema(ctx)
	if err != nil {
		t.Fatalf("InspectSchema(v17) error = %v", err)
	}
	if after.State != SchemaCurrent ||
		after.InstalledVersion != SupportedSchemaVersion {
		t.Fatalf("upgraded inspection = %#v", after)
	}
	if err := store.migrate(ctx, MinimumReadCompatibleSchemaVersion); err == nil ||
		!strings.Contains(err.Error(), "newer than supported version 14") {
		t.Fatalf("2.0-compatible writer accepted v17 metadata: %v", err)
	}
}
