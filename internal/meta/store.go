package meta

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

const migrationLockID int64 = 0x6167656672656967
const defaultMigrationTimeout = 30 * time.Second

type Database interface {
	Begin(context.Context) (pgx.Tx, error)
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
	QueryRow(context.Context, string, ...any) pgx.Row
}

type Store struct {
	database Database
}

func New(database Database) (*Store, error) {
	if database == nil {
		return nil, errors.New("metadata database is required")
	}
	return &Store{database: database}, nil
}

func (store *Store) Migrate(ctx context.Context) error {
	if store == nil || store.database == nil {
		return errors.New("metadata store is required")
	}
	migrationCtx := ctx
	cancel := func() {}
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		migrationCtx, cancel = context.WithTimeout(ctx, defaultMigrationTimeout)
	}
	defer cancel()
	timeout := remainingTimeout(migrationCtx, defaultMigrationTimeout)
	tx, err := store.database.Begin(migrationCtx)
	if err != nil {
		return fmt.Errorf("begin metadata migration: %w", err)
	}
	defer rollbackWithTimeout(migrationCtx, tx, timeout)

	if _, err := tx.Exec(
		migrationCtx,
		`SELECT
			pg_catalog.set_config('lock_timeout', $1, true),
			pg_catalog.set_config('statement_timeout', $1, true)`,
		postgresTimeout(timeout),
	); err != nil {
		return fmt.Errorf("bound metadata migration transaction: %w", err)
	}
	if _, err := tx.Exec(
		migrationCtx,
		`SELECT pg_catalog.pg_advisory_xact_lock($1)`,
		migrationLockID,
	); err != nil {
		return fmt.Errorf("lock metadata migration: %w", err)
	}
	if _, err := tx.Exec(
		migrationCtx,
		`CREATE SCHEMA IF NOT EXISTS agefreighter_meta`,
	); err != nil {
		return fmt.Errorf("create metadata schema: %w", err)
	}
	if _, err := tx.Exec(
		migrationCtx,
		`CREATE TABLE IF NOT EXISTS agefreighter_meta.schema_migration (
			version integer PRIMARY KEY CHECK (version > 0),
			applied_at timestamp with time zone NOT NULL DEFAULT clock_timestamp()
		)`,
	); err != nil {
		return fmt.Errorf("create metadata migration table: %w", err)
	}

	var current int
	if err := tx.QueryRow(
		migrationCtx,
		`SELECT COALESCE(MAX(version), 0)
		 FROM agefreighter_meta.schema_migration`,
	).Scan(&current); err != nil {
		return fmt.Errorf("read metadata schema version: %w", err)
	}
	if current > schemaVersion {
		return fmt.Errorf(
			"metadata schema version %d is newer than supported version %d",
			current,
			schemaVersion,
		)
	}
	for version := current + 1; version <= schemaVersion; version++ {
		for _, statement := range migrations[version-1] {
			if _, err := tx.Exec(migrationCtx, statement); err != nil {
				return fmt.Errorf("apply metadata migration %d: %w", version, err)
			}
		}
		if _, err := tx.Exec(
			migrationCtx,
			`INSERT INTO agefreighter_meta.schema_migration (version) VALUES ($1)`,
			version,
		); err != nil {
			return fmt.Errorf("record metadata migration %d: %w", version, err)
		}
	}
	if err := tx.Commit(migrationCtx); err != nil {
		return fmt.Errorf("commit metadata migration: %w", err)
	}
	return nil
}

func remainingTimeout(ctx context.Context, fallback time.Duration) time.Duration {
	if deadline, ok := ctx.Deadline(); ok {
		if remaining := time.Until(deadline); remaining > 0 {
			return remaining
		}
		return time.Millisecond
	}
	return fallback
}

func postgresTimeout(timeout time.Duration) string {
	return fmt.Sprintf("%dms", max(timeout.Milliseconds(), 1))
}

func rollback(ctx context.Context, tx pgx.Tx) {
	_ = tx.Rollback(context.WithoutCancel(ctx))
}

func rowsAffectedOne(tag pgconn.CommandTag, operation string) error {
	if tag.RowsAffected() != 1 {
		return fmt.Errorf("%w: %s affected %d rows", ErrConflict, operation, tag.RowsAffected())
	}
	return nil
}
