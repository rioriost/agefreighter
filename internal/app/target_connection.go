package app

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/config"
)

func targetCredentialRefresher(reference config.SecretRef) func(context.Context, *pgx.ConnConfig) error {
	if reference.File == "" {
		return nil
	}
	return func(ctx context.Context, connection *pgx.ConnConfig) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		dsn, err := resolveSecret(reference)
		if err != nil {
			return fmt.Errorf("refresh target credential: %w", err)
		}
		refreshed, err := pgx.ParseConfig(dsn)
		if err != nil {
			// A malformed credential file may contain an unredactable token.
			return errors.New("rotated target connection is invalid")
		}
		if refreshed.Host != connection.Host || refreshed.Port != connection.Port ||
			refreshed.User != connection.User || refreshed.Database != connection.Database {
			return errors.New("rotated target connection must retain host, port, user and database")
		}
		connection.Password = refreshed.Password
		return nil
	}
}
