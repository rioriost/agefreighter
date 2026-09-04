package pggraph

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"
)

const minimumServerVersion = 190000

type Capabilities struct {
	ServerVersion       string
	ServerVersionNumber int
	PropertyGraph       bool
}

func ValidateServerVersion(version int) error {
	if version < minimumServerVersion {
		return fmt.Errorf(
			"PostgreSQL property graphs require server_version_num >= %d, got %d",
			minimumServerVersion,
			version,
		)
	}
	return nil
}

// Probe connects to PostgreSQL and rolls back a temporary SQL/PGQ object. The
// DDL probe catches development builds that report version 19 but do not carry
// the property graph feature.
func Probe(ctx context.Context, dsn string) (Capabilities, error) {
	if strings.TrimSpace(dsn) == "" {
		return Capabilities{}, errors.New("PostgreSQL connection string is required")
	}
	connection, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return Capabilities{}, fmt.Errorf("connect to PostgreSQL property graph target: %w", err)
	}
	defer connection.Close(context.Background())

	capabilities := Capabilities{}
	var versionNumber string
	if err := connection.QueryRow(ctx,
		`SELECT current_setting('server_version'), current_setting('server_version_num')`,
	).Scan(&capabilities.ServerVersion, &versionNumber); err != nil {
		return Capabilities{}, fmt.Errorf("read PostgreSQL version: %w", err)
	}
	capabilities.ServerVersionNumber, err = strconv.Atoi(versionNumber)
	if err != nil {
		return Capabilities{}, fmt.Errorf("parse server_version_num %q: %w", versionNumber, err)
	}
	if err := ValidateServerVersion(capabilities.ServerVersionNumber); err != nil {
		return capabilities, err
	}

	transaction, err := connection.Begin(ctx)
	if err != nil {
		return capabilities, fmt.Errorf("begin property graph capability probe: %w", err)
	}
	defer transaction.Rollback(context.Background())
	statements := []string{
		`CREATE TEMP TABLE agefreighter_pggraph_probe_vertex (` +
			`id bigint PRIMARY KEY, external_id text, properties jsonb)`,
		`CREATE TEMP PROPERTY GRAPH agefreighter_pggraph_probe ` +
			`VERTEX TABLES (agefreighter_pggraph_probe_vertex ` +
			`LABEL vertex PROPERTIES (external_id, properties))`,
		`SELECT external_id FROM GRAPH_TABLE (` +
			`agefreighter_pggraph_probe MATCH (v IS vertex) ` +
			`COLUMNS (v.external_id AS external_id)) LIMIT 0`,
	}
	for _, statement := range statements {
		if _, err := transaction.Exec(ctx, statement); err != nil {
			return capabilities, fmt.Errorf("probe PostgreSQL SQL/PGQ support: %w", err)
		}
	}
	capabilities.PropertyGraph = true
	return capabilities, nil
}
