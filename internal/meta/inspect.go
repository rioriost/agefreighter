package meta

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5/pgconn"
)

const SupportedSchemaVersion = schemaVersion

type SchemaState string

const (
	SchemaAbsent  SchemaState = "absent"
	SchemaCurrent SchemaState = "current"
	SchemaPending SchemaState = "pending"
	SchemaNewer   SchemaState = "newer"
	SchemaInvalid SchemaState = "invalid"
	SchemaUnknown SchemaState = "unknown"
)

type SchemaInspection struct {
	State            SchemaState `json:"state"`
	InstalledVersion int         `json:"installedVersion,omitempty"`
	SupportedVersion int         `json:"supportedVersion"`
	PendingVersions  int         `json:"pendingVersions,omitempty"`
	Detail           string      `json:"detail,omitempty"`
}

func (inspection SchemaInspection) RequireCurrent() error {
	if inspection.State == SchemaCurrent {
		return nil
	}
	return fmt.Errorf(
		"metadata schema is %s (installed version %d, supported version %d)",
		inspection.State,
		inspection.InstalledVersion,
		inspection.SupportedVersion,
	)
}

// InspectSchema reads migration metadata without creating or changing any
// database object.
func (store *Store) InspectSchema(ctx context.Context) (SchemaInspection, error) {
	inspection := SchemaInspection{
		State:            SchemaUnknown,
		SupportedVersion: SupportedSchemaVersion,
	}
	if store == nil || store.database == nil {
		return inspection, errors.New("metadata store is required")
	}
	var schemaExists, tableExists bool
	err := store.database.QueryRow(
		ctx,
		`SELECT
			pg_catalog.to_regnamespace('agefreighter_meta') IS NOT NULL,
			pg_catalog.to_regclass(
				'agefreighter_meta.schema_migration'
			) IS NOT NULL`,
	).Scan(&schemaExists, &tableExists)
	if err != nil {
		if metadataPermissionDenied(err) {
			inspection.Detail = "permission denied while inspecting metadata catalog"
			return inspection, nil
		}
		return inspection, fmt.Errorf("inspect metadata catalog: %w", err)
	}
	if !schemaExists {
		inspection.State = SchemaAbsent
		return inspection, nil
	}
	if !tableExists {
		inspection.State = SchemaInvalid
		inspection.Detail = "metadata schema exists without schema_migration table"
		return inspection, nil
	}

	var minimum, maximum, count int
	err = store.database.QueryRow(
		ctx,
		`SELECT
			COALESCE(MIN(version), 0),
			COALESCE(MAX(version), 0),
			COUNT(*)::integer
		 FROM agefreighter_meta.schema_migration`,
	).Scan(&minimum, &maximum, &count)
	if err != nil {
		if metadataPermissionDenied(err) {
			inspection.Detail = "permission denied while reading metadata schema version"
			return inspection, nil
		}
		return inspection, fmt.Errorf("read metadata schema version: %w", err)
	}
	inspection.InstalledVersion = maximum
	if count == 0 {
		inspection.State = SchemaPending
		inspection.PendingVersions = SupportedSchemaVersion
		return inspection, nil
	}
	if minimum != 1 || maximum != count {
		inspection.State = SchemaInvalid
		inspection.Detail = "metadata migration history is not contiguous from version 1"
		return inspection, nil
	}
	switch {
	case maximum < SupportedSchemaVersion:
		inspection.State = SchemaPending
		inspection.PendingVersions = SupportedSchemaVersion - maximum
	case maximum == SupportedSchemaVersion:
		inspection.State = SchemaCurrent
	default:
		inspection.State = SchemaNewer
	}
	return inspection, nil
}

func metadataPermissionDenied(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "42501"
}
