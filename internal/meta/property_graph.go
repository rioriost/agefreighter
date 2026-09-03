package meta

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

type PropertyGraphState string

const (
	PropertyGraphLoading PropertyGraphState = "loading"
	PropertyGraphActive  PropertyGraphState = "active"
)

type PropertyGraphLabel struct {
	Name       string
	Kind       LabelKind
	Table      string
	StartLabel string
	EndLabel   string
}

type PropertyGraphGeneration struct {
	JobID                 string
	Schema                string
	Graph                 string
	DefinitionFingerprint string
	State                 PropertyGraphState
	DigestRoot            string
	DigestRows            int64
	DigestRangeCount      int
	Labels                []PropertyGraphLabel
}

type PropertyGraphDigestRange struct {
	JobID     string
	LabelName string
	Kind      LabelKind
	RangeID   uint8
	Rows      int64
	Digest    string
}

func (store *Store) RegisterPropertyGraph(
	ctx context.Context,
	value PropertyGraphGeneration,
) error {
	if err := validatePropertyGraph(value); err != nil {
		return err
	}
	tx, existing := store.database.(pgx.Tx)
	owns := !existing
	var err error
	if owns {
		tx, err = store.database.Begin(ctx)
		if err != nil {
			return fmt.Errorf("begin property graph registration: %w", err)
		}
		defer rollback(ctx, tx)
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO agefreighter_meta.property_graph_generation (
			job_id, target_schema, graph_name, definition_fingerprint, state
		) VALUES ($1::uuid, $2, $3, $4, $5)`,
		value.JobID, value.Schema, value.Graph,
		value.DefinitionFingerprint, value.State,
	); err != nil {
		return fmt.Errorf("register property graph generation: %w", err)
	}
	for _, label := range value.Labels {
		var start, end any
		if label.Kind == EdgeLabel {
			start, end = label.StartLabel, label.EndLabel
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO agefreighter_meta.property_graph_label (
				job_id, label_name, kind, table_name, start_label, end_label
			) VALUES ($1::uuid, $2, $3, $4, $5, $6)`,
			value.JobID, label.Name, string(label.Kind), label.Table, start, end,
		); err != nil {
			return fmt.Errorf("register property graph label %q: %w", label.Name, err)
		}
	}
	if owns {
		if err := tx.Commit(ctx); err != nil {
			return fmt.Errorf("commit property graph registration: %w", err)
		}
	}
	return nil
}

func (store *Store) GetPropertyGraph(
	ctx context.Context,
	jobID string,
) (PropertyGraphGeneration, error) {
	if err := validateJobID(jobID); err != nil {
		return PropertyGraphGeneration{}, err
	}
	var value PropertyGraphGeneration
	var encodedLabels []byte
	query := `
		SELECT generation.job_id::text, generation.target_schema,
		       generation.graph_name, generation.definition_fingerprint::text,
		       generation.state,
		       COALESCE(generation.digest_root::text, ''),
		       COALESCE(generation.digest_rows, 0),
		       COALESCE(generation.digest_range_count, 0),
		       COALESCE(
		         jsonb_agg(jsonb_build_object(
		           'name', label.label_name,
		           'kind', label.kind,
		           'table', label.table_name,
		           'startLabel', COALESCE(label.start_label, ''),
		           'endLabel', COALESCE(label.end_label, '')
		         ) ORDER BY label.kind DESC, label.label_name)
		           FILTER (WHERE label.job_id IS NOT NULL),
		         '[]'::jsonb
		       )
		FROM agefreighter_meta.property_graph_generation generation
		LEFT JOIN agefreighter_meta.property_graph_label label
		  ON label.job_id = generation.job_id
		WHERE generation.job_id = $1::uuid
		GROUP BY generation.job_id, generation.target_schema,
		         generation.graph_name, generation.definition_fingerprint,
		         generation.state`
	err := store.database.QueryRow(ctx, query, jobID).Scan(
		&value.JobID, &value.Schema, &value.Graph,
		&value.DefinitionFingerprint, &value.State, &value.DigestRoot,
		&value.DigestRows, &value.DigestRangeCount, &encodedLabels,
	)
	var pgError *pgconn.PgError
	if errors.As(err, &pgError) && pgError.Code == "42703" {
		legacyQuery := strings.Replace(query,
			"COALESCE(generation.digest_root::text, ''),\n\t\t       COALESCE(generation.digest_rows, 0),\n\t\t       COALESCE(generation.digest_range_count, 0)",
			"''::text, 0::bigint, 0::integer", 1)
		err = store.database.QueryRow(ctx, legacyQuery, jobID).Scan(
			&value.JobID, &value.Schema, &value.Graph,
			&value.DefinitionFingerprint, &value.State, &value.DigestRoot,
			&value.DigestRows, &value.DigestRangeCount, &encodedLabels,
		)
	}
	if errors.Is(err, pgx.ErrNoRows) {
		return PropertyGraphGeneration{}, fmt.Errorf(
			"%w: property graph for job %q", ErrNotFound, jobID,
		)
	}
	if err != nil {
		return PropertyGraphGeneration{}, fmt.Errorf("read property graph generation: %w", err)
	}
	var labels []struct {
		Name       string `json:"name"`
		Kind       string `json:"kind"`
		Table      string `json:"table"`
		StartLabel string `json:"startLabel"`
		EndLabel   string `json:"endLabel"`
	}
	if err := json.Unmarshal(encodedLabels, &labels); err != nil {
		return PropertyGraphGeneration{}, fmt.Errorf("decode property graph labels: %w", err)
	}
	for _, encoded := range labels {
		var kind LabelKind
		if err := kind.Scan(encoded.Kind); err != nil {
			return PropertyGraphGeneration{}, err
		}
		value.Labels = append(value.Labels, PropertyGraphLabel{
			Name: encoded.Name, Kind: kind, Table: encoded.Table,
			StartLabel: encoded.StartLabel, EndLabel: encoded.EndLabel,
		})
	}
	return value, nil
}

func (store *Store) ReplacePropertyGraphDigests(
	ctx context.Context,
	jobID string,
	ranges []PropertyGraphDigestRange,
	root string,
	rows int64,
	rangeCount int,
) error {
	if err := validateJobID(jobID); err != nil {
		return err
	}
	if err := validateFingerprint(root); err != nil {
		return fmt.Errorf("property graph digest root: %w", err)
	}
	if rows < 0 || rangeCount <= 0 {
		return errors.New("property graph digest rows and range count are invalid")
	}
	ordered := slices.Clone(ranges)
	slices.SortFunc(ordered, comparePropertyGraphDigestRanges)
	for index, value := range ordered {
		if value.JobID != jobID || !validTargetIdentifier(value.LabelName) ||
			(value.Kind != VertexLabel && value.Kind != EdgeLabel) || value.Rows <= 0 {
			return fmt.Errorf("property graph digest range %d is invalid", index)
		}
		if err := validateFingerprint(value.Digest); err != nil {
			return fmt.Errorf("property graph digest range %d: %w", index, err)
		}
		if index > 0 && comparePropertyGraphDigestRanges(ordered[index-1], value) == 0 {
			return fmt.Errorf("duplicate property graph digest range for %q/%d",
				value.LabelName, value.RangeID)
		}
	}
	tx, existing := store.database.(pgx.Tx)
	owns := !existing
	var err error
	if owns {
		tx, err = store.database.Begin(ctx)
		if err != nil {
			return fmt.Errorf("begin property graph digest replacement: %w", err)
		}
		defer rollback(ctx, tx)
	}
	if _, err := tx.Exec(ctx,
		`DELETE FROM agefreighter_meta.property_graph_digest_range
		 WHERE job_id = $1::uuid`, jobID); err != nil {
		return fmt.Errorf("clear property graph digest ranges: %w", err)
	}
	for _, value := range ordered {
		if _, err := tx.Exec(ctx, `
			INSERT INTO agefreighter_meta.property_graph_digest_range (
				job_id, label_name, kind, range_id, row_count, digest
			) VALUES ($1::uuid, $2, $3, $4, $5, $6)`,
			jobID, value.LabelName, string(value.Kind), int(value.RangeID),
			value.Rows, value.Digest,
		); err != nil {
			return fmt.Errorf("store property graph digest range: %w", err)
		}
	}
	tag, err := tx.Exec(ctx, `
		UPDATE agefreighter_meta.property_graph_generation
		SET digest_root = $2, digest_rows = $3, digest_range_count = $4,
		    updated_at = clock_timestamp()
		WHERE job_id = $1::uuid`, jobID, root, rows, rangeCount)
	if err != nil {
		return fmt.Errorf("store property graph digest root: %w", err)
	}
	if err := rowsAffectedOne(tag, "store property graph digest root"); err != nil {
		return err
	}
	if owns {
		if err := tx.Commit(ctx); err != nil {
			return fmt.Errorf("commit property graph digest replacement: %w", err)
		}
	}
	return nil
}

func (store *Store) ListPropertyGraphDigests(
	ctx context.Context,
	jobID string,
) ([]PropertyGraphDigestRange, error) {
	if err := validateJobID(jobID); err != nil {
		return nil, err
	}
	var encoded []byte
	err := store.database.QueryRow(ctx, `
		SELECT COALESCE(jsonb_agg(jsonb_build_object(
			'labelName', label_name, 'kind', kind, 'rangeId', range_id,
			'rows', row_count, 'digest', digest::text
		) ORDER BY kind DESC, label_name, range_id), '[]'::jsonb)
		FROM agefreighter_meta.property_graph_digest_range
		WHERE job_id = $1::uuid`, jobID).Scan(&encoded)
	if err != nil {
		return nil, fmt.Errorf("read property graph digest ranges: %w", err)
	}
	var values []struct {
		LabelName string `json:"labelName"`
		Kind      string `json:"kind"`
		RangeID   int    `json:"rangeId"`
		Rows      int64  `json:"rows"`
		Digest    string `json:"digest"`
	}
	if err := json.Unmarshal(encoded, &values); err != nil {
		return nil, fmt.Errorf("decode property graph digest ranges: %w", err)
	}
	result := make([]PropertyGraphDigestRange, len(values))
	for index, value := range values {
		var kind LabelKind
		if err := kind.Scan(value.Kind); err != nil {
			return nil, err
		}
		if value.RangeID < 0 || value.RangeID > 255 {
			return nil, errors.New("stored property graph digest range is invalid")
		}
		result[index] = PropertyGraphDigestRange{
			JobID: jobID, LabelName: value.LabelName, Kind: kind,
			RangeID: uint8(value.RangeID), Rows: value.Rows, Digest: value.Digest,
		}
	}
	return result, nil
}

func comparePropertyGraphDigestRanges(left, right PropertyGraphDigestRange) int {
	if left.Kind != right.Kind {
		return int(right.Kind) - int(left.Kind)
	}
	if compared := strings.Compare(left.LabelName, right.LabelName); compared != 0 {
		return compared
	}
	return int(left.RangeID) - int(right.RangeID)
}

func (store *Store) ActivatePropertyGraph(ctx context.Context, jobID string) error {
	if err := validateJobID(jobID); err != nil {
		return err
	}
	tag, err := store.database.Exec(ctx, `
		UPDATE agefreighter_meta.property_graph_generation
		SET state = 'active', updated_at = clock_timestamp()
		WHERE job_id = $1::uuid AND state = 'loading'`, jobID)
	if err != nil {
		return fmt.Errorf("activate property graph: %w", err)
	}
	return rowsAffectedOne(tag, "activate property graph")
}

func validatePropertyGraph(value PropertyGraphGeneration) error {
	if err := validateJobID(value.JobID); err != nil {
		return err
	}
	if !validTargetIdentifier(value.Schema) || !validTargetIdentifier(value.Graph) {
		return errors.New("property graph schema and graph must be valid identifiers")
	}
	if err := validateFingerprint(value.DefinitionFingerprint); err != nil {
		return fmt.Errorf("property graph definition fingerprint: %w", err)
	}
	if value.State != PropertyGraphLoading && value.State != PropertyGraphActive {
		return fmt.Errorf("unsupported property graph state %q", value.State)
	}
	if len(value.Labels) == 0 {
		return errors.New("property graph requires labels")
	}
	labels := slices.Clone(value.Labels)
	slices.SortFunc(labels, func(left, right PropertyGraphLabel) int {
		return strings.Compare(left.Name, right.Name)
	})
	seenTables := make(map[string]struct{}, len(labels))
	vertexLabels := make(map[string]struct{}, len(labels))
	for index, label := range labels {
		if !validTargetIdentifier(label.Name) || !validTargetIdentifier(label.Table) {
			return fmt.Errorf("property graph label %d has invalid name or table", index)
		}
		if index > 0 && labels[index-1].Name == label.Name {
			return fmt.Errorf("duplicate property graph label %q", label.Name)
		}
		if _, exists := seenTables[label.Table]; exists {
			return fmt.Errorf("duplicate property graph table %q", label.Table)
		}
		seenTables[label.Table] = struct{}{}
		switch label.Kind {
		case VertexLabel:
			if label.StartLabel != "" || label.EndLabel != "" {
				return fmt.Errorf("vertex label %q has edge endpoints", label.Name)
			}
			vertexLabels[label.Name] = struct{}{}
		case EdgeLabel:
			if label.StartLabel == "" || label.EndLabel == "" {
				return fmt.Errorf("edge label %q requires endpoints", label.Name)
			}
		default:
			return fmt.Errorf("unsupported property graph label kind %q", label.Kind)
		}
	}
	for _, label := range labels {
		if label.Kind != EdgeLabel {
			continue
		}
		if _, ok := vertexLabels[label.StartLabel]; !ok {
			return fmt.Errorf("edge label %q has unknown start label %q", label.Name, label.StartLabel)
		}
		if _, ok := vertexLabels[label.EndLabel]; !ok {
			return fmt.Errorf("edge label %q has unknown end label %q", label.Name, label.EndLabel)
		}
	}
	return nil
}
