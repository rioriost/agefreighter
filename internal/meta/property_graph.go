package meta

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/jackc/pgx/v5"
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
	Labels                []PropertyGraphLabel
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
	err := store.database.QueryRow(ctx, `
		SELECT generation.job_id::text, generation.target_schema,
		       generation.graph_name, generation.definition_fingerprint::text,
		       generation.state,
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
		         generation.state`, jobID,
	).Scan(
		&value.JobID, &value.Schema, &value.Graph,
		&value.DefinitionFingerprint, &value.State, &encodedLabels,
	)
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
