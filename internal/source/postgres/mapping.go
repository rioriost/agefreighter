package postgres

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/sqlquery"
	"github.com/rioriost/agefreighter/pkg/model"
)

type mappingKind uint8

const (
	vertexMapping mappingKind = iota + 1
	edgeMapping
)

func (kind mappingKind) String() string {
	switch kind {
	case vertexMapping:
		return "vertex"
	case edgeMapping:
		return "edge"
	default:
		return "unknown"
	}
}

type compiledProperty struct {
	name  string
	field string
}

type compiledMapping struct {
	kind      mappingKind
	kindIndex int
	label     model.Label
	namespace model.Namespace
	query     string
	keyField  string

	idField string

	externalIDField string
	start           config.EndpointMapping
	end             config.EndpointMapping
	properties      []compiledProperty
}

func buildMappings(
	ctx context.Context,
	namespace string,
	source config.PostgreSQLSource,
	maxProperties int,
) ([]compiledMapping, error) {
	if strings.TrimSpace(namespace) == "" {
		return nil, errors.New("PostgreSQL namespace is required")
	}
	mappings := make([]compiledMapping, 0, len(source.Vertices)+len(source.Edges))
	for index, vertex := range source.Vertices {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		resource := fmt.Sprintf("PostgreSQL vertex mapping %d", index)
		if strings.TrimSpace(vertex.Label) == "" {
			return nil, fmt.Errorf("%s label is required", resource)
		}
		if strings.TrimSpace(vertex.IDField) == "" {
			return nil, fmt.Errorf("%s idField is required", resource)
		}
		if err := validateQuery(vertex.Query, vertex.KeyField, source.ReadMode, resource); err != nil {
			return nil, err
		}
		properties, err := compileProperties(vertex.Properties, maxProperties)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", resource, err)
		}
		mappings = append(mappings, compiledMapping{
			kind:       vertexMapping,
			kindIndex:  index,
			label:      model.Label(vertex.Label),
			namespace:  model.Namespace(namespace),
			query:      strings.TrimSpace(vertex.Query),
			keyField:   vertex.KeyField,
			idField:    vertex.IDField,
			properties: properties,
		})
	}
	for index, edge := range source.Edges {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		resource := fmt.Sprintf("PostgreSQL edge mapping %d", index)
		if strings.TrimSpace(edge.Label) == "" {
			return nil, fmt.Errorf("%s label is required", resource)
		}
		if err := validateEndpoint(edge.Start, namespace, resource+" start"); err != nil {
			return nil, err
		}
		if err := validateEndpoint(edge.End, namespace, resource+" end"); err != nil {
			return nil, err
		}
		if err := validateQuery(edge.Query, edge.KeyField, source.ReadMode, resource); err != nil {
			return nil, err
		}
		properties, err := compileProperties(edge.Properties, maxProperties)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", resource, err)
		}
		mappings = append(mappings, compiledMapping{
			kind:            edgeMapping,
			kindIndex:       index,
			label:           model.Label(edge.Label),
			namespace:       model.Namespace(namespace),
			query:           strings.TrimSpace(edge.Query),
			keyField:        edge.KeyField,
			externalIDField: edge.ExternalIDField,
			start:           edge.Start,
			end:             edge.End,
			properties:      properties,
		})
	}
	if len(mappings) == 0 {
		return nil, errors.New("PostgreSQL source has no mappings")
	}
	return mappings, nil
}

func validateQuery(
	query, keyField string,
	mode config.PostgreSQLReadMode,
	resource string,
) error {
	trimmed := strings.TrimSpace(query)
	if trimmed == "" {
		return fmt.Errorf("%s query is required", resource)
	}
	if strings.Contains(trimmed, ";") {
		return fmt.Errorf("%s query must be one statement without a semicolon", resource)
	}
	if !sqlquery.HasTopLevelOrderBy(trimmed) {
		return fmt.Errorf("%s query must contain ORDER BY for deterministic resume", resource)
	}
	fields := strings.Fields(trimmed)
	if len(fields) == 0 ||
		(!strings.EqualFold(fields[0], "select") && !strings.EqualFold(fields[0], "with")) {
		return fmt.Errorf("%s query must be SELECT or WITH", resource)
	}
	if mode == config.PostgreSQLReadKeyset {
		if strings.TrimSpace(keyField) == "" {
			return fmt.Errorf("%s keyField is required in keyset mode", resource)
		}
		if !strings.Contains(query, "$1") || !strings.Contains(query, "$2") {
			return fmt.Errorf("%s query must use $1 and $2 in keyset mode", resource)
		}
	} else if keyField != "" {
		return fmt.Errorf("%s keyField is only valid in keyset mode", resource)
	}
	return nil
}

func validateEndpoint(endpoint config.EndpointMapping, namespace, resource string) error {
	if strings.TrimSpace(endpoint.Label) == "" {
		return fmt.Errorf("%s label is required", resource)
	}
	if strings.TrimSpace(endpoint.Field) == "" {
		return fmt.Errorf("%s field is required", resource)
	}
	if endpoint.Namespace == "" && namespace == "" {
		return fmt.Errorf("%s namespace is required", resource)
	}
	return nil
}

func compileProperties(
	properties map[string]string,
	maxProperties int,
) ([]compiledProperty, error) {
	if len(properties) > maxProperties {
		return nil, fmt.Errorf(
			"has %d properties, maximum is %d", len(properties), maxProperties,
		)
	}
	names := make([]string, 0, len(properties))
	for name := range properties {
		names = append(names, name)
	}
	sort.Strings(names)
	compiled := make([]compiledProperty, 0, len(names))
	for _, name := range names {
		if name == "" {
			return nil, errors.New("property name must not be empty")
		}
		field := properties[name]
		if field == "" {
			return nil, fmt.Errorf("property %q source field is required", name)
		}
		compiled = append(compiled, compiledProperty{name: name, field: field})
	}
	return compiled, nil
}

func (mapping compiledMapping) resource() string {
	return fmt.Sprintf("%s[%d]:%s", mapping.kind.String(), mapping.kindIndex, mapping.label)
}
