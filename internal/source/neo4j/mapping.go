package neo4j

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/sqlquery"
	"github.com/rioriost/agefreighter/pkg/model"
)

var queryFieldPattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

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
	kind         mappingKind
	kindIndex    int
	label        model.Label
	namespace    model.Namespace
	query        string
	initialQuery string
	keyField     string
	paged        bool

	idField         string
	externalIDField string
	start           config.EndpointMapping
	end             config.EndpointMapping
	properties      []compiledProperty
}

func buildMappings(
	ctx context.Context,
	namespace string,
	source config.Neo4jSource,
	maxProperties int,
) ([]compiledMapping, error) {
	if strings.TrimSpace(namespace) == "" {
		return nil, errors.New("Neo4j namespace is required")
	}
	mappings := make([]compiledMapping, 0, len(source.Vertices)+len(source.Edges))
	for index, vertex := range source.Vertices {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		resource := fmt.Sprintf("Neo4j vertex mapping %d", index)
		if strings.TrimSpace(vertex.Label) == "" {
			return nil, fmt.Errorf("%s label is required", resource)
		}
		if strings.TrimSpace(vertex.IDField) == "" {
			return nil, fmt.Errorf("%s idField is required", resource)
		}
		if err := validateQuery(vertex.Query, vertex.KeyField, resource); err != nil {
			return nil, err
		}
		if err := validateInitialQuery(vertex.InitialQuery, vertex.KeyField, resource); err != nil {
			return nil, err
		}
		properties, err := compileProperties(vertex.Properties, maxProperties)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", resource, err)
		}
		mappings = append(mappings, compiledMapping{
			kind: vertexMapping, kindIndex: index,
			label: model.Label(vertex.Label), namespace: model.Namespace(namespace),
			query: strings.TrimSpace(vertex.Query), keyField: vertex.KeyField,
			initialQuery: strings.TrimSpace(vertex.InitialQuery),
			paged:        usesKeysetPages(vertex.Query),
			idField:      vertex.IDField, properties: properties,
		})
	}
	for index, edge := range source.Edges {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		resource := fmt.Sprintf("Neo4j edge mapping %d", index)
		if strings.TrimSpace(edge.Label) == "" {
			return nil, fmt.Errorf("%s label is required", resource)
		}
		if err := validateEndpoint(edge.Start, namespace, resource+" start"); err != nil {
			return nil, err
		}
		if err := validateEndpoint(edge.End, namespace, resource+" end"); err != nil {
			return nil, err
		}
		if err := validateQuery(edge.Query, edge.KeyField, resource); err != nil {
			return nil, err
		}
		if err := validateInitialQuery(edge.InitialQuery, edge.KeyField, resource); err != nil {
			return nil, err
		}
		properties, err := compileProperties(edge.Properties, maxProperties)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", resource, err)
		}
		mappings = append(mappings, compiledMapping{
			kind: edgeMapping, kindIndex: index,
			label: model.Label(edge.Label), namespace: model.Namespace(namespace),
			query: strings.TrimSpace(edge.Query), keyField: edge.KeyField,
			initialQuery:    strings.TrimSpace(edge.InitialQuery),
			paged:           usesKeysetPages(edge.Query),
			externalIDField: edge.ExternalIDField, start: edge.Start, end: edge.End,
			properties: properties,
		})
	}
	if len(mappings) == 0 {
		return nil, errors.New("Neo4j source has no mappings")
	}
	return mappings, nil
}

func validateQuery(query, keyField, resource string) error {
	if strings.TrimSpace(query) == "" {
		return fmt.Errorf("%s query is required", resource)
	}
	if strings.TrimSpace(keyField) == "" {
		return fmt.Errorf("%s keyField is required", resource)
	}
	if !queryFieldPattern.MatchString(keyField) {
		return fmt.Errorf("%s keyField must be an unquoted Cypher result identifier", resource)
	}
	if !sqlquery.HasParameter(query, "afterKey") {
		return fmt.Errorf("%s query must use $afterKey", resource)
	}
	paged := usesKeysetPages(query)
	if (!paged && !sqlquery.HasFinalTopLevelOrderByField(query, keyField)) ||
		(paged && !sqlquery.HasTopLevelOrderByField(query, keyField)) {
		return fmt.Errorf("%s query must end with ascending ORDER BY keyField", resource)
	}
	if sqlquery.HasKeyword(query, "skip") {
		return fmt.Errorf("%s query must not use SKIP", resource)
	}
	if sqlquery.HasKeyword(query, "offset") {
		return fmt.Errorf("%s query must not use OFFSET", resource)
	}
	if sqlquery.HasKeyword(query, "limit") && !paged {
		return fmt.Errorf("%s query must not use LIMIT", resource)
	}
	if sqlquery.HasKeyword(query, "union") {
		return fmt.Errorf("%s query must not use UNION", resource)
	}
	if sqlquery.HasKeyword(query, "collect") {
		return fmt.Errorf("%s query must not use collect", resource)
	}
	if strings.Contains(query, ";") {
		return fmt.Errorf("%s query must be one statement without a semicolon", resource)
	}
	return nil
}

func usesKeysetPages(query string) bool {
	return sqlquery.HasFinalTopLevelLimitParameter(query, "pageRows")
}

func validateInitialQuery(query, keyField, resource string) error {
	if strings.TrimSpace(query) == "" {
		return nil
	}
	if !usesKeysetPages(query) ||
		!sqlquery.HasTopLevelOrderByField(query, keyField) {
		return fmt.Errorf(
			"%s initial query must end with ascending ORDER BY keyField LIMIT $pageRows",
			resource,
		)
	}
	for _, keyword := range []string{"skip", "offset", "union", "collect"} {
		if sqlquery.HasKeyword(query, keyword) {
			return fmt.Errorf("%s initial query must not use %s", resource, keyword)
		}
	}
	if strings.Contains(query, ";") {
		return fmt.Errorf("%s initial query must be one statement without a semicolon", resource)
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

func compileProperties(properties map[string]string, maximum int) ([]compiledProperty, error) {
	if len(properties) > maximum {
		return nil, fmt.Errorf("has %d properties, maximum is %d", len(properties), maximum)
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
		field := strings.TrimSpace(properties[name])
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
