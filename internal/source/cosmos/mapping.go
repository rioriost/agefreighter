package cosmos

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

// mappingKind distinguishes vertex mappings from edge mappings while
// preserving their configured order (vertices before edges).
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

// compiledProperty binds a property name to its parsed JSON Pointer.
type compiledProperty struct {
	name    string
	pointer pointer
}

// compiledMapping is a config.CosmosVertexQuery/CosmosEdgeQuery compiled
// into a form the Iterator can execute directly: pointers are pre-parsed
// and query parameters are pre-converted to their native Go values.
type compiledMapping struct {
	kind       mappingKind
	container  string
	label      model.Label
	namespace  model.Namespace
	query      string
	parameters []Parameter

	idField pointer // vertex only

	hasExternalID   bool
	externalIDField pointer // edge only, when hasExternalID

	start      config.EndpointMapping // edge only
	startField pointer                // edge only
	end        config.EndpointMapping // edge only
	endField   pointer                // edge only

	properties []compiledProperty
}

// buildMappings compiles a Cosmos source's vertex and edge queries, in
// configured order (all vertices before all edges), defensively
// re-validating every JSON Pointer even though config.Validate already
// checked their syntax, since an Iterator can be constructed directly.
func buildMappings(
	ctx context.Context,
	namespace string,
	source config.CosmosSource,
	maxProperties int,
) ([]compiledMapping, error) {
	mappings := make([]compiledMapping, 0, len(source.Vertices)+len(source.Edges))
	for _, vertex := range source.Vertices {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(vertex.Properties) > maxProperties {
			return nil, fmt.Errorf(
				"Cosmos vertex mapping %q has %d properties, maximum is %d",
				vertex.Label, len(vertex.Properties), maxProperties,
			)
		}
		idField, err := parsePointer(vertex.IDField)
		if err != nil {
			return nil, fmt.Errorf("Cosmos vertex mapping %q idField: %w", vertex.Label, err)
		}
		properties, err := compileProperties(vertex.Properties)
		if err != nil {
			return nil, fmt.Errorf("Cosmos vertex mapping %q: %w", vertex.Label, err)
		}
		parameters, err := compileParameters(vertex.Label, vertex.Parameters)
		if err != nil {
			return nil, err
		}
		mappings = append(mappings, compiledMapping{
			kind:       vertexMapping,
			container:  vertex.Container,
			label:      model.Label(vertex.Label),
			namespace:  model.Namespace(namespace),
			query:      vertex.Query,
			parameters: parameters,
			idField:    idField,
			properties: properties,
		})
	}
	for _, edge := range source.Edges {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(edge.Properties) > maxProperties {
			return nil, fmt.Errorf(
				"Cosmos edge mapping %q has %d properties, maximum is %d",
				edge.Label, len(edge.Properties), maxProperties,
			)
		}
		var (
			externalIDField pointer
			hasExternalID   bool
		)
		if edge.ExternalIDField != "" {
			hasExternalID = true
			var err error
			externalIDField, err = parsePointer(edge.ExternalIDField)
			if err != nil {
				return nil, fmt.Errorf("Cosmos edge mapping %q externalIdField: %w", edge.Label, err)
			}
		}
		startField, err := parsePointer(edge.Start.Field)
		if err != nil {
			return nil, fmt.Errorf("Cosmos edge mapping %q start field: %w", edge.Label, err)
		}
		endField, err := parsePointer(edge.End.Field)
		if err != nil {
			return nil, fmt.Errorf("Cosmos edge mapping %q end field: %w", edge.Label, err)
		}
		properties, err := compileProperties(edge.Properties)
		if err != nil {
			return nil, fmt.Errorf("Cosmos edge mapping %q: %w", edge.Label, err)
		}
		parameters, err := compileParameters(edge.Label, edge.Parameters)
		if err != nil {
			return nil, err
		}
		mappings = append(mappings, compiledMapping{
			kind:            edgeMapping,
			container:       edge.Container,
			label:           model.Label(edge.Label),
			namespace:       model.Namespace(namespace),
			query:           edge.Query,
			parameters:      parameters,
			hasExternalID:   hasExternalID,
			externalIDField: externalIDField,
			start:           edge.Start,
			startField:      startField,
			end:             edge.End,
			endField:        endField,
			properties:      properties,
		})
	}
	if len(mappings) == 0 {
		return nil, errors.New("Cosmos source has no mappings")
	}
	return mappings, nil
}

func compileProperties(properties map[string]string) ([]compiledProperty, error) {
	names := make([]string, 0, len(properties))
	for name := range properties {
		names = append(names, name)
	}
	sort.Strings(names)
	compiled := make([]compiledProperty, 0, len(names))
	for _, name := range names {
		parsed, err := parsePointer(properties[name])
		if err != nil {
			return nil, fmt.Errorf("property %q: %w", name, err)
		}
		compiled = append(compiled, compiledProperty{name: name, pointer: parsed})
	}
	return compiled, nil
}

func compileParameters(label string, parameters []config.CosmosQueryParameter) ([]Parameter, error) {
	compiled := make([]Parameter, len(parameters))
	for index, parameter := range parameters {
		if parameter.Name == "" || parameter.Name[0] != '@' {
			return nil, fmt.Errorf(
				"Cosmos mapping %q parameter %q must start with @", label, parameter.Name,
			)
		}
		compiled[index] = Parameter{Name: parameter.Name, Value: parameter.Value.Native()}
	}
	return compiled, nil
}
