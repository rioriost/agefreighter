package cosmos

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/rioriost/agefreighter/internal/config"
)

const maxGremlinMappings = 1_024

const gremlinVertexLabelsQuery = `SELECT VALUE c.label
FROM c
WHERE NOT IS_DEFINED(c._isEdge) AND IS_STRING(c.label)`

const gremlinEdgeMappingsQuery = `SELECT
	c.label AS label,
	c._vertexLabel AS startLabel,
	c._sinkLabel AS endLabel
FROM c
WHERE c._isEdge = true
	AND IS_STRING(c.label)
	AND IS_STRING(c._vertexLabel)
	AND IS_STRING(c._sinkLabel)`

type gremlinEdgeMapping struct {
	label string
	start string
	end   string
}

// InterpretGremlinDocuments discovers the bounded label catalog of a Cosmos
// DB for Apache Gremlin container and converts it to deterministic mappings.
func InterpretGremlinDocuments(
	ctx context.Context,
	source config.CosmosSource,
	client QueryClient,
) (config.CosmosSource, error) {
	if ctx == nil {
		return config.CosmosSource{}, errors.New(
			"Cosmos Gremlin interpretation context is required",
		)
	}
	if client == nil {
		return config.CosmosSource{}, errors.New(
			"Cosmos Gremlin interpretation client is required",
		)
	}
	if source.Gremlin == nil || !source.Gremlin.Enabled {
		return config.CosmosSource{}, errors.New(
			"Cosmos Gremlin interpretation configuration is required",
		)
	}
	options := *source.Gremlin
	labels, err := discoverGremlinLabels(ctx, client, source, options)
	if err != nil {
		return config.CosmosSource{}, err
	}
	if len(labels) == 0 {
		return config.CosmosSource{}, errors.New(
			"Cosmos Gremlin interpretation found no matching vertices",
		)
	}
	edges, err := discoverGremlinEdges(ctx, client, source, options, labels)
	if err != nil {
		return config.CosmosSource{}, err
	}

	source.Gremlin = nil
	source.Vertices = make([]config.CosmosVertexQuery, len(labels))
	for index, label := range labels {
		mapping, err := gremlinVertexQuery(options, label)
		if err != nil {
			return config.CosmosSource{}, err
		}
		source.Vertices[index] = mapping
	}
	source.Edges = make([]config.CosmosEdgeQuery, len(edges))
	for index, edge := range edges {
		mapping, err := gremlinEdgeQuery(options, edge)
		if err != nil {
			return config.CosmosSource{}, err
		}
		source.Edges[index] = mapping
	}
	return source, nil
}

func discoverGremlinLabels(
	ctx context.Context,
	client QueryClient,
	source config.CosmosSource,
	options config.CosmosGremlin,
) ([]string, error) {
	query := gremlinVertexLabelsQuery
	parameters := []Parameter(nil)
	if options.LabelPrefix != "" {
		query += " AND STARTSWITH(c.label, @labelPrefix)"
		parameters = []Parameter{{
			Name: "@labelPrefix", Value: options.LabelPrefix,
		}}
	}
	values := make(map[string]struct{})
	err := visitGremlinDiscovery(
		ctx,
		client,
		options.Container,
		query,
		parameters,
		source.PageSize,
		options.MaxDiscoveryDocuments,
		func(raw []byte) error {
			value, err := decodeDocument(raw)
			if err != nil {
				return err
			}
			label, ok := value.(string)
			if !ok || !validGremlinName(label) {
				return errors.New(
					"Cosmos Gremlin discovery returned an invalid vertex label",
				)
			}
			values[label] = struct{}{}
			if len(values) > options.MaxLabels {
				return fmt.Errorf(
					"Cosmos Gremlin discovery found more than %d vertex labels",
					options.MaxLabels,
				)
			}
			return nil
		},
	)
	if err != nil {
		return nil, fmt.Errorf("discover Cosmos Gremlin vertex labels: %w", err)
	}
	labels := make([]string, 0, len(values))
	for label := range values {
		labels = append(labels, label)
	}
	slices.Sort(labels)
	return labels, nil
}

func discoverGremlinEdges(
	ctx context.Context,
	client QueryClient,
	source config.CosmosSource,
	options config.CosmosGremlin,
	labels []string,
) ([]gremlinEdgeMapping, error) {
	query := gremlinEdgeMappingsQuery
	parameters := make([]Parameter, 0, 2)
	if options.RelationshipTypePrefix != "" {
		query += " AND STARTSWITH(c.label, @relationshipTypePrefix)"
		parameters = append(parameters, Parameter{
			Name:  "@relationshipTypePrefix",
			Value: options.RelationshipTypePrefix,
		})
	}
	if options.LabelPrefix != "" {
		query += " AND STARTSWITH(c._vertexLabel, @labelPrefix)" +
			" AND STARTSWITH(c._sinkLabel, @labelPrefix)"
		parameters = append(parameters, Parameter{
			Name: "@labelPrefix", Value: options.LabelPrefix,
		})
	}
	selectedLabels := make(map[string]struct{}, len(labels))
	for _, label := range labels {
		selectedLabels[label] = struct{}{}
	}
	mappings := make(map[gremlinEdgeMapping]struct{})
	relationshipTypes := make(map[string]struct{})
	err := visitGremlinDiscovery(
		ctx,
		client,
		options.Container,
		query,
		parameters,
		source.PageSize,
		options.MaxDiscoveryDocuments,
		func(raw []byte) error {
			value, err := decodeDocument(raw)
			if err != nil {
				return err
			}
			document, ok := value.(map[string]any)
			if !ok {
				return errors.New(
					"Cosmos Gremlin discovery returned an invalid edge mapping",
				)
			}
			mapping, err := decodeGremlinEdgeMapping(document)
			if err != nil {
				return err
			}
			if _, ok := selectedLabels[mapping.start]; !ok {
				return nil
			}
			if _, ok := selectedLabels[mapping.end]; !ok {
				return nil
			}
			relationshipTypes[mapping.label] = struct{}{}
			if len(relationshipTypes) > options.MaxLabels {
				return fmt.Errorf(
					"Cosmos Gremlin discovery found more than %d relationship types",
					options.MaxLabels,
				)
			}
			mappings[mapping] = struct{}{}
			if len(labels)+len(mappings) > maxGremlinMappings {
				return fmt.Errorf(
					"Cosmos Gremlin discovery exceeds %d generated mappings",
					maxGremlinMappings,
				)
			}
			return nil
		},
	)
	if err != nil {
		return nil, fmt.Errorf("discover Cosmos Gremlin edges: %w", err)
	}
	result := make([]gremlinEdgeMapping, 0, len(mappings))
	for mapping := range mappings {
		result = append(result, mapping)
	}
	slices.SortFunc(result, func(left, right gremlinEdgeMapping) int {
		if compared := strings.Compare(left.label, right.label); compared != 0 {
			return compared
		}
		if compared := strings.Compare(left.start, right.start); compared != 0 {
			return compared
		}
		return strings.Compare(left.end, right.end)
	})
	return result, nil
}

func visitGremlinDiscovery(
	ctx context.Context,
	client QueryClient,
	container string,
	query string,
	parameters []Parameter,
	pageSize int,
	maxDocuments int,
	visit func([]byte) error,
) error {
	hasContinuation := false
	continuation := ""
	documents := 0
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		pager, err := client.NewQueryPager(
			container,
			query,
			parameters,
			QueryOptions{
				PageSizeHint:             int32(pageSize),
				ContinuationToken:        continuation,
				HasContinuationToken:     hasContinuation,
				ContinuationTokenLimitKB: defaultContinuationTokenLimitKB,
			},
		)
		if err != nil {
			return fmt.Errorf("open Cosmos Gremlin discovery query: %w", err)
		}
		page, err := pager.NextPage(ctx)
		if err != nil {
			return fmt.Errorf("fetch Cosmos Gremlin discovery page: %w", err)
		}
		for _, item := range page.Items {
			documents++
			if documents > maxDocuments {
				return fmt.Errorf(
					"Cosmos Gremlin discovery scanned more than %d documents",
					maxDocuments,
				)
			}
			if err := visit(item); err != nil {
				return err
			}
		}
		if !page.HasContinuation {
			return nil
		}
		if hasContinuation && page.ContinuationToken == continuation {
			return errors.New(
				"Cosmos Gremlin discovery returned a repeated continuation token",
			)
		}
		hasContinuation = true
		continuation = page.ContinuationToken
	}
}

func decodeGremlinEdgeMapping(
	document map[string]any,
) (gremlinEdgeMapping, error) {
	mapping := gremlinEdgeMapping{}
	for field, target := range map[string]*string{
		"label":      &mapping.label,
		"startLabel": &mapping.start,
		"endLabel":   &mapping.end,
	} {
		value, ok := document[field].(string)
		if !ok || !validGremlinName(value) {
			return gremlinEdgeMapping{}, fmt.Errorf(
				"Cosmos Gremlin discovery returned an invalid %s",
				field,
			)
		}
		*target = value
	}
	return mapping, nil
}

func gremlinVertexQuery(
	options config.CosmosGremlin,
	label string,
) (config.CosmosVertexQuery, error) {
	labelParameter, err := gremlinParameter("@label", label)
	if err != nil {
		return config.CosmosVertexQuery{}, err
	}
	return config.CosmosVertexQuery{
		Container: options.Container,
		Label:     label,
		Query: gremlinDataQuery(
			"NOT IS_DEFINED(c._isEdge) AND c.label = @label",
		),
		Parameters:           []config.CosmosQueryParameter{labelParameter},
		IDField:              "/id",
		DocumentFormat:       config.CosmosDocumentGremlin,
		PartitionKeyProperty: options.PartitionKeyProperty,
		MaxProperties:        options.MaxProperties,
	}, nil
}

func gremlinEdgeQuery(
	options config.CosmosGremlin,
	mapping gremlinEdgeMapping,
) (config.CosmosEdgeQuery, error) {
	labelParameter, err := gremlinParameter("@label", mapping.label)
	if err != nil {
		return config.CosmosEdgeQuery{}, err
	}
	startParameter, err := gremlinParameter("@startLabel", mapping.start)
	if err != nil {
		return config.CosmosEdgeQuery{}, err
	}
	endParameter, err := gremlinParameter("@endLabel", mapping.end)
	if err != nil {
		return config.CosmosEdgeQuery{}, err
	}
	return config.CosmosEdgeQuery{
		Container: options.Container,
		Label:     mapping.label,
		Query: gremlinDataQuery(
			"c._isEdge = true AND c.label = @label" +
				" AND c._vertexLabel = @startLabel" +
				" AND c._sinkLabel = @endLabel",
		),
		Parameters: []config.CosmosQueryParameter{
			labelParameter,
			startParameter,
			endParameter,
		},
		ExternalIDField: "/id",
		Start: config.EndpointMapping{
			Label: mapping.start,
			Field: "/_vertexId",
		},
		End: config.EndpointMapping{
			Label: mapping.end,
			Field: "/_sink",
		},
		DocumentFormat:       config.CosmosDocumentGremlin,
		PartitionKeyProperty: options.PartitionKeyProperty,
		MaxProperties:        options.MaxProperties,
	}, nil
}

func gremlinDataQuery(predicate string) string {
	return "SELECT * FROM c WHERE " + predicate
}

func gremlinParameter(
	name string,
	value string,
) (config.CosmosQueryParameter, error) {
	parameter, err := config.NewCosmosParamValue(value)
	if err != nil {
		return config.CosmosQueryParameter{}, fmt.Errorf(
			"encode Cosmos Gremlin discovery parameter: %w",
			err,
		)
	}
	return config.CosmosQueryParameter{Name: name, Value: parameter}, nil
}

func validGremlinName(value string) bool {
	if value == "" || len(value) > 256 || !utf8.ValidString(value) {
		return false
	}
	return !strings.ContainsFunc(value, unicode.IsControl)
}
