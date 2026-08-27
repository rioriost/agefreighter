package neo4j

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"unicode"

	"github.com/rioriost/agefreighter/internal/config"
)

const (
	unlabeledTargetLabel = "NO_LABEL"
	maxDiscoveryMappings = 1_024

	discoverLabelsQuery = `MATCH (n)
UNWIND labels(n) AS label
RETURN DISTINCT label AS label
ORDER BY label`
	discoverUnlabeledQuery = `MATCH (n)
WHERE size(labels(n)) = 0
RETURN count(n) AS count`
	discoverRelationshipTypesQuery = `MATCH ()-[r]->()
RETURN DISTINCT type(r) AS relationshipType
ORDER BY relationshipType`
)

type discoveredLabel struct {
	source string
	target string
}

type endpointPair struct {
	start string
	end   string
}

func DiscoverMappings(
	ctx context.Context,
	source config.Neo4jSource,
	client Client,
) (config.Neo4jSource, error) {
	if ctx == nil {
		return config.Neo4jSource{}, errors.New(
			"Neo4j discovery context is required",
		)
	}
	if client == nil {
		return config.Neo4jSource{}, errors.New(
			"Neo4j discovery client is required",
		)
	}
	if source.Discovery == nil || !source.Discovery.Enabled {
		return config.Neo4jSource{}, errors.New(
			"Neo4j discovery configuration is required",
		)
	}
	options := *source.Discovery
	labels, err := discoverLabels(ctx, client, options)
	if err != nil {
		return config.Neo4jSource{}, err
	}
	if len(labels) == 0 {
		return config.Neo4jSource{}, errors.New(
			"Neo4j discovery found no matching vertices",
		)
	}
	vertices := make([]config.VertexQuery, 0, len(labels))
	for _, label := range labels {
		properties, err := discoverProperties(
			ctx,
			client,
			vertexPropertyQuery(label),
			options.MaxProperties,
		)
		if err != nil {
			return config.Neo4jSource{}, fmt.Errorf(
				"discover Neo4j vertex label %q: %w",
				label.target,
				err,
			)
		}
		if err := requireProperties(
			properties,
			options.VertexKeyProperty,
			options.VertexIDProperty,
		); err != nil {
			return config.Neo4jSource{}, fmt.Errorf(
				"Neo4j vertex label %q: %w",
				label.target,
				err,
			)
		}
		vertices = append(
			vertices,
			buildDiscoveredVertex(label, labels, properties, options),
		)
	}

	relationshipTypes, err := discoverStrings(
		ctx,
		client,
		discoverRelationshipTypesQuery,
		"relationshipType",
		options.RelationshipTypePrefix,
		options.MaxLabels,
	)
	if err != nil {
		return config.Neo4jSource{}, fmt.Errorf(
			"discover Neo4j relationship types: %w",
			err,
		)
	}
	edges := make([]config.EdgeQuery, 0, len(relationshipTypes))
	for _, relationshipType := range relationshipTypes {
		pairs, err := discoverEndpointPairs(
			ctx,
			client,
			relationshipType,
			labels,
		)
		if err != nil {
			return config.Neo4jSource{}, err
		}
		if len(pairs) == 0 {
			continue
		}
		properties, err := discoverProperties(
			ctx,
			client,
			relationshipPropertyQuery(relationshipType),
			options.MaxProperties,
		)
		if err != nil {
			return config.Neo4jSource{}, fmt.Errorf(
				"discover Neo4j relationship type %q: %w",
				relationshipType,
				err,
			)
		}
		if err := requireProperties(
			properties,
			options.EdgeKeyProperty,
			options.EdgeIDProperty,
		); err != nil {
			return config.Neo4jSource{}, fmt.Errorf(
				"Neo4j relationship type %q: %w",
				relationshipType,
				err,
			)
		}
		for _, pair := range pairs {
			edges = append(edges, buildDiscoveredEdge(
				relationshipType,
				pair,
				labels,
				properties,
				options,
			))
			if len(vertices)+len(edges) > maxDiscoveryMappings {
				return config.Neo4jSource{}, fmt.Errorf(
					"Neo4j discovery exceeds %d generated mappings",
					maxDiscoveryMappings,
				)
			}
		}
	}
	source.Discovery = nil
	source.Vertices = vertices
	source.Edges = edges
	return source, nil
}

func discoverLabels(
	ctx context.Context,
	client Client,
	options config.Neo4jDiscovery,
) ([]discoveredLabel, error) {
	names, err := discoverStrings(
		ctx,
		client,
		discoverLabelsQuery,
		"label",
		options.LabelPrefix,
		options.MaxLabels,
	)
	if err != nil {
		return nil, fmt.Errorf("discover Neo4j labels: %w", err)
	}
	labels := make([]discoveredLabel, 0, len(names)+1)
	for _, name := range names {
		labels = append(labels, discoveredLabel{source: name, target: name})
	}
	if options.LabelPrefix != "" {
		return labels, nil
	}
	count, err := discoverCount(ctx, client, discoverUnlabeledQuery)
	if err != nil {
		return nil, fmt.Errorf("discover unlabeled Neo4j vertices: %w", err)
	}
	if count > 0 {
		if slices.Contains(names, unlabeledTargetLabel) {
			return nil, fmt.Errorf(
				"Neo4j label %q conflicts with the unlabeled vertex mapping",
				unlabeledTargetLabel,
			)
		}
		labels = append(labels, discoveredLabel{target: unlabeledTargetLabel})
	}
	slices.SortFunc(labels, func(left, right discoveredLabel) int {
		return strings.Compare(left.target, right.target)
	})
	if len(labels) > options.MaxLabels {
		return nil, fmt.Errorf(
			"Neo4j discovery found %d labels, maximum is %d",
			len(labels),
			options.MaxLabels,
		)
	}
	return labels, nil
}

func discoverStrings(
	ctx context.Context,
	client Client,
	query string,
	field string,
	prefix string,
	maximum int,
) ([]string, error) {
	stream, err := client.Query(ctx, query, nil)
	if err != nil {
		return nil, err
	}
	var values []string
	for {
		record, nextErr := stream.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			return nil, errors.Join(nextErr, stream.Close(ctx))
		}
		raw, found := record.Get(field)
		value, ok := raw.(string)
		if !found || !ok || !validDiscoveredIdentifier(value) {
			return nil, errors.Join(
				fmt.Errorf("Neo4j discovery returned an invalid %s", field),
				stream.Close(ctx),
			)
		}
		if !strings.HasPrefix(value, prefix) {
			continue
		}
		values = append(values, value)
		if len(values) > maximum {
			return nil, errors.Join(
				fmt.Errorf(
					"Neo4j discovery found more than %d matching %s values",
					maximum,
					field,
				),
				stream.Close(ctx),
			)
		}
	}
	if err := stream.Close(ctx); err != nil {
		return nil, err
	}
	slices.Sort(values)
	values = slices.Compact(values)
	return values, nil
}

func discoverCount(
	ctx context.Context,
	client Client,
	query string,
) (int64, error) {
	stream, err := client.Query(ctx, query, nil)
	if err != nil {
		return 0, err
	}
	record, err := stream.Next(ctx)
	if err != nil {
		return 0, errors.Join(err, stream.Close(ctx))
	}
	raw, found := record.Get("count")
	count, ok := raw.(int64)
	if !found || !ok || count < 0 {
		return 0, errors.Join(
			errors.New("Neo4j discovery returned an invalid count"),
			stream.Close(ctx),
		)
	}
	if _, err := stream.Next(ctx); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errors.New("Neo4j discovery count returned multiple rows")
		}
		return 0, errors.Join(err, stream.Close(ctx))
	}
	return count, stream.Close(ctx)
}

func discoverProperties(
	ctx context.Context,
	client Client,
	query string,
	maxProperties int,
) ([]string, error) {
	return discoverStrings(
		ctx,
		client,
		query,
		"property",
		"",
		maxProperties,
	)
}

func discoverEndpointPairs(
	ctx context.Context,
	client Client,
	relationshipType string,
	labels []discoveredLabel,
) ([]endpointPair, error) {
	stream, err := client.Query(
		ctx,
		relationshipPairQuery(relationshipType),
		nil,
	)
	if err != nil {
		return nil, err
	}
	pairs := make(map[endpointPair]struct{})
	for {
		record, nextErr := stream.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			return nil, errors.Join(nextErr, stream.Close(ctx))
		}
		start, selected, err := endpointPrimaryLabel(
			record,
			"startLabels",
			labels,
		)
		if err != nil {
			return nil, errors.Join(err, stream.Close(ctx))
		}
		if !selected {
			continue
		}
		end, selected, err := endpointPrimaryLabel(record, "endLabels", labels)
		if err != nil {
			return nil, errors.Join(err, stream.Close(ctx))
		}
		if !selected {
			continue
		}
		pairs[endpointPair{start: start, end: end}] = struct{}{}
		if len(pairs) > maxDiscoveryMappings {
			return nil, errors.Join(
				fmt.Errorf(
					"Neo4j relationship type %q has too many endpoint label pairs",
					relationshipType,
				),
				stream.Close(ctx),
			)
		}
	}
	if err := stream.Close(ctx); err != nil {
		return nil, err
	}
	result := make([]endpointPair, 0, len(pairs))
	for pair := range pairs {
		result = append(result, pair)
	}
	slices.SortFunc(result, func(left, right endpointPair) int {
		if compared := strings.Compare(left.start, right.start); compared != 0 {
			return compared
		}
		return strings.Compare(left.end, right.end)
	})
	return result, nil
}

func endpointPrimaryLabel(
	record Record,
	field string,
	labels []discoveredLabel,
) (string, bool, error) {
	raw, found := record.Get(field)
	if !found {
		return "", false, fmt.Errorf(
			"Neo4j discovery endpoint omitted %s",
			field,
		)
	}
	values, err := stringList(raw)
	if err != nil {
		return "", false, fmt.Errorf(
			"Neo4j discovery endpoint %s is invalid",
			field,
		)
	}
	if len(values) == 0 {
		for _, label := range labels {
			if label.source == "" {
				return label.target, true, nil
			}
		}
		return "", false, nil
	}
	selected := make([]string, 0, len(values))
	for _, value := range values {
		for _, label := range labels {
			if label.source == value {
				selected = append(selected, label.target)
				break
			}
		}
	}
	if len(selected) == 0 {
		return "", false, nil
	}
	slices.Sort(selected)
	return selected[0], true, nil
}

func stringList(value any) ([]string, error) {
	switch typed := value.(type) {
	case []string:
		result := slices.Clone(typed)
		for _, item := range result {
			if !validDiscoveredIdentifier(item) {
				return nil, errors.New("list contains an invalid label")
			}
		}
		return result, nil
	case []any:
		result := make([]string, len(typed))
		for index, item := range typed {
			text, ok := item.(string)
			if !ok || !validDiscoveredIdentifier(text) {
				return nil, errors.New("list contains an invalid label")
			}
			result[index] = text
		}
		return result, nil
	default:
		return nil, errors.New("value is not a string list")
	}
}

func buildDiscoveredVertex(
	label discoveredLabel,
	labels []discoveredLabel,
	properties []string,
	options config.Neo4jDiscovery,
) config.VertexQuery {
	propertyMap, propertyReturns := discoveredPropertyMapping("n", properties)
	match := "MATCH (n:" + quoteCypherIdentifier(label.source) + ")"
	predicates := []string{
		"$afterKey IS NULL OR n." +
			quoteCypherIdentifier(options.VertexKeyProperty) +
			" > $afterKey",
	}
	if label.source == "" {
		match = "MATCH (n)"
		predicates = append([]string{"size(labels(n)) = 0"}, predicates...)
	} else {
		predicates = append(
			[]string{primaryLabelPredicate("n", label.target, labels)},
			predicates...,
		)
	}
	return config.VertexQuery{
		Label: label.target,
		Query: buildDiscoveryQuery(
			match,
			predicates,
			[]string{
				"n." + quoteCypherIdentifier(options.VertexKeyProperty) +
					" AS __key",
				"n." + quoteCypherIdentifier(options.VertexIDProperty) +
					" AS __id",
			},
			propertyReturns,
		),
		KeyField:   "__key",
		IDField:    "__id",
		Properties: propertyMap,
	}
}

func buildDiscoveredEdge(
	relationshipType string,
	pair endpointPair,
	labels []discoveredLabel,
	properties []string,
	options config.Neo4jDiscovery,
) config.EdgeQuery {
	propertyMap, propertyReturns := discoveredPropertyMapping("r", properties)
	return config.EdgeQuery{
		Label: relationshipType,
		Query: buildDiscoveryQuery(
			"MATCH (a)-[r:"+quoteCypherIdentifier(relationshipType)+"]->(b)",
			[]string{
				"$afterKey IS NULL OR r." +
					quoteCypherIdentifier(options.EdgeKeyProperty) +
					" > $afterKey",
				primaryLabelPredicate("a", pair.start, labels),
				primaryLabelPredicate("b", pair.end, labels),
			},
			[]string{
				"r." + quoteCypherIdentifier(options.EdgeKeyProperty) +
					" AS __key",
				"r." + quoteCypherIdentifier(options.EdgeIDProperty) +
					" AS __id",
				endpointIDExpression("a", options.VertexIDProperty) +
					" AS __start",
				endpointIDExpression("b", options.VertexIDProperty) +
					" AS __end",
			},
			propertyReturns,
		),
		KeyField:        "__key",
		ExternalIDField: "__id",
		Start: config.EndpointMapping{
			Label: pair.start,
			Field: "__start",
		},
		End: config.EndpointMapping{
			Label: pair.end,
			Field: "__end",
		},
		Properties: propertyMap,
	}
}

func buildDiscoveryQuery(
	match string,
	predicates []string,
	returns []string,
	propertyReturns []string,
) string {
	allReturns := append(slices.Clone(returns), propertyReturns...)
	return match +
		" WHERE (" + strings.Join(predicates, ") AND (") + ")" +
		" RETURN " + strings.Join(allReturns, ", ") +
		" ORDER BY __key"
}

func discoveredPropertyMapping(
	variable string,
	properties []string,
) (map[string]string, []string) {
	mapping := make(map[string]string, len(properties))
	returns := make([]string, 0, len(properties))
	for index, property := range properties {
		alias := fmt.Sprintf("__property_%04d", index)
		mapping[property] = alias
		returns = append(
			returns,
			variable+"."+quoteCypherIdentifier(property)+" AS "+alias,
		)
	}
	return mapping, returns
}

func primaryLabelPredicate(
	variable string,
	target string,
	labels []discoveredLabel,
) string {
	if target == unlabeledTargetLabel {
		return "size(labels(" + variable + ")) = 0"
	}
	parts := []string{variable + ":" + quoteCypherIdentifier(target)}
	for _, label := range labels {
		if label.target >= target {
			break
		}
		if label.source != "" {
			parts = append(parts,
				"NOT "+variable+":"+quoteCypherIdentifier(label.source),
			)
		}
	}
	return strings.Join(parts, " AND ")
}

func endpointIDExpression(
	variable string,
	idProperty string,
) string {
	return variable + "." + quoteCypherIdentifier(idProperty)
}

func vertexPropertyQuery(label discoveredLabel) string {
	if label.source == "" {
		return `MATCH (n)
WHERE size(labels(n)) = 0
UNWIND keys(n) AS property
RETURN DISTINCT property AS property
ORDER BY property`
	}
	return "MATCH (n:" + quoteCypherIdentifier(label.source) + `)
UNWIND keys(n) AS property
RETURN DISTINCT property AS property
ORDER BY property`
}

func relationshipPropertyQuery(relationshipType string) string {
	return "MATCH ()-[r:" + quoteCypherIdentifier(relationshipType) + `]->()
UNWIND keys(r) AS property
RETURN DISTINCT property AS property
ORDER BY property`
}

func relationshipPairQuery(relationshipType string) string {
	return "MATCH (a)-[r:" + quoteCypherIdentifier(relationshipType) + `]->(b)
RETURN DISTINCT labels(a) AS startLabels, labels(b) AS endLabels`
}

func requireProperties(properties []string, required ...string) error {
	for _, property := range required {
		if !slices.Contains(properties, property) {
			return fmt.Errorf(
				"required stable property %q was not found",
				property,
			)
		}
	}
	return nil
}

func quoteCypherIdentifier(identifier string) string {
	return "`" + strings.ReplaceAll(identifier, "`", "``") + "`"
}

func validDiscoveredIdentifier(value string) bool {
	if value == "" || len(value) > 256 {
		return false
	}
	return !strings.ContainsFunc(value, unicode.IsControl)
}
