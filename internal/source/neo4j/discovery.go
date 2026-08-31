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
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
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
	return DiscoverMappingsBounded(ctx, source, client, nil)
}

func DiscoverMappingsBounded(
	ctx context.Context,
	source config.Neo4jSource,
	client Client,
	budget *sourcecontract.ProfileBudget,
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
	labels, err := discoverLabels(ctx, client, options, budget)
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
			vertexPropertyQuery(label, labels),
			options.MaxProperties,
			budget,
		)
		if err != nil {
			return config.Neo4jSource{}, fmt.Errorf(
				"discover Neo4j vertex label %q: %w",
				label.target,
				err,
			)
		}
		if len(properties) == 0 {
			count, err := discoverCount(
				ctx,
				client,
				vertexPartitionCountQuery(label, labels),
				budget,
			)
			if err != nil {
				return config.Neo4jSource{}, fmt.Errorf(
					"count Neo4j vertex label %q partition: %w",
					label.target,
					err,
				)
			}
			if count == 0 {
				continue
			}
		}
		requiredVertexProperties := []string{options.VertexKeyProperty}
		if options.VertexIdentity != config.Neo4jVertexIdentityInternalID {
			requiredVertexProperties = append(
				requiredVertexProperties,
				options.VertexIDProperty,
			)
		}
		if err := requireProperties(properties, requiredVertexProperties...); err != nil {
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
		budget,
		sourcecontract.ProfileBudgetUsage{Labels: 1},
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
			budget,
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
			budget,
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
				len(pairs) > 1,
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
	budget *sourcecontract.ProfileBudget,
) ([]discoveredLabel, error) {
	names, err := discoverStrings(
		ctx,
		client,
		discoverLabelsQuery,
		"label",
		options.LabelPrefix,
		options.MaxLabels,
		budget,
		sourcecontract.ProfileBudgetUsage{Labels: 1},
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
	count, err := discoverCount(
		ctx,
		client,
		discoverUnlabeledQuery,
		budget,
		sourcecontract.ProfileBudgetLabels,
	)
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
		if err := budget.Charge(sourcecontract.ProfileBudgetUsage{Labels: 1}); err != nil {
			return nil, err
		}
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
	budget *sourcecontract.ProfileBudget,
	catalogCharge sourcecontract.ProfileBudgetUsage,
) ([]string, error) {
	var dimension sourcecontract.ProfileBudgetDimension
	if catalogCharge.Labels != 0 {
		dimension |= sourcecontract.ProfileBudgetLabels
	}
	if catalogCharge.Properties != 0 {
		dimension |= sourcecontract.ProfileBudgetProperties
	}
	if err := budget.Full(dimension); err != nil {
		return nil, err
	}
	stream, err := client.Query(ctx, query, nil)
	if err != nil {
		_ = budget.Charge(sourcecontract.ProfileBudgetUsage{FailedRequestAttempts: 1})
		return nil, err
	}
	if err := budget.Charge(sourcecontract.ProfileBudgetUsage{Pages: 1}); err != nil {
		return nil, errors.Join(err, stream.Close(ctx))
	}
	var values []string
	seen := make(map[string]struct{})
	for {
		if err := budget.CanProcess(); err != nil {
			return nil, errors.Join(err, stream.Close(ctx))
		}
		record, nextErr := stream.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			return nil, errors.Join(nextErr, stream.Close(ctx))
		}
		size, sizeErr := estimateRecordSize(record, int64(^uint64(0)>>1))
		if err := budget.Charge(sourcecontract.ProfileBudgetUsage{
			Rows: 1, DecodedInputBytes: size,
		}); err != nil {
			return nil, errors.Join(err, stream.Close(ctx))
		}
		if sizeErr != nil {
			return nil, errors.Join(sizeErr, stream.Close(ctx))
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
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		if err := budget.Charge(catalogCharge); err != nil {
			return nil, errors.Join(err, stream.Close(ctx))
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
	budget *sourcecontract.ProfileBudget,
	dimensions ...sourcecontract.ProfileBudgetDimension,
) (int64, error) {
	if err := budget.Full(dimensions...); err != nil {
		return 0, err
	}
	stream, err := client.Query(ctx, query, nil)
	if err != nil {
		_ = budget.Charge(sourcecontract.ProfileBudgetUsage{FailedRequestAttempts: 1})
		return 0, err
	}
	if err := budget.Charge(sourcecontract.ProfileBudgetUsage{Pages: 1}); err != nil {
		return 0, errors.Join(err, stream.Close(ctx))
	}
	record, err := stream.Next(ctx)
	if err != nil {
		return 0, errors.Join(err, stream.Close(ctx))
	}
	size, sizeErr := estimateRecordSize(record, int64(^uint64(0)>>1))
	if err := budget.Charge(sourcecontract.ProfileBudgetUsage{
		Rows: 1, DecodedInputBytes: size,
	}); err != nil {
		return 0, errors.Join(err, stream.Close(ctx))
	}
	if sizeErr != nil {
		return 0, errors.Join(sizeErr, stream.Close(ctx))
	}
	raw, found := record.Get("count")
	count, ok := raw.(int64)
	if !found || !ok || count < 0 {
		return 0, errors.Join(
			errors.New("Neo4j discovery returned an invalid count"),
			stream.Close(ctx),
		)
	}
	if err := budget.CanProcess(); err != nil {
		return 0, errors.Join(err, stream.Close(ctx))
	}
	extra, err := stream.Next(ctx)
	if !errors.Is(err, io.EOF) {
		if err == nil {
			size, sizeErr := estimateRecordSize(extra, int64(^uint64(0)>>1))
			if chargeErr := budget.Charge(sourcecontract.ProfileBudgetUsage{
				Rows: 1, DecodedInputBytes: size,
			}); chargeErr != nil {
				return 0, errors.Join(chargeErr, stream.Close(ctx))
			}
			if sizeErr != nil {
				return 0, errors.Join(sizeErr, stream.Close(ctx))
			}
		}
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
	budget *sourcecontract.ProfileBudget,
) ([]string, error) {
	return discoverStrings(
		ctx,
		client,
		query,
		"property",
		"",
		maxProperties,
		budget,
		sourcecontract.ProfileBudgetUsage{Properties: 1},
	)
}

func discoverEndpointPairs(
	ctx context.Context,
	client Client,
	relationshipType string,
	labels []discoveredLabel,
	budget *sourcecontract.ProfileBudget,
) ([]endpointPair, error) {
	if err := budget.Full(sourcecontract.ProfileBudgetLabels); err != nil {
		return nil, err
	}
	stream, err := client.Query(
		ctx,
		relationshipPairQuery(relationshipType),
		nil,
	)
	if err != nil {
		_ = budget.Charge(sourcecontract.ProfileBudgetUsage{FailedRequestAttempts: 1})
		return nil, err
	}
	if err := budget.Charge(sourcecontract.ProfileBudgetUsage{Pages: 1}); err != nil {
		return nil, errors.Join(err, stream.Close(ctx))
	}
	pairs := make(map[endpointPair]struct{})
	for {
		if err := budget.CanProcess(); err != nil {
			return nil, errors.Join(err, stream.Close(ctx))
		}
		record, nextErr := stream.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			return nil, errors.Join(nextErr, stream.Close(ctx))
		}
		size, sizeErr := estimateRecordSize(record, int64(^uint64(0)>>1))
		if err := budget.Charge(sourcecontract.ProfileBudgetUsage{
			Rows: 1, DecodedInputBytes: size,
		}); err != nil {
			return nil, errors.Join(err, stream.Close(ctx))
		}
		if sizeErr != nil {
			return nil, errors.Join(sizeErr, stream.Close(ctx))
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
		pair := endpointPair{start: start, end: end}
		if _, exists := pairs[pair]; !exists {
			if err := budget.Charge(
				sourcecontract.ProfileBudgetUsage{Labels: 1},
			); err != nil {
				return nil, errors.Join(err, stream.Close(ctx))
			}
			pairs[pair] = struct{}{}
		}
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
	key := "n." + quoteCypherIdentifier(options.VertexKeyProperty)
	predicates := []string{}
	if label.source == "" {
		match = "MATCH (n)"
		predicates = append([]string{"size(labels(n)) = 0"}, predicates...)
	} else {
		predicates = append(
			[]string{primaryLabelPredicate("n", label.target, labels)},
			predicates...,
		)
	}
	returns := []string{
		key + " AS __key",
		vertexIdentityExpression("n", options) + " AS __id",
	}
	return config.VertexQuery{
		Label: label.target,
		Query: buildDiscoveryQuery(
			match,
			append(slices.Clone(predicates), key+" > $afterKey"),
			returns,
			propertyReturns,
		),
		InitialQuery: buildDiscoveryQuery(
			match,
			append(slices.Clone(predicates), key+" IS NOT NULL"),
			returns,
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
	filterEndpoints bool,
) config.EdgeQuery {
	propertyMap, propertyReturns := discoveredPropertyMapping("r", properties)
	key := "r." + quoteCypherIdentifier(options.EdgeKeyProperty)
	match := "MATCH (a)-[r:" + quoteCypherIdentifier(relationshipType) + "]->(b)" +
		" USING INDEX r:" + quoteCypherIdentifier(relationshipType) +
		"(" + quoteCypherIdentifier(options.EdgeKeyProperty) + ")"
	predicates := make([]string, 0, 3)
	if filterEndpoints {
		predicates = append(
			predicates,
			primaryLabelPredicate("a", pair.start, labels),
			primaryLabelPredicate("b", pair.end, labels),
		)
	}
	returns := []string{
		key + " AS __key",
		"r." + quoteCypherIdentifier(options.EdgeIDProperty) + " AS __id",
		vertexIdentityExpression("a", options) + " AS __start",
		vertexIdentityExpression("b", options) + " AS __end",
	}
	return config.EdgeQuery{
		Label: relationshipType,
		Query: buildDiscoveryQuery(
			match,
			append(slices.Clone(predicates), key+" > $afterKey"),
			returns,
			propertyReturns,
		),
		InitialQuery: buildDiscoveryQuery(
			match,
			append(slices.Clone(predicates), key+" IS NOT NULL"),
			returns,
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
	// FetchRows is already the Bolt driver's fetch size. Keep one ordered stream
	// open per mapping so large relationship mappings do not pay Cypher planning
	// and index-seek startup costs again for every fetched page.
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

func vertexIdentityExpression(
	variable string,
	options config.Neo4jDiscovery,
) string {
	if options.VertexIdentity == config.Neo4jVertexIdentityInternalID {
		return "id(" + variable + ")"
	}
	return variable + "." + quoteCypherIdentifier(options.VertexIDProperty)
}

func vertexPropertyQuery(
	label discoveredLabel,
	labels []discoveredLabel,
) string {
	if label.source == "" {
		return `MATCH (n)
WHERE size(labels(n)) = 0
UNWIND keys(n) AS property
RETURN DISTINCT property AS property
ORDER BY property`
	}
	return "MATCH (n:" + quoteCypherIdentifier(label.source) + `)
WHERE ` + primaryLabelPredicate("n", label.target, labels) + `
UNWIND keys(n) AS property
RETURN DISTINCT property AS property
ORDER BY property`
}

func vertexPartitionCountQuery(
	label discoveredLabel,
	labels []discoveredLabel,
) string {
	if label.source == "" {
		return discoverUnlabeledQuery
	}
	return "MATCH (n:" + quoteCypherIdentifier(label.source) + ")\n" +
		"WHERE " + primaryLabelPredicate("n", label.target, labels) + "\n" +
		"RETURN count(n) AS count"
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
