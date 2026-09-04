package config

import (
	"fmt"
	"net/url"
	"regexp"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/rioriost/agefreighter/internal/sqlquery"
)

const maxConcurrency = 256

var (
	jobNamePattern         = regexp.MustCompile(`^[a-z][a-z0-9-]{2,62}$`)
	graphNamePattern       = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.-]*[A-Za-z0-9_]$`)
	envNamePattern         = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	cosmosParameterPattern = regexp.MustCompile(`^@[A-Za-z_][A-Za-z0-9_]*$`)
	sourceIdentityPattern  = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)
	queryFieldPattern      = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
)

type ValidationError struct {
	Path    string
	Code    string
	Message string
}

func (err ValidationError) Error() string {
	return fmt.Sprintf("%s [%s]: %s", err.Path, err.Code, err.Message)
}

type ValidationErrors []ValidationError

func (errs ValidationErrors) Error() string {
	messages := make([]string, len(errs))
	for index, err := range errs {
		messages[index] = err.Error()
	}
	return "configuration is invalid:\n- " + strings.Join(messages, "\n- ")
}

func (job LoadJob) Validate() error {
	var errs ValidationErrors
	add := func(valid bool, path, code, message string) {
		if !valid {
			errs = append(errs, ValidationError{Path: path, Code: code, Message: message})
		}
	}

	add(job.APIVersion == APIVersion, "apiVersion", "unsupported", "must be agefreighter.io/v2")
	add(job.Kind == KindLoadJob, "kind", "unsupported", "must be LoadJob")
	add(jobNamePattern.MatchString(job.Metadata.Name), "metadata.name", "format",
		"must be 3-63 characters using lowercase letters, digits, and hyphens")
	validateSource(job.Source, &errs)
	validateTarget(job.Target, &errs)
	validateRuntime(job.Runtime, &errs)
	validateTrial(job.Trial, job.Source, job.Target, job.Runtime, &errs)
	validateErrorPolicies(job.Errors, &errs)
	if job.Target.Type == TargetPostgreSQLPropertyGraph {
		add(
			job.Errors.MissingEndpoint == MissingEndpointError,
			"errors.missingEndpoint",
			"unsupported",
			"postgresql-property-graph loads currently support only error",
		)
		validatePropertyGraphEdgeIdentity(job.Source, &errs)
	}
	if job.Errors.MissingEndpoint == MissingEndpointDefer {
		add(
			job.Target.Mode == LoadAppend || job.Target.Mode == LoadUpsert,
			"errors.missingEndpoint",
			"policy",
			"defer is supported only for append and upsert",
		)
	}
	if job.Target.Mode == LoadUpsert {
		add(
			job.Errors.MaxDeferredEdges > 0,
			"errors.maxDeferredEdges",
			"required",
			"must be positive for upsert FIFO ordering",
		)
		validateUpsertEdgeIdentity(job.Source, &errs)
	}

	if len(errs) != 0 {
		return errs
	}
	return nil
}

func validatePropertyGraphEdgeIdentity(source Source, errs *ValidationErrors) {
	add := validationAdder(errs)
	switch source.Type {
	case SourcePostgreSQL:
		if source.PostgreSQL != nil {
			for index, edge := range source.PostgreSQL.Edges {
				add(edge.ExternalIDField != "",
					fmt.Sprintf("source.postgresql.edges[%d].externalIdField", index),
					"required", "is required for PostgreSQL property graph edge identity")
			}
		}
	case SourceNeo4j:
		if source.Neo4j != nil &&
			(source.Neo4j.Discovery == nil || !source.Neo4j.Discovery.Enabled) {
			for index, edge := range source.Neo4j.Edges {
				add(edge.ExternalIDField != "",
					fmt.Sprintf("source.neo4j.edges[%d].externalIdField", index),
					"required", "is required for PostgreSQL property graph edge identity")
			}
		}
	case SourceCosmos:
		if source.Cosmos != nil &&
			(source.Cosmos.Gremlin == nil || !source.Cosmos.Gremlin.Enabled) {
			for index, edge := range source.Cosmos.Edges {
				add(edge.ExternalIDField != "",
					fmt.Sprintf("source.cosmos.edges[%d].externalIdField", index),
					"required", "is required for PostgreSQL property graph edge identity")
			}
		}
	}
}

func validateTrial(
	trial *TrialOptions,
	source Source,
	target Target,
	runtime Runtime,
	errs *ValidationErrors,
) {
	if trial == nil {
		return
	}
	add := validationAdder(errs)
	add(trial.Enabled, "trial.enabled", "required",
		"must be true when the trial block is present")
	add(trial.MaxVerticesPerLabel >= 1 &&
		trial.MaxVerticesPerLabel <= 100_000,
		"trial.maxVerticesPerLabel", "range",
		"must be from 1 to 100000")
	add(trial.MaxVertices >= 1 && trial.MaxVertices <= 1_000_000,
		"trial.maxVertices", "range",
		"must be from 1 to 1000000")
	add(trial.MaxVerticesPerLabel <= trial.MaxVertices,
		"trial.maxVerticesPerLabel", "range",
		"must not exceed maxVertices")
	add(trial.MaxEdges >= 1 && trial.MaxEdges <= 1_000_000,
		"trial.maxEdges", "range",
		"must be from 1 to 1000000")
	add(trial.MaxBytes > 0, "trial.maxBytes", "range",
		"must be positive")
	add(trial.MaxBytes <= runtime.MemoryLimit, "trial.maxBytes", "range",
		"must not exceed runtime.memoryLimit")
	add(target.Mode == LoadCreate || target.Mode == LoadReplace,
		"trial", "policy",
		"is supported only with create or replace")
	cosmosGremlin := source.Type == SourceCosmos &&
		source.Cosmos != nil &&
		source.Cosmos.Gremlin != nil &&
		source.Cosmos.Gremlin.Enabled
	add(!cosmosGremlin, "trial", "unsupported",
		"is not supported with Cosmos Gremlin interpretation because the Go SDK cannot execute the cross-partition ordering required for deterministic sampling")

	available := configuredVertexLabels(source)
	validateAvailableLabels := !(source.Type == SourceNeo4j &&
		source.Neo4j != nil &&
		source.Neo4j.Discovery != nil &&
		source.Neo4j.Discovery.Enabled) &&
		!cosmosGremlin
	seen := make(map[string]bool, len(trial.IncludeLabels))
	for index, label := range trial.IncludeLabels {
		path := fmt.Sprintf("trial.includeLabels[%d]", index)
		add(label != "", path, "required", "must not be empty")
		add(!seen[label], path, "duplicate", "must be unique")
		if validateAvailableLabels {
			add(available[label], path, "unknown",
				"must match a configured vertex label")
		}
		seen[label] = true
	}
	if source.Type == SourceCosmos && source.Cosmos != nil {
		for index, mapping := range source.Cosmos.Vertices {
			add(sqlquery.HasTopLevelOrderBy(mapping.Query),
				fmt.Sprintf("source.cosmos.vertices[%d].query", index),
				"ordering",
				"must contain ORDER BY on a stable unique key in trial mode")
		}
		for index, mapping := range source.Cosmos.Edges {
			add(sqlquery.HasTopLevelOrderBy(mapping.Query),
				fmt.Sprintf("source.cosmos.edges[%d].query", index),
				"ordering",
				"must contain ORDER BY on a stable unique key in trial mode")
		}
	}
}

func configuredVertexLabels(source Source) map[string]bool {
	labels := make(map[string]bool)
	switch source.Type {
	case SourceCSV:
		if source.CSV != nil {
			for _, mapping := range source.CSV.Vertices {
				labels[mapping.Label] = true
			}
		}
	case SourcePostgreSQL:
		if source.PostgreSQL != nil {
			for _, mapping := range source.PostgreSQL.Vertices {
				labels[mapping.Label] = true
			}
		}
	case SourceNeo4j:
		if source.Neo4j != nil {
			for _, mapping := range source.Neo4j.Vertices {
				labels[mapping.Label] = true
			}
		}
	case SourceCosmos:
		if source.Cosmos != nil {
			for _, mapping := range source.Cosmos.Vertices {
				labels[mapping.Label] = true
			}
		}
	}
	return labels
}

func validateSource(source Source, errs *ValidationErrors) {
	add := validationAdder(errs)
	add(source.Namespace != "", "source.namespace", "required", "must not be empty")

	configured := 0
	for _, present := range []bool{
		source.CSV != nil,
		source.PostgreSQL != nil,
		source.Neo4j != nil,
		source.Cosmos != nil,
	} {
		if present {
			configured++
		}
	}
	add(configured == 1, "source", "discriminator", "must contain exactly one source configuration")

	switch source.Type {
	case SourceCSV:
		add(source.CSV != nil, "source.csv", "required", "is required when source.type is csv")
		if source.CSV != nil {
			validateCSV(*source.CSV, source.Namespace, errs)
		}
	case SourcePostgreSQL:
		add(source.PostgreSQL != nil, "source.postgresql", "required",
			"is required when source.type is postgresql")
		if source.PostgreSQL != nil {
			validatePostgreSQL(*source.PostgreSQL, source.Namespace, errs)
		}
	case SourceNeo4j:
		add(source.Neo4j != nil, "source.neo4j", "required", "is required when source.type is neo4j")
		if source.Neo4j != nil {
			validateNeo4j(*source.Neo4j, source.Namespace, errs)
		}
	case SourceCosmos:
		add(source.Cosmos != nil, "source.cosmos", "required",
			"is required when source.type is cosmos-nosql")
		if source.Cosmos != nil {
			validateCosmos(*source.Cosmos, source.Namespace, errs)
		}
	default:
		add(false, "source.type", "unsupported", "must be csv, postgresql, neo4j, or cosmos-nosql")
	}
}

func validatePostgreSQL(
	source PostgreSQLSource,
	namespace string,
	errs *ValidationErrors,
) {
	add := validationAdder(errs)
	validateSecret(source.Connection, "source.postgresql.connection", errs)
	add(
		source.ReadMode == PostgreSQLReadCopy ||
			source.ReadMode == PostgreSQLReadCursor ||
			source.ReadMode == PostgreSQLReadKeyset,
		"source.postgresql.readMode",
		"unsupported",
		"must be copy, cursor, or keyset",
	)
	add(
		source.FetchRows >= 1 && source.FetchRows <= 100_000,
		"source.postgresql.fetchRows",
		"range",
		"must be from 1 to 100000",
	)
	validateQueries(source.Vertices, source.Edges, namespace, errs)
	for index, vertex := range source.Vertices {
		validatePostgreSQLQuery(
			source.ReadMode,
			vertex.Query,
			vertex.KeyField,
			fmt.Sprintf("source.postgresql.vertices[%d]", index),
			errs,
		)
	}
	for index, edge := range source.Edges {
		validatePostgreSQLQuery(
			source.ReadMode,
			edge.Query,
			edge.KeyField,
			fmt.Sprintf("source.postgresql.edges[%d]", index),
			errs,
		)
	}
}

func validatePostgreSQLQuery(
	mode PostgreSQLReadMode,
	query string,
	keyField string,
	path string,
	errs *ValidationErrors,
) {
	add := validationAdder(errs)
	trimmed := strings.TrimSpace(query)
	fields := strings.Fields(trimmed)
	firstKeyword := ""
	if len(fields) > 0 {
		firstKeyword = strings.ToLower(fields[0])
	}
	add(
		firstKeyword == "select" || firstKeyword == "with",
		path+".query",
		"format",
		"must be a SELECT or WITH query",
	)
	add(
		!strings.Contains(trimmed, ";"),
		path+".query",
		"format",
		"must contain exactly one statement without a semicolon",
	)
	add(
		sqlquery.HasTopLevelOrderBy(query),
		path+".query",
		"ordering",
		"must contain ORDER BY for deterministic resume",
	)
	if mode == PostgreSQLReadKeyset {
		add(keyField != "", path+".keyField", "required",
			"is required in keyset mode")
		add(strings.Contains(query, "$1"), path+".query", "parameter",
			"must use $1 for the prior key in keyset mode")
		add(strings.Contains(query, "$2"), path+".query", "parameter",
			"must use $2 for the fetch limit in keyset mode")
	} else {
		add(keyField == "", path+".keyField", "policy",
			"is supported only in keyset mode")
	}
}

func validateCSV(source CSVSource, namespace string, errs *ValidationErrors) {
	add := validationAdder(errs)
	validateDelimitedOptions(source.Defaults, "source.csv.defaults", errs)
	add(len(source.Vertices) > 0, "source.csv.vertices", "required", "must contain at least one vertex mapping")
	for index, vertex := range source.Vertices {
		path := fmt.Sprintf("source.csv.vertices[%d]", index)
		add(vertex.Label != "", path+".label", "required", "must not be empty")
		add(vertex.Path != "", path+".path", "required", "must not be empty")
		add(vertex.IDColumn != "", path+".idColumn", "required", "must not be empty")
		validatePropertyMapping(vertex.Properties, path+".properties", errs)
		if vertex.Format != nil {
			validateDelimitedOptions(*vertex.Format, path+".format", errs)
		}
	}
	for index, edge := range source.Edges {
		path := fmt.Sprintf("source.csv.edges[%d]", index)
		add(edge.Label != "", path+".label", "required", "must not be empty")
		add(edge.Path != "", path+".path", "required", "must not be empty")
		add(edge.ExternalIDColumn != "", path+".externalIdColumn", "required",
			"is required for resumable edge loading and verification")
		validateEndpoint(edge.Start, namespace, path+".start", errs)
		validateEndpoint(edge.End, namespace, path+".end", errs)
		validatePropertyMapping(edge.Properties, path+".properties", errs)
		if edge.Format != nil {
			validateDelimitedOptions(*edge.Format, path+".format", errs)
		}
	}
}

func validateDelimitedOptions(options DelimitedOptions, path string, errs *ValidationErrors) {
	add := validationAdder(errs)
	add(utf8.RuneCountInString(options.Delimiter) == 1, path+".delimiter", "format",
		"must contain exactly one character")
	add(utf8.RuneCountInString(options.Quote) == 1, path+".quote", "format",
		"must contain exactly one character")
	add(utf8.RuneCountInString(options.Escape) == 1, path+".escape", "format",
		"must contain exactly one character")
	add(options.Delimiter != "\n" && options.Delimiter != "\r", path+".delimiter", "format",
		"must not be a line break")
	add(options.Quote != "\n" && options.Quote != "\r", path+".quote", "format",
		"must not be a line break")
	add(options.Escape != "\n" && options.Escape != "\r", path+".escape", "format",
		"must not be a line break")
	add(options.Delimiter != options.Quote, path+".delimiter", "format",
		"must differ from quote")
	add(options.Encoding == "utf-8", path+".encoding", "unsupported", "only utf-8 is supported in 2.0.0")
}

func validateQueries(vertices []VertexQuery, edges []EdgeQuery, namespace string, errs *ValidationErrors) {
	add := validationAdder(errs)
	add(len(vertices) > 0, "source.vertices", "required", "must contain at least one vertex query")
	for index, vertex := range vertices {
		path := fmt.Sprintf("source.vertices[%d]", index)
		add(vertex.Label != "", path+".label", "required", "must not be empty")
		add(vertex.Query != "", path+".query", "required", "must not be empty")
		add(vertex.IDField != "", path+".idField", "required", "must not be empty")
		validatePropertyMapping(vertex.Properties, path+".properties", errs)
	}
	for index, edge := range edges {
		path := fmt.Sprintf("source.edges[%d]", index)
		add(edge.Label != "", path+".label", "required", "must not be empty")
		add(edge.Query != "", path+".query", "required", "must not be empty")
		validateEndpoint(edge.Start, namespace, path+".start", errs)
		validateEndpoint(edge.End, namespace, path+".end", errs)
		validatePropertyMapping(edge.Properties, path+".properties", errs)
	}
}

func validateNeo4j(source Neo4jSource, namespace string, errs *ValidationErrors) {
	add := validationAdder(errs)
	parsedURI, uriErr := url.Parse(source.URI)
	validScheme := uriErr == nil && (parsedURI.Scheme == "neo4j" ||
		parsedURI.Scheme == "neo4j+s" || parsedURI.Scheme == "neo4j+ssc" ||
		parsedURI.Scheme == "bolt" || parsedURI.Scheme == "bolt+s" ||
		parsedURI.Scheme == "bolt+ssc")
	add(validScheme && parsedURI.Host != "" && parsedURI.User == nil,
		"source.neo4j.uri", "format",
		"must use neo4j, neo4j+s, neo4j+ssc, bolt, bolt+s, or bolt+ssc without embedded credentials")
	add(source.Database != "", "source.neo4j.database", "required", "must not be empty")
	add(sourceIdentityPattern.MatchString(source.SourceID),
		"source.neo4j.sourceId", "format",
		"must be 1-128 characters using letters, digits, dots, underscores, colons, or hyphens")
	add(source.FetchRows >= 1 && source.FetchRows <= 100_000,
		"source.neo4j.fetchRows", "range", "must be from 1 to 100000")
	add(source.MultiLabelPolicy == Neo4jMultiLabelConfigured ||
		source.MultiLabelPolicy == Neo4jMultiLabelReject,
		"source.neo4j.multiLabelPolicy", "unsupported",
		"must be configured or reject")
	add((source.Username == "") == (source.Password == nil), "source.neo4j", "authentication",
		"username and password must either both be set or both be omitted")
	if source.Password != nil {
		validateSecret(*source.Password, "source.neo4j.password", errs)
	}
	if source.Discovery != nil {
		validateNeo4jDiscovery(*source.Discovery, source, errs)
	} else {
		validateQueries(source.Vertices, source.Edges, namespace, errs)
	}
	for index, vertex := range source.Vertices {
		validateNeo4jQuery(vertex.Query, vertex.KeyField,
			fmt.Sprintf("source.neo4j.vertices[%d]", index), errs)
	}
	for index, edge := range source.Edges {
		validateNeo4jQuery(edge.Query, edge.KeyField,
			fmt.Sprintf("source.neo4j.edges[%d]", index), errs)
	}
}

func validateNeo4jDiscovery(
	discovery Neo4jDiscovery,
	source Neo4jSource,
	errs *ValidationErrors,
) {
	add := validationAdder(errs)
	add(discovery.Enabled, "source.neo4j.discovery.enabled", "required",
		"must be true when discovery is configured")
	add(len(source.Vertices) == 0 && len(source.Edges) == 0,
		"source.neo4j.discovery", "policy",
		"cannot be combined with explicit vertex or edge mappings")
	add(source.MultiLabelPolicy == Neo4jMultiLabelConfigured,
		"source.neo4j.multiLabelPolicy", "policy",
		"must be configured when discovery is enabled")
	add(discovery.VertexIdentity == Neo4jVertexIdentityProperty ||
		discovery.VertexIdentity == Neo4jVertexIdentityInternalID,
		"source.neo4j.discovery.vertexIdentity", "unsupported",
		"must be property or internal-id")
	for path, property := range map[string]string{
		"source.neo4j.discovery.vertexKeyProperty": discovery.VertexKeyProperty,
		"source.neo4j.discovery.vertexIdProperty":  discovery.VertexIDProperty,
		"source.neo4j.discovery.edgeKeyProperty":   discovery.EdgeKeyProperty,
		"source.neo4j.discovery.edgeIdProperty":    discovery.EdgeIDProperty,
	} {
		add(validDiscoveryIdentifier(property), path, "format",
			"must be 1-256 UTF-8 bytes without control characters")
	}
	for path, prefix := range map[string]string{
		"source.neo4j.discovery.labelPrefix":            discovery.LabelPrefix,
		"source.neo4j.discovery.relationshipTypePrefix": discovery.RelationshipTypePrefix,
	} {
		add(len(prefix) <= 256 && utf8.ValidString(prefix) &&
			!strings.ContainsFunc(prefix, unicode.IsControl),
			path, "format",
			"must not exceed 256 UTF-8 bytes or contain control characters")
	}
	add(discovery.MaxLabels >= 1 && discovery.MaxLabels <= 256,
		"source.neo4j.discovery.maxLabels", "range",
		"must be from 1 to 256")
	add(discovery.MaxProperties >= 1 && discovery.MaxProperties <= 1_024,
		"source.neo4j.discovery.maxProperties", "range",
		"must be from 1 to 1024")
}

func validDiscoveryIdentifier(value string) bool {
	return value != "" &&
		len(value) <= 256 &&
		utf8.ValidString(value) &&
		!strings.ContainsFunc(value, unicode.IsControl)
}

func validateNeo4jQuery(
	query string,
	keyField string,
	path string,
	errs *ValidationErrors,
) {
	add := validationAdder(errs)
	paged := sqlquery.HasFinalTopLevelLimitParameter(query, "pageRows")
	add(keyField != "", path+".keyField", "required",
		"is required for durable Neo4j resume")
	add(queryFieldPattern.MatchString(keyField), path+".keyField", "format",
		"must be an unquoted Cypher result identifier")
	add((!paged && sqlquery.HasFinalTopLevelOrderByField(query, keyField)) ||
		(paged && sqlquery.HasTopLevelOrderByField(query, keyField)),
		path+".query", "ordering",
		"must end with ascending ORDER BY keyField for deterministic resume")
	add(sqlquery.HasParameter(query, "afterKey"), path+".query", "parameter",
		"must use $afterKey for the prior stable key")
	add(!sqlquery.HasKeyword(query, "skip"), path+".query", "unsupported",
		"must use keyset resume rather than SKIP")
	add(!sqlquery.HasKeyword(query, "offset"), path+".query", "unsupported",
		"must use keyset resume rather than OFFSET")
	add(!sqlquery.HasKeyword(query, "limit") || paged,
		path+".query", "unsupported",
		"must stream the complete mapping or use LIMIT $pageRows")
	add(!sqlquery.HasKeyword(query, "union"), path+".query", "unsupported",
		"must not use UNION because it cannot guarantee one total key order")
	add(!sqlquery.HasKeyword(query, "collect"), path+".query", "unsupported",
		"must not eagerly materialize records with collect")
}

func validateCosmos(source CosmosSource, namespace string, errs *ValidationErrors) {
	add := validationAdder(errs)
	add(strings.HasPrefix(source.Endpoint, "https://"), "source.cosmos.endpoint", "format",
		"must use https")
	add(source.Credential == "default-azure", "source.cosmos.credential", "unsupported",
		"must be default-azure")
	add(source.Database != "", "source.cosmos.database", "required", "must not be empty")
	add(source.PageSize >= 1 && source.PageSize <= 1000, "source.cosmos.pageSize", "range",
		"must be from 1 to 1000")
	if source.Gremlin != nil {
		validateCosmosGremlin(*source.Gremlin, source, errs)
	} else {
		add(len(source.Vertices) > 0, "source.cosmos.vertices", "required",
			"must contain at least one vertex query")
	}
	for index, vertex := range source.Vertices {
		path := fmt.Sprintf("source.cosmos.vertices[%d]", index)
		add(vertex.Container != "", path+".container", "required", "must not be empty")
		add(vertex.Label != "", path+".label", "required", "must not be empty")
		add(vertex.Query != "", path+".query", "required", "must not be empty")
		validateJSONPointer(path+".idField", vertex.IDField, errs)
		validateCosmosParameters(vertex.Parameters, path+".parameters", errs)
		validateCosmosPropertyMapping(vertex.Properties, path+".properties", errs)
		validateCosmosDocumentFormat(
			vertex.DocumentFormat,
			vertex.PartitionKeyProperty,
			vertex.MaxProperties,
			vertex.Properties,
			path,
			errs,
		)
	}
	for index, edge := range source.Edges {
		path := fmt.Sprintf("source.cosmos.edges[%d]", index)
		add(edge.Container != "", path+".container", "required", "must not be empty")
		add(edge.Label != "", path+".label", "required", "must not be empty")
		add(edge.Query != "", path+".query", "required", "must not be empty")
		if edge.ExternalIDField != "" {
			validateJSONPointer(path+".externalIdField", edge.ExternalIDField, errs)
		}
		validateCosmosEndpoint(edge.Start, namespace, path+".start", errs)
		validateCosmosEndpoint(edge.End, namespace, path+".end", errs)
		validateCosmosParameters(edge.Parameters, path+".parameters", errs)
		validateCosmosPropertyMapping(edge.Properties, path+".properties", errs)
		validateCosmosDocumentFormat(
			edge.DocumentFormat,
			edge.PartitionKeyProperty,
			edge.MaxProperties,
			edge.Properties,
			path,
			errs,
		)
	}
}

func validateCosmosDocumentFormat(
	format CosmosDocumentFormat,
	partitionKeyProperty string,
	maxProperties int,
	properties map[string]string,
	path string,
	errs *ValidationErrors,
) {
	if format == "" {
		return
	}
	add := validationAdder(errs)
	add(format == CosmosDocumentGremlin, path+".documentFormat",
		"unsupported", "must be cosmos-gremlin")
	add(validDiscoveryIdentifier(partitionKeyProperty),
		path+".partitionKeyProperty", "format",
		"must be 1-256 UTF-8 bytes without control characters")
	add(maxProperties >= 1 && maxProperties <= 1_024,
		path+".maxProperties", "range", "must be from 1 to 1024")
	add(len(properties) == 0, path+".properties", "policy",
		"must be omitted for automatically interpreted Gremlin documents")
}

func validateCosmosGremlin(
	gremlin CosmosGremlin,
	source CosmosSource,
	errs *ValidationErrors,
) {
	add := validationAdder(errs)
	add(gremlin.Enabled, "source.cosmos.gremlin.enabled", "required",
		"must be true when Gremlin interpretation is configured")
	add(len(source.Vertices) == 0 && len(source.Edges) == 0,
		"source.cosmos.gremlin", "policy",
		"cannot be combined with explicit vertex or edge mappings")
	add(gremlin.Container != "", "source.cosmos.gremlin.container",
		"required", "must not be empty")
	add(validDiscoveryIdentifier(gremlin.PartitionKeyProperty),
		"source.cosmos.gremlin.partitionKeyProperty", "format",
		"must be 1-256 UTF-8 bytes without control characters")
	for path, prefix := range map[string]string{
		"source.cosmos.gremlin.labelPrefix":            gremlin.LabelPrefix,
		"source.cosmos.gremlin.relationshipTypePrefix": gremlin.RelationshipTypePrefix,
	} {
		add(len(prefix) <= 256 && utf8.ValidString(prefix) &&
			!strings.ContainsFunc(prefix, unicode.IsControl),
			path, "format",
			"must not exceed 256 UTF-8 bytes or contain control characters")
	}
	add(gremlin.MaxLabels >= 1 && gremlin.MaxLabels <= 256,
		"source.cosmos.gremlin.maxLabels", "range",
		"must be from 1 to 256")
	add(gremlin.MaxProperties >= 1 && gremlin.MaxProperties <= 1_024,
		"source.cosmos.gremlin.maxProperties", "range",
		"must be from 1 to 1024")
	add(gremlin.MaxDiscoveryDocuments >= 1 &&
		gremlin.MaxDiscoveryDocuments <= 1_000_000,
		"source.cosmos.gremlin.maxDiscoveryDocuments", "range",
		"must be from 1 to 1000000")
}

// validateCosmosEndpoint validates an edge endpoint mapping used by a Cosmos
// source, where Field must be an RFC 6901 JSON Pointer rather than the
// column/property name semantics used by other source types.
func validateCosmosEndpoint(endpoint EndpointMapping, defaultNamespace, path string, errs *ValidationErrors) {
	add := validationAdder(errs)
	add(endpoint.Label != "", path+".label", "required", "must not be empty")
	validateJSONPointer(path+".field", endpoint.Field, errs)
	add(endpoint.Namespace != "" || defaultNamespace != "", path+".namespace", "required",
		"must be set when the source has no default namespace")
}

// validateCosmosPropertyMapping validates a Cosmos property mapping, where
// each value must be an RFC 6901 JSON Pointer into the source document.
func validateCosmosPropertyMapping(properties map[string]string, path string, errs *ValidationErrors) {
	add := validationAdder(errs)
	keys := make([]string, 0, len(properties))
	for property := range properties {
		keys = append(keys, property)
	}
	slices.Sort(keys)
	for _, property := range keys {
		add(property != "", path, "format", "property names must not be empty")
		validateJSONPointer(path+"."+property, properties[property], errs)
	}
}

// validateCosmosParameters validates named Cosmos query parameters. Names
// must use the Cosmos DB "@name" convention and must be unique within a
// single query; values are validated at decode time by CosmosParamValue.
func validateCosmosParameters(parameters []CosmosQueryParameter, path string, errs *ValidationErrors) {
	add := validationAdder(errs)
	seen := make(map[string]bool, len(parameters))
	for index, parameter := range parameters {
		itemPath := fmt.Sprintf("%s[%d]", path, index)
		valid := cosmosParameterPattern.MatchString(parameter.Name)
		add(valid, itemPath+".name", "format",
			"must use @ followed by a letter or underscore and letters, digits, or underscores")
		if valid {
			add(!seen[parameter.Name], itemPath+".name", "duplicate",
				"must be unique within the query")
			seen[parameter.Name] = true
		}
	}
}

// validateJSONPointer validates RFC 6901 JSON Pointer syntax: the value must
// be non-empty, start with "/", and every "~" escape must be followed by "0"
// or "1".
func validateJSONPointer(path, value string, errs *ValidationErrors) {
	add := validationAdder(errs)
	valid := value != "" && strings.HasPrefix(value, "/") && jsonPointerEscapesValid(value)
	add(valid, path, "format", "must be a non-empty RFC 6901 JSON Pointer starting with /")
}

func jsonPointerEscapesValid(pointer string) bool {
	for index := 0; index < len(pointer); index++ {
		if pointer[index] != '~' {
			continue
		}
		if index+1 >= len(pointer) || (pointer[index+1] != '0' && pointer[index+1] != '1') {
			return false
		}
	}
	return true
}

func validateEndpoint(endpoint EndpointMapping, defaultNamespace, path string, errs *ValidationErrors) {
	add := validationAdder(errs)
	add(endpoint.Label != "", path+".label", "required", "must not be empty")
	add(endpoint.Field != "", path+".field", "required", "must not be empty")
	add(endpoint.Namespace != "" || defaultNamespace != "", path+".namespace", "required",
		"must be set when the source has no default namespace")
}

func validatePropertyMapping(properties map[string]string, path string, errs *ValidationErrors) {
	add := validationAdder(errs)
	keys := make([]string, 0, len(properties))
	for property := range properties {
		keys = append(keys, property)
	}
	slices.Sort(keys)
	for _, property := range keys {
		field := properties[property]
		add(property != "", path, "format", "property names must not be empty")
		add(field != "", path+"."+property, "required", "source field must not be empty")
	}
}

func validateTarget(target Target, errs *ValidationErrors) {
	add := validationAdder(errs)
	switch target.Type {
	case TargetApacheAGE:
		add(len(target.Graph) >= 3 && len(target.Graph) <= 63 && graphNamePattern.MatchString(target.Graph),
			"target.graph", "format", "must satisfy the Apache AGE 3-63 byte graph-name rules")
		add(target.Schema == "", "target.schema", "unsupported",
			"is supported only for postgresql-property-graph targets")
	case TargetPostgreSQLPropertyGraph:
		add(validPostgreSQLIdentifier(target.Graph), "target.graph", "format",
			"must be a valid PostgreSQL identifier of at most 63 bytes")
		add(validPostgreSQLIdentifier(target.Schema), "target.schema", "format",
			"must be a valid PostgreSQL identifier of at most 63 bytes")
		add(target.Mode == LoadCreate || target.Mode == LoadReplace ||
			target.Mode == LoadAppend || target.Mode == LoadUpsert,
			"target.mode", "unsupported",
			"must be create, replace, append, or upsert")
		add(target.Mode == LoadAppend || target.AppendDuplicate == "",
			"target.appendDuplicate", "unsupported",
			"is supported only for append loads")
	default:
		add(false, "target.type", "unsupported",
			"must be apache-age or postgresql-property-graph")
	}
	switch target.Mode {
	case LoadCreate, LoadReplace, LoadAppend, LoadUpsert:
	default:
		add(false, "target.mode", "unsupported", "must be create, replace, append, or upsert")
	}
	switch target.PropertyMode {
	case PropertiesReplace, PropertiesMerge, PropertiesMergeDeleteNull:
	default:
		add(false, "target.propertyMode", "unsupported",
			"must be replace, merge, or merge-delete-null")
	}
	if target.Mode == LoadAppend {
		switch target.AppendDuplicate {
		case AppendDuplicateError, AppendDuplicateIgnoreIdentical:
		default:
			add(false, "target.appendDuplicate", "unsupported",
				"must be error or ignore-identical")
		}
	} else {
		add(
			target.AppendDuplicate == "" ||
				target.AppendDuplicate == AppendDuplicateError ||
				target.AppendDuplicate == AppendDuplicateIgnoreIdentical,
			"target.appendDuplicate",
			"unsupported",
			"must be error or ignore-identical",
		)
	}
	validateSecret(target.Connection, "target.connection", errs)
}

func validPostgreSQLIdentifier(identifier string) bool {
	return identifier != "" && len(identifier) <= 63 &&
		utf8.ValidString(identifier) && !strings.ContainsRune(identifier, '\x00')
}

func validateSecret(secret SecretRef, path string, errs *ValidationErrors) {
	add := validationAdder(errs)
	add((secret.Env == "") != (secret.File == ""), path, "secret-reference",
		"must contain exactly one env or file reference")
	if secret.Env != "" {
		add(envNamePattern.MatchString(secret.Env), path+".env", "format",
			"must be a valid environment variable name")
	}
}

func validateRuntime(runtime Runtime, errs *ValidationErrors) {
	add := validationAdder(errs)
	add(runtime.MemoryLimit > 0, "runtime.memoryLimit", "range", "must be positive")
	add(runtime.BatchRows > 0, "runtime.batchRows", "range", "must be positive")
	add(runtime.BatchBytes > 0, "runtime.batchBytes", "range", "must be positive")
	add(runtime.BatchBytes <= runtime.MemoryLimit, "runtime.batchBytes", "range",
		"must not exceed memoryLimit")
	add(runtime.MaxSourceConcurrency == 1,
		"runtime.maxSourceConcurrency", "unsupported",
		"must be 1; connectors preserve ordered checkpoint and resume semantics")
	add(runtime.MaxTransformConcurrency == 1,
		"runtime.maxTransformConcurrency", "unsupported",
		"must be 1; connector transforms are ordered and execute within source iteration")
	validateConcurrency(runtime.MaxTargetConnections, "runtime.maxTargetConnections", errs)
	add(runtime.MaxTargetConnections >= 2, "runtime.maxTargetConnections", "range",
		"must be at least 2 for target loading")
	add(runtime.OperationTimeout > 0, "runtime.operationTimeout", "range", "must be positive")
}

func validateConcurrency(value int, path string, errs *ValidationErrors) {
	validationAdder(errs)(value >= 1 && value <= maxConcurrency, path, "range", "must be from 1 to 256")
}

func validateErrorPolicies(policies ErrorPolicies, errs *ValidationErrors) {
	add := validationAdder(errs)
	switch policies.MalformedRecord {
	case MalformedFail, MalformedQuarantine:
	default:
		add(false, "errors.malformedRecord", "unsupported", "must be fail or quarantine")
	}
	switch policies.MissingEndpoint {
	case MissingEndpointError, MissingEndpointQuarantine, MissingEndpointDefer:
	default:
		add(false, "errors.missingEndpoint", "unsupported", "must be error, quarantine, or defer")
	}
	add(policies.RejectLimit >= 0, "errors.rejectLimit", "range", "must not be negative")
	add(policies.MaxDeferredEdges >= 0, "errors.maxDeferredEdges", "range",
		"must not be negative")
	add(policies.MissingEndpoint != MissingEndpointDefer ||
		policies.MaxDeferredEdges > 0,
		"errors.maxDeferredEdges", "required",
		"must be positive when missing endpoints are deferred")
	add(policies.RejectLimit == 0 || policies.MalformedRecord == MalformedQuarantine,
		"errors.rejectLimit", "policy",
		"must be zero unless malformed records are quarantined")
	quarantine := policies.MalformedRecord == MalformedQuarantine ||
		policies.MissingEndpoint == MissingEndpointQuarantine
	add(!quarantine || policies.QuarantinePath != "", "errors.quarantinePath", "required",
		"is required by a quarantine policy")
}

func validateUpsertEdgeIdentity(source Source, errs *ValidationErrors) {
	add := validationAdder(errs)
	switch source.Type {
	case SourceCSV:
		if source.CSV != nil {
			for index, edge := range source.CSV.Edges {
				add(edge.ExternalIDColumn != "", fmt.Sprintf("source.csv.edges[%d].externalIdColumn", index),
					"required", "is required for edge upsert")
			}
		}
	case SourcePostgreSQL:
		if source.PostgreSQL != nil {
			validateQueryEdgeIdentity(source.PostgreSQL.Edges, "source.postgresql.edges", errs)
		}
	case SourceNeo4j:
		if source.Neo4j != nil {
			validateQueryEdgeIdentity(source.Neo4j.Edges, "source.neo4j.edges", errs)
		}
	case SourceCosmos:
		if source.Cosmos != nil {
			for index, edge := range source.Cosmos.Edges {
				add(edge.ExternalIDField != "", fmt.Sprintf("source.cosmos.edges[%d].externalIdField", index),
					"required", "is required for edge upsert")
			}
		}
	}
}

func validateQueryEdgeIdentity(edges []EdgeQuery, path string, errs *ValidationErrors) {
	add := validationAdder(errs)
	for index, edge := range edges {
		add(edge.ExternalIDField != "", fmt.Sprintf("%s[%d].externalIdField", path, index),
			"required", "is required for edge upsert")
	}
}

func validationAdder(errs *ValidationErrors) func(bool, string, string, string) {
	return func(valid bool, path, code, message string) {
		if !valid {
			*errs = append(*errs, ValidationError{Path: path, Code: code, Message: message})
		}
	}
}
