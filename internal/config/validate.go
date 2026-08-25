package config

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
	"unicode/utf8"
)

const maxConcurrency = 256

var (
	jobNamePattern   = regexp.MustCompile(`^[a-z][a-z0-9-]{2,62}$`)
	graphNamePattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.-]*[A-Za-z0-9_]$`)
	envNamePattern   = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
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
	validateErrorPolicies(job.Errors, &errs)
	if job.Target.Mode == LoadUpsert {
		validateUpsertEdgeIdentity(job.Source, &errs)
	}

	if len(errs) != 0 {
		return errs
	}
	return nil
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
			validateSecret(source.PostgreSQL.Connection, "source.postgresql.connection", errs)
			validateQueries(source.PostgreSQL.Vertices, source.PostgreSQL.Edges, source.Namespace, errs)
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
	add(strings.HasPrefix(source.URI, "neo4j://") || strings.HasPrefix(source.URI, "neo4j+s://") ||
		strings.HasPrefix(source.URI, "bolt://") || strings.HasPrefix(source.URI, "bolt+s://"),
		"source.neo4j.uri", "format", "must use neo4j, neo4j+s, bolt, or bolt+s")
	add(source.Database != "", "source.neo4j.database", "required", "must not be empty")
	add((source.Username == "") == (source.Password == nil), "source.neo4j", "authentication",
		"username and password must either both be set or both be omitted")
	if source.Password != nil {
		validateSecret(*source.Password, "source.neo4j.password", errs)
	}
	validateQueries(source.Vertices, source.Edges, namespace, errs)
}

func validateCosmos(source CosmosSource, namespace string, errs *ValidationErrors) {
	add := validationAdder(errs)
	add(strings.HasPrefix(source.Endpoint, "https://"), "source.cosmos.endpoint", "format",
		"must use https")
	add(source.Credential == "default-azure", "source.cosmos.credential", "unsupported",
		"must be default-azure")
	add(source.Database != "", "source.cosmos.database", "required", "must not be empty")
	add(len(source.Vertices) > 0, "source.cosmos.vertices", "required",
		"must contain at least one vertex query")
	for index, vertex := range source.Vertices {
		path := fmt.Sprintf("source.cosmos.vertices[%d]", index)
		add(vertex.Container != "", path+".container", "required", "must not be empty")
		add(vertex.Label != "", path+".label", "required", "must not be empty")
		add(vertex.Query != "", path+".query", "required", "must not be empty")
		add(vertex.IDField != "", path+".idField", "required", "must not be empty")
		validatePropertyMapping(vertex.Properties, path+".properties", errs)
	}
	for index, edge := range source.Edges {
		path := fmt.Sprintf("source.cosmos.edges[%d]", index)
		add(edge.Container != "", path+".container", "required", "must not be empty")
		add(edge.Label != "", path+".label", "required", "must not be empty")
		add(edge.Query != "", path+".query", "required", "must not be empty")
		validateEndpoint(edge.Start, namespace, path+".start", errs)
		validateEndpoint(edge.End, namespace, path+".end", errs)
		validatePropertyMapping(edge.Properties, path+".properties", errs)
	}
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
	add(target.Type == TargetApacheAGE, "target.type", "unsupported", "must be apache-age")
	add(len(target.Graph) >= 3 && len(target.Graph) <= 63 && graphNamePattern.MatchString(target.Graph),
		"target.graph", "format", "must satisfy the Apache AGE 3-63 byte graph-name rules")
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
	validateSecret(target.Connection, "target.connection", errs)
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
	validateConcurrency(runtime.MaxSourceConcurrency, "runtime.maxSourceConcurrency", errs)
	validateConcurrency(runtime.MaxTransformConcurrency, "runtime.maxTransformConcurrency", errs)
	validateConcurrency(runtime.MaxTargetConnections, "runtime.maxTargetConnections", errs)
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
