package config

const (
	APIVersion  = "agefreighter.io/v2"
	KindLoadJob = "LoadJob"
)

type SourceType string

const (
	SourceCSV        SourceType = "csv"
	SourcePostgreSQL SourceType = "postgresql"
	SourceNeo4j      SourceType = "neo4j"
	SourceCosmos     SourceType = "cosmos-nosql"
)

type TargetType string

const TargetApacheAGE TargetType = "apache-age"

type LoadMode string

const (
	LoadCreate  LoadMode = "create"
	LoadReplace LoadMode = "replace"
	LoadAppend  LoadMode = "append"
	LoadUpsert  LoadMode = "upsert"
)

type PropertyMode string

const (
	PropertiesReplace         PropertyMode = "replace"
	PropertiesMerge           PropertyMode = "merge"
	PropertiesMergeDeleteNull PropertyMode = "merge-delete-null"
)

type MalformedRecordPolicy string

const (
	MalformedFail       MalformedRecordPolicy = "fail"
	MalformedQuarantine MalformedRecordPolicy = "quarantine"
)

type MissingEndpointPolicy string

const (
	MissingEndpointError      MissingEndpointPolicy = "error"
	MissingEndpointQuarantine MissingEndpointPolicy = "quarantine"
	MissingEndpointDefer      MissingEndpointPolicy = "defer"
)

type LoadJob struct {
	APIVersion string        `json:"apiVersion" yaml:"apiVersion"`
	Kind       string        `json:"kind" yaml:"kind"`
	Metadata   Metadata      `json:"metadata" yaml:"metadata"`
	Source     Source        `json:"source" yaml:"source"`
	Target     Target        `json:"target" yaml:"target"`
	Runtime    Runtime       `json:"runtime" yaml:"runtime"`
	Errors     ErrorPolicies `json:"errors" yaml:"errors"`
}

type Metadata struct {
	Name string `json:"name" yaml:"name"`
}

type Source struct {
	Type       SourceType        `json:"type" yaml:"type"`
	Namespace  string            `json:"namespace" yaml:"namespace"`
	CSV        *CSVSource        `json:"csv,omitempty" yaml:"csv,omitempty"`
	PostgreSQL *PostgreSQLSource `json:"postgresql,omitempty" yaml:"postgresql,omitempty"`
	Neo4j      *Neo4jSource      `json:"neo4j,omitempty" yaml:"neo4j,omitempty"`
	Cosmos     *CosmosSource     `json:"cosmos,omitempty" yaml:"cosmos,omitempty"`
}

type CSVSource struct {
	Defaults DelimitedOptions `json:"defaults" yaml:"defaults"`
	Vertices []CSVVertex      `json:"vertices" yaml:"vertices"`
	Edges    []CSVEdge        `json:"edges,omitempty" yaml:"edges,omitempty"`
}

type DelimitedOptions struct {
	Delimiter string  `json:"delimiter" yaml:"delimiter"`
	Quote     string  `json:"quote" yaml:"quote"`
	Escape    string  `json:"escape" yaml:"escape"`
	Header    *bool   `json:"header,omitempty" yaml:"header,omitempty"`
	Encoding  string  `json:"encoding" yaml:"encoding"`
	NullValue *string `json:"nullValue,omitempty" yaml:"nullValue,omitempty"`
}

type CSVVertex struct {
	Label      string            `json:"label" yaml:"label"`
	Path       string            `json:"path" yaml:"path"`
	IDColumn   string            `json:"idColumn" yaml:"idColumn"`
	Properties map[string]string `json:"properties,omitempty" yaml:"properties,omitempty"`
	Format     *DelimitedOptions `json:"format,omitempty" yaml:"format,omitempty"`
}

type CSVEdge struct {
	Label            string            `json:"label" yaml:"label"`
	Path             string            `json:"path" yaml:"path"`
	ExternalIDColumn string            `json:"externalIdColumn,omitempty" yaml:"externalIdColumn,omitempty"`
	Start            EndpointMapping   `json:"start" yaml:"start"`
	End              EndpointMapping   `json:"end" yaml:"end"`
	Properties       map[string]string `json:"properties,omitempty" yaml:"properties,omitempty"`
	Format           *DelimitedOptions `json:"format,omitempty" yaml:"format,omitempty"`
}

type EndpointMapping struct {
	Label     string `json:"label" yaml:"label"`
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`
	Field     string `json:"field" yaml:"field"`
}

type PostgreSQLSource struct {
	Connection SecretRef     `json:"connection" yaml:"connection"`
	Vertices   []VertexQuery `json:"vertices" yaml:"vertices"`
	Edges      []EdgeQuery   `json:"edges,omitempty" yaml:"edges,omitempty"`
}

type Neo4jSource struct {
	URI      string        `json:"uri" yaml:"uri"`
	Database string        `json:"database" yaml:"database"`
	Username string        `json:"username,omitempty" yaml:"username,omitempty"`
	Password *SecretRef    `json:"password,omitempty" yaml:"password,omitempty"`
	Vertices []VertexQuery `json:"vertices" yaml:"vertices"`
	Edges    []EdgeQuery   `json:"edges,omitempty" yaml:"edges,omitempty"`
}

type CosmosSource struct {
	Endpoint   string              `json:"endpoint" yaml:"endpoint"`
	Credential string              `json:"credential" yaml:"credential"`
	Database   string              `json:"database" yaml:"database"`
	PageSize   int                 `json:"pageSize,omitempty" yaml:"pageSize,omitempty"`
	Vertices   []CosmosVertexQuery `json:"vertices" yaml:"vertices"`
	Edges      []CosmosEdgeQuery   `json:"edges,omitempty" yaml:"edges,omitempty"`
}

// CosmosQueryParameter binds a named parameter (which must use the Cosmos DB
// "@name" convention) to a strict JSON value for a parametrized query.
type CosmosQueryParameter struct {
	Name  string           `json:"name" yaml:"name"`
	Value CosmosParamValue `json:"value" yaml:"value"`
}

type VertexQuery struct {
	Label      string            `json:"label" yaml:"label"`
	Query      string            `json:"query" yaml:"query"`
	IDField    string            `json:"idField" yaml:"idField"`
	Properties map[string]string `json:"properties,omitempty" yaml:"properties,omitempty"`
}

type EdgeQuery struct {
	Label           string            `json:"label" yaml:"label"`
	Query           string            `json:"query" yaml:"query"`
	ExternalIDField string            `json:"externalIdField,omitempty" yaml:"externalIdField,omitempty"`
	Start           EndpointMapping   `json:"start" yaml:"start"`
	End             EndpointMapping   `json:"end" yaml:"end"`
	Properties      map[string]string `json:"properties,omitempty" yaml:"properties,omitempty"`
}

// CosmosVertexQuery describes a Cosmos DB for NoSQL vertex source query.
// IDField, and the values of Properties, are RFC 6901 JSON Pointers into
// each returned document.
type CosmosVertexQuery struct {
	Container  string                 `json:"container" yaml:"container"`
	Label      string                 `json:"label" yaml:"label"`
	Query      string                 `json:"query" yaml:"query"`
	Parameters []CosmosQueryParameter `json:"parameters,omitempty" yaml:"parameters,omitempty"`
	IDField    string                 `json:"idField" yaml:"idField"`
	Properties map[string]string      `json:"properties,omitempty" yaml:"properties,omitempty"`
}

// CosmosEdgeQuery describes a Cosmos DB for NoSQL edge source query.
// ExternalIDField, Start.Field, End.Field, and the values of Properties, are
// RFC 6901 JSON Pointers into each returned document.
type CosmosEdgeQuery struct {
	Container       string                 `json:"container" yaml:"container"`
	Label           string                 `json:"label" yaml:"label"`
	Query           string                 `json:"query" yaml:"query"`
	Parameters      []CosmosQueryParameter `json:"parameters,omitempty" yaml:"parameters,omitempty"`
	ExternalIDField string                 `json:"externalIdField,omitempty" yaml:"externalIdField,omitempty"`
	Start           EndpointMapping        `json:"start" yaml:"start"`
	End             EndpointMapping        `json:"end" yaml:"end"`
	Properties      map[string]string      `json:"properties,omitempty" yaml:"properties,omitempty"`
}

type Target struct {
	Type         TargetType   `json:"type" yaml:"type"`
	Graph        string       `json:"graph" yaml:"graph"`
	Mode         LoadMode     `json:"mode" yaml:"mode"`
	Connection   SecretRef    `json:"connection" yaml:"connection"`
	PropertyMode PropertyMode `json:"propertyMode" yaml:"propertyMode"`
}

type SecretRef struct {
	Env  string `json:"env,omitempty" yaml:"env,omitempty"`
	File string `json:"file,omitempty" yaml:"file,omitempty"`
}

type Runtime struct {
	MemoryLimit             ByteSize `json:"memoryLimit" yaml:"memoryLimit"`
	BatchRows               int      `json:"batchRows" yaml:"batchRows"`
	BatchBytes              ByteSize `json:"batchBytes" yaml:"batchBytes"`
	MaxSourceConcurrency    int      `json:"maxSourceConcurrency" yaml:"maxSourceConcurrency"`
	MaxTransformConcurrency int      `json:"maxTransformConcurrency" yaml:"maxTransformConcurrency"`
	MaxTargetConnections    int      `json:"maxTargetConnections" yaml:"maxTargetConnections"`
	OperationTimeout        Duration `json:"operationTimeout" yaml:"operationTimeout"`
}

type ErrorPolicies struct {
	MalformedRecord MalformedRecordPolicy `json:"malformedRecord" yaml:"malformedRecord"`
	MissingEndpoint MissingEndpointPolicy `json:"missingEndpoint" yaml:"missingEndpoint"`
	RejectLimit     int                   `json:"rejectLimit" yaml:"rejectLimit"`
	QuarantinePath  string                `json:"quarantinePath,omitempty" yaml:"quarantinePath,omitempty"`
}
