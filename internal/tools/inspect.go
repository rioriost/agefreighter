package tools

import (
	"fmt"
	"slices"

	"github.com/rioriost/agefreighter/internal/config"
)

const InspectionFormatVersion = 1

type Inspection struct {
	FormatVersion int              `json:"formatVersion"`
	Job           string           `json:"job"`
	Source        SourceInspection `json:"source"`
	Target        TargetInspection `json:"target"`
	Warnings      []string         `json:"warnings,omitempty"`
}

type SourceInspection struct {
	Type           config.SourceType            `json:"type"`
	Namespace      string                       `json:"namespace"`
	Consistency    string                       `json:"consistency"`
	ReadMode       config.PostgreSQLReadMode    `json:"readMode,omitempty"`
	FetchRows      int                          `json:"fetchRows,omitempty"`
	MultiLabel     config.Neo4jMultiLabelPolicy `json:"multiLabelPolicy,omitempty"`
	Database       string                       `json:"database,omitempty"`
	SourceID       string                       `json:"sourceId,omitempty"`
	PageSize       int                          `json:"pageSize,omitempty"`
	Credential     string                       `json:"credential,omitempty"`
	Connection     string                       `json:"connection,omitempty"`
	VertexMappings []MappingInspection          `json:"vertexMappings"`
	EdgeMappings   []MappingInspection          `json:"edgeMappings"`
}

type TargetInspection struct {
	Type            config.TargetType            `json:"type"`
	Graph           string                       `json:"graph"`
	Mode            config.LoadMode              `json:"mode"`
	PropertyMode    config.PropertyMode          `json:"propertyMode"`
	AppendDuplicate config.AppendDuplicatePolicy `json:"appendDuplicate,omitempty"`
	Connection      string                       `json:"connection"`
}

type MappingInspection struct {
	Label          string   `json:"label"`
	Location       string   `json:"location,omitempty"`
	IdentityField  string   `json:"identityField,omitempty"`
	ResumeKey      string   `json:"resumeKey,omitempty"`
	StartLabel     string   `json:"startLabel,omitempty"`
	StartNamespace string   `json:"startNamespace,omitempty"`
	StartField     string   `json:"startField,omitempty"`
	EndLabel       string   `json:"endLabel,omitempty"`
	EndNamespace   string   `json:"endNamespace,omitempty"`
	EndField       string   `json:"endField,omitempty"`
	PropertyFields []string `json:"propertyFields"`
}

func InspectConfiguration(path string) (Inspection, error) {
	job, err := config.Load(path)
	if err != nil {
		return Inspection{}, fmt.Errorf("inspect job: %w", err)
	}
	return BuildInspection(job), nil
}

func BuildInspection(job config.LoadJob) Inspection {
	plan := config.BuildStaticPlan(job)
	inspection := Inspection{
		FormatVersion: InspectionFormatVersion,
		Job:           job.Metadata.Name,
		Source: SourceInspection{
			Type:        job.Source.Type,
			Namespace:   job.Source.Namespace,
			Consistency: plan.Source.Consistency,
		},
		Target: TargetInspection{
			Type:            job.Target.Type,
			Graph:           job.Target.Graph,
			Mode:            job.Target.Mode,
			PropertyMode:    job.Target.PropertyMode,
			AppendDuplicate: job.Target.AppendDuplicate,
			Connection:      secretKind(job.Target.Connection),
		},
		Warnings: slices.Clone(plan.Warnings),
	}
	switch job.Source.Type {
	case config.SourceCSV:
		inspection.Source.Consistency = "files-at-open-time"
		buildCSVInspection(&inspection.Source, job.Source.CSV)
	case config.SourcePostgreSQL:
		buildPostgreSQLInspection(&inspection.Source, job.Source.PostgreSQL)
	case config.SourceNeo4j:
		buildNeo4jInspection(&inspection.Source, job.Source.Neo4j)
	case config.SourceCosmos:
		buildCosmosInspection(&inspection.Source, job.Source.Cosmos)
	}
	return inspection
}

func buildCSVInspection(target *SourceInspection, source *config.CSVSource) {
	if source == nil {
		return
	}
	for _, mapping := range source.Vertices {
		target.VertexMappings = append(target.VertexMappings, MappingInspection{
			Label:          mapping.Label,
			Location:       mapping.Path,
			IdentityField:  mapping.IDColumn,
			PropertyFields: sortedValues(mapping.Properties),
		})
	}
	for _, mapping := range source.Edges {
		target.EdgeMappings = append(target.EdgeMappings, edgeInspection(
			target.Namespace,
			mapping.Label,
			mapping.Path,
			mapping.ExternalIDColumn,
			"",
			mapping.Start,
			mapping.End,
			mapping.Properties,
		))
	}
}

func buildPostgreSQLInspection(
	target *SourceInspection,
	source *config.PostgreSQLSource,
) {
	if source == nil {
		return
	}
	target.ReadMode = source.ReadMode
	target.FetchRows = source.FetchRows
	target.Connection = secretKind(source.Connection)
	for _, mapping := range source.Vertices {
		target.VertexMappings = append(target.VertexMappings, queryVertexInspection(mapping))
	}
	for _, mapping := range source.Edges {
		target.EdgeMappings = append(target.EdgeMappings, edgeInspection(
			target.Namespace,
			mapping.Label,
			"",
			mapping.ExternalIDField,
			mapping.KeyField,
			mapping.Start,
			mapping.End,
			mapping.Properties,
		))
	}
}

func buildNeo4jInspection(target *SourceInspection, source *config.Neo4jSource) {
	if source == nil {
		return
	}
	target.Database = source.Database
	target.SourceID = source.SourceID
	target.FetchRows = source.FetchRows
	target.MultiLabel = source.MultiLabelPolicy
	if source.Password != nil {
		target.Credential = secretKind(*source.Password)
	}
	for _, mapping := range source.Vertices {
		target.VertexMappings = append(target.VertexMappings, queryVertexInspection(mapping))
	}
	for _, mapping := range source.Edges {
		target.EdgeMappings = append(target.EdgeMappings, edgeInspection(
			target.Namespace,
			mapping.Label,
			"",
			mapping.ExternalIDField,
			mapping.KeyField,
			mapping.Start,
			mapping.End,
			mapping.Properties,
		))
	}
}

func buildCosmosInspection(target *SourceInspection, source *config.CosmosSource) {
	if source == nil {
		return
	}
	target.Database = source.Database
	target.PageSize = source.PageSize
	target.Credential = source.Credential
	target.Consistency = "connector-verified"
	for _, mapping := range source.Vertices {
		target.VertexMappings = append(target.VertexMappings, MappingInspection{
			Label:          mapping.Label,
			Location:       mapping.Container,
			IdentityField:  mapping.IDField,
			PropertyFields: sortedValues(mapping.Properties),
		})
	}
	for _, mapping := range source.Edges {
		target.EdgeMappings = append(target.EdgeMappings, edgeInspection(
			target.Namespace,
			mapping.Label,
			mapping.Container,
			mapping.ExternalIDField,
			"",
			mapping.Start,
			mapping.End,
			mapping.Properties,
		))
	}
}

func queryVertexInspection(mapping config.VertexQuery) MappingInspection {
	return MappingInspection{
		Label:          mapping.Label,
		IdentityField:  mapping.IDField,
		ResumeKey:      mapping.KeyField,
		PropertyFields: sortedValues(mapping.Properties),
	}
}

func edgeInspection(
	defaultNamespace string,
	label string,
	location string,
	identity string,
	resumeKey string,
	start config.EndpointMapping,
	end config.EndpointMapping,
	properties map[string]string,
) MappingInspection {
	return MappingInspection{
		Label:          label,
		Location:       location,
		IdentityField:  identity,
		ResumeKey:      resumeKey,
		StartLabel:     start.Label,
		StartNamespace: endpointNamespace(defaultNamespace, start.Namespace),
		StartField:     start.Field,
		EndLabel:       end.Label,
		EndNamespace:   endpointNamespace(defaultNamespace, end.Namespace),
		EndField:       end.Field,
		PropertyFields: sortedValues(properties),
	}
}

func endpointNamespace(defaultNamespace, configured string) string {
	if configured != "" {
		return configured
	}
	return defaultNamespace
}

func sortedValues(properties map[string]string) []string {
	values := make([]string, 0, len(properties))
	for _, sourceField := range properties {
		values = append(values, sourceField)
	}
	slices.Sort(values)
	return values
}

func secretKind(reference config.SecretRef) string {
	if reference.Env != "" {
		return "environment"
	}
	if reference.File != "" {
		return "file"
	}
	return "none"
}
