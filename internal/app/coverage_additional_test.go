package app

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
	"go.yaml.in/yaml/v3"
)

func TestAdditionalAppPreOpenBranches(t *testing.T) {
	job := testLoadJob("graph", "vertices.csv", "edges.csv")
	job.Trial = &config.TrialOptions{Enabled: true}
	if _, err := execute(
		t.Context(), job, "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", true,
	); err == nil || !strings.Contains(err.Error(), "trial jobs cannot be resumed") {
		t.Fatalf("execute(trial resume) error = %v", err)
	}

	job.Trial = nil
	job.Target.Mode = config.LoadMode("future")
	if _, err := execute(
		t.Context(), job, "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", false,
	); err == nil || !strings.Contains(err.Error(), "not implemented") {
		t.Fatalf("execute(invalid mode) error = %v", err)
	}

	job.Target.Mode = config.LoadReplace
	if err := promoteReplace(
		t.Context(), nil, job,
		"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
		meta.GraphGeneration{},
		meta.ConnectorTelemetry{},
		meta.ConnectorTelemetry{},
	); err == nil {
		t.Fatal("promoteReplace() accepted multiple telemetry summaries")
	}

	job.Target.Mode = config.LoadCreate
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "job.yaml")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Cleanup(
		t.Context(), path, "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
	); err == nil || !strings.Contains(err.Error(), "replace load job") {
		t.Fatalf("Cleanup(create) error = %v", err)
	}
}

func TestAdditionalResolvedMappingBranches(t *testing.T) {
	job := testLoadJob("graph", "vertices.csv", "edges.csv")
	valid := []age.LoadLabel{
		{Generation: verificationLabel(7, "Person", meta.VertexLabel)},
		{Generation: verificationLabel(8, "KNOWS", meta.EdgeLabel)},
	}
	missingSource := job
	missingSource.Source.CSV = nil
	if _, err := resolvedMappingSummary(missingSource, valid); err == nil {
		t.Fatal("resolvedMappingSummary() accepted a missing source")
	}
	if _, err := resolvedIdentityCoverage(config.LoadJob{
		Source: config.Source{Type: "future"},
	}); err == nil {
		t.Fatal("resolvedIdentityCoverage() accepted an unsupported source")
	}

	invalid := append([]age.LoadLabel(nil), valid...)
	invalid[0].Generation.ID = 0
	if _, err := resolvedMappingSummary(job, invalid); err == nil {
		t.Fatal("resolvedMappingSummary() accepted incomplete label identity")
	}
	duplicate := []age.LoadLabel{valid[0], valid[0]}
	if _, err := resolvedMappingSummary(job, duplicate); err == nil {
		t.Fatal("resolvedMappingSummary() accepted duplicate labels")
	}
	if _, err := resolvedMappingSummary(job, valid[:1]); err == nil {
		t.Fatal("resolvedMappingSummary() accepted a missing label")
	}
	wrongKind := append([]age.LoadLabel(nil), valid...)
	wrongKind[0].Generation.Kind = meta.EdgeLabel
	if _, err := resolvedMappingSummary(job, wrongKind); err == nil {
		t.Fatal("resolvedMappingSummary() accepted a kind mismatch")
	}
}

func TestAdditionalProfileIteratorBranches(t *testing.T) {
	accumulator := newProfileAccumulator(profileLimits{rows: 10}, nil)
	accumulator.budget = sourcecontract.NewProfileBudget(
		sourcecontract.ProfileBudgetLimits{},
	)
	cancel := func(error) {}

	csvJob := testLoadJob("graph", "vertices.csv", "edges.csv")
	csvJob.Source.Namespace = ""
	if _, err := newProfileSourceIterator(
		t.Context(), csvJob, accumulator, cancel,
	); err == nil {
		t.Fatal("newProfileSourceIterator() accepted an empty CSV namespace")
	}

	const dsnEnv = "AGEFREIGHTER_PROFILE_INVALID_DSN"
	t.Setenv(dsnEnv, "postgres://example.invalid/database")
	postgresJob := config.LoadJob{
		Source: config.Source{
			Type:      config.SourcePostgreSQL,
			Namespace: "crm",
			PostgreSQL: &config.PostgreSQLSource{
				Connection: config.SecretRef{Env: dsnEnv},
				ReadMode:   "invalid",
				FetchRows:  10,
			},
		},
	}
	if _, err := newProfileSourceIterator(
		t.Context(), postgresJob, accumulator, cancel,
	); err == nil {
		t.Fatal("newProfileSourceIterator() accepted an invalid PostgreSQL mode")
	}

	neo4jJob := config.LoadJob{Source: config.Source{
		Type:      config.SourceNeo4j,
		Namespace: "crm",
		Neo4j: &config.Neo4jSource{
			URI: "not-a-uri", Database: "neo4j", FetchRows: 10,
		},
	}}
	if _, err := newProfileSourceIterator(
		t.Context(), neo4jJob, accumulator, cancel,
	); err == nil {
		t.Fatal("newProfileSourceIterator() accepted an invalid Neo4j URI")
	}

	cosmosJob := config.LoadJob{Source: config.Source{
		Type:      config.SourceCosmos,
		Namespace: "crm",
		Cosmos: &config.CosmosSource{
			Endpoint: "https://example.documents.azure.com:443/",
			Database: "database",
			PageSize: 0,
			Vertices: []config.CosmosVertexQuery{{
				Container: "items", Label: "Person",
				Query: "SELECT * FROM c", IDField: "/id",
			}},
		},
	}}
	if _, err := newProfileSourceIterator(
		t.Context(), cosmosJob, accumulator, cancel,
	); err == nil {
		t.Fatal("newProfileSourceIterator() accepted an invalid Cosmos page size")
	}
}

func TestProfileIteratorMalformedLimitAndSourceFailureReport(t *testing.T) {
	directory := t.TempDir()
	vertices := filepath.Join(directory, "vertices.csv")
	edges := filepath.Join(directory, "edges.csv")
	if err := os.WriteFile(vertices, []byte("id,name\n,Ada\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(edges, []byte("id,start,end\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	job := testLoadJob("graph", vertices, edges)
	accumulator := newProfileAccumulator(profileLimits{rows: 10}, nil)
	accumulator.malformed = maxProfileMalformedRows - 1
	accumulator.budget = sourcecontract.NewProfileBudget(
		sourcecontract.ProfileBudgetLimits{},
	)
	ctx, cancel := context.WithCancelCause(t.Context())
	iterator, err := newProfileSourceIterator(ctx, job, accumulator, cancel)
	if err != nil {
		t.Fatal(err)
	}
	defer iterator.Close()
	if _, err := iterator.Next(ctx); !errors.Is(err, errProfileLimit) {
		t.Fatalf("profile iterator Next() error = %v", err)
	}

	job.Source.CSV.Vertices[0].Path = filepath.Join(directory, "missing.csv")
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "missing-job.yaml")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	document, err := SourceProfile(t.Context(), path, ProfileOptions{})
	if err != nil {
		t.Fatalf("SourceProfile() error = %v", err)
	}
	if document.Command != "profile" || document.Outcome == "" {
		t.Fatalf("SourceProfile() = %#v", document)
	}
}

func TestAdditionalSourceIteratorPreOpenBranches(t *testing.T) {
	csvJob := testLoadJob("graph", "vertices.csv", "edges.csv")
	csvJob.Source.Namespace = ""
	if _, err := newSourceIterator(
		t.Context(), csvJob, "", nil,
	); err == nil {
		t.Fatal("newSourceIterator() accepted an empty CSV namespace")
	}

	cosmosJob := config.LoadJob{Source: config.Source{
		Type:      config.SourceCosmos,
		Namespace: "crm",
		Cosmos: &config.CosmosSource{
			Endpoint: "https://example.documents.azure.com:443/",
			Database: "database",
			PageSize: 0,
			Vertices: []config.CosmosVertexQuery{{
				Container: "items", Label: "Person",
				Query: "SELECT * FROM c", IDField: "/id",
			}},
		},
	}}
	if _, err := newSourceIterator(
		t.Context(), cosmosJob, "", nil,
	); err == nil {
		t.Fatal("newSourceIterator() accepted an invalid Cosmos page size")
	}

	neo4jJob := config.LoadJob{Source: config.Source{
		Type: config.SourceNeo4j,
		Neo4j: &config.Neo4jSource{
			URI: "not-a-uri", Database: "neo4j", FetchRows: 10,
		},
	}}
	if _, err := resolveNeo4jDiscovery(
		t.Context(), neo4jJob, nil,
	); err != nil {
		t.Fatalf("disabled resolveNeo4jDiscovery() error = %v", err)
	}
	neo4jJob.Source.Neo4j.Discovery = &config.Neo4jDiscovery{Enabled: true}
	if _, err := resolveNeo4jDiscovery(
		t.Context(), neo4jJob, nil,
	); err == nil {
		t.Fatal("resolveNeo4jDiscovery() accepted an invalid URI")
	}
}

func TestAdditionalProfilePureBranches(t *testing.T) {
	accumulator := newProfileAccumulator(
		profileLimits{},
		[]profileMapping{{
			kind: model.RecordVertex, label: "Person", dynamic: true,
		}},
	)
	record := model.VertexRecord(model.Vertex{
		Label: "Person",
		Properties: model.Properties{
			"name": {Kind: model.ValueString, String: "Ada"},
		},
	})
	accumulator.add(record, 1)
	accumulator.add(record, 1)

	full := newProfileAccumulator(
		profileLimits{},
		[]profileMapping{{
			kind: model.RecordVertex, label: "Person", dynamic: true,
		}},
	)
	for index := 0; index < maxProfileProperties; index++ {
		full.properties[profilePropertyKey{
			kind: model.RecordVertex, label: "Existing",
			property: string(rune(index + 1)),
		}] = &profilePropertyStats{}
	}
	full.add(record, 1)
	if !full.propertiesTruncated {
		t.Fatal("dynamic property hard cap was not reported")
	}

	properties := make(map[string]string, maxProfileProperties+1)
	for index := 0; index < maxProfileProperties+1; index++ {
		name := string(rune(0x1000 + index))
		properties[name] = name
	}
	mappings, _, truncated := profileMappings(config.LoadJob{Source: config.Source{
		Type: config.SourceCosmos,
		Cosmos: &config.CosmosSource{Edges: []config.CosmosEdgeQuery{
			{
				Label:      "CONNECTS",
				Start:      config.EndpointMapping{Label: "A"},
				End:        config.EndpointMapping{Label: "Z"},
				Properties: properties,
			},
			{
				Label:      "CONNECTS",
				Start:      config.EndpointMapping{Label: "B"},
				End:        config.EndpointMapping{Label: "A"},
				Properties: map[string]string{"extra": "extra"},
			},
			{
				Label: "CONNECTS",
				Start: config.EndpointMapping{Label: "B"},
				End:   config.EndpointMapping{Label: "B"},
			},
		}},
	}})
	if len(mappings) != 3 || !truncated {
		t.Fatalf("profileMappings() = %d, truncated=%t", len(mappings), truncated)
	}

	for _, job := range []config.LoadJob{
		{Source: config.Source{
			Type: config.SourcePostgreSQL,
			PostgreSQL: &config.PostgreSQLSource{
				ReadMode: config.PostgreSQLReadKeyset,
				Vertices: []config.VertexQuery{{Query: "SELECT 1"}},
			},
		}},
		{Source: config.Source{
			Type: config.SourcePostgreSQL,
			PostgreSQL: &config.PostgreSQLSource{
				ReadMode: config.PostgreSQLReadKeyset,
				Edges:    []config.EdgeQuery{{Query: "SELECT 1"}},
			},
		}},
		{Source: config.Source{
			Type:  config.SourceNeo4j,
			Neo4j: &config.Neo4jSource{Vertices: []config.VertexQuery{{Query: "RETURN 1"}}},
		}},
		{Source: config.Source{
			Type:  config.SourceNeo4j,
			Neo4j: &config.Neo4jSource{Edges: []config.EdgeQuery{{Query: "RETURN 1"}}},
		}},
	} {
		if profileDeterministic(job) {
			t.Fatalf("profileDeterministic(%s) = true", job.Source.Type)
		}
	}

	sectionAccumulator := newProfileAccumulator(profileLimits{}, nil)
	sectionAccumulator.properties[profilePropertyKey{
		kind: model.RecordVertex, label: "B", property: "value",
	}] = &profilePropertyStats{observed: 1, distinctLimit: true}
	sectionAccumulator.properties[profilePropertyKey{
		kind: model.RecordVertex, label: "A", property: "value",
	}] = &profilePropertyStats{observed: 1}
	section := profilePropertySection(profileRun{accumulator: sectionAccumulator})
	if len(section.Fields) != 2 {
		t.Fatalf("profilePropertySection() = %#v", section)
	}
	emptySection := profilePropertySection(profileRun{
		accumulator: newProfileAccumulator(profileLimits{}, nil),
	})
	if len(emptySection.Fields) != 1 {
		t.Fatalf("empty profilePropertySection() = %#v", emptySection)
	}
	if got := profileConnectorMode(config.LoadJob{Source: config.Source{
		Type: config.SourceCosmos, Cosmos: &config.CosmosSource{},
	}}); got != "nosql" {
		t.Fatalf("profileConnectorMode() = %q", got)
	}
}
