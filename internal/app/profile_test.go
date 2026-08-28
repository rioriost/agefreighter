package app

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/report"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
	"go.yaml.in/yaml/v3"
)

func TestSourceProfileCSVIsBoundedSourceOnlyAndRedacted(t *testing.T) {
	directory := t.TempDir()
	vertices := filepath.Join(directory, "vertices.csv")
	edges := filepath.Join(directory, "edges.csv")
	if err := os.WriteFile(
		vertices,
		[]byte("id,name,score\np1,Alice,1\np2,,2\np3,Carol,3\n"),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		edges,
		[]byte("id,start,end,since\ne1,p1,p2,2020\n"),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	job := testLoadJob("unreachable_target", vertices, edges)
	job.Source.CSV.Vertices[0].Properties = map[string]string{
		"name": "name", "score": "score",
	}
	job.Source.CSV.Edges[0].Properties = map[string]string{"since": "since"}
	job.Target.Connection = config.SecretRef{Env: "PROFILE_TARGET_MUST_NOT_BE_READ"}
	job.Errors.MalformedRecord = config.MalformedQuarantine
	job.Errors.RejectLimit = 10
	job.Errors.QuarantinePath = filepath.Join(directory, "must-not-exist.jsonl")
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "job.yaml")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}

	document, err := SourceProfile(t.Context(), path, ProfileOptions{
		Mode: ProfileSample, SampleSize: 2,
		GeneratedAt: time.Date(2026, 8, 28, 0, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatalf("SourceProfile() error = %v", err)
	}
	if document.Command != "profile" || document.Outcome != report.OutcomeIncomplete {
		t.Fatalf("SourceProfile() = %#v", document)
	}
	rendered, err := report.Render(document, report.FormatJSON)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{
		"Alice", "Carol", "p1", "p2", "PROFILE_TARGET_MUST_NOT_BE_READ",
		vertices, edges,
	} {
		if strings.Contains(string(rendered), secret) {
			t.Fatalf("profile disclosed %q:\n%s", secret, rendered)
		}
	}
	if _, err := os.Stat(job.Errors.QuarantinePath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("profile created quarantine output: %v", err)
	}
	var parsed struct {
		Sections []report.Section `json:"sections"`
	}
	if err := json.Unmarshal(rendered, &parsed); err != nil {
		t.Fatal(err)
	}
	signals := sectionByTitle(t, parsed.Sections, "Mapping signals")
	if fieldByName(t, signals.Fields, "sampledRows").Value != "2" {
		t.Fatalf("signals = %#v", signals)
	}
	properties := sectionByTitle(t, parsed.Sections, "Property observations")
	if len(properties.Fields) != 3 ||
		!strings.Contains(properties.Fields[0].Value+properties.Fields[1].Value, "null=1") {
		t.Fatalf("property observations = %#v", properties)
	}
	telemetry := sectionByTitle(t, parsed.Sections, "Connector telemetry")
	if fieldByName(t, telemetry.Fields, "rawInputBytes").Value == "0" ||
		fieldByName(t, telemetry.Fields, "decodedInputBytes").Value == "0" {
		t.Fatalf("connector telemetry = %#v", telemetry)
	}
}

func TestProfileMappingInventoryConnectorMatrix(t *testing.T) {
	tests := []struct {
		name string
		job  config.LoadJob
		mode string
	}{
		{
			name: "csv",
			job: config.LoadJob{Source: config.Source{
				Type: config.SourceCSV,
				CSV: &config.CSVSource{
					Vertices: []config.CSVVertex{{Label: "Person"}},
					Edges: []config.CSVEdge{{
						Label: "KNOWS",
						Start: config.EndpointMapping{Label: "Person"},
						End:   config.EndpointMapping{Label: "Person"},
					}},
				},
			}},
			mode: "delimited",
		},
		{
			name: "postgresql",
			job: config.LoadJob{Source: config.Source{
				Type: config.SourcePostgreSQL,
				PostgreSQL: &config.PostgreSQLSource{
					ReadMode: config.PostgreSQLReadCursor,
					Vertices: []config.VertexQuery{{Label: "Person"}},
					Edges: []config.EdgeQuery{{
						Label: "KNOWS",
						Start: config.EndpointMapping{Label: "Person"},
						End:   config.EndpointMapping{Label: "Person"},
					}},
				},
			}},
			mode: "cursor",
		},
		{
			name: "neo4j",
			job: config.LoadJob{Source: config.Source{
				Type: config.SourceNeo4j,
				Neo4j: &config.Neo4jSource{
					Discovery: &config.Neo4jDiscovery{Enabled: true},
				},
			}},
			mode: "discovered-cypher",
		},
		{
			name: "cosmos",
			job: config.LoadJob{Source: config.Source{
				Type: config.SourceCosmos,
				Cosmos: &config.CosmosSource{
					Gremlin: &config.CosmosGremlin{Enabled: true},
				},
			}},
			mode: "gremlin-nosql",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := profileConnectorMode(test.job); got != test.mode {
				t.Fatalf("profileConnectorMode() = %q, want %q", got, test.mode)
			}
			if test.name == "cosmos" && profileDeterministic(test.job) {
				t.Fatal("Cosmos Gremlin discovery was reported as deterministic")
			}
			if test.name == "neo4j" || test.name == "cosmos" {
				return
			}
			mappings, truncated, propertiesTruncated := profileMappings(test.job)
			if len(mappings) != 2 || truncated || propertiesTruncated {
				t.Fatalf("profileMappings() = %#v, %t, %t", mappings, truncated, propertiesTruncated)
			}
		})
	}
}

func TestProfileAccumulatorTypedFactsAndCaps(t *testing.T) {
	mappings := []profileMapping{{
		kind: model.RecordVertex, label: "Person",
		properties: []string{"active", "name", "tags"},
	}}
	accumulator := newProfileAccumulator(profileLimits{
		rows: 10, bytes: 1 << 20, pages: 10, requestCharge: 10,
	}, mappings)
	accumulator.add(model.VertexRecord(model.Vertex{
		Label: "Person",
		Properties: model.Properties{
			"active": {Kind: model.ValueBoolean, Boolean: true},
			"name":   {Kind: model.ValueString, String: "sensitive"},
			"tags": {Kind: model.ValueList, List: []model.Value{{
				Kind: model.ValueString, String: "private",
			}}},
		},
	}), 100)
	accumulator.add(model.VertexRecord(model.Vertex{
		Label: "Person",
		Properties: model.Properties{
			"active": {Kind: model.ValueBoolean, Boolean: false},
			"name":   {Kind: model.ValueNull},
		},
	}), 80)
	accumulator.malformedRow(errors.New("edge endpoints must not be null"))
	if accumulator.rows != 2 || accumulator.bytes != 180 ||
		accumulator.malformed != 1 || accumulator.missingEnds != 1 {
		t.Fatalf("accumulator = %#v", accumulator)
	}
	name := accumulator.properties[profilePropertyKey{
		kind: model.RecordVertex, label: "Person", property: "name",
	}]
	if name.observed != 2 || name.present != 2 || name.nulls != 1 ||
		name.typeCounts[model.ValueString] != 1 ||
		name.typeCounts[model.ValueNull] != 1 ||
		len(name.distinct) != 1 {
		t.Fatalf("name stats = %#v", name)
	}
}

func TestBoundedProfileJobDoesNotMutateDefinition(t *testing.T) {
	properties := make(map[string]string)
	for index := 0; index < maxProfileProperties+1; index++ {
		name := "property-" + string(rune(0x1000+index))
		properties[name] = "field-" + string(rune(0x1000+index))
	}
	vertices := make([]config.VertexQuery, maxProfileLabels+1)
	for index := range vertices {
		vertices[index] = config.VertexQuery{
			Label: "Label-" + string(rune(0x1000+index)),
		}
	}
	vertices[0].Properties = properties
	source := &config.PostgreSQLSource{Vertices: vertices}
	job := config.LoadJob{Source: config.Source{
		Type: config.SourcePostgreSQL, PostgreSQL: source,
	}}

	bounded, mappingsTruncated, propertiesTruncated := boundedProfileJob(job)
	if !mappingsTruncated || !propertiesTruncated ||
		len(bounded.Source.PostgreSQL.Vertices) != maxProfileLabels ||
		len(bounded.Source.PostgreSQL.Vertices[0].Properties) != maxProfileProperties {
		t.Fatalf(
			"boundedProfileJob() = mappings=%d properties=%d truncated=(%t,%t)",
			len(bounded.Source.PostgreSQL.Vertices),
			len(bounded.Source.PostgreSQL.Vertices[0].Properties),
			mappingsTruncated,
			propertiesTruncated,
		)
	}
	if len(job.Source.PostgreSQL.Vertices) != maxProfileLabels+1 ||
		len(job.Source.PostgreSQL.Vertices[0].Properties) != maxProfileProperties+1 {
		t.Fatal("boundedProfileJob() mutated the supplied job definition")
	}
}

func TestProfileAccumulatorDiscoversBoundedGremlinProperties(t *testing.T) {
	mapping := profileMapping{
		kind: model.RecordVertex, label: "Person", dynamic: true,
	}
	accumulator := newProfileAccumulator(profileLimits{
		rows: 10, bytes: 1 << 20, pages: 10, requestCharge: 10,
	}, []profileMapping{mapping})
	accumulator.add(model.VertexRecord(model.Vertex{
		Label: "Person",
		Properties: model.Properties{
			"name": {Kind: model.ValueString, String: "secret"},
			"age":  {Kind: model.ValueInteger, Integer: 42},
		},
	}), 50)
	if len(accumulator.properties) != 2 || accumulator.propertiesTruncated {
		t.Fatalf("dynamic properties = %#v", accumulator.properties)
	}
}

func TestProfileAccumulatorDynamicEdgePropertyUsesLabelWideDenominator(t *testing.T) {
	accumulator := newProfileAccumulator(profileLimits{
		rows: 10, bytes: 1 << 20, pages: 10, requestCharge: 10,
	}, []profileMapping{
		{
			kind: model.RecordEdge, label: "CONNECTS",
			start: "Person", end: "Team", dynamic: true,
		},
		{
			kind: model.RecordEdge, label: "CONNECTS",
			start: "Service", end: "Region", dynamic: true,
		},
	})
	accumulator.add(model.EdgeRecord(model.Edge{
		Label: "CONNECTS",
		Start: model.Endpoint{Label: "Person"},
		End:   model.Endpoint{Label: "Team"},
	}), 10)
	accumulator.add(model.EdgeRecord(model.Edge{
		Label: "CONNECTS",
		Start: model.Endpoint{Label: "Service"},
		End:   model.Endpoint{Label: "Region"},
		Properties: model.Properties{
			"latency": {Kind: model.ValueInteger, Integer: 5},
		},
	}), 10)

	stats := accumulator.properties[profilePropertyKey{
		kind: model.RecordEdge, label: "CONNECTS", property: "latency",
	}]
	if stats == nil || stats.observed != 2 || stats.present != 1 {
		t.Fatalf("latency stats = %#v", stats)
	}
}

func TestProfileDeterminismRequiresDeclaredUniqueTotalOrder(t *testing.T) {
	orderedPostgreSQL := config.LoadJob{Source: config.Source{
		Type: config.SourcePostgreSQL,
		PostgreSQL: &config.PostgreSQLSource{
			ReadMode: config.PostgreSQLReadCursor,
			Vertices: []config.VertexQuery{{
				Query: "SELECT id FROM people ORDER BY id",
			}},
		},
	}}
	if profileDeterministic(orderedPostgreSQL) {
		t.Fatal("PostgreSQL ORDER BY without unique key semantics was deterministic")
	}

	keysetPostgreSQL := orderedPostgreSQL
	keysetSource := *orderedPostgreSQL.Source.PostgreSQL
	keysetSource.ReadMode = config.PostgreSQLReadKeyset
	keysetSource.Vertices = []config.VertexQuery{{
		Query: "SELECT id FROM people WHERE ($1::bigint IS NULL OR id > $1) " +
			"ORDER BY id LIMIT $2",
		KeyField: "id",
	}}
	keysetPostgreSQL.Source.PostgreSQL = &keysetSource
	if !profileDeterministic(keysetPostgreSQL) {
		t.Fatal("PostgreSQL keyset unique ordering contract was not deterministic")
	}

	orderedCosmos := config.LoadJob{Source: config.Source{
		Type: config.SourceCosmos,
		Cosmos: &config.CosmosSource{Vertices: []config.CosmosVertexQuery{{
			Query: "SELECT * FROM c ORDER BY c.id",
		}}},
	}}
	if profileDeterministic(orderedCosmos) {
		t.Fatal("Cosmos ORDER BY without unique key semantics was deterministic")
	}
}

func TestProfileTelemetryUsesCumulativeThrottleBudget(t *testing.T) {
	budget := sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{})
	if err := budget.Charge(sourcecontract.ProfileBudgetUsage{
		ThrottledRequests: 5,
	}); err != nil {
		t.Fatal(err)
	}
	section := profileTelemetrySection(profileRun{
		job:    config.LoadJob{Source: config.Source{Type: config.SourceCosmos}},
		budget: budget,
		telemetry: &sourcecontract.Telemetry{
			Connector: "cosmos-nosql", ThrottledRequests: 3,
		},
	})
	if got := fieldByName(t, section.Fields, "throttledRequests").Value; got != "5" {
		t.Fatalf("reported throttles = %s, want 5", got)
	}
}

func TestSourceProfilePropagatesCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := SourceProfile(ctx, "unused", ProfileOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("SourceProfile() error = %v", err)
	}
}

func TestProfileValueHashEncodesContainerBoundaries(t *testing.T) {
	integer := func(value int64) model.Value {
		return model.Value{Kind: model.ValueInteger, Integer: value}
	}
	leftList := model.Value{Kind: model.ValueList, List: []model.Value{
		{Kind: model.ValueList, List: []model.Value{integer(1)}},
		integer(2),
	}}
	rightList := model.Value{Kind: model.ValueList, List: []model.Value{
		{Kind: model.ValueList, List: []model.Value{integer(1), integer(2)}},
	}}
	if profileValueHash(leftList) == profileValueHash(rightList) {
		t.Fatal("list nesting boundaries collided")
	}
	leftObject := model.Value{Kind: model.ValueObject, Object: map[string]model.Value{
		"a": {Kind: model.ValueObject, Object: map[string]model.Value{
			"b": integer(1),
		}},
		"c": integer(2),
	}}
	rightObject := model.Value{Kind: model.ValueObject, Object: map[string]model.Value{
		"a": {Kind: model.ValueObject, Object: map[string]model.Value{
			"b": integer(1),
			"c": integer(2),
		}},
	}}
	if profileValueHash(leftObject) == profileValueHash(rightObject) {
		t.Fatal("object nesting boundaries collided")
	}
}

func TestProfileCapacityRemainsLowerBoundWhenMappingsTruncated(t *testing.T) {
	accumulator := newProfileAccumulator(
		profileLimits{}, []profileMapping{{kind: model.RecordVertex, label: "Person"}},
	)
	accumulator.add(model.VertexRecord(model.Vertex{Label: "Person"}), 100)
	section := profileCapacitySection(profileRun{
		complete: true, mappingsTruncated: true, accumulator: accumulator,
	})
	if got := fieldByName(t, section.Fields, "method").Value; got != "sampled-lower-bound-range" {
		t.Fatalf("capacity method = %q", got)
	}
	if got := fieldByName(t, section.Fields, "estimatedTargetRows").Value; got != ">=1" {
		t.Fatalf("estimated target rows = %q", got)
	}
}

func sectionByTitle(t *testing.T, sections []report.Section, title string) report.Section {
	t.Helper()
	for _, section := range sections {
		if section.Title == title {
			return section
		}
	}
	t.Fatalf("section %q not found", title)
	return report.Section{}
}

func fieldByName(t *testing.T, fields []report.Field, name string) report.Field {
	t.Helper()
	for _, field := range fields {
		if field.Name == name {
			return field
		}
	}
	t.Fatalf("field %q not found", name)
	return report.Field{}
}
