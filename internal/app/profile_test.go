package app

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
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

type scriptedProfileIterator struct {
	items []sourcecontract.Item
	err   error
	next  int
}

func (iterator *scriptedProfileIterator) Next(context.Context) (sourcecontract.Item, error) {
	if iterator.next < len(iterator.items) {
		item := iterator.items[iterator.next]
		iterator.next++
		return item, nil
	}
	if iterator.err != nil {
		return sourcecontract.Item{}, iterator.err
	}
	return sourcecontract.Item{}, io.EOF
}

func (*scriptedProfileIterator) Close() error { return nil }

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

func TestSourceProfileOptionAndLoadValidation(t *testing.T) {
	for _, test := range []struct {
		name    string
		options ProfileOptions
		path    string
		want    string
	}{
		{"invalid mode", ProfileOptions{Mode: "everything"}, "unused", "mode"},
		{"negative sample", ProfileOptions{SampleSize: -1}, "unused", "sample size"},
		{"oversized sample", ProfileOptions{SampleSize: MaxProfileSampleSize + 1}, "unused", "sample size"},
		{"missing configuration", ProfileOptions{}, "missing-profile.yaml", "load profile configuration"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := SourceProfile(t.Context(), test.path, test.options)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("SourceProfile() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestConsumeProfileOutcomes(t *testing.T) {
	record := model.VertexRecord(model.Vertex{Label: "Person"})
	newRun := func(limit int64) profileRun {
		limits := profileLimits{rows: limit}
		return profileRun{
			budget: sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{
				Rows: limit,
			}),
			accumulator: newProfileAccumulator(limits, nil),
		}
	}

	run := newRun(2)
	err := consumeProfile(t.Context(), &scriptedProfileIterator{
		items: []sourcecontract.Item{{Record: record}},
	}, &run)
	if err != nil || !run.complete || run.accumulator.rows != 1 {
		t.Fatalf("completed run = %#v, err=%v", run, err)
	}

	sourceErr := errors.New("source failed")
	run = newRun(2)
	err = consumeProfile(t.Context(), &scriptedProfileIterator{err: sourceErr}, &run)
	if !errors.Is(err, sourceErr) || run.complete {
		t.Fatalf("source error = %v, complete=%t", err, run.complete)
	}

	run = newRun(1)
	if err := run.budget.Charge(sourcecontract.ProfileBudgetUsage{Rows: 1}); err != nil {
		t.Fatal(err)
	}
	err = consumeProfile(t.Context(), &scriptedProfileIterator{}, &run)
	if !errors.Is(err, sourcecontract.ErrProfileBudget) || run.limitReason != "rows" {
		t.Fatalf("budget result = %v, reason=%q", err, run.limitReason)
	}

	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(errProfileLimit)
	run = newRun(2)
	err = consumeProfile(ctx, &scriptedProfileIterator{err: context.Canceled}, &run)
	if !errors.Is(err, errProfileLimit) || run.limitReason != "rows" {
		t.Fatalf("row limit result = %v, reason=%q", err, run.limitReason)
	}
}

func TestProfileAccumulatorBranchMatrix(t *testing.T) {
	accumulator := newProfileAccumulator(
		profileLimits{},
		[]profileMapping{{
			kind: model.RecordEdge, label: "KNOWS", start: "Person", end: "Person",
			properties: []string{"weight"},
		}},
	)
	accumulator.add(model.Record{}, 9)
	accumulator.add(model.EdgeRecord(model.Edge{
		Label: "KNOWS",
		Start: model.Endpoint{Label: "Person"},
		End:   model.Endpoint{Label: "Person"},
		Properties: model.Properties{
			"weight": {Kind: model.ValueFloat, Float: 1.5},
		},
	}), 11)
	if accumulator.otherBad != 1 || accumulator.edges != 1 ||
		accumulator.rows != 2 || accumulator.bytes != 20 {
		t.Fatalf("accumulator = %#v", accumulator)
	}
	for _, test := range []struct {
		message string
		field   *int64
	}{
		{"external id missing", &accumulator.missingID},
		{"idfield missing", &accumulator.missingID},
		{"id field missing", &accumulator.missingID},
		{"endpoint missing", &accumulator.missingEnds},
		{"start field missing", &accumulator.missingEnds},
		{"end field missing", &accumulator.missingEnds},
		{"property missing", &accumulator.missingProp},
		{"field missing", &accumulator.missingProp},
		{"other malformed input", &accumulator.otherBad},
	} {
		before := *test.field
		accumulator.malformedRow(errors.New(test.message))
		if *test.field != before+1 {
			t.Fatalf("%q did not increment expected class", test.message)
		}
	}
}

func TestProfileDynamicCapsAndDistinctLimit(t *testing.T) {
	accumulator := newProfileAccumulator(
		profileLimits{},
		[]profileMapping{{kind: model.RecordVertex, label: "Person", dynamic: true}},
	)
	accumulator.budget = sourcecontract.NewProfileBudget(
		sourcecontract.ProfileBudgetLimits{Properties: 1},
	)
	accumulator.add(model.VertexRecord(model.Vertex{
		Label: "Person",
		Properties: model.Properties{
			"a": {Kind: model.ValueInteger, Integer: 1},
			"b": {Kind: model.ValueInteger, Integer: 2},
		},
	}), 1)
	if !accumulator.propertiesTruncated || len(accumulator.properties) != 1 {
		t.Fatalf("dynamic cap = %#v", accumulator.properties)
	}

	distinct := newProfileAccumulator(profileLimits{}, []profileMapping{{
		kind: model.RecordVertex, label: "Person", properties: []string{"id"},
	}})
	for index := 0; index < maxProfileDistinct; index++ {
		distinct.add(model.VertexRecord(model.Vertex{
			Label: "Person",
			Properties: model.Properties{
				"id": {Kind: model.ValueInteger, Integer: int64(index)},
			},
		}), 1)
	}
	stats := distinct.properties[profilePropertyKey{
		kind: model.RecordVertex, label: "Person", property: "id",
	}]
	if !stats.distinctLimit || len(stats.distinct) != maxProfileDistinct {
		t.Fatalf("distinct stats = %#v", stats)
	}
}

func TestProfileMappingAndDeterminismMatrices(t *testing.T) {
	endpoint := config.EndpointMapping{Label: "Person"}
	jobs := []config.LoadJob{
		{Source: config.Source{Type: config.SourceNeo4j, Neo4j: &config.Neo4jSource{
			Vertices: []config.VertexQuery{{Label: "Person", Properties: map[string]string{"name": "name"}}},
			Edges:    []config.EdgeQuery{{Label: "KNOWS", Start: endpoint, End: endpoint}},
		}}},
		{Source: config.Source{Type: config.SourceCosmos, Cosmos: &config.CosmosSource{
			Vertices: []config.CosmosVertexQuery{{
				Label: "Person", DocumentFormat: config.CosmosDocumentGremlin,
				Properties: map[string]string{"name": "/name"},
			}},
			Edges: []config.CosmosEdgeQuery{{
				Label: "KNOWS", Start: endpoint, End: endpoint,
				DocumentFormat: config.CosmosDocumentGremlin,
			}},
		}}},
	}
	for _, job := range jobs {
		mappings, truncated, propertyTruncated := profileMappings(job)
		if len(mappings) != 2 || truncated || propertyTruncated {
			t.Fatalf("profileMappings(%s) = %#v, %t, %t",
				job.Source.Type, mappings, truncated, propertyTruncated)
		}
	}

	neo := jobs[0]
	neo.Source.Neo4j.Vertices[0].KeyField = "id"
	neo.Source.Neo4j.Vertices[0].Query = "MATCH (n) RETURN n ORDER BY id"
	neo.Source.Neo4j.Edges[0].KeyField = "id"
	neo.Source.Neo4j.Edges[0].Query = "MATCH ()-[r]->() RETURN r ORDER BY id"
	if !profileDeterministic(neo) {
		t.Fatal("ordered Neo4j mappings were not deterministic")
	}
	neo.Source.Neo4j.Edges[0].KeyField = ""
	if profileDeterministic(neo) {
		t.Fatal("unordered Neo4j edge was deterministic")
	}
	discovered := config.LoadJob{Source: config.Source{
		Type:  config.SourceNeo4j,
		Neo4j: &config.Neo4jSource{Discovery: &config.Neo4jDiscovery{Enabled: true}},
	}}
	if !profileDeterministic(discovered) {
		t.Fatal("Neo4j discovery should be deterministic")
	}
	if profileConnectorMode(config.LoadJob{}) != "unknown" {
		t.Fatal("unknown connector mode was not reported")
	}
}

func TestBuildSourceProfileStateMatrix(t *testing.T) {
	base := func() profileRun {
		limits := profileLimits{rows: 10, bytes: 100, pages: 2, requestCharge: 3}
		accumulator := newProfileAccumulator(limits, []profileMapping{{
			kind: model.RecordVertex, label: "Person", properties: []string{"name"},
		}})
		accumulator.budget = sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{})
		accumulator.add(model.VertexRecord(model.Vertex{Label: "Person"}), 5)
		return profileRun{
			job: config.LoadJob{
				Source:  config.Source{Type: config.SourceCSV},
				Target:  config.Target{Mode: config.LoadCreate},
				Runtime: config.Runtime{OperationTimeout: config.Duration(time.Second)},
			},
			mode: ProfileSample, generatedAt: time.Unix(1, 0), limits: limits,
			mappings: []profileMapping{{
				kind: model.RecordVertex, label: "Person", properties: []string{"name"},
			}},
			accumulator: accumulator, budget: accumulator.budget,
			deterministic: true, connectorMode: "delimited",
		}
	}
	tests := []struct {
		name string
		edit func(*profileRun)
		want report.Outcome
	}{
		{"complete", func(r *profileRun) { r.complete = true }, report.OutcomeIncomplete},
		{"source error no rows", func(r *profileRun) {
			r.sourceError = true
			r.sourceErrorDetail = "safe"
			r.accumulator = nil
		}, report.OutcomeIncomplete},
		{"truncated nondeterministic", func(r *profileRun) {
			r.limitReason = "rows"
			r.deterministic = false
		}, report.OutcomeIncomplete},
		{"mapping caps", func(r *profileRun) {
			r.mappingsTruncated = true
			r.propertiesTruncated = true
		}, report.OutcomeIncomplete},
		{"malformed exact replace", func(r *profileRun) {
			r.complete = true
			r.mode = ProfileExact
			r.job.Target.Mode = config.LoadReplace
			r.accumulator.malformedRow(errors.New("property missing"))
		}, report.OutcomeIncomplete},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			run := base()
			test.edit(&run)
			document, err := buildSourceProfile(run)
			if err != nil {
				t.Fatal(err)
			}
			if document.Outcome != test.want || len(document.Sections) != 8 {
				t.Fatalf("document outcome=%s sections=%d", document.Outcome, len(document.Sections))
			}
		})
	}
}

func TestProfileValueAndSaturationHelpers(t *testing.T) {
	values := []struct {
		value model.Value
		width int64
	}{
		{model.Value{Kind: model.ValueNull}, 0},
		{model.Value{Kind: model.ValueBoolean, Boolean: true}, 1},
		{model.Value{Kind: model.ValueInteger, Integer: 1}, 8},
		{model.Value{Kind: model.ValueFloat, Float: 1.5}, 8},
		{model.Value{Kind: model.ValueString, String: "abc"}, 3},
		{model.Value{Kind: model.ValueList, List: []model.Value{{Kind: model.ValueString, String: "x"}}}, 3},
		{model.Value{Kind: model.ValueObject, Object: map[string]model.Value{
			"k": {Kind: model.ValueBoolean},
		}}, 4},
		{model.Value{Kind: model.ValueKind(99)}, 0},
	}
	for _, test := range values {
		if got := profileValueWidth(test.value); got != test.width {
			t.Fatalf("profileValueWidth(%d) = %d, want %d", test.value.Kind, got, test.width)
		}
		_ = profileValueHash(test.value)
	}
	if saturatingProfileAdd(math.MaxInt64, 1) != math.MaxInt64 ||
		saturatingProfileAdd(2, -1) != 1 ||
		saturatingProfileMultiply(0, 9) != 0 ||
		saturatingProfileMultiply(math.MaxInt64, 2) != math.MaxInt64 ||
		saturatingProfileMultiply(3, 4) != 12 ||
		profileSum(math.MaxInt64, 1) != math.MaxInt64 {
		t.Fatal("saturating arithmetic failed")
	}
	if profileTypeCounts([7]int64{}) != "none" ||
		profileTypeCounts([7]int64{1, 2}) != "null:1|boolean:2" ||
		profileKindName(model.RecordEdge) != "edge" ||
		profileKindName(model.RecordVertex) != "vertex" {
		t.Fatal("profile formatting helpers failed")
	}
	vertex := model.VertexRecord(model.Vertex{
		Label: "P", Namespace: "n", ExternalID: "id",
		Properties: model.Properties{"x": {Kind: model.ValueString, String: "abc"}},
	})
	edge := model.EdgeRecord(model.Edge{
		Label: "E", Namespace: "n", ExternalID: "id",
		Start:      model.Endpoint{Label: "P", Namespace: "n", ExternalID: "s"},
		End:        model.Endpoint{Label: "P", Namespace: "n", ExternalID: "e"},
		Properties: model.Properties{"x": {Kind: model.ValueInteger, Integer: 1}},
	})
	if profileRecordWidth(vertex) <= 0 || profileRecordWidth(edge) <= profileRecordWidth(vertex) ||
		profileRecordWidth(model.Record{}) != 0 {
		t.Fatal("record width calculation failed")
	}
}

func TestBoundedProfileJobConnectorMatrix(t *testing.T) {
	properties := map[string]string{"b": "b", "a": "a"}
	endpoint := config.EndpointMapping{Label: "Person"}
	tests := []struct {
		name string
		job  config.LoadJob
		edit func(*config.LoadJob)
	}{
		{
			name: "csv",
			job: config.LoadJob{Source: config.Source{
				Type: config.SourceCSV,
				CSV: &config.CSVSource{
					Vertices: []config.CSVVertex{{Label: "Person", Properties: properties}},
					Edges:    []config.CSVEdge{{Label: "KNOWS", Start: endpoint, End: endpoint, Properties: properties}},
				},
			}},
			edit: func(job *config.LoadJob) { job.Source.CSV.Vertices[0].Properties["a"] = "changed" },
		},
		{
			name: "neo4j",
			job: config.LoadJob{Source: config.Source{
				Type: config.SourceNeo4j,
				Neo4j: &config.Neo4jSource{
					Vertices: []config.VertexQuery{{Label: "Person", Properties: properties}},
					Edges:    []config.EdgeQuery{{Label: "KNOWS", Start: endpoint, End: endpoint, Properties: properties}},
				},
			}},
			edit: func(job *config.LoadJob) { job.Source.Neo4j.Vertices[0].Properties["a"] = "changed" },
		},
		{
			name: "cosmos",
			job: config.LoadJob{Source: config.Source{
				Type: config.SourceCosmos,
				Cosmos: &config.CosmosSource{
					Vertices: []config.CosmosVertexQuery{{Label: "Person", Properties: properties}},
					Edges:    []config.CosmosEdgeQuery{{Label: "KNOWS", Start: endpoint, End: endpoint, Properties: properties}},
				},
			}},
			edit: func(job *config.LoadJob) { job.Source.Cosmos.Vertices[0].Properties["a"] = "changed" },
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			original := test.job
			bounded, mappingsTruncated, propertiesTruncated := boundedProfileJob(test.job)
			if mappingsTruncated || propertiesTruncated {
				t.Fatal("small definition was truncated")
			}
			test.edit(&bounded)
			switch original.Source.Type {
			case config.SourceCSV:
				if original.Source.CSV.Vertices[0].Properties["a"] != "a" {
					t.Fatal("CSV source was mutated")
				}
			case config.SourceNeo4j:
				if original.Source.Neo4j.Vertices[0].Properties["a"] != "a" {
					t.Fatal("Neo4j source was mutated")
				}
			case config.SourceCosmos:
				if original.Source.Cosmos.Vertices[0].Properties["a"] != "a" {
					t.Fatal("Cosmos source was mutated")
				}
			}
		})
	}

	vertices, edges, truncated := capProfileMappings([]int{1}, []int{2}, 3)
	if truncated || len(vertices) != 1 || len(edges) != 1 {
		t.Fatal("small mappings were truncated")
	}
	vertices, edges, truncated = capProfileMappings([]int{1, 2, 3}, []int{4}, 2)
	if !truncated || len(vertices) != 2 || edges != nil {
		t.Fatal("vertex-heavy mappings were not capped")
	}
	vertices, edges, truncated = capProfileMappings([]int{1}, []int{2, 3}, 2)
	if !truncated || len(vertices) != 1 || len(edges) != 1 {
		t.Fatal("mixed mappings were not capped")
	}
}

func TestProfileMetadataAndUnavailableSections(t *testing.T) {
	directory := t.TempDir()
	file := filepath.Join(directory, "input.csv")
	if err := os.WriteFile(file, []byte("id\n1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	job := config.LoadJob{Source: config.Source{
		Type: config.SourceCSV,
		CSV: &config.CSVSource{
			Vertices: []config.CSVVertex{{Path: file}},
			Edges:    []config.CSVEdge{{Path: file}},
		},
	}}
	bytes, _, known, timestampKnown := profileCSVMetadata(job)
	if bytes == 0 || !known || !timestampKnown {
		t.Fatalf("metadata = %d %t %t", bytes, known, timestampKnown)
	}
	job.Source.CSV.Vertices[0].Path = directory
	if _, _, known, timestampKnown := profileCSVMetadata(job); known || timestampKnown {
		t.Fatal("directory was accepted as CSV input")
	}
	job.Source.CSV.Vertices = nil
	job.Source.CSV.Edges = nil
	if bytes, _, known, timestampKnown := profileCSVMetadata(job); bytes != 0 || !known || timestampKnown {
		t.Fatalf("empty metadata = %d %t %t", bytes, known, timestampKnown)
	}

	run := profileRun{
		job: config.LoadJob{
			Source:  config.Source{Type: config.SourceCSV},
			Runtime: config.Runtime{OperationTimeout: config.Duration(time.Second)},
		},
		mode: ProfileExact, limits: profileLimits{},
		budget: sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{}),
	}
	if fieldByName(t, profileSourceSection(run).Fields, "sourceTimestamp").Status != report.CheckUnavailable {
		t.Fatal("unknown source timestamp was not unavailable")
	}
	if profileLabelSection(run, model.RecordVertex).Fields[0].Status != report.CheckUnavailable ||
		profilePropertySection(run).Fields[0].Status != report.CheckUnavailable ||
		profileSignalsSection(run).Fields[0].Status != report.CheckUnavailable ||
		profileCapacitySection(run).Fields[0].Status != report.CheckUnavailable {
		t.Fatal("missing profile evidence was not unavailable")
	}
}

func TestNewProfileSourceIteratorPreOpenFailures(t *testing.T) {
	accumulator := newProfileAccumulator(profileLimits{rows: 1}, nil)
	accumulator.budget = sourcecontract.NewProfileBudget(sourcecontract.ProfileBudgetLimits{})
	cancel := func(error) {}
	if _, err := newProfileSourceIterator(t.Context(), config.LoadJob{
		Source: config.Source{Type: config.SourceType("unknown")},
	}, accumulator, cancel); err == nil {
		t.Fatal("unsupported profile connector was accepted")
	}
	missingEnv := "AGEFREIGHTER_PROFILE_TEST_MISSING_SECRET"
	t.Setenv(missingEnv, "")
	for _, test := range []struct {
		name string
		job  config.LoadJob
		want string
	}{
		{
			name: "postgresql",
			job: config.LoadJob{Source: config.Source{
				Type: config.SourcePostgreSQL,
				PostgreSQL: &config.PostgreSQLSource{
					Connection: config.SecretRef{Env: missingEnv},
				},
			}},
			want: "resolve PostgreSQL source credential",
		},
		{
			name: "neo4j",
			job: config.LoadJob{Source: config.Source{
				Type: config.SourceNeo4j,
				Neo4j: &config.Neo4jSource{
					Password: &config.SecretRef{Env: missingEnv},
				},
			}},
			want: "resolve Neo4j source credential",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := newProfileSourceIterator(
				t.Context(), test.job, accumulator, cancel,
			); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("newProfileSourceIterator() error = %v", err)
			}
		})
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
