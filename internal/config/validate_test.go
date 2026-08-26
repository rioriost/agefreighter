package config

import (
	"strings"
	"testing"
)

func TestValidationErrors(t *testing.T) {
	tests := []struct {
		name string
		edit func(*LoadJob)
		path string
	}{
		{
			name: "api version",
			edit: func(job *LoadJob) { job.APIVersion = "v1" },
			path: "apiVersion",
		},
		{
			name: "source discriminator",
			edit: func(job *LoadJob) { job.Source.Neo4j = &Neo4jSource{} },
			path: "source [discriminator]",
		},
		{
			name: "graph name",
			edit: func(job *LoadJob) { job.Target.Graph = "x" },
			path: "target.graph",
		},
		{
			name: "batch exceeds memory",
			edit: func(job *LoadJob) { job.Runtime.BatchBytes = job.Runtime.MemoryLimit + 1 },
			path: "runtime.batchBytes",
		},
		{
			name: "secret has two references",
			edit: func(job *LoadJob) { job.Target.Connection.File = "/secret" },
			path: "target.connection",
		},
		{
			name: "concurrency",
			edit: func(job *LoadJob) { job.Runtime.MaxTargetConnections = 257 },
			path: "runtime.maxTargetConnections",
		},
		{
			name: "quarantine path",
			edit: func(job *LoadJob) { job.Errors.QuarantinePath = "" },
			path: "errors.quarantinePath",
		},
		{
			name: "delimiter equals quote",
			edit: func(job *LoadJob) { job.Source.CSV.Defaults.Delimiter = `"` },
			path: "source.csv.defaults.delimiter",
		},
		{
			name: "line-break quote",
			edit: func(job *LoadJob) { job.Source.CSV.Defaults.Quote = "\n" },
			path: "source.csv.defaults.quote",
		},
		{
			name: "line-break escape",
			edit: func(job *LoadJob) { job.Source.CSV.Defaults.Escape = "\r" },
			path: "source.csv.defaults.escape",
		},
		{
			name: "load mode",
			edit: func(job *LoadJob) { job.Target.Mode = "truncate" },
			path: "target.mode",
		},
		{
			name: "property mode",
			edit: func(job *LoadJob) { job.Target.PropertyMode = "patch" },
			path: "target.propertyMode",
		},
		{
			name: "append duplicate policy",
			edit: func(job *LoadJob) { job.Target.AppendDuplicate = "overwrite" },
			path: "target.appendDuplicate",
		},
		{
			name: "malformed policy",
			edit: func(job *LoadJob) { job.Errors.MalformedRecord = "ignore" },
			path: "errors.malformedRecord",
		},
		{
			name: "endpoint policy",
			edit: func(job *LoadJob) { job.Errors.MissingEndpoint = "skip" },
			path: "errors.missingEndpoint",
		},
		{
			name: "reject limit",
			edit: func(job *LoadJob) { job.Errors.RejectLimit = -1 },
			path: "errors.rejectLimit",
		},
		{
			name: "deferred edge limit",
			edit: func(job *LoadJob) { job.Errors.MaxDeferredEdges = -1 },
			path: "errors.maxDeferredEdges",
		},
		{
			name: "defer without capacity",
			edit: func(job *LoadJob) {
				job.Errors.MissingEndpoint = MissingEndpointDefer
				job.Errors.MaxDeferredEdges = 0
			},
			path: "errors.maxDeferredEdges",
		},
		{
			name: "defer outside incremental mode",
			edit: func(job *LoadJob) {
				job.Errors.MissingEndpoint = MissingEndpointDefer
				job.Errors.MaxDeferredEdges = 1
			},
			path: "errors.missingEndpoint",
		},
		{
			name: "reject limit without malformed quarantine",
			edit: func(job *LoadJob) { job.Errors.MalformedRecord = MalformedFail },
			path: "errors.rejectLimit",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			job := validCSVJob(t)
			test.edit(&job)

			err := job.Validate()

			if err == nil || !strings.Contains(err.Error(), test.path) {
				t.Fatalf("Validate() error = %v, want path %q", err, test.path)
			}
		})
	}
}

func TestUpsertRequiresEdgeIdentity(t *testing.T) {
	job := validCSVJob(t)
	job.Target.Mode = LoadUpsert
	job.Errors.MaxDeferredEdges = 100_000
	job.Source.CSV.Edges[0].ExternalIDColumn = ""

	err := job.Validate()

	if err == nil || !strings.Contains(err.Error(), "externalIdColumn") {
		t.Fatalf("Validate() error = %v, want edge identity error", err)
	}
}

func TestUpsertRequiresDeferredEdgeCapacity(t *testing.T) {
	job := validCSVJob(t)
	job.Target.Mode = LoadUpsert
	job.Errors.MaxDeferredEdges = 0

	err := job.Validate()

	if err == nil || !strings.Contains(err.Error(), "errors.maxDeferredEdges") {
		t.Fatalf("Validate() error = %v, want deferred edge capacity", err)
	}
}

func TestPostgreSQLReadModeValidation(t *testing.T) {
	job, err := Load("testdata/valid/postgresql.yaml")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	tests := []struct {
		name string
		edit func(*PostgreSQLSource)
		path string
	}{
		{
			name: "unsupported mode",
			edit: func(source *PostgreSQLSource) {
				source.ReadMode = "offset"
			},
			path: "source.postgresql.readMode",
		},
		{
			name: "fetch rows",
			edit: func(source *PostgreSQLSource) {
				source.FetchRows = 100_001
			},
			path: "source.postgresql.fetchRows",
		},
		{
			name: "non read query",
			edit: func(source *PostgreSQLSource) {
				source.Vertices[0].Query = "DELETE FROM person"
			},
			path: "source.postgresql.vertices[0].query",
		},
		{
			name: "multiple statements",
			edit: func(source *PostgreSQLSource) {
				source.Vertices[0].Query = "SELECT 1; SELECT 2"
			},
			path: "source.postgresql.vertices[0].query",
		},
		{
			name: "key outside keyset",
			edit: func(source *PostgreSQLSource) {
				source.Vertices[0].KeyField = "person_id"
			},
			path: "source.postgresql.vertices[0].keyField",
		},
		{
			name: "keyset key",
			edit: func(source *PostgreSQLSource) {
				source.ReadMode = PostgreSQLReadKeyset
				source.Vertices[0].Query =
					"SELECT person_id FROM person WHERE ($1::bigint IS NULL OR person_id > $1) ORDER BY person_id LIMIT $2"
			},
			path: "source.postgresql.vertices[0].keyField",
		},
		{
			name: "keyset parameters",
			edit: func(source *PostgreSQLSource) {
				source.ReadMode = PostgreSQLReadKeyset
				source.Vertices[0].KeyField = "person_id"
				source.Vertices[0].Query = "SELECT person_id FROM person"
			},
			path: "source.postgresql.vertices[0].query",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := job
			source := *job.Source.PostgreSQL
			source.Vertices = append([]VertexQuery(nil), source.Vertices...)
			source.Edges = append([]EdgeQuery(nil), source.Edges...)
			candidate.Source.PostgreSQL = &source
			test.edit(&source)
			if err := candidate.Validate(); err == nil ||
				!strings.Contains(err.Error(), test.path) {
				t.Fatalf("Validate() error = %v, want %q", err, test.path)
			}
		})
	}
}

func TestCSVRequiresEdgeIdentityForCreate(t *testing.T) {
	job := validCSVJob(t)
	job.Target.Mode = LoadCreate
	job.Source.CSV.Edges[0].ExternalIDColumn = ""

	err := job.Validate()

	if err == nil || !strings.Contains(err.Error(), "externalIdColumn") {
		t.Fatalf("Validate() error = %v, want edge identity error", err)
	}
}

func TestValidationAcceptsSupportedAGEGraphNames(t *testing.T) {
	for _, graph := range []string{"my.graph", "my-graph", "_my.graph-name_"} {
		t.Run(graph, func(t *testing.T) {
			job := validCSVJob(t)
			job.Target.Graph = graph
			if err := job.Validate(); err != nil {
				t.Fatalf("Validate() error = %v", err)
			}
		})
	}
}

func TestQueryUpsertRequiresEdgeIdentity(t *testing.T) {
	tests := []struct {
		name string
		path string
		edit func(*LoadJob)
	}{
		{
			name: "postgresql",
			path: "testdata/valid/postgresql.yaml",
			edit: func(job *LoadJob) {
				job.Target.Mode = LoadUpsert
				job.Source.PostgreSQL.Edges[0].ExternalIDField = ""
			},
		},
		{
			name: "neo4j",
			path: "testdata/valid/neo4j.yaml",
			edit: func(job *LoadJob) {
				job.Target.Mode = LoadUpsert
				job.Source.Neo4j.Edges = []EdgeQuery{{
					Label: "KNOWS",
					Query: "MATCH ()-[r:KNOWS]->() " +
						"WHERE $afterKey IS NULL OR r.id > $afterKey " +
						"RETURN r.id AS id, r.from AS from, r.to AS to ORDER BY id",
					KeyField: "id",
					Start:    EndpointMapping{Label: "Person", Field: "from"},
					End:      EndpointMapping{Label: "Person", Field: "to"},
				}}
			},
		},
		{
			name: "cosmos",
			path: "testdata/valid/cosmos.json",
			edit: func(job *LoadJob) {
				job.Source.Cosmos.Edges[0].ExternalIDField = ""
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			job, err := Load(test.path)
			if err != nil {
				t.Fatalf("Load() error = %v", err)
			}
			test.edit(&job)

			err = job.Validate()

			if err == nil || !strings.Contains(err.Error(), "required for edge upsert") {
				t.Fatalf("Validate() error = %v, want edge identity error", err)
			}
		})
	}
}

func TestNeo4jValidation(t *testing.T) {
	job, err := Load("testdata/valid/neo4j.yaml")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	job.Source.Neo4j.URI = "neo4j://user:password@example.invalid"
	job.Source.Neo4j.SourceID = "invalid source id"
	job.Source.Neo4j.FetchRows = 100_001
	job.Source.Neo4j.MultiLabelPolicy = "first"
	job.Source.Neo4j.Vertices[0].KeyField = ""
	job.Source.Neo4j.Vertices[0].Query =
		"MATCH (n) RETURN '$afterKey ORDER BY id' AS value SKIP 1"

	err = job.Validate()

	for _, want := range []string{
		"without embedded credentials",
		"sourceId",
		"fetchRows",
		"multiLabelPolicy",
		"keyField",
		"ascending ORDER BY",
		"$afterKey",
		"rather than SKIP",
	} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("Validate() error = %v, want %q", err, want)
		}
	}
}

func TestCosmosValidation(t *testing.T) {
	job, err := Load("testdata/valid/cosmos.json")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	job.Source.Cosmos.Endpoint = "http://insecure.invalid"
	job.Source.Cosmos.Credential = "key"
	job.Source.Cosmos.Database = ""
	job.Source.Cosmos.Vertices[0].Container = ""
	job.Source.Cosmos.Vertices[0].Label = ""
	job.Source.Cosmos.Vertices[0].Query = ""
	job.Source.Cosmos.Vertices[0].IDField = ""
	job.Source.Cosmos.Edges[0].Container = ""
	job.Source.Cosmos.Edges[0].Label = ""
	job.Source.Cosmos.Edges[0].Query = ""

	err = job.Validate()

	for _, path := range []string{
		"source.cosmos.endpoint",
		"source.cosmos.credential",
		"source.cosmos.database",
		"source.cosmos.vertices[0].container",
		"source.cosmos.edges[0].container",
	} {
		if !strings.Contains(err.Error(), path) {
			t.Errorf("Validate() error = %v, want %s", err, path)
		}
	}
}

func TestCosmosJSONPointerAndParameterValidation(t *testing.T) {
	job, err := Load("testdata/valid/cosmos.json")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	job.Source.Cosmos.Vertices[0].IDField = "id"               // missing leading slash
	job.Source.Cosmos.Vertices[0].Properties["name"] = "name~" // dangling escape
	job.Source.Cosmos.Edges[0].ExternalIDField = "/id~2"       // invalid escape
	job.Source.Cosmos.Edges[0].Start.Field = "fromId"          // missing leading slash
	job.Source.Cosmos.Vertices[0].Parameters = []CosmosQueryParameter{
		{Name: "kind"},                 // missing @ prefix
		{Name: "@"},                    // no name after @
		{Name: "@dup"}, {Name: "@dup"}, // duplicate
	}

	err = job.Validate()
	if err == nil {
		t.Fatal("Validate() error = nil, want validation errors")
	}
	for _, path := range []string{
		"source.cosmos.vertices[0].idField",
		"source.cosmos.vertices[0].properties.name",
		"source.cosmos.edges[0].externalIdField",
		"source.cosmos.edges[0].start.field",
		"source.cosmos.vertices[0].parameters[0].name",
		"source.cosmos.vertices[0].parameters[1].name",
		"source.cosmos.vertices[0].parameters[3].name",
	} {
		if !strings.Contains(err.Error(), path) {
			t.Errorf("Validate() error = %v, want %s", err, path)
		}
	}
}

func TestCosmosPageSizeValidation(t *testing.T) {
	for _, pageSize := range []int{0, -1, 1001} {
		job, err := Load("testdata/valid/cosmos.json")
		if err != nil {
			t.Fatalf("Load() error = %v", err)
		}
		job.Source.Cosmos.PageSize = pageSize
		if err := job.Validate(); err == nil || !strings.Contains(err.Error(), "source.cosmos.pageSize") {
			t.Fatalf("Validate() pageSize=%d error = %v, want pageSize range error", pageSize, err)
		}
	}
}

func TestJSONPointerEscapeValidity(t *testing.T) {
	tests := []struct {
		pointer string
		valid   bool
	}{
		{"/id", true},
		{"/a~0b", true},
		{"/a~1b", true},
		{"/", true},
		{"", false},
		{"id", false},
		{"/a~2b", false},
		{"/a~", false},
	}
	for _, test := range tests {
		if got := jsonPointerEscapesValid(test.pointer); test.pointer != "" && got != test.valid && strings.HasPrefix(test.pointer, "/") {
			t.Errorf("jsonPointerEscapesValid(%q) = %v, want %v", test.pointer, got, test.valid)
		}
		var errs ValidationErrors
		validateJSONPointer("path", test.pointer, &errs)
		if (len(errs) == 0) != test.valid {
			t.Errorf("validateJSONPointer(%q) errs = %v, want valid=%v", test.pointer, errs, test.valid)
		}
	}
}

func TestValidationErrorFormatting(t *testing.T) {
	errs := ValidationErrors{
		{Path: "first", Code: "required", Message: "missing"},
		{Path: "second", Code: "format", Message: "invalid"},
	}
	want := "configuration is invalid:\n- first [required]: missing\n- second [format]: invalid"
	if got := errs.Error(); got != want {
		t.Fatalf("ValidationErrors.Error() = %q, want %q", got, want)
	}
}

func TestCSVOptionalFormatValidationBranches(t *testing.T) {
	job := validCSVJob(t)
	job.Source.Type = SourceType("invalid")
	if err := job.Validate(); err == nil ||
		!strings.Contains(err.Error(), "source.type") {
		t.Fatalf("Validate(invalid source type) = %v", err)
	}

	job = validCSVJob(t)
	job.Source.CSV.Vertices[0].Format = &DelimitedOptions{
		Delimiter: "\n", Quote: `"`, Escape: `"`,
		Header:   job.Source.CSV.Defaults.Header,
		Encoding: "utf-8", NullValue: job.Source.CSV.Defaults.NullValue,
	}
	if err := job.Validate(); err == nil ||
		!strings.Contains(err.Error(), "vertices[0].format") {
		t.Fatalf("Validate(vertex format) = %v", err)
	}

	job = validCSVJob(t)
	job.Source.CSV.Edges[0].Format = &DelimitedOptions{
		Delimiter: ",", Quote: "\n", Escape: `"`,
		Header:   job.Source.CSV.Defaults.Header,
		Encoding: "utf-8", NullValue: job.Source.CSV.Defaults.NullValue,
	}
	if err := job.Validate(); err == nil ||
		!strings.Contains(err.Error(), "edges[0].format") {
		t.Fatalf("Validate(edge format) = %v", err)
	}
}

func TestBuildStaticPlanWarnings(t *testing.T) {
	job, err := Load("testdata/valid/cosmos.json")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	plan := BuildStaticPlan(job)

	if len(plan.Warnings) != 2 {
		t.Fatalf("BuildStaticPlan() warnings = %v, want connector and incremental warnings", plan.Warnings)
	}
	if plan.Limits.MemoryLimit != "1GiB" || plan.Target.Mode != LoadUpsert {
		t.Fatalf("BuildStaticPlan() = %#v, want configured limits and mode", plan)
	}
}

func validCSVJob(t *testing.T) LoadJob {
	t.Helper()
	job, err := Load("testdata/valid/csv.yaml")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	return job
}
