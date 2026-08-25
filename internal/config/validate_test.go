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
					Label: "KNOWS", Query: "MATCH ()-[r:KNOWS]->() RETURN r",
					Start: EndpointMapping{Label: "Person", Field: "from"},
					End:   EndpointMapping{Label: "Person", Field: "to"},
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
