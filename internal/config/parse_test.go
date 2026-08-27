package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/santhosh-tekuri/jsonschema/v6"
	"go.yaml.in/yaml/v3"
)

func TestCosmosDefaults(t *testing.T) {
	job := validCSVJob(t)
	job.Source.CSV = nil
	job.Source.Type = SourceCosmos
	job.Source.Cosmos = &CosmosSource{}
	job.applyDefaults()
	if job.Source.Cosmos.Credential != "default-azure" ||
		job.Source.Cosmos.PageSize != defaultCosmosPageSize {
		t.Fatalf("Cosmos defaults = %#v", job.Source.Cosmos)
	}
}

func TestCosmosGremlinDefaultsAndStaticPlan(t *testing.T) {
	job, err := Load("testdata/valid/cosmos-gremlin.json")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	gremlin := job.Source.Cosmos.Gremlin
	if gremlin == nil ||
		gremlin.MaxLabels != 256 ||
		gremlin.MaxProperties != 1_024 ||
		gremlin.MaxDiscoveryDocuments != 100_000 {
		t.Fatalf("Cosmos Gremlin defaults = %#v", gremlin)
	}
	plan := BuildStaticPlan(job)
	if !plan.Source.CosmosGremlin ||
		len(plan.Warnings) < 2 ||
		!strings.Contains(plan.Warnings[1], "interpreted") {
		t.Fatalf("Cosmos Gremlin plan = %#v", plan)
	}
}

func TestNeo4jDefaultsAndStaticPlan(t *testing.T) {
	job, err := Load("testdata/valid/neo4j.yaml")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if job.Source.Neo4j.FetchRows != 1_000 ||
		job.Source.Neo4j.MultiLabelPolicy != Neo4jMultiLabelConfigured {
		t.Fatalf("Neo4j defaults = %#v", job.Source.Neo4j)
	}
	plan := BuildStaticPlan(job)
	if plan.Source.FetchRows != 1_000 ||
		plan.Source.Neo4jMultiLabel != Neo4jMultiLabelConfigured ||
		plan.Source.Consistency != "per-mapping-read-transaction" {
		t.Fatalf("Neo4j plan source = %#v", plan.Source)
	}
	if len(plan.Warnings) == 0 ||
		!strings.Contains(plan.Warnings[0], "point-in-time") {
		t.Fatalf("Neo4j plan warnings = %#v", plan.Warnings)
	}
}

func TestNeo4jDiscoveryDefaultsAndStaticPlan(t *testing.T) {
	job, err := Load("testdata/valid/neo4j-discovery.yaml")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	discovery := job.Source.Neo4j.Discovery
	if discovery == nil ||
		discovery.VertexIDProperty != "sequence" ||
		discovery.EdgeIDProperty != "sequence" ||
		discovery.MaxLabels != 256 ||
		discovery.MaxProperties != 1_024 {
		t.Fatalf("Neo4j discovery defaults = %#v", discovery)
	}
	plan := BuildStaticPlan(job)
	if !plan.Source.Neo4jDiscovery ||
		len(plan.Warnings) < 2 ||
		!strings.Contains(plan.Warnings[1], "resolved") {
		t.Fatalf("Neo4j discovery plan = %#v", plan)
	}
}

func TestInactiveIncrementalDefaultsAreOmitted(t *testing.T) {
	job := validCSVJob(t)
	encoded, err := json.Marshal(job)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	for _, field := range []string{"appendDuplicate", "maxDeferredEdges"} {
		if strings.Contains(string(encoded), `"`+field+`"`) {
			t.Fatalf("inactive field %q was serialized: %s", field, encoded)
		}
	}

	job.Target.Mode = LoadAppend
	job.Errors.MissingEndpoint = MissingEndpointDefer
	job.applyDefaults()
	if job.Target.AppendDuplicate != AppendDuplicateError ||
		job.Errors.MaxDeferredEdges != 100_000 {
		t.Fatalf("active incremental defaults = %#v, %#v", job.Target, job.Errors)
	}

	job = validCSVJob(t)
	job.Target.Mode = LoadUpsert
	job.Errors.MaxDeferredEdges = 0
	job.applyDefaults()
	if job.Errors.MaxDeferredEdges != 100_000 {
		t.Fatalf(
			"upsert deferred edge limit = %d, want 100000",
			job.Errors.MaxDeferredEdges,
		)
	}
}

func TestParseReportsSemanticValidation(t *testing.T) {
	if _, err := Parse([]byte(`
apiVersion: agefreighter.io/v2
kind: LoadJob
metadata:
  name: INVALID
`)); err == nil || !strings.Contains(err.Error(), "metadata.name") {
		t.Fatalf("Parse(semantic error) = %v", err)
	}
}

func TestValidFixtures(t *testing.T) {
	for _, path := range validFixturePaths(t) {
		t.Run(filepath.Base(path), func(t *testing.T) {
			job, err := Load(path)
			if err != nil {
				t.Fatalf("Load() error = %v", err)
			}
			if job.Runtime.MemoryLimit <= 0 {
				t.Fatal("Load() did not apply runtime defaults")
			}
		})
	}
}

func TestDefaults(t *testing.T) {
	job, err := Load("testdata/valid/postgresql.yaml")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if job.Target.Mode != LoadAppend {
		t.Fatalf("Target.Mode = %q, want append", job.Target.Mode)
	}
	if job.Runtime.BatchRows != 10_000 {
		t.Fatalf("Runtime.BatchRows = %d, want 10000", job.Runtime.BatchRows)
	}
	if job.Runtime.BatchBytes != 16*mebibyte {
		t.Fatalf("Runtime.BatchBytes = %s, want 16MiB", job.Runtime.BatchBytes)
	}
	if job.Runtime.OperationTimeout.String() != "30s" {
		t.Fatalf("Runtime.OperationTimeout = %s, want 30s", job.Runtime.OperationTimeout)
	}
	if job.Runtime.MaxTransformConcurrency != 1 {
		t.Fatalf(
			"Runtime.MaxTransformConcurrency = %d, want 1",
			job.Runtime.MaxTransformConcurrency,
		)
	}
	if job.Target.AppendDuplicate != AppendDuplicateError {
		t.Fatalf(
			"Target.AppendDuplicate = %q, want error",
			job.Target.AppendDuplicate,
		)
	}
	if job.Errors.MaxDeferredEdges != 0 {
		t.Fatalf(
			"Errors.MaxDeferredEdges = %d, want 0",
			job.Errors.MaxDeferredEdges,
		)
	}
	if job.Source.PostgreSQL.ReadMode != PostgreSQLReadCopy ||
		job.Source.PostgreSQL.FetchRows != 1_000 {
		t.Fatalf("PostgreSQL defaults = %#v", job.Source.PostgreSQL)
	}
}

func TestTrialDefaultsAndStaticPlan(t *testing.T) {
	job := validCSVJob(t)
	job.Trial = &TrialOptions{Enabled: true}
	job.applyDefaults()

	if job.Trial.MaxVerticesPerLabel != 1_000 ||
		job.Trial.MaxVertices != 10_000 ||
		job.Trial.MaxEdges != 10_000 ||
		job.Trial.MaxBytes != 64*mebibyte {
		t.Fatalf("trial defaults = %#v", job.Trial)
	}
	if err := job.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	plan := BuildStaticPlan(job)
	if plan.Trial == nil ||
		plan.Trial.MaxVertices != 10_000 ||
		plan.Trial.MaxBytes != "64MiB" ||
		len(plan.Warnings) == 0 ||
		!strings.Contains(plan.Warnings[0], "non-resumable") {
		t.Fatalf("trial plan = %#v", plan)
	}
}

func TestTrialSupportsEveryConfiguredSource(t *testing.T) {
	for _, path := range []string{
		"testdata/valid/csv.yaml",
		"testdata/valid/postgresql.yaml",
		"testdata/valid/neo4j.yaml",
		"testdata/valid/cosmos.json",
	} {
		t.Run(filepath.Base(path), func(t *testing.T) {
			job, err := Load(path)
			if err != nil {
				t.Fatalf("Load() error = %v", err)
			}
			job.Target.Mode = LoadCreate
			job.Target.AppendDuplicate = ""
			job.Errors.MissingEndpoint = MissingEndpointError
			if job.Source.Cosmos != nil {
				for index := range job.Source.Cosmos.Vertices {
					job.Source.Cosmos.Vertices[index].Query += " ORDER BY c.id"
				}
				for index := range job.Source.Cosmos.Edges {
					job.Source.Cosmos.Edges[index].Query += " ORDER BY c.id"
				}
			}
			labels := configuredVertexLabels(job.Source)
			var include string
			for label := range labels {
				include = label
				break
			}
			job.Trial = &TrialOptions{
				Enabled:       true,
				IncludeLabels: []string{include},
			}
			job.applyDefaults()
			if err := job.Validate(); err != nil {
				t.Fatalf("Validate() error = %v", err)
			}
		})
	}
}

func TestPostgreSQLStaticPlan(t *testing.T) {
	job, err := Load("testdata/valid/postgresql.yaml")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	plan := BuildStaticPlan(job)
	if plan.Source.PostgreSQLReadMode != PostgreSQLReadCopy ||
		plan.Source.FetchRows != 1_000 ||
		plan.Source.Consistency != "exported-repeatable-read-snapshot" {
		t.Fatalf("PostgreSQL plan source = %#v", plan.Source)
	}
}

func TestResolvePostgreSQLConnectionFile(t *testing.T) {
	job := LoadJob{Source: Source{PostgreSQL: &PostgreSQLSource{
		Connection: SecretRef{File: "secrets/source.dsn"},
	}}}
	base := t.TempDir()
	resolveJobPaths(&job, base)
	want := filepath.Join(base, "secrets/source.dsn")
	if job.Source.PostgreSQL.Connection.File != want {
		t.Fatalf(
			"PostgreSQL connection file = %q, want %q",
			job.Source.PostgreSQL.Connection.File,
			want,
		)
	}
}

func TestResolveNeo4jPasswordFile(t *testing.T) {
	job := LoadJob{Source: Source{Neo4j: &Neo4jSource{
		Password: &SecretRef{File: "secrets/neo4j-password"},
	}}}
	base := t.TempDir()
	resolveJobPaths(&job, base)
	want := filepath.Join(base, "secrets/neo4j-password")
	if job.Source.Neo4j.Password.File != want {
		t.Fatalf("Neo4j password file = %q, want %q",
			job.Source.Neo4j.Password.File, want)
	}
}

func TestLoadResolvesRelativePathsFromConfigurationDirectory(t *testing.T) {
	dir := t.TempDir()
	job := validCSVJob(t)
	job.Source.CSV.Vertices[0].Path = "data/vertices.csv"
	job.Source.CSV.Edges[0].Path = "data/edges.csv"
	job.Target.Connection.Env = ""
	job.Target.Connection.File = "secrets/age.dsn"
	job.Errors.QuarantinePath = "rejects/quarantine.jsonl"
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	path := filepath.Join(dir, "job.yaml")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	for name, got := range map[string]string{
		"vertex":     loaded.Source.CSV.Vertices[0].Path,
		"edge":       loaded.Source.CSV.Edges[0].Path,
		"connection": loaded.Target.Connection.File,
		"quarantine": loaded.Errors.QuarantinePath,
	} {
		if !filepath.IsAbs(got) || !strings.HasPrefix(got, dir+string(filepath.Separator)) {
			t.Fatalf("%s path = %q, want path under %q", name, got, dir)
		}
	}
}

func TestParseRejectsMultipleDocuments(t *testing.T) {
	data := []byte(`
apiVersion: agefreighter.io/v2
kind: LoadJob
---
apiVersion: agefreighter.io/v2
kind: LoadJob
`)
	if _, err := Parse(data); err == nil || !strings.Contains(err.Error(), "exactly one document") {
		t.Fatalf("Parse() error = %v, want single-document error", err)
	}
}

func TestParseRejectsMalformedTrailingDocument(t *testing.T) {
	data := []byte(`
apiVersion: agefreighter.io/v2
kind: LoadJob
---
invalid: [
`)
	if _, err := Parse(data); err == nil || !strings.Contains(err.Error(), "trailing") {
		t.Fatalf("Parse() error = %v, want trailing decode error", err)
	}
}

func TestParseRejectsEmptyAndMalformedDocuments(t *testing.T) {
	for _, data := range [][]byte{
		nil,
		[]byte("apiVersion: ["),
		[]byte("apiVersion: agefreighter.io/v2\nunknown: true\n"),
	} {
		if _, err := Parse(data); err == nil {
			t.Fatalf("Parse(%q) error = nil, want error", data)
		}
	}
}

func TestLoadErrors(t *testing.T) {
	if _, err := Load("testdata/does-not-exist.yaml"); err == nil {
		t.Fatal("Load() missing file error = nil")
	}

	path := filepath.Join(t.TempDir(), "large.yaml")
	data := make([]byte, MaxDocumentBytes+1)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	if _, err := Load(path); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("Load() large file error = %v, want size error", err)
	}
	if _, err := Load(t.TempDir()); err == nil || !strings.Contains(err.Error(), "read configuration") {
		t.Fatalf("Load() directory error = %v, want read error", err)
	}
}

func TestDelimitedDefaultsApplyToFileOverrides(t *testing.T) {
	job := validCSVJob(t)
	nullValue := "NULL"
	job.Source.CSV.Defaults.Delimiter = ";"
	job.Source.CSV.Defaults.NullValue = &nullValue
	job.Source.CSV.Vertices[0].Format = &DelimitedOptions{Delimiter: "\t"}
	job.Source.CSV.Edges[0].Format = &DelimitedOptions{}

	job.applyDefaults()

	if job.Source.CSV.Vertices[0].Format.Delimiter != "\t" ||
		job.Source.CSV.Vertices[0].Format.Quote != `"` ||
		*job.Source.CSV.Vertices[0].Format.NullValue != "NULL" ||
		job.Source.CSV.Edges[0].Format.Delimiter != ";" ||
		job.Source.CSV.Edges[0].Format.Encoding != "utf-8" {
		t.Fatalf("applyDefaults() did not default file formats: %#v", job.Source.CSV)
	}
}

func TestApplyDelimitedDefaultsFillsEveryEmptyOption(t *testing.T) {
	var options DelimitedOptions

	applyDelimitedDefaults(&options)

	if options.Delimiter != "," ||
		options.Quote != `"` ||
		options.Escape != `"` ||
		options.Header == nil ||
		!*options.Header ||
		options.Encoding != "utf-8" ||
		options.NullValue == nil ||
		*options.NullValue != "" {
		t.Fatalf("applyDelimitedDefaults() = %#v", options)
	}
}

func TestLiteralSecretIsRejectedWithoutDisclosure(t *testing.T) {
	data, err := os.ReadFile("testdata/invalid/literal-secret.yaml")
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}

	_, err = Parse(data)

	if err == nil {
		t.Fatal("Parse() error = nil, want literal secret rejection")
	}
	if strings.Contains(err.Error(), "supersecret") || strings.Contains(err.Error(), "admin") {
		t.Fatalf("Parse() disclosed credential: %v", err)
	}
}

func TestFixturesConformToJSONSchema(t *testing.T) {
	root := moduleRoot(t)
	compiler := jsonschema.NewCompiler()
	schema, err := compiler.Compile(filepath.Join(root, "docs/reference/load-job.schema.json"))
	if err != nil {
		t.Fatalf("Compile() schema error = %v", err)
	}

	for _, path := range validFixturePaths(t) {
		t.Run(filepath.Base(path), func(t *testing.T) {
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("ReadFile() error = %v", err)
			}
			var yamlValue any
			if err := yaml.Unmarshal(data, &yamlValue); err != nil {
				t.Fatalf("yaml.Unmarshal() error = %v", err)
			}
			encoded, err := json.Marshal(yamlValue)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			var document any
			if err := json.Unmarshal(encoded, &document); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}
			if err := schema.Validate(document); err != nil {
				t.Fatalf("schema.Validate() error = %v", err)
			}

			job, err := Parse(data)
			if err != nil {
				t.Fatalf("Parse() error = %v", err)
			}
			encoded, err = json.Marshal(job)
			if err != nil {
				t.Fatalf("json.Marshal(job) error = %v", err)
			}
			if err := json.Unmarshal(encoded, &document); err != nil {
				t.Fatalf("json.Unmarshal(job) error = %v", err)
			}
			if err := schema.Validate(document); err != nil {
				t.Fatalf("schema.Validate(defaulted job) error = %v", err)
			}
		})
	}
}

func TestInvalidSecretFailsJSONSchema(t *testing.T) {
	root := moduleRoot(t)
	compiler := jsonschema.NewCompiler()
	schema, err := compiler.Compile(filepath.Join(root, "docs/reference/load-job.schema.json"))
	if err != nil {
		t.Fatalf("Compile() schema error = %v", err)
	}
	data, err := os.ReadFile("testdata/invalid/literal-secret.yaml")
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	var document any
	if err := yaml.Unmarshal(data, &document); err != nil {
		t.Fatalf("yaml.Unmarshal() error = %v", err)
	}
	if err := schema.Validate(document); err == nil {
		t.Fatal("schema.Validate() error = nil, want invalid secret rejection")
	}
}

func TestJSONSchemaAGEGraphAndSecretPatterns(t *testing.T) {
	root := moduleRoot(t)
	compiler := jsonschema.NewCompiler()
	schema, err := compiler.Compile(filepath.Join(root, "docs/reference/load-job.schema.json"))
	if err != nil {
		t.Fatalf("Compile() schema error = %v", err)
	}
	job := validCSVJob(t)
	job.Target.Graph = "my.graph-name"

	document := schemaDocument(t, job)
	if err := schema.Validate(document); err != nil {
		t.Fatalf("schema rejected supported AGE graph name: %v", err)
	}

	job.Target.Connection.Env = "INVALID-ENV"
	document = schemaDocument(t, job)
	if err := schema.Validate(document); err == nil {
		t.Fatal("schema accepted invalid environment variable name")
	}
}

func schemaDocument(t *testing.T, value any) any {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var document any
	if err := json.Unmarshal(encoded, &document); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	return document
}

func FuzzParse(f *testing.F) {
	for _, path := range validFixturePaths(f) {
		data, err := os.ReadFile(path)
		if err != nil {
			f.Fatalf("ReadFile() error = %v", err)
		}
		f.Add(data)
	}
	f.Add([]byte("not: [valid"))

	f.Fuzz(func(t *testing.T, data []byte) {
		job, err := Parse(data)
		if err == nil {
			if err := job.Validate(); err != nil {
				t.Fatalf("Parse() returned invalid job: %v", err)
			}
		}
	})
}

type fixtureTB interface {
	Helper()
	Fatalf(string, ...any)
}

func validFixturePaths(tb fixtureTB) []string {
	tb.Helper()
	paths, err := filepath.Glob("testdata/valid/*")
	if err != nil {
		tb.Fatalf("Glob() error = %v", err)
	}
	return paths
}

func moduleRoot(t *testing.T) string {
	t.Helper()
	current, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd() error = %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(current, "go.mod")); err == nil {
			return current
		}
		parent := filepath.Dir(current)
		if parent == current {
			t.Fatal("go.mod not found")
		}
		current = parent
	}
}
