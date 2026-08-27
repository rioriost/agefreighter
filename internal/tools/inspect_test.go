package tools

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
)

func TestBuildInspectionCSVDoesNotExposeSecrets(t *testing.T) {
	job := inspectionJob()
	job.Trial = &config.TrialOptions{
		Enabled:             true,
		MaxVerticesPerLabel: 2,
		MaxVertices:         4,
		MaxEdges:            3,
		MaxBytes:            1 << 20,
		IncludeLabels:       []string{"Person"},
	}
	report := BuildInspection(job)

	if report.FormatVersion != InspectionFormatVersion ||
		report.Job != "inspect-test" ||
		report.Trial == nil ||
		report.Trial.MaxVertices != 4 ||
		report.Source.Type != config.SourceCSV ||
		report.Source.Consistency != "files-at-open-time" {
		t.Fatalf("BuildInspection() = %#v", report)
	}
	if report.Target.Connection != "environment" {
		t.Fatalf("target connection = %q", report.Target.Connection)
	}
	if len(report.Source.VertexMappings) != 1 ||
		len(report.Source.EdgeMappings) != 1 {
		t.Fatalf("source mappings = %#v", report.Source)
	}
	vertex := report.Source.VertexMappings[0]
	if vertex.Label != "Person" || vertex.IdentityField != "id" ||
		!slices.Equal(vertex.PropertyFields, []string{"display_name", "score"}) {
		t.Fatalf("vertex inspection = %#v", vertex)
	}
	edge := report.Source.EdgeMappings[0]
	if edge.StartLabel != "Person" || edge.StartField != "start_id" ||
		edge.StartNamespace != "crm" ||
		edge.EndLabel != "Person" || edge.EndField != "end_id" ||
		edge.EndNamespace != "crm" {
		t.Fatalf("edge inspection = %#v", edge)
	}
}

func TestBuildInspectionConnectorContracts(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*config.LoadJob)
		check  func(*testing.T, Inspection)
	}{
		{
			name: "postgresql",
			mutate: func(job *config.LoadJob) {
				job.Source.Type = config.SourcePostgreSQL
				job.Source.CSV = nil
				job.Source.PostgreSQL = &config.PostgreSQLSource{
					Connection: config.SecretRef{File: "/secret/dsn"},
					ReadMode:   config.PostgreSQLReadKeyset,
					FetchRows:  321,
					Vertices: []config.VertexQuery{{
						Label: "Person", IDField: "id", KeyField: "sequence",
						Query: "secret query", Properties: map[string]string{"name": "display_name"},
					}},
					Edges: []config.EdgeQuery{{
						Label: "KNOWS", KeyField: "sequence",
						Start: config.EndpointMapping{
							Label: "Person", Namespace: "external", Field: "start_id",
						},
						End: config.EndpointMapping{Label: "Person", Field: "end_id"},
					}},
				}
			},
			check: func(t *testing.T, report Inspection) {
				if report.Source.ReadMode != config.PostgreSQLReadKeyset ||
					report.Source.FetchRows != 321 ||
					report.Source.Connection != "file" ||
					report.Source.VertexMappings[0].ResumeKey != "sequence" ||
					report.Source.EdgeMappings[0].StartNamespace != "external" ||
					report.Source.EdgeMappings[0].EndNamespace != "crm" {
					t.Fatalf("PostgreSQL inspection = %#v", report.Source)
				}
			},
		},
		{
			name: "neo4j",
			mutate: func(job *config.LoadJob) {
				job.Source.Type = config.SourceNeo4j
				job.Source.CSV = nil
				job.Source.Neo4j = &config.Neo4jSource{
					URI:      "neo4j://user:secret@example.invalid",
					Database: "neo4j", SourceID: "logical-graph",
					Password:  &config.SecretRef{Env: "VERY_SECRET_PASSWORD"},
					FetchRows: 500, MultiLabelPolicy: config.Neo4jMultiLabelReject,
					Discovery: &config.Neo4jDiscovery{Enabled: true},
				}
			},
			check: func(t *testing.T, report Inspection) {
				if report.Source.Database != "neo4j" ||
					report.Source.SourceID != "logical-graph" ||
					report.Source.Credential != "environment" ||
					report.Source.MultiLabel != config.Neo4jMultiLabelReject ||
					!report.Source.Discovery {
					t.Fatalf("Neo4j inspection = %#v", report.Source)
				}
			},
		},
		{
			name: "cosmos",
			mutate: func(job *config.LoadJob) {
				job.Source.Type = config.SourceCosmos
				job.Source.CSV = nil
				job.Source.Cosmos = &config.CosmosSource{
					Endpoint: "https://private.invalid", Credential: "default-azure",
					Database: "graph", PageSize: 42,
					Vertices: []config.CosmosVertexQuery{{
						Container: "people", Label: "Person", IDField: "/id",
						Query: "secret query", Properties: map[string]string{"name": "/profile/name"},
					}},
				}
			},
			check: func(t *testing.T, report Inspection) {
				if report.Source.Consistency != "connector-verified" ||
					report.Source.Database != "graph" ||
					report.Source.PageSize != 42 ||
					report.Source.VertexMappings[0].Location != "people" {
					t.Fatalf("Cosmos inspection = %#v", report.Source)
				}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			job := inspectionJob()
			test.mutate(&job)
			report := BuildInspection(job)
			test.check(t, report)
		})
	}
}

func TestInspectConfigurationLoadsAndResolvesPaths(t *testing.T) {
	root := t.TempDir()
	job := inspectionJob()
	job.Source.CSV.Vertices[0].Path = "data/vertices.csv"
	data := []byte(`apiVersion: agefreighter.io/v2
kind: LoadJob
metadata:
  name: inspect-file
source:
  type: csv
  namespace: crm
  csv:
    vertices:
      - label: Person
        path: data/vertices.csv
        idColumn: id
target:
  type: apache-age
  graph: people
  connection:
    env: TEST_DSN
runtime:
  batchRows: 100
`)
	path := filepath.Join(root, "job.yaml")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write job: %v", err)
	}

	report, err := InspectConfiguration(path)
	if err != nil {
		t.Fatalf("InspectConfiguration() error = %v", err)
	}
	want := filepath.Join(root, "data/vertices.csv")
	if got := report.Source.VertexMappings[0].Location; got != want {
		t.Fatalf("mapping location = %q, want %q", got, want)
	}
}

func TestInspectConfigurationRejectsInvalidJob(t *testing.T) {
	_, err := InspectConfiguration(filepath.Join(t.TempDir(), "missing.yaml"))
	if err == nil {
		t.Fatal("InspectConfiguration() accepted missing job")
	}
}

func TestInspectCommand(t *testing.T) {
	path := filepath.Join("..", "config", "testdata", "valid", "csv.yaml")
	var output bytes.Buffer
	command := NewInspectCommand()
	command.SetOut(&output)
	command.SetArgs([]string{path})
	if err := command.Execute(); err != nil {
		t.Fatalf("inspect command: %v", err)
	}
	if !bytes.Contains(output.Bytes(), []byte(`"formatVersion": 1`)) {
		t.Fatalf("inspect output = %q", output.String())
	}

	command = NewInspectCommand()
	command.SetOut(failingInspectionWriter{})
	command.SetArgs([]string{path})
	if err := command.Execute(); err == nil {
		t.Fatal("inspect command ignored output error")
	}
	command = NewInspectCommand()
	command.SetArgs([]string{"missing.yaml"})
	if err := command.Execute(); err == nil {
		t.Fatal("inspect command accepted missing job")
	}
}

type failingInspectionWriter struct{}

func (failingInspectionWriter) Write([]byte) (int, error) {
	return 0, errors.New("write failed")
}

func TestInspectionHelpers(t *testing.T) {
	if got := secretKind(config.SecretRef{}); got != "none" {
		t.Fatalf("secretKind(empty) = %q", got)
	}
	if got := endpointNamespace("default", "configured"); got != "configured" {
		t.Fatalf("endpointNamespace() = %q", got)
	}
	if got := sortedValues(nil); got == nil || len(got) != 0 {
		t.Fatalf("sortedValues(nil) = %#v, want non-nil empty slice", got)
	}

	job := inspectionJob()
	job.Source.Type = "future-source"
	job.Source.CSV = nil
	report := BuildInspection(job)
	if len(report.Source.VertexMappings) != 0 || report.Source.Type != "future-source" {
		t.Fatalf("unknown source inspection = %#v", report.Source)
	}
}

func inspectionJob() config.LoadJob {
	header := true
	return config.LoadJob{
		APIVersion: config.APIVersion,
		Kind:       config.KindLoadJob,
		Metadata:   config.Metadata{Name: "inspect-test"},
		Source: config.Source{
			Type: config.SourceCSV, Namespace: "crm",
			CSV: &config.CSVSource{
				Defaults: config.DelimitedOptions{Delimiter: ",", Quote: `"`, Header: &header, Encoding: "utf-8"},
				Vertices: []config.CSVVertex{{
					Label: "Person", Path: "/data/vertices.csv", IDColumn: "id",
					Properties: map[string]string{"name": "display_name", "score": "score"},
				}},
				Edges: []config.CSVEdge{{
					Label: "KNOWS", Path: "/data/edges.csv", ExternalIDColumn: "edge_id",
					Start: config.EndpointMapping{Label: "Person", Field: "start_id"},
					End:   config.EndpointMapping{Label: "Person", Field: "end_id"},
				}},
			},
		},
		Target: config.Target{
			Type: config.TargetApacheAGE, Graph: "people", Mode: config.LoadAppend,
			Connection:      config.SecretRef{Env: "SECRET_DSN"},
			PropertyMode:    config.PropertiesMerge,
			AppendDuplicate: config.AppendDuplicateIgnoreIdentical,
		},
		Runtime: config.Runtime{
			MemoryLimit: 1 << 30, BatchRows: 100, BatchBytes: 1 << 20,
			MaxSourceConcurrency: 1, MaxTransformConcurrency: 1,
			MaxTargetConnections: 2, OperationTimeout: config.Duration(30_000_000_000),
		},
		Errors: config.ErrorPolicies{
			MalformedRecord: config.MalformedFail,
			MissingEndpoint: config.MissingEndpointError,
		},
	}
}
