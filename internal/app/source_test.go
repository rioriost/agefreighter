package app

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/reject"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
)

type diagnosticIterator struct {
	checkpoint model.SourcePosition
}

func (*diagnosticIterator) Next(context.Context) (sourcecontract.Item, error) {
	return sourcecontract.Item{}, io.EOF
}

func (*diagnosticIterator) Close() error {
	return nil
}

func (iterator *diagnosticIterator) RejectionCheckpoint() (int64, model.SourcePosition) {
	return 2, iterator.checkpoint
}

func (*diagnosticIterator) Telemetry() sourcecontract.Telemetry {
	return sourcecontract.Telemetry{Connector: "fake", Pages: 3}
}

type plainIterator struct{}

func (*plainIterator) Next(context.Context) (sourcecontract.Item, error) {
	return sourcecontract.Item{}, io.EOF
}

func (*plainIterator) Close() error {
	return nil
}

func TestSourceContracts(t *testing.T) {
	position := model.SourcePosition{Connector: "fake", Token: "checkpoint"}
	iterator := &diagnosticIterator{checkpoint: position}
	count, gotPosition := sourceRejectionCheckpoint(iterator)
	if count != 2 || gotPosition != position {
		t.Fatalf("sourceRejectionCheckpoint = (%d, %#v)", count, gotPosition)
	}
	telemetry := sourceTelemetry(iterator)
	if telemetry == nil || telemetry.Connector != "fake" || telemetry.Pages != 3 {
		t.Fatalf("sourceTelemetry = %#v", telemetry)
	}

	plain := &plainIterator{}
	count, gotPosition = sourceRejectionCheckpoint(plain)
	if count != 0 || gotPosition != (model.SourcePosition{}) {
		t.Fatalf("plain checkpoint = (%d, %#v)", count, gotPosition)
	}
	if telemetry := sourceTelemetry(plain); telemetry != nil {
		t.Fatalf("plain telemetry = %#v", telemetry)
	}
}

func TestValidateImplementedSource(t *testing.T) {
	job := testLoadJob("graph", "vertices.csv", "edges.csv")
	if err := validateImplementedSource(job); err != nil {
		t.Fatalf("CSV source error = %v", err)
	}
	job.Source.CSV = nil
	if err := validateImplementedSource(job); err == nil {
		t.Fatal("accepted missing CSV configuration")
	}
	job.Source.Type = config.SourceCosmos
	job.Source.Cosmos = &config.CosmosSource{}
	if err := validateImplementedSource(job); err != nil {
		t.Fatalf("Cosmos source error = %v", err)
	}
	job.Source.Cosmos = nil
	if err := validateImplementedSource(job); err == nil {
		t.Fatal("accepted missing Cosmos configuration")
	}
	job.Source.Type = config.SourcePostgreSQL
	job.Source.PostgreSQL = &config.PostgreSQLSource{}
	if err := validateImplementedSource(job); err != nil {
		t.Fatalf("PostgreSQL source error = %v", err)
	}
	job.Source.PostgreSQL = nil
	if err := validateImplementedSource(job); err == nil {
		t.Fatal("accepted missing PostgreSQL configuration")
	}
	job.Source.Type = config.SourceNeo4j
	job.Source.Neo4j = &config.Neo4jSource{}
	if err := validateImplementedSource(job); err != nil {
		t.Fatalf("Neo4j source error = %v", err)
	}
	job.Source.Neo4j = nil
	if err := validateImplementedSource(job); err == nil {
		t.Fatal("accepted missing Neo4j configuration")
	}
}

func TestConfiguredCosmosLabels(t *testing.T) {
	job := testLoadJob("graph", "vertices.csv", "edges.csv")
	job.Source = config.Source{
		Type: config.SourceCosmos, Namespace: "crm",
		Cosmos: &config.CosmosSource{
			Vertices: []config.CosmosVertexQuery{
				{Label: "Person"},
				{Label: "Organization"},
			},
			Edges: []config.CosmosEdgeQuery{{
				Label: "WORKS_AT",
				Start: config.EndpointMapping{Label: "Person"},
				End:   config.EndpointMapping{Label: "Organization"},
			}},
		},
	}

	labels, err := configuredLabels(job)
	if err != nil {
		t.Fatalf("configuredLabels: %v", err)
	}
	if labels["Person"] != age.VertexLabel ||
		labels["Organization"] != age.VertexLabel ||
		labels["WORKS_AT"] != age.EdgeLabel {
		t.Fatalf("configuredLabels = %#v", labels)
	}

	job.Source.Cosmos.Edges[0].Label = "Person"
	if _, err := configuredLabels(job); err == nil {
		t.Fatal("accepted Cosmos vertex/edge label collision")
	}
	job.Source.Cosmos.Edges[0].Label = "WORKS_AT"
	job.Source.Cosmos.Edges[0].Start.Label = "Missing"
	if _, err := configuredLabels(job); err == nil {
		t.Fatal("accepted missing Cosmos endpoint label")
	}
	job.Source.Cosmos = nil
	if _, err := configuredLabels(job); err == nil {
		t.Fatal("accepted missing Cosmos source")
	}
}

func TestConfiguredPostgreSQLLabels(t *testing.T) {
	job := config.LoadJob{Source: config.Source{
		Type: config.SourcePostgreSQL,
		PostgreSQL: &config.PostgreSQLSource{
			Vertices: []config.VertexQuery{{Label: "Person"}},
			Edges: []config.EdgeQuery{{
				Label: "KNOWS",
				Start: config.EndpointMapping{Label: "Person"},
				End:   config.EndpointMapping{Label: "Person"},
			}},
		},
	}}
	labels, err := configuredLabels(job)
	if err != nil {
		t.Fatalf("configuredLabels() error = %v", err)
	}

	if labels["Person"] != age.VertexLabel ||
		labels["KNOWS"] != age.EdgeLabel {
		t.Fatalf("configuredLabels() = %#v", labels)
	}
	job.Source.PostgreSQL = nil
	if _, err := configuredLabels(job); err == nil {
		t.Fatal("configuredLabels() accepted missing PostgreSQL source")
	}
}

func TestConfiguredNeo4jLabels(t *testing.T) {
	job := config.LoadJob{Source: config.Source{
		Type: config.SourceNeo4j,
		Neo4j: &config.Neo4jSource{
			Vertices: []config.VertexQuery{{Label: "Person"}},
			Edges: []config.EdgeQuery{{
				Label: "KNOWS",
				Start: config.EndpointMapping{Label: "Person"},
				End:   config.EndpointMapping{Label: "Person"},
			}},
		},
	}}
	labels, err := configuredLabels(job)
	if err != nil {
		t.Fatalf("configuredLabels() error = %v", err)
	}
	if labels["Person"] != age.VertexLabel ||
		labels["KNOWS"] != age.EdgeLabel {
		t.Fatalf("configuredLabels() = %#v", labels)
	}
	job.Source.Neo4j = nil
	if _, err := configuredLabels(job); err == nil {
		t.Fatal("configuredLabels() accepted missing Neo4j source")
	}
}

func TestNewCosmosSourceIteratorConstruction(t *testing.T) {
	job := testLoadJob("graph", "vertices.csv", "edges.csv")
	job.Source = config.Source{
		Type: config.SourceCosmos, Namespace: "crm",
		Cosmos: &config.CosmosSource{
			Endpoint:   "https://example.documents.azure.com:443/",
			Credential: "default-azure", Database: "graphdb", PageSize: 10,
			Vertices: []config.CosmosVertexQuery{{
				Container: "vertices", Label: "Person",
				Query: "SELECT * FROM c", IDField: "/id",
			}},
		},
	}
	iterator, err := newSourceIterator(t.Context(), job, "", nil)
	if err != nil {
		t.Fatalf("newSourceIterator: %v", err)
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestWriteSourceRejectionRequiresWriter(t *testing.T) {
	if err := writeSourceRejection(
		t.Context(),
		nil,
		model.SourcePosition{Token: "token"},
		nil,
		errors.New("bad record"),
	); err == nil {
		t.Fatal("writeSourceRejection accepted a nil writer")
	}
}

func TestWriteSourceRejectionPreservesCSVFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rejects.jsonl")
	writer, err := reject.NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter: %v", err)
	}
	if err := writeSourceRejection(
		t.Context(),
		writer,
		model.SourcePosition{Connector: "csv", Token: "token"},
		[]string{"broken", "row"},
		errors.New("bad record"),
	); err != nil {
		t.Fatalf("writeSourceRejection: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if !strings.Contains(string(content), `"fields":["broken","row"]`) {
		t.Fatalf("quarantine entry lost CSV fields: %s", content)
	}
}
