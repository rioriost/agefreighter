package app

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/report"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
)

func networkInventoryJob(sourceType config.SourceType) config.LoadJob {
	job := config.LoadJob{
		Source: config.Source{Type: sourceType, Namespace: "inventory"},
		Target: config.Target{Type: config.TargetApacheAGE, Mode: config.LoadCreate},
	}
	if sourceType == config.SourcePostgreSQL {
		job.Source.PostgreSQL = &config.PostgreSQLSource{
			Vertices: []config.VertexQuery{{Label: "Person"}},
			Edges:    []config.EdgeQuery{{Label: "KNOWS"}},
		}
	} else {
		job.Source.Cosmos = &config.CosmosSource{
			Vertices: []config.CosmosVertexQuery{{Label: "Person"}},
			Edges:    []config.CosmosEdgeQuery{{Label: "KNOWS"}},
		}
	}
	return job
}

func networkInventoryItems() []sourcecontract.Item {
	person := func(id string) sourcecontract.Item {
		return sourcecontract.Item{Record: model.VertexRecord(model.Vertex{
			Label: "Person", ExternalID: model.ExternalID(id),
			Properties: model.Properties{"name": {Kind: model.ValueString, String: "redacted"}},
		})}
	}
	return []sourcecontract.Item{
		person("p1"), person("p2"),
		{Record: model.EdgeRecord(model.Edge{
			Label: "KNOWS", ExternalID: "e1",
			Start: model.Endpoint{Label: "Person", ExternalID: "p1"},
			End:   model.Endpoint{Label: "Person", ExternalID: "p2"},
		})},
	}
}

func TestNetworkInventoryConsumesEveryMappedRecordWithoutDisclosingValues(t *testing.T) {
	for _, test := range []struct {
		typeName config.SourceType
		method   string
	}{
		{config.SourcePostgreSQL, "postgresql-repeatable-read-complete-stream"},
		{config.SourceCosmos, "cosmos-nosql-complete-stream"},
	} {
		t.Run(string(test.typeName), func(t *testing.T) {
			iterator := &scriptedProfileIterator{items: networkInventoryItems()}
			doc, err := consumeNetworkInventory(t.Context(), networkInventoryJob(test.typeName), iterator, InventoryOptions{
				GeneratedAt: time.Date(2026, 9, 7, 0, 0, 0, 0, time.UTC),
			})
			if err != nil || doc.Outcome != report.OutcomePass || doc.Command != "inventory" {
				t.Fatalf("inventory = %#v, %v", doc, err)
			}
			inventory := sectionByTitle(t, doc.Sections, "Source inventory")
			for name, want := range map[string]string{
				"vertices": "2", "edges": "1", "totalRows": "3", "countMethod": test.method,
			} {
				if got := fieldByName(t, inventory.Fields, name).Value; got != want {
					t.Fatalf("%s = %q, want %q", name, got, want)
				}
			}
			labels := sectionByTitle(t, doc.Sections, "Mapped record counts")
			if fieldByName(t, labels.Fields, "vertex:Person").Value != "2" || fieldByName(t, labels.Fields, "edge:KNOWS").Value != "1" {
				t.Fatalf("labels = %#v", labels)
			}
			capacity := sectionByTitle(t, doc.Sections, "Capacity indicators")
			if fieldByName(t, capacity.Fields, "method").Value != "complete-stream-range" {
				t.Fatalf("capacity = %#v", capacity)
			}
			rendered, renderErr := report.Render(doc, report.FormatJSON)
			if renderErr != nil || strings.Contains(string(rendered), "redacted") || strings.Contains(string(rendered), "p1") {
				t.Fatalf("inventory disclosed source values: %v", renderErr)
			}
		})
	}
}

func TestNetworkInventoryRejectsPartialOrInvalidStreams(t *testing.T) {
	job := networkInventoryJob(config.SourcePostgreSQL)
	for _, iterator := range []*scriptedProfileIterator{
		{items: networkInventoryItems()[:1], err: errors.New("source changed")},
		{items: []sourcecontract.Item{{Record: model.Record{}}}},
	} {
		if _, err := consumeNetworkInventory(t.Context(), job, iterator, InventoryOptions{}); err == nil {
			t.Fatal("partial or invalid stream was accepted as exact")
		}
	}
}

type inventoryCloseFailure struct {
	*scriptedProfileIterator
	closed bool
}

func (iterator *inventoryCloseFailure) Close() error {
	iterator.closed = true
	return errors.New("PRIVATE-CLOSE-ERROR")
}

func TestNetworkInventoryCloseFailureCannotPass(t *testing.T) {
	iterator := &inventoryCloseFailure{
		scriptedProfileIterator: &scriptedProfileIterator{items: networkInventoryItems()},
	}
	doc, err := consumeNetworkInventory(t.Context(), networkInventoryJob(config.SourcePostgreSQL), iterator, InventoryOptions{})
	if err == nil || err.Error() != "network inventory source close failed" ||
		doc.Outcome == report.OutcomePass || !iterator.closed {
		t.Fatalf("close failure = %#v, %v; closed=%v", doc, err, iterator.closed)
	}
}

func TestSourceInventoryMissingCredentialsFailBeforeSourceAccess(t *testing.T) {
	t.Setenv("INVENTORY_MISSING_SECRET", "")
	for _, connector := range []string{"postgresql", "neo4j-discovery"} {
		t.Run(connector, func(t *testing.T) {
			job, err := config.Load(filepath.Join("..", "config", "testdata", "valid", connector+".yaml"))
			if err != nil {
				t.Fatal(err)
			}
			ref := config.SecretRef{Env: "INVENTORY_MISSING_SECRET"}
			want := "network inventory initialization failed"
			if job.Source.PostgreSQL != nil {
				job.Source.PostgreSQL.Connection = ref
			} else {
				job.Source.Neo4j.Password = &ref
				want = "resolve Neo4j source password:"
			}
			path := writeLoadJob(t, t.TempDir(), "inventory.yaml", job)
			doc, err := SourceInventory(t.Context(), path, InventoryOptions{})
			if err == nil || !strings.HasPrefix(err.Error(), want) || doc.Outcome == report.OutcomePass {
				t.Fatalf("missing credentials = %#v, %v", doc, err)
			}
		})
	}
}
