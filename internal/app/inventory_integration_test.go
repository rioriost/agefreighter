package app

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/report"
)

func TestSourceInventoryPostgreSQLIntegration(t *testing.T) {
	dsn := os.Getenv("AGEFREIGHTER_POSTGRES_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_POSTGRES_TEST_DSN")
	}
	t.Setenv("INVENTORY_SOURCE_DSN", dsn)
	t.Setenv("INVENTORY_TARGET_MUST_NOT_BE_READ", "")
	job, err := config.Load(filepath.Join("..", "config", "testdata", "valid", "postgresql.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	job.Target.Connection = config.SecretRef{Env: "INVENTORY_TARGET_MUST_NOT_BE_READ"}
	job.Source.PostgreSQL.Connection = config.SecretRef{Env: "INVENTORY_SOURCE_DSN"}
	job.Source.PostgreSQL.Vertices[0].Query = `SELECT person_id, full_name
		FROM (VALUES ('p1', 'PRIVATE-INVENTORY-VALUE'), ('p2', 'PRIVATE-INVENTORY-VALUE'))
		AS people(person_id, full_name) ORDER BY person_id`
	job.Source.PostgreSQL.Edges[0].Query = `SELECT 'e1'::text AS relationship_id,
		'p1'::text AS from_id, 'p2'::text AS to_id ORDER BY relationship_id`
	job.Errors.MalformedRecord = config.MalformedQuarantine
	job.Errors.RejectLimit = 10
	job.Errors.QuarantinePath = filepath.Join(t.TempDir(), "must-not-exist.jsonl")
	at := time.Date(2026, 9, 25, 0, 0, 0, 0, time.UTC)

	t.Run("complete-source-only", func(t *testing.T) {
		path := writeLoadJob(t, t.TempDir(), "inventory.yaml", job)
		doc, err := SourceInventory(t.Context(), path, InventoryOptions{GeneratedAt: at})
		if err != nil {
			t.Fatal(err)
		}
		if doc.Outcome != report.OutcomePass || doc.Command != "inventory" || !doc.GeneratedAt.Equal(at) ||
			doc.Target != nil || doc.Job != nil || len(doc.Errors) != 0 || len(doc.IncompleteChecks) != 0 {
			t.Fatalf("unexpected inventory: %#v", doc)
		}
		fields := sectionByTitle(t, doc.Sections, "Source inventory").Fields
		for key, want := range map[string]string{
			"connector": "postgresql", "countMethod": "postgresql-repeatable-read-complete-stream",
			"vertices": "2", "edges": "1", "totalRows": "3",
		} {
			if got := fieldByName(t, fields, key); got.Value != want || got.Status != report.CheckPass {
				t.Fatalf("%s = %#v, want %s", key, got, want)
			}
		}
		counts := sectionByTitle(t, doc.Sections, "Mapped record counts").Fields
		if len(counts) != 2 || counts[0].Name != "edge:KNOWS" || counts[0].Value != "1" ||
			counts[1].Name != "vertex:Person" || counts[1].Value != "2" {
			t.Fatalf("mapped counts = %#v", counts)
		}
		for _, field := range sectionByTitle(t, doc.Sections, "Capacity indicators").Fields {
			if field.Name == "estimatedMigrationTime" {
				t.Fatal("inventory claimed migration throughput")
			}
		}
		data, err := report.Render(doc, report.FormatJSON)
		if err != nil {
			t.Fatal(err)
		}
		for _, private := range []string{"PRIVATE-INVENTORY-VALUE", "INVENTORY_SOURCE_DSN", dsn, path} {
			if strings.Contains(string(data), private) {
				t.Fatal("inventory disclosed private source evidence")
			}
		}
	})

	t.Run("malformed-fails-without-quarantine", func(t *testing.T) {
		job.Source.PostgreSQL.Vertices[0].Query = `SELECT NULL::text AS person_id,
			'PRIVATE-INVENTORY-VALUE'::text AS full_name ORDER BY person_id`
		path := writeLoadJob(t, t.TempDir(), "malformed.yaml", job)
		doc, err := SourceInventory(t.Context(), path, InventoryOptions{})
		if err == nil || doc.Outcome == report.OutcomePass {
			t.Fatalf("malformed inventory = %#v, %v", doc, err)
		}
		if strings.Contains(err.Error(), "PRIVATE-INVENTORY-VALUE") || strings.Contains(err.Error(), dsn) {
			t.Fatal("inventory error disclosed source evidence")
		}
	})
	if _, err := os.Stat(job.Errors.QuarantinePath); !os.IsNotExist(err) {
		t.Fatalf("inventory wrote quarantine: %v", err)
	}
}

func TestSourceInventoryNeo4jIntegration(t *testing.T) {
	uri := os.Getenv("AGEFREIGHTER_NEO4J_TEST_URI")
	database := os.Getenv("AGEFREIGHTER_NEO4J_TEST_DATABASE")
	if uri == "" || database == "" {
		t.Skip("set Neo4j source test settings")
	}
	job, err := config.Load(filepath.Join("..", "config", "testdata", "valid", "neo4j-discovery.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	job.Source.Neo4j.URI = uri
	job.Source.Neo4j.Database = database
	job.Source.Neo4j.Username = os.Getenv("AGEFREIGHTER_NEO4J_TEST_USERNAME")
	if job.Source.Neo4j.Username == "" {
		job.Source.Neo4j.Password = nil
	} else {
		t.Setenv("INVENTORY_NEO4J_PASSWORD", os.Getenv("AGEFREIGHTER_NEO4J_TEST_PASSWORD"))
		job.Source.Neo4j.Password = &config.SecretRef{Env: "INVENTORY_NEO4J_PASSWORD"}
	}
	t.Setenv("INVENTORY_TARGET_MUST_NOT_BE_READ", "")
	job.Target.Connection = config.SecretRef{Env: "INVENTORY_TARGET_MUST_NOT_BE_READ"}
	path := writeLoadJob(t, t.TempDir(), "inventory.yaml", job)
	for _, at := range []time.Time{{}, time.Date(2026, 9, 25, 0, 0, 0, 0, time.UTC)} {
		doc, err := SourceInventory(t.Context(), path, InventoryOptions{GeneratedAt: at})
		if err != nil {
			t.Fatal(err)
		}
		if doc.Outcome != report.OutcomePass || doc.Command != "inventory" || doc.GeneratedAt.IsZero() ||
			doc.Target != nil || doc.Job != nil || len(doc.Errors) != 0 || len(doc.IncompleteChecks) != 0 {
			t.Fatalf("unexpected inventory: %#v", doc)
		}
		if !at.IsZero() && !doc.GeneratedAt.Equal(at) {
			t.Fatalf("generatedAt = %v, want %v", doc.GeneratedAt, at)
		}
		fields := sectionByTitle(t, doc.Sections, "Source inventory").Fields
		if fieldByName(t, fields, "countMethod").Value != "neo4j-transactional-count-store" {
			t.Fatalf("unexpected count method: %#v", fields)
		}
		counts := map[string]int64{}
		for _, key := range []string{"vertices", "edges", "totalRows"} {
			field := fieldByName(t, fields, key)
			count, err := strconv.ParseInt(field.Value, 10, 64)
			if err != nil || count < 0 || field.Status != report.CheckPass {
				t.Fatalf("%s = %#v: %v", key, field, err)
			}
			counts[key] = count
		}
		if counts["totalRows"] != counts["vertices"]+counts["edges"] {
			t.Fatalf("inconsistent totals: %v", counts)
		}
	}
}
