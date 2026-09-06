package app

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/report"
	"go.yaml.in/yaml/v3"
)

func csvInventoryJob(t *testing.T, vertices, edges string) (config.LoadJob, string) {
	t.Helper()
	dir := t.TempDir()
	v := filepath.Join(dir, "v.csv")
	e := filepath.Join(dir, "e.csv")
	for path, data := range map[string]string{v: vertices, e: edges} {
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	job := testLoadJob("unused", v, e)
	job.Target.Connection = config.SecretRef{Env: "INVENTORY_TARGET_MUST_NOT_BE_READ"}
	job.Source.CSV.Vertices[0].Properties = map[string]string{"name": "name"}
	job.Source.CSV.Edges[0].Properties = map[string]string{}
	return job, filepath.Join(dir, "job.yaml")
}

func TestSourceInventoryCSVCompleteMappedCounts(t *testing.T) {
	job, path := csvInventoryJob(t, "id,name\np1,\"日本語\nquoted\"\np2,B\n", "id,start,end\ne1,p1,p2\n")
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	doc, err := SourceInventory(t.Context(), path, InventoryOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if doc.Outcome != report.OutcomePass || doc.Command != "inventory" {
		t.Fatalf("%+v", doc)
	}
	fields := sectionByTitle(t, doc.Sections, "Source inventory").Fields
	for key, value := range map[string]string{"vertices": "2", "edges": "1", "totalRows": "3", "countMethod": "csv-complete-stream"} {
		if fieldByName(t, fields, key).Value != value {
			t.Fatalf("wrong %s", key)
		}
	}
	capacity := sectionByTitle(t, doc.Sections, "Capacity indicators")
	if fieldByName(t, capacity.Fields, "method").Value != "complete-stream-range" {
		t.Fatal("not full capacity")
	}
	b, err := report.Render(doc, report.FormatJSON)
	if err != nil {
		t.Fatal(err)
	}
	for _, private := range []string{"日本語", "quoted", path, "INVENTORY_TARGET_MUST_NOT_BE_READ"} {
		if strings.Contains(string(b), private) {
			t.Fatalf("leaked %q", private)
		}
	}
}

func TestCSVInventoryRejectsMalformedDespiteQuarantinePolicy(t *testing.T) {
	job, _ := csvInventoryJob(t, "id,name\np1,A,extra\n", "id,start,end\n")
	job.Errors.MalformedRecord = config.MalformedQuarantine
	job.Errors.RejectLimit = 10
	job.Errors.QuarantinePath = filepath.Join(t.TempDir(), "must-not-exist")
	doc, err := csvSourceInventory(t.Context(), job, InventoryOptions{})
	if err == nil || doc.Outcome == report.OutcomePass {
		t.Fatal("malformed inventory passed")
	}
	if _, err := os.Stat(job.Errors.QuarantinePath); !os.IsNotExist(err) {
		t.Fatal("quarantine written")
	}
}

func TestCSVInventoryEmptyAndCancellation(t *testing.T) {
	job, _ := csvInventoryJob(t, "id,name\n", "id,start,end\n")
	doc, err := csvSourceInventory(t.Context(), job, InventoryOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if fieldByName(t, sectionByTitle(t, doc.Sections, "Source inventory").Fields, "totalRows").Value != "0" {
		t.Fatal("not empty")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := csvSourceInventory(ctx, job, InventoryOptions{}); err == nil {
		t.Fatal("canceled inventory passed")
	}
	job.Runtime.OperationTimeout = config.Duration(time.Nanosecond)
	if _, err := csvSourceInventory(t.Context(), job, InventoryOptions{}); err == nil {
		t.Fatal("timeout inventory passed")
	}
}

func TestCSVInventoryFingerprintChangesAndBounds(t *testing.T) {
	job, _ := csvInventoryJob(t, "id,name\np1,A\n", "id,start,end\n")
	first, _, err := csvInventorySnapshot(t.Context(), *job.Source.CSV)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(job.Source.CSV.Vertices[0].Path, []byte("id,name\np1,B\n"), 0600); err != nil {
		t.Fatal(err)
	}
	second, _, err := csvInventorySnapshot(t.Context(), *job.Source.CSV)
	if err != nil || first == second {
		t.Fatal("changed bytes accepted")
	}
	job.Source.CSV.Vertices[0].Path = t.TempDir()
	if _, _, err = csvInventorySnapshot(t.Context(), *job.Source.CSV); err == nil {
		t.Fatal("directory accepted")
	}
	if _, _, err = csvInventorySnapshot(t.Context(), config.CSVSource{}); err == nil {
		t.Fatal("empty mapping accepted")
	}
}
