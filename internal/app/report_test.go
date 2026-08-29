package app

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/report"
)

func TestBuildMigrationReportDeterministicAndCompatible(t *testing.T) {
	generatedAt := time.Date(2026, 8, 28, 6, 0, 0, 0, time.UTC)
	snapshot := testMigrationReportSnapshot(14)
	document, err := buildMigrationReport(snapshot, false, generatedAt)
	if err != nil {
		t.Fatalf("buildMigrationReport() error = %v", err)
	}
	firstJSON, err := report.Render(document, report.FormatJSON)
	if err != nil {
		t.Fatalf("Render(JSON) error = %v", err)
	}
	firstMarkdown, err := report.Render(document, report.FormatMarkdown)
	if err != nil {
		t.Fatalf("Render(Markdown) error = %v", err)
	}
	document, err = buildMigrationReport(snapshot, false, generatedAt)
	if err != nil {
		t.Fatalf("second buildMigrationReport() error = %v", err)
	}
	secondJSON, _ := report.Render(document, report.FormatJSON)
	secondMarkdown, _ := report.Render(document, report.FormatMarkdown)
	if !bytes.Equal(firstJSON, secondJSON) ||
		!bytes.Equal(firstMarkdown, secondMarkdown) {
		t.Fatal("migration report is not deterministic")
	}
	for _, expected := range [][]byte{
		[]byte(`"value": "14"`),
		[]byte(`connector telemetry requires metadata schema v15`),
		[]byte(`"outcome": "incomplete"`),
	} {
		if !bytes.Contains(firstJSON, expected) {
			t.Fatalf("report missing %q:\n%s", expected, firstJSON)
		}
	}
	if bytes.Contains(firstJSON, []byte("private-resume-token")) {
		t.Fatalf("report leaked a raw resume token:\n%s", firstJSON)
	}
}

func TestBuildMigrationReportIncludesTelemetryAndExactCounts(t *testing.T) {
	snapshot := testMigrationReportSnapshot(meta.SupportedSchemaVersion)
	snapshot.TelemetryAvailable = true
	snapshot.Telemetry = meta.ConnectorTelemetry{
		JobID: snapshot.Job.ID, Connector: "cosmos-nosql",
		Pages: 5, RequestCharge: 9.25, ThrottledRequests: 1,
		ContinuationDigest: "abcdef123456",
		RecordedAt:         snapshot.Job.UpdatedAt,
	}
	snapshot.Counts[snapshot.Labels[0].ID] = countResult{
		Count: 2, Status: report.CheckPass,
	}
	document, err := buildMigrationReport(
		snapshot,
		true,
		time.Date(2026, 8, 28, 6, 0, 0, 0, time.UTC),
	)
	if err != nil {
		t.Fatalf("buildMigrationReport() error = %v", err)
	}
	output, err := report.Render(document, report.FormatJSON)
	if err != nil {
		t.Fatalf("Render() error = %v", err)
	}
	if document.Outcome != report.OutcomePass {
		t.Fatalf("Outcome = %q, want %q", document.Outcome, report.OutcomePass)
	}
	for _, expected := range [][]byte{
		[]byte(`"connector"`),
		[]byte(`"cosmos-nosql"`),
		[]byte(`exactIdentityCount`),
		[]byte(`"value": "2"`),
	} {
		if !bytes.Contains(output, expected) {
			t.Fatalf("report missing %q:\n%s", expected, output)
		}
	}
}

func TestClassifyCountNeverTreatsFailureAsZero(t *testing.T) {
	cancelled := classifyCount(0, context.DeadlineExceeded)
	if cancelled.Status != report.CheckUnknown || cancelled.Detail == "" {
		t.Fatalf("deadline count result = %#v", cancelled)
	}
	permission := classifyCount(0, &pgconn.PgError{Code: "42501"})
	if permission.Status != report.CheckUnknown || permission.Detail == "" {
		t.Fatalf("permission count result = %#v", permission)
	}
	failed := classifyCount(0, errors.New("corrupt relation"))
	if failed.Status != report.CheckFail || failed.Detail == "" {
		t.Fatalf("failed count result = %#v", failed)
	}
}

func TestMigrationReportOptionValidation(t *testing.T) {
	if _, err := MigrationReport(
		t.Context(),
		"missing.yaml",
		"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
		ReportOptions{LimitBatches: -1},
	); err == nil || !strings.Contains(err.Error(), "batch report limit") {
		t.Fatalf("MigrationReport(invalid limit) error = %v", err)
	}
	if _, err := MigrationReport(
		t.Context(),
		"missing.yaml",
		"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
		ReportOptions{LimitBatches: MaxReportBatches + 1},
	); err == nil {
		t.Fatal("MigrationReport() accepted oversized batch limit")
	}
}

func testMigrationReportSnapshot(schemaVersion int) migrationReportSnapshot {
	created := time.Date(2026, 8, 28, 5, 0, 0, 0, time.UTC)
	completed := created.Add(time.Minute)
	jobID := "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
	state := meta.SchemaCurrent
	if schemaVersion < meta.SupportedSchemaVersion {
		state = meta.SchemaPending
	}
	return migrationReportSnapshot{
		Probe: age.DegradedProbe{
			PostgreSQLVersion: "17.9", PostgreSQLStatus: age.ProbePass,
			AGEVersion: "1.6.0", AGEVersionStatus: age.ProbePass,
		},
		Schema: meta.SchemaInspection{
			State: state, InstalledVersion: schemaVersion,
			SupportedVersion: meta.SupportedSchemaVersion,
		},
		Job: meta.Job{
			ID: jobID, Name: "people", SourceType: "csv", LoadMode: "create",
			TargetGraph: "people", ConfigFingerprint: strings.Repeat("a", 64),
			Status: meta.JobCommitted, GraphGenerationID: 1, NextBatchID: 2,
			CommittedRows: 2, CommittedBytes: 100, CreatedAt: created,
			StartedAt: &created, UpdatedAt: completed, CompletedAt: &completed,
		},
		Graph: meta.GraphGeneration{
			ID: 1, JobID: jobID, GraphName: "people",
			GraphOID: 42, NamespaceOID: 42, Generation: 1,
			State: meta.GenerationActive, CreatedAt: created, UpdatedAt: completed,
		},
		GraphAvailable: true,
		Labels: []meta.LabelGeneration{{
			ID: 2, GraphGenerationID: 1, LabelName: "Person",
			Kind: meta.VertexLabel, GraphNamespaceOID: 42,
			LabelID: 1, RelationOID: 43, SequenceOID: 44,
			MappingGeneration: 1, CreatedAt: created, UpdatedAt: completed,
		}},
		LatestBatch: meta.BatchAttempt{
			JobID: jobID, BatchID: 1, Attempt: 1,
			Status: meta.BatchCommitted, Rows: 2, Bytes: 100,
			Last: meta.Position{
				Resource: "vertices.csv", Line: 3, ByteOffset: 42,
				Token: "private-resume-token",
			},
			StartedAt: created, FinishedAt: &completed,
		},
		LatestBatchAvailable: true,
		Rejects: meta.RejectSummaryPage{
			Summaries:   []meta.RejectSummary{{ErrorClass: "mapping", Count: 1}},
			ScannedRows: 1,
		},
		Counts: make(map[int64]countResult),
	}
}
