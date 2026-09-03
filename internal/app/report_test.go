package app

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/report"
	"go.yaml.in/yaml/v3"
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
		"not-a-job-id",
		ReportOptions{},
	); err == nil {
		t.Fatal("MigrationReport() accepted invalid job ID")
	}
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
	if _, err := MigrationReport(
		t.Context(),
		"missing.yaml",
		"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
		ReportOptions{},
	); err == nil || !strings.Contains(err.Error(), "load target configuration") {
		t.Fatalf("MigrationReport(missing config) error = %v", err)
	}
}

func TestMigrationReportCredentialFailure(t *testing.T) {
	const missing = "AGEFREIGHTER_REPORT_MISSING_DSN"
	t.Setenv(missing, "")
	job := testLoadJob("graph", "vertices.csv", "edges.csv")
	job.Target.Connection = config.SecretRef{Env: missing}
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "job.yaml")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := MigrationReport(
		t.Context(), path,
		"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
		ReportOptions{},
	); err == nil {
		t.Fatal("missing target credential was accepted")
	}
}

func TestMigrationReportStateMatrix(t *testing.T) {
	tests := []struct {
		name  string
		edit  func(*migrationReportSnapshot)
		count bool
		want  report.Outcome
		code  string
	}{
		{"failed job", func(s *migrationReportSnapshot) {
			s.Job.Status = meta.JobFailed
		}, false, report.OutcomeFail, ""},
		{"running job", func(s *migrationReportSnapshot) {
			s.Job.Status = meta.JobRunning
		}, false, report.OutcomePass, ""},
		{"missing graph", func(s *migrationReportSnapshot) {
			s.GraphAvailable = false
			s.Labels = nil
		}, false, report.OutcomeIncomplete, ""},
		{"labels truncated", func(s *migrationReportSnapshot) {
			s.LabelsTruncated = true
		}, false, report.OutcomeIncomplete, "LABELS_TRUNCATED"},
		{"batches truncated", func(s *migrationReportSnapshot) {
			s.BatchLimit = 1
			s.BatchesTruncated = true
			s.Batches = []meta.BatchAttempt{s.LatestBatch}
		}, false, report.OutcomePass, "BATCHES_TRUNCATED"},
		{"rejects truncated", func(s *migrationReportSnapshot) {
			s.Rejects.Truncated = true
			s.Rejects.ScannedRows = 100
		}, false, report.OutcomeIncomplete, "REJECTS_TRUNCATED"},
		{"unknown count", func(s *migrationReportSnapshot) {
			s.Counts[s.Labels[0].ID] = countResult{
				Status: report.CheckUnknown, Detail: "timeout",
			}
		}, true, report.OutcomeIncomplete, ""},
		{"failed count", func(s *migrationReportSnapshot) {
			s.Counts[s.Labels[0].ID] = countResult{
				Status: report.CheckFail, Detail: "corrupt",
			}
		}, true, report.OutcomeFail, "IDENTITY_COUNT_ERROR"},
		{"missing count", func(*migrationReportSnapshot) {}, true, report.OutcomeIncomplete, ""},
		{"no labels", func(s *migrationReportSnapshot) {
			s.Labels = nil
		}, true, report.OutcomePass, ""},
		{"no latest batch", func(s *migrationReportSnapshot) {
			s.LatestBatchAvailable = false
		}, false, report.OutcomePass, ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			snapshot := testMigrationReportSnapshot(meta.SupportedSchemaVersion)
			snapshot.TelemetryAvailable = true
			test.edit(&snapshot)
			document, err := buildMigrationReport(snapshot, test.count, time.Unix(1, 0))
			if err != nil {
				t.Fatal(err)
			}
			if document.Outcome != test.want {
				t.Fatalf("outcome = %s, want %s", document.Outcome, test.want)
			}
			if test.code != "" {
				found := false
				for _, finding := range append(document.Warnings, document.Errors...) {
					found = found || finding.Code == test.code
				}
				if !found {
					t.Fatalf("finding %q missing: %#v %#v", test.code, document.Warnings, document.Errors)
				}
			}
		})
	}
}

func TestReportFormattingHelpers(t *testing.T) {
	if got := classifyCount(7, nil); got.Count != 7 || got.Status != report.CheckPass {
		t.Fatalf("successful count = %#v", got)
	}
	if got := classifyCount(0, &pgconn.PgError{Code: "57014"}); got.Status != report.CheckUnknown {
		t.Fatalf("statement timeout = %#v", got)
	}
	if got := classifyCount(0, context.Canceled); got.Status != report.CheckUnknown {
		t.Fatalf("canceled count = %#v", got)
	}

	invalid := "prefix" + strings.Repeat("界", 400) + "\x00suffix"
	bounded := boundedReportValue(invalid)
	if len(bounded) > reportValueBytes || !strings.HasSuffix(bounded, "...") ||
		!strings.Contains(bounded, "prefix") {
		t.Fatalf("bounded report value length=%d value=%q", len(bounded), bounded)
	}
	if boundedReportValue("a\x00b") != "a\uFFFDb" {
		t.Fatal("NUL was not sanitized")
	}
	if tokenDigest("") != "" || len(tokenDigest("secret")) != 64 {
		t.Fatal("token digest contract failed")
	}
	if optionalTimeField("at", nil).Status != report.CheckUnavailable {
		t.Fatal("nil time was not unavailable")
	}
	now := time.Unix(1, 2)
	if optionalTimeField("at", &now).Status != report.CheckPass {
		t.Fatal("recorded time was not available")
	}
	for _, test := range []struct {
		status age.ProbeStatus
		want   report.CheckStatus
	}{
		{age.ProbePass, report.CheckPass},
		{age.ProbeFail, report.CheckFail},
		{age.ProbeUnavailable, report.CheckUnavailable},
		{age.ProbeUnknown, report.CheckUnknown},
	} {
		if got := versionValue("v", test.status); got.Status != test.want {
			t.Fatalf("version status = %s, want %s", got.Status, test.want)
		}
	}
}

func TestMigrationReportSectionHelpers(t *testing.T) {
	for _, test := range []struct {
		status meta.JobStatus
		want   report.CheckStatus
	}{
		{meta.JobCommitted, report.CheckPass},
		{meta.JobFailed, report.CheckFail},
		{meta.JobRunning, report.CheckWarning},
	} {
		if got := jobStatusCheck(meta.Job{Status: test.status}); got.Status != test.want {
			t.Fatalf("job status %s = %s", test.status, got.Status)
		}
		current := schemaCheck(meta.SchemaInspection{State: meta.SchemaCurrent})
		if current.Status != report.CheckPass {
			t.Fatalf("current schema = %#v", current)
		}
		pending := schemaCheck(meta.SchemaInspection{
			State: meta.SchemaPending, InstalledVersion: 16, SupportedVersion: 17,
		})
		if pending.Status != report.CheckWarning || pending.Detail == "" {
			t.Fatalf("pending schema = %#v", pending)
		}

		snapshot := testMigrationReportSnapshot(meta.SupportedSchemaVersion)
		snapshot.BatchLimit = 2
		snapshot.Batches = []meta.BatchAttempt{snapshot.LatestBatch}
		if fields := batchSection(snapshot).Fields; len(fields) != 8 {
			t.Fatalf("batch fields = %#v", fields)
		}
		snapshot.LatestBatchAvailable = false
		if fieldByName(t, batchSection(snapshot).Fields, "latest").Value != "none" {
			t.Fatal("missing latest batch was not reported")
		}

		snapshot = testMigrationReportSnapshot(meta.SupportedSchemaVersion)
		snapshot.Counts[snapshot.Labels[0].ID] = countResult{
			Status: report.CheckUnknown, Detail: "timeout",
		}
		field := fieldByName(t, labelSection(snapshot, true).Fields, "001.Person.exactIdentityCount")
		if field.Status != report.CheckUnknown || field.Value != "timeout" {
			t.Fatalf("unknown label count = %#v", field)
		}
		snapshot.GraphAvailable = false
		if labelSection(snapshot, true).Fields[0].Status != report.CheckUnavailable {
			t.Fatal("missing graph label section was not unavailable")
		}

		cleaned := time.Unix(10, 0)
		backup := backupSection(meta.Job{
			BackupGraphName: "old", BackupCleanedAt: &cleaned,
		}, meta.GraphGeneration{ReplacesGraphOID: 42})
		if fieldByName(t, backup.Fields, "retained").Value != "false" ||
			fieldByName(t, backup.Fields, "backupGraphOid").Value != "42" ||
			fieldByName(t, backup.Fields, "cleanedAt").Value == "" {
			t.Fatalf("backup section = %#v", backup)
		}
		if valueOrNone("") != "none" || valueOrNone("x") != "x" {
			t.Fatal("valueOrNone failed")
		}
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
			TargetBackend: meta.TargetBackendApacheAGE,
			TargetGraph:   "people", ConfigFingerprint: strings.Repeat("a", 64),
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
