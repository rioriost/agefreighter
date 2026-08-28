package app

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/report"
)

func TestDoctorDegradedPostgreSQLIntegration(t *testing.T) {
	dsn := strings.TrimSpace(os.Getenv("AGEFREIGHTER_POSTGRES_TEST_DSN"))
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_POSTGRES_TEST_DSN to run degraded doctor integration tests")
	}
	t.Setenv("AGEFREIGHTER_DEGRADED_TEST_DSN", dsn)
	job := testLoadJob("degraded_doctor", "unused-vertices", "unused-edges")
	job.Target.Connection.Env = "AGEFREIGHTER_DEGRADED_TEST_DSN"
	path := writeLoadJob(t, t.TempDir(), "degraded-doctor.yaml", job)

	document, err := Doctor(t.Context(), path, DoctorOptions{})
	if err != nil {
		t.Fatalf("Doctor() error = %v", err)
	}
	if document.Outcome != report.OutcomeIncomplete ||
		document.Target == nil ||
		document.Target.PostgreSQL.Status != report.CheckPass ||
		document.Target.AGE.Status != report.CheckUnavailable {
		t.Fatalf("degraded doctor report = %#v", document)
	}
	rendered, err := report.Render(document, report.FormatJSON)
	if err != nil {
		t.Fatalf("render degraded doctor: %v", err)
	}
	if bytes.Contains(rendered, []byte(dsn)) {
		t.Fatal("degraded doctor report disclosed its DSN")
	}
}

func TestReadOnlyDiagnosticsRaceIntegration(t *testing.T) {
	dsn := strings.TrimSpace(os.Getenv("AGEFREIGHTER_AGE_TEST_DSN"))
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run read-only diagnostic race tests")
	}
	path, jobID := loadReleaseReadinessFixture(t, dsn, "diagnostic_race")
	pool, err := pgx.Connect(t.Context(), dsn)
	if err != nil {
		t.Fatalf("connect metadata version check: %v", err)
	}
	defer pool.Close(context.Background())
	schemaVersion := readMetadataVersion(t, pool)

	operations := []func(context.Context) error{
		func(ctx context.Context) error {
			_, err := MigrationReport(ctx, path, jobID, ReportOptions{IncludeCounts: true})
			return err
		},
		func(ctx context.Context) error {
			_, err := VerificationReport(ctx, path, jobID, VerifyOptions{
				Counts: true, Integrity: true, Limit: 100,
			})
			return err
		},
		func(ctx context.Context) error {
			_, err := Doctor(ctx, path, DoctorOptions{})
			return err
		},
		func(ctx context.Context) error {
			_, err := OptimizationReport(ctx, path, OptimizeOptions{})
			return err
		},
		func(ctx context.Context) error {
			_, err := Verify(ctx, path, jobID)
			return err
		},
	}
	start := make(chan struct{})
	failures := make(chan error, len(operations)*2)
	var workers sync.WaitGroup
	for iteration := 0; iteration < 2; iteration++ {
		for index, operation := range operations {
			workers.Add(1)
			go func(index int, operation func(context.Context) error) {
				defer workers.Done()
				<-start
				ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
				defer cancel()
				if err := operation(ctx); err != nil {
					failures <- fmt.Errorf("diagnostic %d: %w", index, err)
				}
			}(index, operation)
		}
	}
	close(start)
	workers.Wait()
	close(failures)
	for err := range failures {
		t.Error(err)
	}
	if current := readMetadataVersion(t, pool); current != schemaVersion {
		t.Fatalf("read-only diagnostics migrated metadata from v%d to v%d", schemaVersion, current)
	}
}

func TestDeepVerificationDetectsCorruptionIntegration(t *testing.T) {
	dsn := strings.TrimSpace(os.Getenv("AGEFREIGHTER_AGE_TEST_DSN"))
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run corruption integration tests")
	}
	path, jobID := loadReleaseReadinessFixture(t, dsn, "corruption")
	connection, err := pgx.Connect(t.Context(), dsn)
	if err != nil {
		t.Fatalf("connect corruption fixture: %v", err)
	}
	defer connection.Close(context.Background())
	if _, err := connection.Exec(t.Context(), "LOAD 'age'"); err != nil {
		t.Fatalf("load AGE: %v", err)
	}
	if _, err := connection.Exec(
		t.Context(),
		`SET search_path = ag_catalog, "$user", public`,
	); err != nil {
		t.Fatalf("set AGE search path: %v", err)
	}

	var graph string
	var graphGenerationID int64
	if err := connection.QueryRow(t.Context(), `
		SELECT generation.graph_name, generation.graph_generation_id
		FROM agefreighter_meta.graph_generation generation
		WHERE generation.job_id = $1::uuid`,
		jobID,
	).Scan(&graph, &graphGenerationID); err != nil {
		t.Fatalf("read graph generation: %v", err)
	}
	type labelIdentity struct {
		generationID int64
		labelID      int32
	}
	labels := make(map[string]labelIdentity)
	rows, err := connection.Query(t.Context(), `
		SELECT label_name, label_generation_id, label_id
		FROM agefreighter_meta.label_generation
		WHERE graph_generation_id = $1`,
		graphGenerationID,
	)
	if err != nil {
		t.Fatalf("read label generations: %v", err)
	}
	for rows.Next() {
		var name string
		var value labelIdentity
		if err := rows.Scan(&name, &value.generationID, &value.labelID); err != nil {
			rows.Close()
			t.Fatalf("scan label generation: %v", err)
		}
		labels[name] = value
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		t.Fatalf("iterate label generations: %v", err)
	}
	rows.Close()

	vertexIDs := make(map[string]int64)
	rows, err = connection.Query(t.Context(), `
		SELECT external_id, graph_id
		FROM agefreighter_meta.vertex_identity
		WHERE graph_generation_id = $1`,
		graphGenerationID,
	)
	if err != nil {
		t.Fatalf("read vertex identities: %v", err)
	}
	for rows.Next() {
		var externalID string
		var graphID int64
		if err := rows.Scan(&externalID, &graphID); err != nil {
			rows.Close()
			t.Fatalf("scan vertex identity: %v", err)
		}
		vertexIDs[externalID] = graphID
	}
	rows.Close()
	edgeIDs := make(map[string]int64)
	rows, err = connection.Query(t.Context(), `
		SELECT external_id, graph_id
		FROM agefreighter_meta.edge_identity
		WHERE graph_generation_id = $1`,
		graphGenerationID,
	)
	if err != nil {
		t.Fatalf("read edge identities: %v", err)
	}
	for rows.Next() {
		var externalID string
		var graphID int64
		if err := rows.Scan(&externalID, &graphID); err != nil {
			rows.Close()
			t.Fatalf("scan edge identity: %v", err)
		}
		edgeIDs[externalID] = graphID
	}
	rows.Close()

	vertexTable := pgx.Identifier{graph, "Person"}.Sanitize()
	edgeTable := pgx.Identifier{graph, "KNOWS"}.Sanitize()
	wrongEdgeID := (int64(labels["Person"].labelID) << 48) |
		(edgeIDs["e1"] & ((int64(1) << 48) - 1))
	statements := []struct {
		sql  string
		args []any
	}{
		{
			sql: fmt.Sprintf(
				"UPDATE %s SET id = $1::text::graphid WHERE id = $2::text::graphid",
				edgeTable,
			),
			args: []any{fmt.Sprint(wrongEdgeID), fmt.Sprint(edgeIDs["e1"])},
		},
		{
			sql: fmt.Sprintf(
				"UPDATE %s SET start_id = $1::text::graphid WHERE id = $2::text::graphid",
				edgeTable,
			),
			args: []any{fmt.Sprint(vertexIDs["p3"]), fmt.Sprint(edgeIDs["e2"])},
		},
		{
			sql:  fmt.Sprintf("DELETE FROM %s WHERE id = $1::text::graphid", vertexTable),
			args: []any{fmt.Sprint(vertexIDs["p1"])},
		},
		{
			sql: `DELETE FROM agefreighter_meta.vertex_identity
				WHERE graph_generation_id = $1 AND external_id = 'p2'`,
			args: []any{graphGenerationID},
		},
		{
			sql: `UPDATE agefreighter_meta.load_job
				SET load_mode = 'replace', backup_graph_name = 'contaminated_backup'
				WHERE job_id = $1::uuid`,
			args: []any{jobID},
		},
	}
	for _, statement := range statements {
		if _, err := connection.Exec(t.Context(), statement.sql, statement.args...); err != nil {
			t.Fatalf("apply corruption %q: %v", statement.sql, err)
		}
	}

	document, err := VerificationReport(
		t.Context(),
		path,
		jobID,
		VerifyOptions{Counts: true, Integrity: true, Limit: 100},
	)
	if err != nil {
		t.Fatalf("VerificationReport() error = %v", err)
	}
	if document.Outcome != report.OutcomeFail {
		t.Fatalf("corruption outcome = %s, want fail", document.Outcome)
	}
	assertReportCheckStatus(t, document, "generation-ownership", report.CheckFail)
	assertReportCheckStatus(t, document, "catalog.e.KNOWS", report.CheckFail)
	integrity := releaseReadinessSection(t, document, "Bounded integrity")
	person := releaseReadinessField(t, integrity, "v.Person")
	knows := releaseReadinessField(t, integrity, "e.KNOWS")
	for _, evidence := range []string{"missingPhysicalRows=1", "orphanPhysicalRows=1"} {
		if !strings.Contains(person.Value, evidence) {
			t.Fatalf("person integrity %q missing %q", person.Value, evidence)
		}
	}
	for _, evidence := range []string{
		"missingPhysicalRows=1",
		"missingEndpointRows=2",
		"changedEndpointRows=1",
		"orphanPhysicalRows=1",
	} {
		if !strings.Contains(knows.Value, evidence) {
			t.Fatalf("edge integrity %q missing %q", knows.Value, evidence)
		}
	}
}

func loadReleaseReadinessFixture(t *testing.T, dsn, prefix string) (string, string) {
	t.Helper()
	dir := t.TempDir()
	vertices := filepath.Join(dir, "vertices.csv")
	edges := filepath.Join(dir, "edges.csv")
	if err := os.WriteFile(
		vertices,
		[]byte("id,name\np1,Ada\np2,Grace\np3,Katherine\n"),
		0o600,
	); err != nil {
		t.Fatalf("write release-readiness vertices: %v", err)
	}
	if err := os.WriteFile(
		edges,
		[]byte("id,start,end\ne1,p1,p2\ne2,p2,p3\n"),
		0o600,
	); err != nil {
		t.Fatalf("write release-readiness edges: %v", err)
	}
	graph := fmt.Sprintf("af_%s_%d", prefix, time.Now().UnixNano())
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)
	job := testLoadJob(graph, vertices, edges)
	path := writeLoadJob(t, dir, "job.yaml", job)
	result, err := Load(t.Context(), path)
	if result.JobID != "" {
		registerCleanup(t, dsn, graph, result.JobID)
	}
	if err != nil {
		t.Fatalf("load release-readiness fixture: %v", err)
	}
	if result.Status != meta.JobCommitted {
		t.Fatalf("release-readiness fixture status = %s", result.Status)
	}
	return path, result.JobID
}

func readMetadataVersion(t *testing.T, connection *pgx.Conn) int {
	t.Helper()
	var version int
	if err := connection.QueryRow(
		t.Context(),
		`SELECT COALESCE(MAX(version), 0)
		 FROM agefreighter_meta.schema_migration`,
	).Scan(&version); err != nil {
		t.Fatalf("read metadata version: %v", err)
	}
	return version
}

func assertReportCheckStatus(
	t *testing.T,
	document report.Document,
	id string,
	status report.CheckStatus,
) {
	t.Helper()
	for _, check := range document.Checks {
		if check.ID == id {
			if check.Status != status {
				t.Fatalf("check %s status = %s, want %s", id, check.Status, status)
			}
			return
		}
	}
	t.Fatalf("report missing check %s", id)
}

func releaseReadinessSection(
	t *testing.T,
	document report.Document,
	title string,
) report.Section {
	t.Helper()
	for _, section := range document.Sections {
		if section.Title == title {
			return section
		}
	}
	t.Fatalf("report missing section %s", title)
	return report.Section{}
}

func releaseReadinessField(
	t *testing.T,
	section report.Section,
	name string,
) report.Field {
	t.Helper()
	for _, field := range section.Fields {
		if field.Name == name {
			return field
		}
	}
	t.Fatalf("section %s missing field %s", section.Title, name)
	return report.Field{}
}
