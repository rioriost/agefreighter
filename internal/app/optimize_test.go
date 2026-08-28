package app

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/cypher"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/report"
)

func TestDuplicateIndexGroupsAreDeterministic(t *testing.T) {
	indexes := []indexEvidence{
		{Name: "z_index", Signature: "same", Valid: true, Ready: true},
		{Name: "a_index", Signature: "same", Valid: true, Ready: true},
		{Name: "ignored", Signature: "same", Valid: false, Ready: true},
		{Name: "single", Signature: "other", Valid: true, Ready: true},
	}
	groups := duplicateIndexGroups(indexes)
	if len(groups) != 1 ||
		strings.Join(groups[0], ",") != "a_index,z_index" {
		t.Fatalf("duplicate groups = %#v", groups)
	}
}

func TestOptimizerMetadataAllowlistVersions(t *testing.T) {
	v14 := optimizerMetadataAllowlist(14)
	v17 := optimizerMetadataAllowlist(17)
	if len(v14) != 9 || len(v17) != 15 {
		t.Fatalf("allowlist sizes v14=%d v17=%d", len(v14), len(v17))
	}
	if slicesContains(v14, "diagnostic_history") ||
		!slicesContains(v17, "job_label_counter") {
		t.Fatalf("versioned allowlists v14=%v v17=%v", v14, v17)
	}
}

func TestAnalyzePreconditionsFailClosed(t *testing.T) {
	snapshot := optimizationGoldenSnapshot()
	if err := validateAnalyzePreconditions(snapshot); err != nil {
		t.Fatalf("valid preconditions rejected: %v", err)
	}
	snapshot.Schema.InstalledVersion--
	snapshot.Schema.State = meta.SchemaPending
	if err := validateAnalyzePreconditions(snapshot); err == nil {
		t.Fatal("pending metadata accepted for --apply-analyze")
	}
	snapshot = optimizationGoldenSnapshot()
	snapshot.MetadataRelationsMissing = []string{"load_job"}
	if err := validateAnalyzePreconditions(snapshot); err == nil {
		t.Fatal("missing allowlisted metadata accepted for --apply-analyze")
	}
	snapshot = optimizationGoldenSnapshot()
	snapshot.Probe.AGEVersionStatus = age.ProbeFail
	if err := validateAnalyzePreconditions(snapshot); err == nil {
		t.Fatal("incompatible AGE accepted for --apply-analyze")
	}
}

func TestAnalyzeFailureRedaction(t *testing.T) {
	pgErr := &pgconn.PgError{
		Code:    "42501",
		Message: "secret table and SQL body",
	}
	if got := safeAnalyzeFailure(pgErr); got != "failed: permission denied" {
		t.Fatalf("permission failure = %q", got)
	}
	if got := safeAnalyzeFailure(&pgconn.PgError{
		Code: "55P03", Message: "relation lock timeout",
	}); got != "failed: statement or lock timeout" {
		t.Fatalf("lock timeout failure = %q", got)
	}
	if got := safeAnalyzeFailure(errors.New("DSN password=secret")); strings.Contains(got, "secret") || strings.Contains(got, "password") {
		t.Fatalf("failure leaked input: %q", got)
	}
}

func TestOptimizerRecoverableEvidenceErrors(t *testing.T) {
	for _, code := range []string{"42501", "57014", "55P03"} {
		if !optimizerEvidenceUnknown(&pgconn.PgError{Code: code}) {
			t.Fatalf("SQLSTATE %s was not recoverable evidence", code)
		}
	}
	if optimizerEvidenceUnknown(&pgconn.PgError{Code: "08006"}) {
		t.Fatal("connection failure was treated as recoverable evidence")
	}
}

func TestOptimizerGoldenReports(t *testing.T) {
	document, err := buildOptimizationReport(
		optimizationGoldenSnapshot(),
		false,
		time.Date(2026, 8, 28, 9, 0, 0, 0, time.UTC),
	)
	if err != nil {
		t.Fatalf("buildOptimizationReport() error = %v", err)
	}
	for _, format := range []report.Format{
		report.FormatJSON,
		report.FormatMarkdown,
	} {
		got, err := report.Render(document, format)
		if err != nil {
			t.Fatalf("Render(%s) error = %v", format, err)
		}
		path := filepath.Join(
			"testdata",
			"optimizer.golden."+string(format),
		)
		if os.Getenv("UPDATE_GOLDEN") == "1" {
			if err := os.WriteFile(path, got, 0o600); err != nil {
				t.Fatalf("WriteFile() error = %v", err)
			}
		}
		want, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile(%s) error = %v", path, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("optimizer golden %s differs:\n%s", format, got)
		}
	}
}

func TestOptimizerReportsPartialAnalyzeCompletion(t *testing.T) {
	snapshot := optimizationGoldenSnapshot()
	snapshot.AnalyzeResults = []analyzeResult{
		{Scope: "agefreighter_meta.load_job", Status: "succeeded", Detail: "completed"},
		{Scope: "people.Person", Status: "failed", Detail: "failed: lock timeout"},
	}
	document, err := buildOptimizationReport(
		snapshot,
		true,
		time.Date(2026, 8, 28, 9, 0, 0, 0, time.UTC),
	)
	if err != nil {
		t.Fatalf("buildOptimizationReport() error = %v", err)
	}
	if document.Outcome != report.OutcomeIncomplete {
		t.Fatalf("partial ANALYZE outcome = %s", document.Outcome)
	}
	found := false
	for _, check := range document.Checks {
		if check.ID == "analyze" {
			found = strings.Contains(
				check.Detail,
				"attempted=2 succeeded=1 failed=1",
			)
		}
	}
	if !found {
		t.Fatalf("partial ANALYZE counts = %#v", document.Checks)
	}
}

func TestOptimizerUnknownMetadataCannotPass(t *testing.T) {
	snapshot := optimizationGoldenSnapshot()
	snapshot.Schema = meta.SchemaInspection{
		State: meta.SchemaUnknown, SupportedVersion: meta.SupportedSchemaVersion,
	}
	snapshot.GraphAvailable = false
	snapshot.GraphStatus = report.CheckUnknown
	snapshot.MetadataIndexStatus = report.CheckUnknown
	snapshot.Relations = nil
	snapshot.Labels = nil
	document, err := buildOptimizationReport(
		snapshot,
		false,
		time.Date(2026, 8, 28, 9, 0, 0, 0, time.UTC),
	)
	if err != nil {
		t.Fatalf("buildOptimizationReport() error = %v", err)
	}
	if document.Outcome != report.OutcomeIncomplete {
		t.Fatalf("unknown metadata outcome = %s", document.Outcome)
	}
}

func TestOptimizerPropertyEvidenceIsExplicitlyUnavailable(t *testing.T) {
	document, err := buildOptimizationReport(
		optimizationGoldenSnapshot(),
		false,
		time.Date(2026, 8, 28, 9, 0, 0, 0, time.UTC),
	)
	if err != nil {
		t.Fatalf("buildOptimizationReport() error = %v", err)
	}
	foundCheck := false
	for _, check := range document.Checks {
		if check.ID != "property-statistics" {
			continue
		}
		foundCheck = check.Status == report.CheckUnavailable &&
			check.Detail == propertyEvidenceUnavailable
	}
	if !foundCheck {
		t.Fatalf("property unavailable check = %#v", document.Checks)
	}
	foundSection := false
	for _, section := range document.Sections {
		for _, field := range section.Fields {
			if section.Title == "AGE property evidence" &&
				field.Status == report.CheckUnavailable &&
				field.Value == propertyEvidenceUnavailable {
				foundSection = true
			}
			if section.Title == "Recommendations" &&
				(strings.Contains(field.Value, "property-index") ||
					strings.Contains(field.Value, "expression-index")) {
				t.Fatalf("property recommendation emitted without evidence: %q", field.Value)
			}
		}
	}
	if !foundSection {
		t.Fatal("explicit unavailable property evidence section not found")
	}
}

func TestOptimizerKeepsNonPropertyRecommendations(t *testing.T) {
	snapshot := optimizationGoldenSnapshot()
	snapshot.Relations[0].LastAnalyze = nil
	snapshot.Relations[0].Indexes = append(
		snapshot.Relations[0].Indexes,
		indexEvidence{
			Name: "Person_pkey_copy", Signature: "one",
			Valid: true, Ready: true, Primary: true, Unique: true,
		},
	)
	section := optimizationRecommendationSection(snapshot)
	var recommendations string
	for _, field := range section.Fields {
		recommendations += field.Value + "\n"
	}
	for _, expected := range []string{
		"action=ANALYZE scope=people.Person",
		"action=review-exact-duplicate-indexes scope=people.Person",
	} {
		if !strings.Contains(recommendations, expected) {
			t.Fatalf("missing non-property recommendation %q: %s", expected, recommendations)
		}
	}
}

func TestOptimizerQueryEvidenceDeduplicatesAcrossFiles(t *testing.T) {
	directory := t.TempDir()
	paths := []string{
		fileWithContents(t, directory, "one.cypher",
			"MATCH (n:Person) WHERE n.name = $value RETURN n"),
		fileWithContents(t, directory, "two.cypher",
			"MATCH (n:Person) WHERE n.name = $other RETURN n ORDER BY n.name"),
	}
	queryReport, err := cypher.AnalyzeFiles(
		t.Context(),
		paths,
		cypher.Options{},
	)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := optimizationGoldenSnapshot()
	snapshot.QueryReport = &queryReport
	candidates := workloadIndexCandidates(snapshot)
	if len(candidates) != 1 {
		t.Fatalf("candidates = %#v", candidates)
	}
	candidate := candidates[0]
	if candidate.QueryCount != 2 ||
		!slices.Equal(candidate.Operators, []string{"=", "ORDER BY"}) ||
		candidate.Schema != "people" ||
		candidate.Label != "Person" ||
		candidate.Property != "name" {
		t.Fatalf("candidate = %#v", candidate)
	}
	if !strings.Contains(
		candidate.SQL,
		`ON "people"."Person" USING btree`,
	) || !strings.Contains(
		candidate.SQL,
		`ag_catalog.agtype_access_operator(properties, '"name"'::ag_catalog.agtype)`,
	) {
		t.Fatalf("candidate SQL = %s", candidate.SQL)
	}
	document, err := buildOptimizationReport(
		snapshot,
		false,
		time.Date(2026, 8, 28, 9, 0, 0, 0, time.UTC),
	)
	if err != nil {
		t.Fatal(err)
	}
	if optimizerSectionByTitle(document, "Cypher query evidence") == nil {
		t.Fatal("query evidence section not emitted")
	}
	recommendations := optimizerSectionByTitle(document, "Recommendations")
	if recommendations == nil ||
		!strings.Contains(
			recommendations.Fields[len(recommendations.Fields)-1].Value,
			"confidence=medium",
		) {
		t.Fatalf("recommendations = %#v", recommendations)
	}
}

func TestOptimizerUnknownQueryNeverRecommendsIndex(t *testing.T) {
	path := fileWithContents(
		t, t.TempDir(), "unknown.cypher",
		"MATCH (n:Person) WHERE n.name = vendorMagic($secret) RETURN n",
	)
	queryReport, err := cypher.AnalyzeFiles(
		t.Context(),
		[]string{path},
		cypher.Options{},
	)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := optimizationGoldenSnapshot()
	snapshot.QueryReport = &queryReport
	if candidates := workloadIndexCandidates(snapshot); len(candidates) != 0 {
		t.Fatalf("unknown query candidates = %#v", candidates)
	}
	document, err := buildOptimizationReport(
		snapshot,
		false,
		time.Date(2026, 8, 28, 9, 0, 0, 0, time.UTC),
	)
	if err != nil {
		t.Fatal(err)
	}
	if document.Outcome != report.OutcomeIncomplete {
		t.Fatalf("unknown query outcome = %s", document.Outcome)
	}
}

func TestWorkloadExpressionIndexSQLQuotesIdentifiersAndProperty(t *testing.T) {
	sql := workloadExpressionIndexSQL(
		`graph"name`,
		`Label"; DROP TABLE secret`,
		`prop'erty`,
	)
	for _, expected := range []string{
		`ON "graph""name"."Label""; DROP TABLE secret"`,
		`'"prop''erty"'::ag_catalog.agtype`,
		`CREATE INDEX "agefreighter_q_`,
	} {
		if !strings.Contains(sql, expected) {
			t.Fatalf("safe SQL lacks %q: %s", expected, sql)
		}
	}
}

func fileWithContents(
	t *testing.T,
	directory, name, contents string,
) string {
	t.Helper()
	path := filepath.Join(directory, name)
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func optimizerSectionByTitle(
	document report.Document,
	title string,
) *report.Section {
	for index := range document.Sections {
		if document.Sections[index].Title == title {
			return &document.Sections[index]
		}
	}
	return nil
}

func optimizationGoldenSnapshot() optimizationSnapshot {
	analyzed := time.Date(2026, 8, 28, 8, 0, 0, 0, time.UTC)
	reset := time.Date(2026, 8, 27, 0, 0, 0, 0, time.UTC)
	return optimizationSnapshot{
		Probe: age.DegradedProbe{
			PostgreSQLVersion:    "17.6",
			PostgreSQLStatus:     age.ProbePass,
			AGEPresenceStatus:    age.ProbePass,
			AGEVersion:           "1.6.0",
			AGEVersionStatus:     age.ProbePass,
			AGELoadabilityStatus: age.ProbePass,
		},
		Schema: meta.SchemaInspection{
			State: meta.SchemaCurrent, InstalledVersion: meta.SupportedSchemaVersion,
			SupportedVersion: meta.SupportedSchemaVersion,
		},
		Graph: meta.GraphGeneration{
			ID: 1, JobID: "11111111-2222-4333-8444-555555555555",
			GraphName: "people", GraphOID: 10, NamespaceOID: 10,
			State: meta.GenerationActive,
		},
		GraphAvailable:      true,
		GraphStatus:         report.CheckPass,
		MetadataIndexStatus: report.CheckPass,
		Labels: []meta.LabelGeneration{{
			ID: 2, GraphGenerationID: 1, LabelName: "Person",
			Kind: meta.VertexLabel, GraphNamespaceOID: 10, RelationOID: 42,
		}},
		Job: meta.Job{
			ID:     "11111111-2222-4333-8444-555555555555",
			Status: meta.JobCommitted, SourceType: "csv", LoadMode: "create",
			CommittedRows: 100, CommittedBytes: 8192, NextBatchID: 2,
		},
		JobAvailable:          true,
		MigrationStatus:       report.CheckPass,
		LatestBatch:           meta.BatchAttempt{Status: meta.BatchCommitted, Rows: 100, Bytes: 8192},
		LatestBatchAvailable:  true,
		BatchAttemptsObserved: 1,
		BatchAttemptsStatus:   report.CheckPass,
		Telemetry: meta.ConnectorTelemetry{
			Connector: "csv", Pages: 1,
		},
		TelemetryAvailable: true,
		Counters: []meta.LabelCounter{{
			LabelGenerationID: 2, Kind: meta.VertexLabel,
			Completeness:  meta.CounterComplete,
			CommittedRows: int64Pointer(100), AcceptedRows: int64Pointer(100),
			RejectedRows: int64Pointer(0), CommittedBytes: int64Pointer(8192),
		}},
		CountersAvailable: true,
		Relations: []relationEvidence{{
			Schema: "people", Name: "Person", OID: 42, Kind: meta.VertexLabel,
			EstimatedRows: 100, LiveRows: 100, TotalBytes: 8192, IndexBytes: 4096,
			LastAnalyze: &analyzed, Status: report.CheckPass,
			RequiredIndexStatus: report.CheckPass,
			Indexes: []indexEvidence{{
				Name: "Person_pkey", Signature: "one", Valid: true, Ready: true,
				Primary: true, Unique: true, AccessMethod: "btree",
				KeyNames: []string{"id"}, KeyOptions: []int16{0}, Scans: 12,
			}},
		}},
		MetadataRelations: []relationEvidence{{
			Schema: "agefreighter_meta", Name: "load_job", OID: 7,
			EstimatedRows: 1, LiveRows: 1, TotalBytes: 8192, IndexBytes: 4096,
			LastAutoAnalyze: &analyzed, Status: report.CheckPass,
			Indexes: []indexEvidence{{
				Name: "load_job_pkey", Signature: "meta", Valid: true, Ready: true,
				Primary: true, Unique: true, KeyNames: []string{"job_id"}, Scans: 2,
			}},
		}},
		DatabaseBytes:      1048576,
		DatabaseSizeStatus: report.CheckPass,
		WALBytes:           4096,
		WALReset:           &reset,
		WALStatus:          report.CheckPass,
		StatsReset:         &reset,
		StatsResetStatus:   report.CheckPass,
		GINStatus:          report.CheckPass,
	}
}

func TestRequiredAGEIndexesRequireExactDefinitions(t *testing.T) {
	valid := []indexEvidence{
		{
			Name: "edge_pkey", AccessMethod: "btree", Valid: true, Ready: true,
			Unique: true, Primary: true, KeyNames: []string{"id"},
			KeyOptions: []int16{0},
		},
		{
			Name: "edge_start_id_idx", AccessMethod: "btree", Valid: true, Ready: true,
			KeyNames: []string{"start_id"}, KeyOptions: []int16{0},
		},
		{
			Name: "edge_end_id_idx", AccessMethod: "btree", Valid: true, Ready: true,
			KeyNames: []string{"end_id"}, KeyOptions: []int16{0},
		},
	}
	if missing := requiredAGEIndexColumns(meta.EdgeLabel, valid); len(missing) != 0 {
		t.Fatalf("valid AGE indexes rejected: %v", missing)
	}

	tests := []struct {
		name   string
		mutate func([]indexEvidence)
		want   string
	}{
		{"wrong access method", func(values []indexEvidence) {
			values[0].AccessMethod = "hash"
		}, "id"},
		{"partial", func(values []indexEvidence) {
			values[1].Predicate = "start_id IS NOT NULL"
		}, "start_id"},
		{"expression", func(values []indexEvidence) {
			values[2].KeyNames = []string{""}
		}, "end_id"},
		{"composite key", func(values []indexEvidence) {
			values[1].KeyNames = []string{"start_id", "id"}
			values[1].KeyOptions = []int16{0, 0}
		}, "start_id"},
		{"descending key", func(values []indexEvidence) {
			values[2].KeyOptions = []int16{3}
		}, "end_id"},
		{"invalid", func(values []indexEvidence) {
			values[0].Valid = false
		}, "id"},
		{"not ready", func(values []indexEvidence) {
			values[1].Ready = false
		}, "start_id"},
		{"id not unique", func(values []indexEvidence) {
			values[0].Unique = false
		}, "id"},
		{"id not primary", func(values []indexEvidence) {
			values[0].Primary = false
		}, "id"},
		{"endpoint unique", func(values []indexEvidence) {
			values[1].Unique = true
		}, "start_id"},
		{"endpoint primary", func(values []indexEvidence) {
			values[2].Primary = true
		}, "end_id"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			indexes := append([]indexEvidence(nil), valid...)
			for index := range indexes {
				indexes[index].KeyNames = append([]string(nil), valid[index].KeyNames...)
				indexes[index].KeyOptions = append([]int16(nil), valid[index].KeyOptions...)
			}
			test.mutate(indexes)
			missing := requiredAGEIndexColumns(meta.EdgeLabel, indexes)
			if !slicesContains(missing, test.want) {
				t.Fatalf("missing indexes = %v, want %s rejected", missing, test.want)
			}
		})
	}
}

func TestRequiredAGEIndexProbeIsTargetedAndUncapped(t *testing.T) {
	for _, fragment := range []string{
		"index_metadata.indrelid = $1",
		"index_metadata.indisvalid",
		"index_metadata.indisready",
		"index_metadata.indislive",
		"NOT index_metadata.indisexclusion",
		"index_metadata.indimmediate",
		"NOT index_metadata.indnullsnotdistinct",
		"access_method.amname = 'btree'",
		"index_metadata.indpred IS NULL",
		"index_metadata.indexprs IS NULL",
		"index_metadata.indnatts = 1",
		"index_metadata.indnkeyatts = 1",
		"index_metadata.indisunique",
		"index_metadata.indisprimary",
	} {
		if !strings.Contains(requiredAGEIndexesSQL, fragment) {
			t.Fatalf("targeted required-index query lacks %q", fragment)
		}
	}
	if strings.Contains(strings.ToUpper(requiredAGEIndexesSQL), "LIMIT") {
		t.Fatal("targeted required-index query depends on a catalog display limit")
	}
}

func TestOptimizerProbeRollsBackRecoverableError(t *testing.T) {
	probeTx := &optimizerTestTx{}
	parentTx := &optimizerTestTx{nested: probeTx}
	permissionErr := &pgconn.PgError{Code: "42501"}
	err := runOptimizerProbe(t.Context(), parentTx, func(pgx.Tx) error {
		return permissionErr
	})
	if !errors.Is(err, permissionErr) {
		t.Fatalf("runOptimizerProbe() error = %v", err)
	}
	if probeTx.rollbacks != 1 || probeTx.commits != 0 {
		t.Fatalf(
			"savepoint commits=%d rollbacks=%d",
			probeTx.commits,
			probeTx.rollbacks,
		)
	}
}

func TestOptimizerProbeRollbackFailureIsFatal(t *testing.T) {
	probeTx := &optimizerTestTx{rollbackErr: errors.New("connection lost")}
	parentTx := &optimizerTestTx{nested: probeTx}
	err := runOptimizerProbe(t.Context(), parentTx, func(pgx.Tx) error {
		return &pgconn.PgError{Code: "57014"}
	})
	if !errors.Is(err, errOptimizerSavepointRecovery) {
		t.Fatalf("runOptimizerProbe() error = %v", err)
	}
	if fatalErr := optimizerProbeFatal(t.Context(), err); fatalErr == nil {
		t.Fatal("savepoint recovery failure was treated as recoverable")
	}
}

func TestOptimizerProbeSavepointControlFailureIsFatal(t *testing.T) {
	probeTx := &optimizerTestTx{commitErr: errors.New("release failed")}
	parentTx := &optimizerTestTx{nested: probeTx}
	err := runOptimizerProbe(t.Context(), parentTx, func(pgx.Tx) error {
		return nil
	})
	if !errors.Is(err, errOptimizerSavepointControl) {
		t.Fatalf("runOptimizerProbe() error = %v", err)
	}
	if fatalErr := optimizerProbeFatal(t.Context(), err); fatalErr == nil {
		t.Fatal("savepoint control failure was treated as recoverable")
	}
	if probeTx.rollbacks != 1 {
		t.Fatalf("savepoint rollbacks = %d", probeTx.rollbacks)
	}
}

func TestOptimizerProbeCancellationRollsBackAndAborts(t *testing.T) {
	probeTx := &optimizerTestTx{}
	parentTx := &optimizerTestTx{nested: probeTx}
	ctx, cancel := context.WithCancel(t.Context())
	err := runOptimizerProbe(ctx, parentTx, func(pgx.Tx) error {
		cancel()
		return &pgconn.PgError{Code: "57014"}
	})
	if probeTx.rollbacks != 1 {
		t.Fatalf("savepoint rollbacks = %d", probeTx.rollbacks)
	}
	if fatalErr := optimizerProbeFatal(ctx, err); !errors.Is(
		fatalErr,
		context.Canceled,
	) {
		t.Fatalf("cancellation result = %v", fatalErr)
	}
}

func TestRequiredAGEIndexCheckUnknownWhenInspectionUnknown(t *testing.T) {
	snapshot := optimizationGoldenSnapshot()
	snapshot.Relations[0].RequiredIndexStatus = report.CheckUnknown
	document, err := buildOptimizationReport(
		snapshot,
		false,
		time.Date(2026, 8, 28, 9, 0, 0, 0, time.UTC),
	)
	if err != nil {
		t.Fatalf("buildOptimizationReport() error = %v", err)
	}
	for _, check := range document.Checks {
		if check.ID == "required-age-indexes" {
			if check.Status != report.CheckUnknown {
				t.Fatalf("required AGE index status = %s", check.Status)
			}
			return
		}
	}
	t.Fatal("required-age-indexes check not found")
}

func TestAnalyzeRelationLockAndAllowlistAreFailClosed(t *testing.T) {
	lockSQL := analyzeRelationLockSQL(`graph"name`, `label; DROP TABLE secret`)
	if lockSQL != `LOCK TABLE "graph""name"."label; DROP TABLE secret" IN SHARE UPDATE EXCLUSIVE MODE` {
		t.Fatalf("unsafe relation lock SQL: %s", lockSQL)
	}
	for _, target := range []relationEvidence{
		{Schema: "public", Name: "load_job", OID: 1},
		{Schema: "agefreighter_meta", Name: "not_owned", OID: 1},
		{Schema: "agefreighter_meta", Name: "load_job"},
		{Schema: "", Name: "Person", OID: 42, Kind: meta.VertexLabel},
	} {
		if err := validateAnalyzeTargetAllowlist(target); err == nil {
			t.Fatalf("unsafe ANALYZE target accepted: %#v", target)
		}
	}
	if err := validateAnalyzeTargetAllowlist(relationEvidence{
		Schema: "agefreighter_meta", Name: "load_job", OID: 7,
	}); err != nil {
		t.Fatalf("owned metadata relation rejected: %v", err)
	}
}

func TestGraphAggregatesUnknownWhenEvidenceIncomplete(t *testing.T) {
	snapshot := optimizationGoldenSnapshot()
	snapshot.LabelsTruncated = true
	section := optimizationGraphSection(snapshot)
	for _, name := range []string{
		"estimatedVertexRows",
		"estimatedEdgeRows",
		"estimatedEdgeDensity",
		"labelRelationBytes",
		"labelIndexBytes",
	} {
		field := sectionField(t, section, name)
		if field.Status != report.CheckUnknown || !strings.Contains(field.Value, "unknown") {
			t.Fatalf("%s = %#v, want unknown", name, field)
		}
	}

	snapshot = optimizationGoldenSnapshot()
	snapshot.Relations[0].Status = report.CheckUnknown
	section = optimizationGraphSection(snapshot)
	if field := sectionField(t, section, "labelRelationBytes"); field.Status != report.CheckUnknown {
		t.Fatalf("unavailable relation bytes = %#v", field)
	}

	snapshot = optimizationGoldenSnapshot()
	snapshot.GraphStatus = report.CheckUnknown
	section = optimizationGraphSection(snapshot)
	if field := sectionField(t, section, "labelRelationBytes"); field.Status != report.CheckUnknown {
		t.Fatalf("unknown graph evidence bytes = %#v", field)
	}
}

func sectionField(t *testing.T, section report.Section, name string) report.Field {
	t.Helper()
	for _, field := range section.Fields {
		if field.Name == name {
			return field
		}
	}
	t.Fatalf("field %q not found in %#v", name, section.Fields)
	return report.Field{}
}

func int64Pointer(value int64) *int64 {
	return &value
}

func slicesContains(values []string, expected string) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}

type optimizerTestTx struct {
	nested      pgx.Tx
	beginErr    error
	commitErr   error
	rollbackErr error
	commits     int
	rollbacks   int
}

func (tx *optimizerTestTx) Begin(context.Context) (pgx.Tx, error) {
	return tx.nested, tx.beginErr
}

func (tx *optimizerTestTx) Commit(context.Context) error {
	tx.commits++
	return tx.commitErr
}

func (tx *optimizerTestTx) Rollback(context.Context) error {
	tx.rollbacks++
	return tx.rollbackErr
}

func (*optimizerTestTx) CopyFrom(
	context.Context,
	pgx.Identifier,
	[]string,
	pgx.CopyFromSource,
) (int64, error) {
	panic("unexpected CopyFrom")
}

func (*optimizerTestTx) SendBatch(context.Context, *pgx.Batch) pgx.BatchResults {
	panic("unexpected SendBatch")
}

func (*optimizerTestTx) LargeObjects() pgx.LargeObjects {
	panic("unexpected LargeObjects")
}

func (*optimizerTestTx) Prepare(
	context.Context,
	string,
	string,
) (*pgconn.StatementDescription, error) {
	panic("unexpected Prepare")
}

func (*optimizerTestTx) Exec(
	context.Context,
	string,
	...any,
) (pgconn.CommandTag, error) {
	panic("unexpected Exec")
}

func (*optimizerTestTx) Query(
	context.Context,
	string,
	...any,
) (pgx.Rows, error) {
	panic("unexpected Query")
}

func (*optimizerTestTx) QueryRow(
	context.Context,
	string,
	...any,
) pgx.Row {
	panic("unexpected QueryRow")
}

func (*optimizerTestTx) Conn() *pgx.Conn {
	return nil
}
