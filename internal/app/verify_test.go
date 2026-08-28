package app

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/report"
)

func TestCountVerificationFieldTriState(t *testing.T) {
	label := meta.LabelGeneration{
		ID: 7, LabelName: "Person", Kind: meta.VertexLabel,
	}

	live := liveCountResult{
		IdentityRows: 3, PhysicalRows: 3, Status: report.CheckPass,
	}
	missing := countVerificationField(
		label, identityCoverageFull, live, nil, 16,
	)
	if missing.Status != report.CheckUnavailable ||
		!strings.Contains(missing.Value, "not recorded") {
		t.Fatalf("legacy field = %#v", missing)
	}
	counters := map[int64]meta.LabelCounter{
		7: {
			LabelGenerationID: 7, Kind: meta.VertexLabel,
			Completeness: meta.CounterComplete,
			Provenance:   meta.CounterProvenanceLifecycle,
			AcceptedRows: counterValue(4), CommittedRows: counterValue(2),
			RejectedRows: counterValue(1),
		},
	}
	mismatch := countVerificationField(
		label, identityCoverageFull, live, counters, 17,
	)
	if mismatch.Status != report.CheckFail ||
		!strings.Contains(mismatch.Value, "liveIdentityRows=3") {
		t.Fatalf("mismatch field = %#v", mismatch)
	}
	counters[7] = meta.LabelCounter{
		LabelGenerationID: 7, Kind: meta.VertexLabel,
		Completeness: meta.CounterComplete,
		Provenance:   meta.CounterProvenanceLifecycle,
		AcceptedRows: counterValue(4), CommittedRows: counterValue(3),
		RejectedRows: counterValue(1),
	}
	if field := countVerificationField(
		label, identityCoverageFull, live, counters, 17,
	); field.Status != report.CheckPass {
		t.Fatalf("matching field = %#v", field)
	}
	counters[7] = meta.LabelCounter{
		LabelGenerationID: 7, Kind: meta.VertexLabel,
		Completeness: meta.CounterIncomplete,
		Provenance:   meta.CounterProvenanceLegacyResume,
	}
	legacy := countVerificationField(
		label, identityCoverageFull, live, counters, 17,
	)
	if legacy.Status != report.CheckUnavailable ||
		!strings.Contains(legacy.Value, "legacy-resume") {
		t.Fatalf("legacy field = %#v", legacy)
	}
}

func TestAnonymousEdgeCountUsesPhysicalCounterAndStaysIncomplete(t *testing.T) {
	label := verificationLabel(8, "KNOWS", meta.EdgeLabel)
	live := liveCountResult{
		IdentityRows: 0, PhysicalRows: 4, Status: report.CheckPass,
	}
	counters := map[int64]meta.LabelCounter{
		label.ID: {
			LabelGenerationID: label.ID, Kind: meta.EdgeLabel,
			Completeness: meta.CounterComplete,
			Provenance:   meta.CounterProvenanceLifecycle,
			AcceptedRows: counterValue(4), CommittedRows: counterValue(4),
			RejectedRows: counterValue(0),
		},
	}
	field := countVerificationField(
		label, identityCoverageOptional, live, counters, 17,
	)
	if field.Status != report.CheckUnavailable ||
		!strings.Contains(field.Value, "livePhysicalRows=4") ||
		!strings.Contains(field.Value, "liveIdentityRows=0") ||
		!strings.Contains(field.Value, "physicalIdentityEquality=unavailable") {
		t.Fatalf("anonymous edge field = %#v", field)
	}
	live.PhysicalRows = 3
	field = countVerificationField(
		label, identityCoverageOptional, live, counters, 17,
	)
	if field.Status != report.CheckFail {
		t.Fatalf("physical counter mismatch field = %#v", field)
	}
	field = countVerificationField(
		label, identityCoverageUnknown, live, counters, 17,
	)
	if field.Status != report.CheckUnavailable ||
		!strings.Contains(field.Value, "storedPhysicalComparison=unavailable") {
		t.Fatalf("legacy edge counter field = %#v", field)
	}
}

func TestIntegrityResultHonorsEdgeIdentityCoverage(t *testing.T) {
	clean := age.IntegrityResult{
		IdentityRows: 2, PhysicalCoverageChecked: false,
	}
	field, incomplete := integrityResultField(
		"e.KNOWS", identityCoverageOptional, 100, clean,
	)
	if field.Status != report.CheckUnavailable || !incomplete ||
		!strings.Contains(field.Value, "reversePhysicalCoverage=unavailable") {
		t.Fatalf("anonymous edge integrity field = %#v, incomplete=%t", field, incomplete)
	}
	clean.MissingPhysicalRows = 1
	field, _ = integrityResultField(
		"e.KNOWS", identityCoverageOptional, 100, clean,
	)
	if field.Status != report.CheckFail {
		t.Fatalf("orphan identity field = %#v", field)
	}
	full := age.IntegrityResult{
		IdentityRows: 2, PhysicalRows: 2, PhysicalCoverageChecked: true,
		OrphanPhysicalRows: 1,
	}
	field, _ = integrityResultField(
		"e.KNOWS", identityCoverageFull, 100, full,
	)
	if field.Status != report.CheckFail {
		t.Fatalf("full coverage orphan field = %#v", field)
	}
}

func counterValue(value int64) *int64 {
	return &value
}

func TestReconcileExpectedLabelsFailsMissingExpectedAndIgnoresUnrelated(t *testing.T) {
	person := verificationLabel(7, "Person", meta.VertexLabel)
	knows := verificationLabel(8, "KNOWS", meta.EdgeLabel)
	unrelated := verificationLabel(99, "Unrelated", meta.VertexLabel)

	matched, mismatches := reconcileExpectedLabels(
		[]meta.LabelGeneration{person, knows},
		[]meta.LabelGeneration{unrelated, person},
	)
	if len(matched) != 1 || matched[0].ID != person.ID {
		t.Fatalf("matched labels = %#v", matched)
	}
	if len(mismatches) != 1 || mismatches[0] != "KNOWS(missing)" {
		t.Fatalf("mismatches = %#v", mismatches)
	}
}

func TestParseResolvedMappingSummaryRequiresVersionedExactLabels(t *testing.T) {
	person := verificationLabel(7, "Person", meta.VertexLabel)
	snapshot := resolvedMappingSnapshot{
		SchemaVersion: resolvedMappingSummaryVersion,
		SourceType:    "csv",
		Labels: []resolvedLabelSnapshot{{
			ID: person.ID, GraphGenerationID: person.GraphGenerationID,
			Name: person.LabelName, Kind: "v",
			GraphNamespaceOID: person.GraphNamespaceOID,
			LabelID:           person.LabelID,
			RelationOID:       person.RelationOID,
			SequenceOID:       person.SequenceOID,
			MappingGeneration: person.MappingGeneration,
			IdentityCoverage:  identityCoverageFull,
		}},
	}

	raw, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	parsed, labels, coverage, err := parseResolvedMappingSummary(raw)
	if err != nil || parsed.SchemaVersion != resolvedMappingSummaryVersion ||
		len(labels) != 1 || !sameResolvedLabel(person, labels[0]) ||
		coverage[person.ID] != identityCoverageFull {
		t.Fatalf("parsed=%#v labels=%#v err=%v", parsed, labels, err)
	}
	raw = append(raw[:len(raw)-1], []byte(`,"unexpected":true}`)...)
	if _, _, _, err := parseResolvedMappingSummary(raw); err == nil {
		t.Fatal("summary with unknown field was accepted")
	}
}

func TestParseLegacyResolvedMappingDoesNotAssumeEdgeCoverage(t *testing.T) {
	person := verificationLabel(7, "Person", meta.VertexLabel)
	knows := verificationLabel(8, "KNOWS", meta.EdgeLabel)
	snapshot := resolvedMappingSnapshot{
		SchemaVersion: legacyResolvedMappingSummaryVersion,
		SourceType:    "postgresql",
		Labels: []resolvedLabelSnapshot{
			resolvedSnapshotLabel(person),
			resolvedSnapshotLabel(knows),
		},
	}
	raw, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	_, _, coverage, err := parseResolvedMappingSummary(raw)
	if err != nil {
		t.Fatalf("parse legacy summary: %v", err)
	}
	if coverage[person.ID] != identityCoverageFull ||
		coverage[knows.ID] != identityCoverageUnknown {
		t.Fatalf("legacy coverage = %#v", coverage)
	}
}

func TestResolvedIdentityCoverageUsesResolvedSourceMappings(t *testing.T) {
	endpoint := config.EndpointMapping{Label: "Person", Field: "id"}
	tests := []struct {
		name string
		job  config.LoadJob
		want identityCoverage
	}{
		{
			name: "csv full",
			job: config.LoadJob{Source: config.Source{
				Type: config.SourceCSV,
				CSV: &config.CSVSource{
					Vertices: []config.CSVVertex{{Label: "Person"}},
					Edges: []config.CSVEdge{{
						Label: "KNOWS", ExternalIDColumn: "edge_id",
						Start: endpoint, End: endpoint,
					}},
				},
			}},
			want: identityCoverageFull,
		},
		{
			name: "postgres optional across mappings",
			job: config.LoadJob{Source: config.Source{
				Type: config.SourcePostgreSQL,
				PostgreSQL: &config.PostgreSQLSource{
					Vertices: []config.VertexQuery{{Label: "Person"}},
					Edges: []config.EdgeQuery{
						{
							Label: "KNOWS", ExternalIDField: "edge_id",
							Start: endpoint, End: endpoint,
						},
						{Label: "KNOWS", Start: endpoint, End: endpoint},
					},
				},
			}},
			want: identityCoverageOptional,
		},
		{
			name: "resolved neo4j discovery mapping full",
			job: config.LoadJob{Source: config.Source{
				Type: config.SourceNeo4j,
				Neo4j: &config.Neo4jSource{
					Vertices: []config.VertexQuery{{Label: "Person"}},
					Edges: []config.EdgeQuery{{
						Label: "KNOWS", ExternalIDField: "discovered_id",
						Start: endpoint, End: endpoint,
					}},
				},
			}},
			want: identityCoverageFull,
		},
		{
			name: "resolved cosmos mapping anonymous",
			job: config.LoadJob{Source: config.Source{
				Type: config.SourceCosmos,
				Cosmos: &config.CosmosSource{
					Vertices: []config.CosmosVertexQuery{{Label: "Person"}},
					Edges: []config.CosmosEdgeQuery{{
						Label: "KNOWS", Start: endpoint, End: endpoint,
					}},
				},
			}},
			want: identityCoverageOptional,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			coverage, err := resolvedIdentityCoverage(test.job)
			if err != nil {
				t.Fatalf("resolvedIdentityCoverage() error = %v", err)
			}
			if coverage["Person"] != identityCoverageFull ||
				coverage["KNOWS"] != test.want {
				t.Fatalf("coverage = %#v", coverage)
			}
		})
	}
}

func TestResolvedMappingSummaryPersistsCoverageAndLegacyResumeValidates(t *testing.T) {
	person := verificationLabel(7, "Person", meta.VertexLabel)
	knows := verificationLabel(8, "KNOWS", meta.EdgeLabel)
	endpoint := config.EndpointMapping{Label: "Person", Field: "/id"}
	job := config.LoadJob{Source: config.Source{
		Type: config.SourceCosmos,
		Cosmos: &config.CosmosSource{
			Vertices: []config.CosmosVertexQuery{{Label: "Person"}},
			Edges: []config.CosmosEdgeQuery{{
				Label: "KNOWS", Start: endpoint, End: endpoint,
			}},
		},
	}}
	loadLabels := []age.LoadLabel{
		{Generation: person},
		{Generation: knows},
	}
	raw, err := resolvedMappingSummary(job, loadLabels)
	if err != nil {
		t.Fatalf("resolvedMappingSummary() error = %v", err)
	}
	snapshot, _, coverage, err := parseResolvedMappingSummary(raw)
	if err != nil || snapshot.SchemaVersion != resolvedMappingSummaryVersion ||
		coverage[person.ID] != identityCoverageFull ||
		coverage[knows.ID] != identityCoverageOptional {
		t.Fatalf("snapshot=%#v coverage=%#v err=%v", snapshot, coverage, err)
	}

	legacy := resolvedMappingSnapshot{
		SchemaVersion: legacyResolvedMappingSummaryVersion,
		SourceType:    string(config.SourceCosmos),
		Labels: []resolvedLabelSnapshot{
			resolvedSnapshotLabel(person),
			resolvedSnapshotLabel(knows),
		},
	}
	legacyRaw, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(legacyRaw)
	stored := meta.JobVerification{
		SubmittedConfigFingerprint: strings.Repeat("a", 64),
		ResolvedMappingFingerprint: hex.EncodeToString(digest[:]),
		ResolvedMappingSummary:     legacyRaw,
	}
	if err := validateLegacyJobVerification(
		stored, job, strings.Repeat("a", 64), loadLabels,
	); err != nil {
		t.Fatalf("validateLegacyJobVerification() error = %v", err)
	}
}

func resolvedSnapshotLabel(label meta.LabelGeneration) resolvedLabelSnapshot {
	return resolvedLabelSnapshot{
		ID: label.ID, GraphGenerationID: label.GraphGenerationID,
		Name: label.LabelName, Kind: string(byte(label.Kind)),
		GraphNamespaceOID: label.GraphNamespaceOID,
		LabelID:           label.LabelID,
		RelationOID:       label.RelationOID,
		SequenceOID:       label.SequenceOID,
		MappingGeneration: label.MappingGeneration,
	}
}

func TestVerificationLabelLimitProducesValidIncompleteReport(t *testing.T) {
	document := report.New("verify", time.Unix(1, 0))
	document.Job = &report.Job{
		ID:                "11111111-2222-4333-8444-555555555555",
		ConfigFingerprint: strings.Repeat("a", 64),
	}
	document.Checks = append(document.Checks, report.Check{
		ID: "job-status", Status: report.CheckPass,
		Summary: "load job is committed",
	})
	addVerificationLabelLimit(
		&document,
		VerifyOptions{Counts: true, Integrity: true},
		maxVerifyLabels+1,
	)
	document.Outcome = ""
	got, err := validatedVerificationReport(document)
	if err != nil {
		t.Fatalf("validatedVerificationReport() error = %v", err)
	}
	if got.Outcome != report.OutcomeIncomplete ||
		len(got.Sections) != 2 ||
		got.Checks[len(got.Checks)-1].Status != report.CheckUnknown {
		t.Fatalf("report = %#v", got)
	}
}

func verificationLabel(
	id int64,
	name string,
	kind meta.LabelKind,
) meta.LabelGeneration {
	return meta.LabelGeneration{
		ID: id, GraphGenerationID: 3, LabelName: name, Kind: kind,
		GraphNamespaceOID: 10, LabelID: uint16(id),
		RelationOID: uint32(100 + id), SequenceOID: uint32(200 + id),
		MappingGeneration: 1,
	}
}

func TestClassifiedVerificationCheckDoesNotTurnUnknownIntoZero(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want report.CheckStatus
	}{
		{"deadline", context.DeadlineExceeded, report.CheckUnknown},
		{"permission", &pgconn.PgError{Code: "42501"}, report.CheckUnknown},
		{"statement timeout", &pgconn.PgError{Code: "57014"}, report.CheckUnknown},
		{"missing metadata", meta.ErrNotFound, report.CheckUnavailable},
		{"mismatch", errors.New("count mismatch"), report.CheckFail},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			check := classifiedVerificationCheck("check", "summary", test.err)
			if check.Status != test.want || check.Detail == "0" {
				t.Fatalf("check = %#v", check)
			}
		})
	}
}

func TestVerificationOptionsRequireBounds(t *testing.T) {
	if _, err := VerificationReport(
		t.Context(), "unused", "bad", VerifyOptions{Counts: true},
	); err == nil {
		t.Fatal("VerificationReport accepted invalid job ID")
	}
}
