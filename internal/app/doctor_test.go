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

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/report"
)

func TestDoctorProbeReportOutcomes(t *testing.T) {
	at := time.Date(2026, 8, 28, 7, 0, 0, 0, time.UTC)
	healthy := age.DegradedProbe{
		PostgreSQLVersion:    "17.6",
		PostgreSQLStatus:     age.ProbePass,
		AGEPresenceStatus:    age.ProbePass,
		AGEVersion:           "1.6.0",
		AGEVersionStatus:     age.ProbePass,
		AGELoadabilityStatus: age.ProbePass,
		AGEPreloadStatus:     age.PreloadConfigured,
	}

	document := newDoctorDocument(healthy, at)
	finalizeDoctor(&document)
	if document.Outcome != report.OutcomePass {
		t.Fatalf("healthy outcome = %s", document.Outcome)
	}

	if _, err := report.Render(document, report.FormatJSON); err != nil {
		t.Fatalf("Render(healthy) error = %v", err)
	}
	assertDoctorGolden(t, document, report.FormatJSON, "doctor.golden.json")
	assertDoctorGolden(t, document, report.FormatMarkdown, "doctor.golden.markdown")

	missing := healthy
	missing.AGEPresenceStatus = age.ProbeUnavailable
	missing.AGEVersion = ""
	missing.AGEVersionStatus = age.ProbeUnavailable
	missing.AGELoadabilityStatus = age.ProbeUnavailable
	document = newDoctorDocument(missing, at)
	addUnavailableAGEChecks(&document)
	finalizeDoctor(&document)
	if document.Outcome != report.OutcomeIncomplete {
		t.Fatalf("missing AGE outcome = %s", document.Outcome)
	}
	if _, err := report.Render(document, report.FormatMarkdown); err != nil {
		t.Fatalf("Render(missing AGE) error = %v", err)
	}
}

func TestValidateMetadataIndexesRejectsMalformedDefinitions(t *testing.T) {
	valid := make([]metadataIndexInspection, 0, len(requiredMetadataIndexes))
	for _, definition := range requiredMetadataIndexes {
		valid = append(valid, metadataIndexInspection{
			metadataIndexDefinition: metadataIndexDefinition{
				Name:       definition.Name,
				Relation:   definition.Relation,
				Unique:     definition.Unique,
				Keys:       slices.Clone(definition.Keys),
				KeyOptions: slices.Clone(definition.KeyOptions),
				Include:    slices.Clone(definition.Include),
				Predicate:  definition.Predicate,
			},
			Valid: true,
			Ready: true,
		})
	}
	if invalid := validateMetadataIndexes(valid); len(invalid) != 0 {
		t.Fatalf("valid metadata indexes rejected: %v", invalid)
	}
	tests := map[string]func([]metadataIndexInspection) []metadataIndexInspection{
		"missing": func(values []metadataIndexInspection) []metadataIndexInspection {
			return values[1:]
		},
		"wrong relation": func(values []metadataIndexInspection) []metadataIndexInspection {
			values[0].Relation = "wrong"
			return values
		},
		"invalid": func(values []metadataIndexInspection) []metadataIndexInspection {
			values[0].Valid = false
			return values
		},
		"not ready": func(values []metadataIndexInspection) []metadataIndexInspection {
			values[0].Ready = false
			return values
		},
		"wrong uniqueness": func(values []metadataIndexInspection) []metadataIndexInspection {
			values[0].Unique = !values[0].Unique
			return values
		},
		"wrong keys": func(values []metadataIndexInspection) []metadataIndexInspection {
			values[0].Keys = []string{"job_id"}
			return values
		},
		"wrong include": func(values []metadataIndexInspection) []metadataIndexInspection {
			values[0].Include = []string{"unexpected"}
			return values
		},
		"wrong ordering": func(values []metadataIndexInspection) []metadataIndexInspection {
			values[0].KeyOptions[0]++
			return values
		},
		"wrong predicate": func(values []metadataIndexInspection) []metadataIndexInspection {
			values[0].Predicate = "status = 'running'"
			return values
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			values := make([]metadataIndexInspection, len(valid))
			for index, value := range valid {
				values[index] = value
				values[index].Keys = slices.Clone(value.Keys)
				values[index].KeyOptions = slices.Clone(value.KeyOptions)
				values[index].Include = slices.Clone(value.Include)
			}
			if invalid := validateMetadataIndexes(mutate(values)); len(invalid) == 0 {
				t.Fatal("malformed metadata index passed validation")
			}
		})
	}
}

func TestRequiredMetadataIndexesMatchInstalledVersion(t *testing.T) {
	v15 := requiredMetadataIndexesForVersion(15)
	if len(v15) != len(requiredMetadataIndexes)-6 {
		t.Fatalf("v15 required indexes = %d, want %d", len(v15), len(requiredMetadataIndexes)-6)
	}
	for _, definition := range v15 {
		if definition.Name == "diagnostic_history_recent_idx" {
			t.Fatal("v15 unexpectedly requires the v16 diagnostic history index")
		}
	}
	v16 := requiredMetadataIndexesForVersion(16)
	if len(v16) != len(requiredMetadataIndexes)-5 {
		t.Fatalf("v16 required indexes = %d, want %d", len(v16), len(requiredMetadataIndexes)-5)
	}
}

func assertDoctorGolden(
	t *testing.T,
	document report.Document,
	format report.Format,
	name string,
) {
	t.Helper()
	got, err := report.Render(document, format)
	if err != nil {
		t.Fatalf("Render(%s) error = %v", format, err)
	}
	path := filepath.Join("testdata", name)
	if os.Getenv("UPDATE_GOLDEN") == "1" {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("MkdirAll() error = %v", err)
		}
		if err := os.WriteFile(path, got, 0o600); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v", path, err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("doctor golden %s differs:\n%s", name, got)
	}
}

func TestDoctorUnknownCannotPass(t *testing.T) {
	document := report.New(
		"doctor",
		time.Date(2026, 8, 28, 7, 0, 0, 0, time.UTC),
	)
	document.Target = &report.Target{
		PostgreSQL: report.VersionValue{
			Value: "17.6", Status: report.CheckPass,
		},
		AGE: report.VersionValue{
			Status: report.CheckUnknown,
		},
	}
	addCheck(&document, "permissions", report.CheckUnknown,
		"permissions are unknown", "permission denied")
	finalizeDoctor(&document)
	if document.Outcome != report.OutcomeIncomplete {
		t.Fatalf("unknown outcome = %s", document.Outcome)
	}
}

func TestSearchPathContains(t *testing.T) {
	if !searchPathContains(`ag_catalog, "$user", public`, "ag_catalog") {
		t.Fatal("AGE search path was not recognized")
	}
	if searchPathContains(`"$user", public`, "ag_catalog") {
		t.Fatal("missing AGE search path was recognized")
	}
}

func TestDoctorClassificationHelpers(t *testing.T) {
	for _, test := range []struct {
		name string
		got  report.CheckStatus
		want report.CheckStatus
	}{
		{"probe pass", probeCheckStatus(age.ProbePass), report.CheckPass},
		{"probe fail", probeCheckStatus(age.ProbeFail), report.CheckFail},
		{"probe unavailable", probeCheckStatus(age.ProbeUnavailable), report.CheckUnavailable},
		{"probe unknown", probeCheckStatus(age.ProbeUnknown), report.CheckUnknown},
		{"diagnostic permission", diagnosticErrorStatus(&pgconn.PgError{Code: "42501"}), report.CheckUnknown},
		{"diagnostic missing table", diagnosticErrorStatus(&pgconn.PgError{Code: "42P01"}), report.CheckUnavailable},
		{"diagnostic missing schema", diagnosticErrorStatus(&pgconn.PgError{Code: "3F000"}), report.CheckUnavailable},
		{"diagnostic missing function", diagnosticErrorStatus(&pgconn.PgError{Code: "42883"}), report.CheckUnavailable},
		{"diagnostic generic", diagnosticErrorStatus(errors.New("boom")), report.CheckUnknown},
		{"catalog missing", catalogErrorStatus(age.ErrCatalogEntryNotFound), report.CheckFail},
		{"catalog mismatch", catalogErrorStatus(errors.New("catalog mismatch")), report.CheckFail},
		{"catalog kind", catalogErrorStatus(errors.New("invalid kind")), report.CheckFail},
		{"catalog id", catalogErrorStatus(errors.New("invalid ID")), report.CheckFail},
		{"catalog permission", catalogErrorStatus(&pgconn.PgError{Code: "42501"}), report.CheckUnknown},
	} {
		t.Run(test.name, func(t *testing.T) {
			if test.got != test.want {
				t.Fatalf("status = %s, want %s", test.got, test.want)
			}
		})
	}
}

func TestSafeDatabaseDetailMatrix(t *testing.T) {
	tests := []struct {
		err  error
		want string
	}{
		{&pgconn.PgError{Code: "42501", Message: "secret"}, "permission denied"},
		{&pgconn.PgError{Code: "42P01"}, "required catalog object"},
		{&pgconn.PgError{Code: "3F000"}, "required catalog object"},
		{&pgconn.PgError{Code: "42883"}, "required catalog object"},
		{&pgconn.PgError{Code: "57014"}, "configured deadline"},
		{&pgconn.PgError{Code: "08006"}, "fallback (SQLSTATE 08006)"},
		{context.DeadlineExceeded, "configured deadline"},
		{context.Canceled, "operation was canceled"},
		{errors.New("sensitive detail"), "fallback"},
	}
	for _, test := range tests {
		if got := safeDatabaseDetail(test.err, "fallback"); !strings.Contains(got, test.want) {
			t.Fatalf("safeDatabaseDetail(%v) = %q, want %q", test.err, got, test.want)
		}
	}
}

func TestDoctorCheckBuildersAndFinalization(t *testing.T) {
	document := report.New("doctor", time.Unix(1, 0))
	addValueCheck(t.Context(), &document, "ok", nil, "value", "detail")
	addValueCheck(t.Context(), &document, "missing",
		&pgconn.PgError{Code: "42P01"}, "catalog", "")
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	addClassifiedCheck(ctx, &document, "canceled", "operation", errors.New("hidden"))
	if len(document.Checks) != 3 ||
		document.Checks[0].Status != report.CheckPass ||
		document.Checks[1].Status != report.CheckUnavailable ||
		document.Checks[2].Status != report.CheckUnknown ||
		!strings.Contains(document.Checks[2].Detail, "canceled") {
		t.Fatalf("checks = %#v", document.Checks)
	}

	long := strings.Repeat("x", 20)
	if got := boundedTypedValue(long, 5); got != "xxxxx" {
		t.Fatalf("boundedTypedValue() = %q", got)
	}
	if got := boundedTypedValue("short", 10); got != "short" {
		t.Fatalf("boundedTypedValue(short) = %q", got)
	}

	for _, test := range []struct {
		name string
		edit func(*report.Document)
		want report.Outcome
	}{
		{"pass", func(*report.Document) {}, report.OutcomePass},
		{"fail check", func(d *report.Document) {
			addCheck(d, "fail", report.CheckFail, "failed", "")
		}, report.OutcomeFail},
		{"error", func(d *report.Document) {
			d.Errors = append(d.Errors, report.Finding{Code: "E", Message: "failed"})
		}, report.OutcomeFail},
		{"unknown", func(d *report.Document) {
			addCheck(d, "unknown", report.CheckUnknown, "unknown", "")
		}, report.OutcomeIncomplete},
		{"unavailable", func(d *report.Document) {
			addCheck(d, "unavailable", report.CheckUnavailable, "unavailable", "")
		}, report.OutcomeIncomplete},
		{"incomplete", func(d *report.Document) {
			d.IncompleteChecks = append(d.IncompleteChecks, "x")
		}, report.OutcomeIncomplete},
	} {
		t.Run(test.name, func(t *testing.T) {
			value := report.New("doctor", time.Unix(1, 0))
			test.edit(&value)
			finalizeDoctor(&value)
			if value.Outcome != test.want {
				t.Fatalf("outcome = %s, want %s", value.Outcome, test.want)
			}
		})
	}
}

func TestDoctorUnavailableAndPreloadBranches(t *testing.T) {
	document := report.New("doctor", time.Unix(1, 0))
	addUnavailableMetadataChecks(&document)
	if len(document.Checks) != 4 {
		t.Fatalf("metadata unavailable checks = %#v", document.Checks)
	}

	for _, test := range []struct {
		name   string
		status age.PreloadStatus
		want   report.CheckStatus
	}{
		{"not configured", age.PreloadNotConfigured, report.CheckWarning},
		{"unknown", age.PreloadUnknown, report.CheckUnknown},
	} {
		t.Run(test.name, func(t *testing.T) {
			value := newDoctorDocument(age.DegradedProbe{
				PostgreSQLStatus:     age.ProbePass,
				AGEPresenceStatus:    age.ProbePass,
				AGEVersionStatus:     age.ProbePass,
				AGELoadabilityStatus: age.ProbePass,
				AGEPreloadStatus:     test.status,
			}, time.Unix(1, 0))
			if got := value.Checks[len(value.Checks)-1].Status; got != test.want {
				t.Fatalf("preload status = %s, want %s", got, test.want)
			}
		})
	}

	for _, state := range []meta.SchemaState{meta.SchemaPending, meta.SchemaUnknown} {
		document = report.New("doctor", time.Unix(1, 0))
		addDoctorHistoryStorageCheck(
			t.Context(), nil,
			meta.SchemaInspection{
				State: state, InstalledVersion: 16,
				SupportedVersion: meta.SupportedSchemaVersion,
			},
			time.Second, &document,
		)
		if len(document.Checks) != 1 {
			t.Fatalf("history check for %s = %#v", state, document.Checks)
		}
	}
}

func TestMetadataIndexCheckUnavailableStates(t *testing.T) {
	for _, test := range []struct {
		state meta.SchemaState
		want  report.CheckStatus
	}{
		{meta.SchemaAbsent, report.CheckUnavailable},
		{meta.SchemaUnknown, report.CheckUnknown},
		{meta.SchemaInvalid, report.CheckUnknown},
		{meta.SchemaNewer, report.CheckUnknown},
	} {
		document := report.New("doctor", time.Unix(1, 0))
		addMetadataIndexCheck(
			t.Context(), nil,
			meta.SchemaInspection{
				State: test.state, SupportedVersion: meta.SupportedSchemaVersion,
			},
			time.Second, &document,
		)
		if len(document.Checks) != 1 || document.Checks[0].Status != test.want {
			t.Fatalf("state %s checks = %#v", test.state, document.Checks)
		}
	}
}

func TestDoctorPoolRejectsInvalidDSN(t *testing.T) {
	if pool, err := openDoctorPool(t.Context(), "://bad"); err == nil || pool != nil {
		t.Fatalf("openDoctorPool() = %#v, %v", pool, err)
	}
}
