package app

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/internal/age"
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
	if len(v15) != len(requiredMetadataIndexes)-1 {
		t.Fatalf("v15 required indexes = %d, want %d", len(v15), len(requiredMetadataIndexes)-1)
	}
	for _, definition := range v15 {
		if definition.Name == "diagnostic_history_recent_idx" {
			t.Fatal("v15 unexpectedly requires the v16 diagnostic history index")
		}
	}
	v16 := requiredMetadataIndexesForVersion(16)
	if len(v16) != len(requiredMetadataIndexes) {
		t.Fatalf("v16 required indexes = %d, want %d", len(v16), len(requiredMetadataIndexes))
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
