package age

import (
	"errors"
	"strings"
	"testing"
)

func TestAGENameValidation(t *testing.T) {
	for _, name := range []string{"abc", "_ab", "graph.with-hyphen", strings.Repeat("a", 63)} {
		if err := ValidateGraphName(name); err != nil {
			t.Errorf("ValidateGraphName(%q) error = %v", name, err)
		}
	}
	for _, name := range []string{"ab", "1graph", "graph-", "graph space", strings.Repeat("a", 64)} {
		if !errors.Is(ValidateGraphName(name), ErrInvalidName) {
			t.Errorf("ValidateGraphName(%q) succeeded", name)
		}
	}
	for _, name := range []string{"a", "_label", "Mixed_123", strings.Repeat("l", 63)} {
		if err := ValidateLabelName(name); err != nil {
			t.Errorf("ValidateLabelName(%q) error = %v", name, err)
		}
	}
	for _, name := range []string{"", "label-name", "label.name", strings.Repeat("l", 64)} {
		if !errors.Is(ValidateLabelName(name), ErrInvalidName) {
			t.Errorf("ValidateLabelName(%q) succeeded", name)
		}
	}
}

func TestDerivedGraphNamesAreStableAndBounded(t *testing.T) {
	base := strings.Repeat("a", 63)
	shadow, err := DeriveGraphName(base, ShadowName, "job-1")
	if err != nil {
		t.Fatalf("DeriveGraphName() error = %v", err)
	}
	repeated, err := DeriveGraphName(base, ShadowName, "job-1")
	if err != nil || repeated != shadow {
		t.Fatalf("repeated name = %q, %v", repeated, err)
	}
	backup, err := DeriveGraphName(base, BackupName, "job-1")
	if err != nil {
		t.Fatalf("DeriveGraphName(backup) error = %v", err)
	}
	if len(shadow) > MaxNameBytes || shadow == backup || shadow == base {
		t.Fatalf("derived names = %q, %q", shadow, backup)
	}
	if _, err := DeriveGraphName(base, "other", "job-1"); !errors.Is(err, ErrInvalidName) {
		t.Fatalf("DeriveGraphName(other) error = %v", err)
	}
}

func FuzzAGEGraphNames(f *testing.F) {
	f.Add("graph")
	f.Add("graph.with-hyphen")
	f.Fuzz(func(t *testing.T, name string) {
		err := ValidateGraphName(name)
		if err == nil {
			if len(name) < MinGraphNameBytes || len(name) > MaxNameBytes {
				t.Fatalf("accepted graph name with %d bytes", len(name))
			}
			if !graphNamePattern.MatchString(name) {
				t.Fatalf("accepted graph name outside pattern: %q", name)
			}
		}
	})
}
