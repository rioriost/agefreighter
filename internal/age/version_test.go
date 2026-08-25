package age

import "testing"

func TestVersionParsingAndValidation(t *testing.T) {
	version, err := ParseVersion("1.6.0")
	if err != nil {
		t.Fatalf("ParseVersion() error = %v", err)
	}
	if version != (Version{Major: 1, Minor: 6, Patch: 0}) ||
		version.String() != "1.6.0" {
		t.Fatalf("version = %#v (%s)", version, version)
	}
	if err := ValidateVersions(170007, version); err != nil {
		t.Fatalf("ValidateVersions() error = %v", err)
	}
}

func TestVersionRejectsUnsupportedAndMalformedValues(t *testing.T) {
	for _, value := range []string{"", "1", "1.6", "1.6.x", "-1.6.0", "1.6.0.1"} {
		if _, err := ParseVersion(value); err == nil {
			t.Errorf("ParseVersion(%q) succeeded", value)
		}
	}
	if err := ValidateVersions(160009, Version{Major: 1, Minor: 6}); err == nil {
		t.Fatal("ValidateVersions() accepted PostgreSQL 16")
	}
	if err := ValidateVersions(170007, Version{Major: 1, Minor: 7}); err == nil {
		t.Fatal("ValidateVersions() accepted AGE 1.7")
	}
}
