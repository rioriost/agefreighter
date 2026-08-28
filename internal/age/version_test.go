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

func TestSupportedTargetVersionSeries(t *testing.T) {
	tests := []struct {
		postgresVersion int
		ageVersion      Version
	}{
		{140024, Version{Major: 1, Minor: 6}},
		{150019, Version{Major: 1, Minor: 6}},
		{160015, Version{Major: 1, Minor: 6}},
		{170011, Version{Major: 1, Minor: 6}},
		{170011, Version{Major: 1, Minor: 7}},
		{180006, Version{Major: 1, Minor: 7}},
		{180006, Version{Major: 1, Minor: 8}},
	}
	for _, test := range tests {
		if err := ValidateVersions(test.postgresVersion, test.ageVersion); err != nil {
			t.Errorf(
				"ValidateVersions(%d, %s) error = %v",
				test.postgresVersion,
				test.ageVersion,
				err,
			)
		}
	}
}

func TestSupportedAGEVersionSeries(t *testing.T) {
	for _, version := range []Version{
		{Major: 1, Minor: 6},
		{Major: 1, Minor: 7},
		{Major: 1, Minor: 8},
	} {
		if !isSupportedAGEVersion(version) {
			t.Errorf("isSupportedAGEVersion(%s) = false", version)
		}
	}
	for _, version := range []Version{
		{Major: 1, Minor: 5},
		{Major: 1, Minor: 9},
		{Major: 2},
	} {
		if isSupportedAGEVersion(version) {
			t.Errorf("isSupportedAGEVersion(%s) = true", version)
		}
	}
}

func TestVersionRejectsUnsupportedAndMalformedValues(t *testing.T) {
	for _, value := range []string{"", "1", "1.6", "1.6.x", "-1.6.0", "1.6.0.1"} {
		if _, err := ParseVersion(value); err == nil {
			t.Errorf("ParseVersion(%q) succeeded", value)
		}
	}
	if err := ValidateVersions(130023, Version{Major: 1, Minor: 6}); err == nil {
		t.Fatal("ValidateVersions() accepted PostgreSQL 13")
	}
	for _, test := range []struct {
		postgresVersion int
		ageVersion      Version
	}{
		{140024, Version{Major: 1, Minor: 7}},
		{160015, Version{Major: 1, Minor: 8}},
		{170011, Version{Major: 1, Minor: 8}},
		{180006, Version{Major: 1, Minor: 6}},
	} {
		if err := ValidateVersions(test.postgresVersion, test.ageVersion); err == nil {
			t.Errorf(
				"ValidateVersions(%d, %s) accepted unsupported pair",
				test.postgresVersion,
				test.ageVersion,
			)
		}
	}
}
