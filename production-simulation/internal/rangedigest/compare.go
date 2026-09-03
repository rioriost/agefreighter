package rangedigest

import "fmt"

func Compare(expected, actual Manifest) (Comparison, error) {
	result := Comparison{
		Status: "fail", FixtureRoot: expected.FixtureRoot,
		ExpectedRoot: expected.RootSHA256, ActualRoot: actual.RootSHA256,
		ExpectedRows: expected.RecordCount, ActualRows: actual.RecordCount,
		Leaves: len(expected.Leaves),
	}
	fail := func(message string) (Comparison, error) {
		result.Mismatch = message
		return result, fmt.Errorf("canonical digest mismatch: %s", message)
	}
	if expected.Version != ManifestVersion || actual.Version != ManifestVersion ||
		expected.CanonicalVersion != CanonicalVersion || actual.CanonicalVersion != CanonicalVersion {
		return fail("manifest version")
	}
	if expected.Source != "fixture" || actual.Source != "apache-age" {
		return fail("manifest source roles")
	}
	if expected.FixtureRoot != actual.FixtureRoot {
		return fail("fixture root")
	}
	if expected.RangeRows != actual.RangeRows {
		return fail("range size")
	}
	if expected.RecordCount != actual.RecordCount {
		return fail("record count")
	}
	if len(expected.Leaves) != len(actual.Leaves) {
		return fail("leaf count")
	}
	for index := range expected.Leaves {
		if expected.Leaves[index] != actual.Leaves[index] {
			return fail(fmt.Sprintf(
				"leaf %d (%s/%s range %d)",
				index,
				expected.Leaves[index].Kind,
				expected.Leaves[index].Name,
				expected.Leaves[index].RangeIndex,
			))
		}
	}
	if expected.RootSHA256 != actual.RootSHA256 {
		return fail("root")
	}
	result.Status = "pass"
	return result, nil
}
