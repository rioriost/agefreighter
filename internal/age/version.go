package age

import (
	"fmt"
	"strconv"
	"strings"
)

type targetVersionSeries struct {
	PostgreSQLMajor int
	AGEMajor        int
	AGEMinor        int
}

var supportedTargetVersionSeries = [...]targetVersionSeries{
	{PostgreSQLMajor: 14, AGEMajor: 1, AGEMinor: 6},
	{PostgreSQLMajor: 15, AGEMajor: 1, AGEMinor: 6},
	{PostgreSQLMajor: 16, AGEMajor: 1, AGEMinor: 6},
	{PostgreSQLMajor: 17, AGEMajor: 1, AGEMinor: 6},
	{PostgreSQLMajor: 17, AGEMajor: 1, AGEMinor: 7},
	{PostgreSQLMajor: 18, AGEMajor: 1, AGEMinor: 7},
	{PostgreSQLMajor: 18, AGEMajor: 1, AGEMinor: 8},
}

type Version struct {
	Major int
	Minor int
	Patch int
}

func ParseVersion(value string) (Version, error) {
	parts := strings.Split(value, ".")
	if len(parts) != 3 {
		return Version{}, fmt.Errorf("invalid version %q", value)
	}
	values := [3]int{}
	for index, part := range parts {
		number, err := strconv.Atoi(part)
		if err != nil || number < 0 {
			return Version{}, fmt.Errorf("invalid version %q", value)
		}
		values[index] = number
	}
	return Version{Major: values[0], Minor: values[1], Patch: values[2]}, nil
}

func (version Version) String() string {
	return fmt.Sprintf("%d.%d.%d", version.Major, version.Minor, version.Patch)
}

func ValidateVersions(postgreSQLVersionNumber int, ageVersion Version) error {
	postgresMajor := postgreSQLVersionNumber / 10000
	if !isSupportedPostgreSQLMajor(postgresMajor) {
		return fmt.Errorf(
			"unsupported PostgreSQL major version %d: agefreighter supports majors 14 through 18",
			postgresMajor,
		)
	}
	if !isSupportedTargetVersion(postgresMajor, ageVersion) {
		return fmt.Errorf(
			"unsupported Apache AGE version %s for PostgreSQL %d: agefreighter supports %s",
			ageVersion,
			postgresMajor,
			supportedAGESeries(postgresMajor),
		)
	}
	return nil
}

func isSupportedPostgreSQLMajor(postgresMajor int) bool {
	for _, target := range supportedTargetVersionSeries {
		if target.PostgreSQLMajor == postgresMajor {
			return true
		}
	}
	return false
}

func isSupportedTargetVersion(postgresMajor int, ageVersion Version) bool {
	for _, target := range supportedTargetVersionSeries {
		if target.PostgreSQLMajor == postgresMajor &&
			target.AGEMajor == ageVersion.Major &&
			target.AGEMinor == ageVersion.Minor {
			return true
		}
	}
	return false
}

func isSupportedAGEVersion(ageVersion Version) bool {
	for _, target := range supportedTargetVersionSeries {
		if target.AGEMajor == ageVersion.Major &&
			target.AGEMinor == ageVersion.Minor {
			return true
		}
	}
	return false
}

func supportedAGESeries(postgresMajor int) string {
	series := make([]string, 0, 2)
	for _, target := range supportedTargetVersionSeries {
		if target.PostgreSQLMajor == postgresMajor {
			series = append(series, fmt.Sprintf("%d.%d.x", target.AGEMajor, target.AGEMinor))
		}
	}
	return strings.Join(series, " or ")
}
