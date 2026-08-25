package age

import (
	"fmt"
	"strconv"
	"strings"
)

const (
	SupportedPostgreSQLMajor = 17
	SupportedAGEMajor        = 1
	SupportedAGEMinor        = 6
)

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
	if postgresMajor != SupportedPostgreSQLMajor {
		return fmt.Errorf(
			"unsupported PostgreSQL major version %d: agefreighter supports %d",
			postgresMajor,
			SupportedPostgreSQLMajor,
		)
	}
	if ageVersion.Major != SupportedAGEMajor ||
		ageVersion.Minor != SupportedAGEMinor {
		return fmt.Errorf(
			"unsupported Apache AGE version %s: agefreighter supports %d.%d.x",
			ageVersion,
			SupportedAGEMajor,
			SupportedAGEMinor,
		)
	}
	return nil
}
