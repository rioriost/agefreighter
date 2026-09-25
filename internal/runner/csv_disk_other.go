//go:build !linux && !darwin

package runner

import "errors"

func csvDiskCapacity(string) (float64, float64, error) {
	return 0, 0, errors.New("CSV guest import requires a supported Unix runner")
}
