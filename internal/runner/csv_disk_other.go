//go:build !linux && !darwin

package runner

import "errors"

func csvDiskGate(string, int64) error {
	return errors.New("CSV guest import requires a supported Unix runner")
}
