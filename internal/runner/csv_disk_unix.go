//go:build linux || darwin

package runner

import (
	"errors"
	"golang.org/x/sys/unix"
)

func csvDiskGate(root string, bytes int64) error {
	var s unix.Statfs_t
	if unix.Statfs(root, &s) != nil {
		return errors.New("CSV disk capacity is unavailable")
	}
	total := float64(s.Blocks) * float64(s.Bsize)
	free := float64(s.Bavail) * float64(s.Bsize)
	if total <= 0 || free-float64(bytes) < total*0.2 {
		return errors.New("CSV import would exceed the 80 percent storage gate")
	}
	return nil
}
