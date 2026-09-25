//go:build linux || darwin

package runner

import (
	"errors"
	"golang.org/x/sys/unix"
)

func csvDiskCapacity(root string) (float64, float64, error) {
	var s unix.Statfs_t
	if unix.Statfs(root, &s) != nil {
		return 0, 0, errors.New("CSV disk capacity is unavailable")
	}
	total := float64(s.Blocks) * float64(s.Bsize)
	free := float64(s.Bavail) * float64(s.Bsize)
	return total, free, nil
}
