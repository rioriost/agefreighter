//go:build linux || darwin

package runner

import (
	"errors"
	"golang.org/x/sys/unix"
	"os"
	"path/filepath"
)

// Kernel-held admission lock is released on process death. Durable operation and
// resume claims are separate: releasing this lock never authorizes replay.
func (m Manager) dispatchLock() (func(), error) {
	if err := os.MkdirAll(m.Root, 0700); err != nil {
		return nil, err
	}
	if err := privateDirectory(m.Root); err != nil {
		return nil, err
	}
	fd, err := unix.Open(filepath.Join(m.Root, ".dispatch.lock"), unix.O_CREAT|unix.O_RDWR|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0600)
	if err != nil {
		return nil, err
	}
	if err := unix.Flock(fd, unix.LOCK_EX|unix.LOCK_NB); err != nil {
		unix.Close(fd)
		return nil, errors.New("another runner admission is in progress")
	}
	return func() { unix.Flock(fd, unix.LOCK_UN); unix.Close(fd) }, nil
}
