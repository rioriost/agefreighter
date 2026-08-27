//go:build windows

package reject

import (
	"context"
	"errors"
	"os"
	"time"

	"golang.org/x/sys/windows"
)

func lockExclusive(ctx context.Context, file *os.File) error {
	timer := time.NewTimer(0)
	if !timer.Stop() {
		<-timer.C
	}
	defer timer.Stop()
	for {
		overlapped := new(windows.Overlapped)
		err := windows.LockFileEx(
			windows.Handle(file.Fd()),
			windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY,
			0,
			^uint32(0),
			^uint32(0),
			overlapped,
		)
		if err == nil {
			return nil
		}
		if !errors.Is(err, windows.ERROR_LOCK_VIOLATION) {
			return err
		}
		timer.Reset(10 * time.Millisecond)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timer.C:
		}
	}
}

func unlockExclusive(file *os.File) error {
	return windows.UnlockFileEx(
		windows.Handle(file.Fd()),
		0,
		^uint32(0),
		^uint32(0),
		new(windows.Overlapped),
	)
}
