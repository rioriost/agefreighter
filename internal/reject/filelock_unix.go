//go:build !windows

package reject

import (
	"context"
	"errors"
	"os"
	"syscall"
	"time"
)

func lockExclusive(ctx context.Context, file *os.File) error {
	timer := time.NewTimer(0)
	if !timer.Stop() {
		<-timer.C
	}
	defer timer.Stop()
	for {
		err := syscall.Flock(
			int(file.Fd()),
			syscall.LOCK_EX|syscall.LOCK_NB,
		)
		if err == nil {
			return nil
		}
		if !errors.Is(err, syscall.EWOULDBLOCK) &&
			!errors.Is(err, syscall.EAGAIN) &&
			!errors.Is(err, syscall.EINTR) {
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
	return syscall.Flock(int(file.Fd()), syscall.LOCK_UN)
}
