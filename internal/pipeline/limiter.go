package pipeline

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

var (
	ErrInvalidMemoryLimit = errors.New("invalid memory limit")
	ErrMemoryOverRelease  = errors.New("memory permit over-release")
)

type MemorySnapshot struct {
	Limit int64
	Used  int64
	Peak  int64
}

type MemoryLimiter struct {
	mu      sync.Mutex
	limit   int64
	used    int64
	peak    int64
	waiters int
	notify  chan struct{}
}

func NewMemoryLimiter(limit int64) (*MemoryLimiter, error) {
	if limit <= 0 {
		return nil, fmt.Errorf("%w: must be positive", ErrInvalidMemoryLimit)
	}
	return &MemoryLimiter{
		limit:  limit,
		notify: make(chan struct{}),
	}, nil
}

func (limiter *MemoryLimiter) Acquire(ctx context.Context, bytes int64) error {
	if bytes <= 0 || bytes > limiter.limit {
		return fmt.Errorf(
			"%w: requested %d bytes with limit %d",
			ErrInvalidMemoryLimit,
			bytes,
			limiter.limit,
		)
	}

	for {
		limiter.mu.Lock()
		if bytes <= limiter.limit-limiter.used {
			limiter.used += bytes
			if limiter.used > limiter.peak {
				limiter.peak = limiter.used
			}
			limiter.mu.Unlock()
			return nil
		}
		notify := limiter.notify
		limiter.waiters++
		limiter.mu.Unlock()

		var err error
		select {
		case <-ctx.Done():
			err = ctx.Err()
		case <-notify:
		}
		limiter.mu.Lock()
		limiter.waiters--
		limiter.mu.Unlock()
		if err != nil {
			return err
		}
	}
}

func (limiter *MemoryLimiter) Release(bytes int64) error {
	if bytes <= 0 {
		return fmt.Errorf("%w: released %d bytes", ErrMemoryOverRelease, bytes)
	}

	limiter.mu.Lock()
	defer limiter.mu.Unlock()
	if bytes > limiter.used {
		return fmt.Errorf(
			"%w: released %d bytes with %d in use",
			ErrMemoryOverRelease,
			bytes,
			limiter.used,
		)
	}
	limiter.used -= bytes
	if limiter.waiters > 0 {
		close(limiter.notify)
		limiter.notify = make(chan struct{})
	}
	return nil
}

func (limiter *MemoryLimiter) Snapshot() MemorySnapshot {
	limiter.mu.Lock()
	defer limiter.mu.Unlock()
	return MemorySnapshot{
		Limit: limiter.limit,
		Used:  limiter.used,
		Peak:  limiter.peak,
	}
}
