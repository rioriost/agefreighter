package pipeline

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestMemoryLimiterBlocksUntilRelease(t *testing.T) {
	limiter, err := NewMemoryLimiter(10)
	if err != nil {
		t.Fatalf("NewMemoryLimiter() error = %v", err)
	}
	if err := limiter.Acquire(context.Background(), 8); err != nil {
		t.Fatalf("Acquire(8) error = %v", err)
	}

	acquired := make(chan error, 1)
	go func() {
		acquired <- limiter.Acquire(context.Background(), 4)
	}()
	select {
	case err := <-acquired:
		t.Fatalf("Acquire(4) returned before release: %v", err)
	case <-time.After(10 * time.Millisecond):
	}

	if err := limiter.Release(8); err != nil {
		t.Fatalf("Release(8) error = %v", err)
	}
	if err := <-acquired; err != nil {
		t.Fatalf("Acquire(4) error = %v", err)
	}
	snapshot := limiter.Snapshot()
	if snapshot.Used != 4 || snapshot.Peak != 8 || snapshot.Limit != 10 {
		t.Fatalf("Snapshot() = %#v", snapshot)
	}
	if err := limiter.Release(4); err != nil {
		t.Fatalf("Release(4) error = %v", err)
	}
}

func TestMemoryLimiterCancellation(t *testing.T) {
	limiter, err := NewMemoryLimiter(1)
	if err != nil {
		t.Fatalf("NewMemoryLimiter() error = %v", err)
	}
	if err := limiter.Acquire(context.Background(), 1); err != nil {
		t.Fatalf("Acquire(1) error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := limiter.Acquire(ctx, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("Acquire(cancelled) error = %v", err)
	}
	if err := limiter.Release(1); err != nil {
		t.Fatalf("Release(1) error = %v", err)
	}
}

func TestMemoryLimiterRejectsInvalidOperations(t *testing.T) {
	if _, err := NewMemoryLimiter(0); !errors.Is(err, ErrInvalidMemoryLimit) {
		t.Fatalf("NewMemoryLimiter(0) error = %v", err)
	}
	limiter, err := NewMemoryLimiter(10)
	if err != nil {
		t.Fatalf("NewMemoryLimiter() error = %v", err)
	}
	for _, bytes := range []int64{-1, 0, 11} {
		if err := limiter.Acquire(context.Background(), bytes); !errors.Is(err, ErrInvalidMemoryLimit) {
			t.Errorf("Acquire(%d) error = %v", bytes, err)
		}
	}
	for _, bytes := range []int64{-1, 0, 1} {
		if err := limiter.Release(bytes); !errors.Is(err, ErrMemoryOverRelease) {
			t.Errorf("Release(%d) error = %v", bytes, err)
		}
	}
}
