package pipeline

import "sync/atomic"

type MetricsSnapshot struct {
	RecordsRead      uint64
	RecordsCommitted uint64
	BytesRead        uint64
	BytesCommitted   uint64
	BatchesStarted   uint64
	BatchesCommitted uint64
	BatchesFailed    uint64
	Memory           MemorySnapshot
}

type MetricsProvider interface {
	Snapshot() MetricsSnapshot
}

type counters struct {
	recordsRead      atomic.Uint64
	recordsCommitted atomic.Uint64
	bytesRead        atomic.Uint64
	bytesCommitted   atomic.Uint64
	batchesStarted   atomic.Uint64
	batchesCommitted atomic.Uint64
	batchesFailed    atomic.Uint64
}

func (counters *counters) snapshot(memory MemorySnapshot) MetricsSnapshot {
	return MetricsSnapshot{
		RecordsRead:      counters.recordsRead.Load(),
		RecordsCommitted: counters.recordsCommitted.Load(),
		BytesRead:        counters.bytesRead.Load(),
		BytesCommitted:   counters.bytesCommitted.Load(),
		BatchesStarted:   counters.batchesStarted.Load(),
		BatchesCommitted: counters.batchesCommitted.Load(),
		BatchesFailed:    counters.batchesFailed.Load(),
		Memory:           memory,
	}
}
