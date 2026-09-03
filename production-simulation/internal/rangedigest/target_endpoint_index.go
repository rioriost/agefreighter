package rangedigest

import "fmt"

const (
	targetEndpointChunkBits = 20
	targetEndpointChunkSize = 1 << targetEndpointChunkBits
	targetEndpointEntryMask = uint64(1<<48) - 1
)

type targetEndpointLabel struct {
	labelID uint64
	chunks  map[uint64][]int64
}

type targetEndpointIndex struct {
	labels map[string]*targetEndpointLabel
}

func newTargetEndpointIndex() *targetEndpointIndex {
	return &targetEndpointIndex{labels: make(map[string]*targetEndpointLabel)}
}

func (index *targetEndpointIndex) add(label string, graphID, sourceKey int64) error {
	if label == "" || graphID <= 0 || sourceKey <= 0 {
		return fmt.Errorf("invalid endpoint identity graph_id=%d source_key=%d", graphID, sourceKey)
	}
	encoded := uint64(graphID)
	labelID := encoded >> 48
	entryID := encoded & targetEndpointEntryMask
	if labelID == 0 || entryID == 0 {
		return fmt.Errorf("invalid AGE graph ID %d", graphID)
	}
	entry, ok := index.labels[label]
	if !ok {
		entry = &targetEndpointLabel{labelID: labelID, chunks: make(map[uint64][]int64)}
		index.labels[label] = entry
	} else if entry.labelID != labelID {
		return fmt.Errorf("label graph ID changed from %d to %d", entry.labelID, labelID)
	}
	chunkID := entryID >> targetEndpointChunkBits
	offset := entryID & (targetEndpointChunkSize - 1)
	chunk := entry.chunks[chunkID]
	if chunk == nil {
		chunk = make([]int64, targetEndpointChunkSize)
		entry.chunks[chunkID] = chunk
	}
	if chunk[offset] != 0 {
		return fmt.Errorf("duplicate AGE graph ID %d", graphID)
	}
	chunk[offset] = sourceKey
	return nil
}

func (index *targetEndpointIndex) lookup(label string, graphID int64) (int64, error) {
	if graphID <= 0 {
		return 0, fmt.Errorf("invalid AGE graph ID %d", graphID)
	}
	entry := index.labels[label]
	if entry == nil {
		return 0, fmt.Errorf("unknown endpoint label %q", label)
	}
	encoded := uint64(graphID)
	labelID := encoded >> 48
	entryID := encoded & targetEndpointEntryMask
	if labelID != entry.labelID || entryID == 0 {
		return 0, fmt.Errorf("AGE graph ID %d does not belong to %q", graphID, label)
	}
	chunk := entry.chunks[entryID>>targetEndpointChunkBits]
	if chunk == nil {
		return 0, fmt.Errorf("AGE graph ID %d is not a migrated vertex", graphID)
	}
	sourceKey := chunk[entryID&(targetEndpointChunkSize-1)]
	if sourceKey == 0 {
		return 0, fmt.Errorf("AGE graph ID %d is not a migrated vertex", graphID)
	}
	return sourceKey, nil
}
