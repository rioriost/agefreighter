package age

import (
	"context"
	"fmt"
	"strconv"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/pkg/model"
)

const (
	denseEndpointChunkBits  = 18
	denseEndpointChunkSize  = 1 << denseEndpointChunkBits
	denseEndpointChunkBytes = denseEndpointChunkSize * 8
)

type denseEndpointCache struct {
	chunks    map[uint64][]GraphID
	maxBytes  int64
	usedBytes int64
}

type pendingDenseIdentity struct {
	chunk   []GraphID
	offset  uint64
	graphID GraphID
}

func newDenseEndpointCache(maxBytes int64) (*denseEndpointCache, error) {
	if maxBytes < denseEndpointChunkBytes {
		return nil, fmt.Errorf(
			"dense endpoint cache limit %d is below one %d-byte chunk",
			maxBytes,
			denseEndpointChunkBytes,
		)
	}
	return &denseEndpointCache{
		chunks:   make(map[uint64][]GraphID),
		maxBytes: maxBytes,
	}, nil
}

func (cache *denseEndpointCache) prepare(
	rows []vertexIdentityRow,
) ([]pendingDenseIdentity, error) {
	pending := make([]pendingDenseIdentity, len(rows))
	for index, row := range rows {
		entry, err := cache.prepareOne(row.externalID, row.graphID)
		if err != nil {
			return nil, err
		}
		pending[index] = entry
	}
	return pending, nil
}

func (cache *denseEndpointCache) prepareOne(
	externalID model.ExternalID,
	graphID GraphID,
) (pendingDenseIdentity, error) {
	internalID, err := parseDenseInternalID(externalID)
	if err != nil {
		return pendingDenseIdentity{}, err
	}
	if err := graphID.Validate(); err != nil {
		return pendingDenseIdentity{}, err
	}
	chunk, err := cache.chunk(internalID >> denseEndpointChunkBits)
	if err != nil {
		return pendingDenseIdentity{}, err
	}
	return pendingDenseIdentity{
		chunk:   chunk,
		offset:  internalID & (denseEndpointChunkSize - 1),
		graphID: graphID,
	}, nil
}

func (cache *denseEndpointCache) apply(rows []pendingDenseIdentity) {
	for _, row := range rows {
		cache.applyOne(row)
	}
}

func (cache *denseEndpointCache) applyOne(row pendingDenseIdentity) {
	row.chunk[row.offset] = row.graphID
}

func (cache *denseEndpointCache) lookup(
	externalID model.ExternalID,
	labelID uint16,
) (GraphID, bool) {
	internalID, err := parseDenseInternalID(externalID)
	if err != nil {
		return 0, false
	}
	chunk, ok := cache.chunks[internalID>>denseEndpointChunkBits]
	if !ok {
		return 0, false
	}
	graphID := chunk[internalID&(denseEndpointChunkSize-1)]
	return graphID, graphID != 0 && graphID.LabelID() == labelID
}

func (cache *denseEndpointCache) chunk(index uint64) ([]GraphID, error) {
	if chunk, ok := cache.chunks[index]; ok {
		return chunk, nil
	}
	if cache.usedBytes+denseEndpointChunkBytes > cache.maxBytes {
		return nil, fmt.Errorf(
			"dense endpoint cache exceeded %d-byte limit at internal ID chunk %d",
			cache.maxBytes,
			index,
		)
	}
	chunk := make([]GraphID, denseEndpointChunkSize)
	cache.chunks[index] = chunk
	cache.usedBytes += denseEndpointChunkBytes
	return chunk, nil
}

func (cache *denseEndpointCache) load(
	ctx context.Context,
	pool *pgxpool.Pool,
	graphGenerationID int64,
) error {
	rows, err := pool.Query(
		ctx,
		`SELECT external_id, graph_id
		 FROM agefreighter_meta.vertex_identity
		 WHERE graph_generation_id = $1`,
		graphGenerationID,
	)
	if err != nil {
		return fmt.Errorf("load dense endpoint cache: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var externalID string
		var graphID GraphID
		if err := rows.Scan(&externalID, &graphID); err != nil {
			return fmt.Errorf("scan dense endpoint cache: %w", err)
		}
		pending, err := cache.prepareOne(model.ExternalID(externalID), graphID)
		if err != nil {
			return fmt.Errorf("load dense endpoint cache: %w", err)
		}
		cache.applyOne(pending)
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate dense endpoint cache: %w", err)
	}
	return nil
}

func parseDenseInternalID(externalID model.ExternalID) (uint64, error) {
	internalID, err := strconv.ParseUint(string(externalID), 10, 63)
	if err != nil {
		return 0, fmt.Errorf(
			"dense endpoint identity %q is not a non-negative Neo4j internal ID: %w",
			externalID,
			err,
		)
	}
	return internalID, nil
}

func (transaction *loadTransaction) resolveEdgesDense(
	edges []stagedEdge,
) ([]resolvedEdge, bool) {
	if transaction.sink.endpointCache == nil {
		return nil, false
	}
	resolved := make([]resolvedEdge, len(edges))
	for index, edge := range edges {
		startID, startOK := transaction.sink.endpointCache.lookup(
			edge.record.Start.ExternalID,
			edge.startLabelID,
		)
		endID, endOK := transaction.sink.endpointCache.lookup(
			edge.record.End.ExternalID,
			edge.endLabelID,
		)
		if !startOK || !endOK {
			return nil, false
		}
		resolved[index] = resolvedEdge{
			stagedEdge: edge,
			startID:    startID,
			endID:      endID,
		}
	}
	return resolved, true
}
