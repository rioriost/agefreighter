package neo4j

import (
	"context"
	"errors"
	"fmt"
	"io"
)

const (
	countVerticesQuery = "MATCH (n) RETURN count(n) AS count"
	countEdgesQuery    = "MATCH ()-[r]->() RETURN count(r) AS count"
)

type Inventory struct {
	Vertices int64
	Edges    int64
}

func (inventory Inventory) TotalRows() int64 {
	if inventory.Vertices > int64(^uint64(0)>>1)-inventory.Edges {
		return int64(^uint64(0) >> 1)
	}
	return inventory.Vertices + inventory.Edges
}

// CountInventory executes two read-only count queries. Neo4j can satisfy these
// simple, unfiltered patterns from its transactional count store; callers must
// still present the operation as source work and apply an operation timeout.
func CountInventory(ctx context.Context, client Client) (Inventory, error) {
	if ctx == nil {
		return Inventory{}, errors.New("Neo4j inventory context is required")
	}
	if client == nil {
		return Inventory{}, errors.New("Neo4j inventory client is required")
	}
	vertices, err := readInventoryCount(ctx, client, countVerticesQuery)
	if err != nil {
		return Inventory{}, fmt.Errorf("count Neo4j vertices: %w", err)
	}
	edges, err := readInventoryCount(ctx, client, countEdgesQuery)
	if err != nil {
		return Inventory{}, fmt.Errorf("count Neo4j relationships: %w", err)
	}
	return Inventory{Vertices: vertices, Edges: edges}, nil
}

func readInventoryCount(ctx context.Context, client Client, query string) (int64, error) {
	stream, err := client.Query(ctx, query, nil)
	if err != nil {
		return 0, err
	}
	record, nextErr := stream.Next(ctx)
	if nextErr != nil {
		return 0, errors.Join(nextErr, stream.Close(ctx))
	}
	raw, found := record.Get("count")
	count, ok := raw.(int64)
	if !found || !ok || count < 0 {
		return 0, errors.Join(errors.New("Neo4j count query returned an invalid count"), stream.Close(ctx))
	}
	_, nextErr = stream.Next(ctx)
	if !errors.Is(nextErr, io.EOF) {
		if nextErr == nil {
			nextErr = errors.New("Neo4j count query returned more than one row")
		}
		return 0, errors.Join(nextErr, stream.Close(ctx))
	}
	if err := stream.Close(ctx); err != nil {
		return 0, err
	}
	return count, nil
}
