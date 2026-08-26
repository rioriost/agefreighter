package neo4j

import (
	"context"
	"errors"
	"io"
	"sync"
)

type fakeRecord struct {
	keys   []string
	values map[string]any
}

func record(values map[string]any, keys ...string) fakeRecord {
	if len(keys) == 0 {
		for key := range values {
			keys = append(keys, key)
		}
	}
	return fakeRecord{keys: keys, values: values}
}

func (record fakeRecord) Get(name string) (any, bool) {
	value, ok := record.values[name]
	return value, ok
}

func (record fakeRecord) Keys() []string {
	return record.keys
}

type fakeStream struct {
	mu         sync.Mutex
	records    []Record
	index      int
	nextErr    error
	closeErr   error
	nextCalls  int
	closeCalls int
	block      bool
	started    chan struct{}
	startOnce  sync.Once
}

func (stream *fakeStream) Next(ctx context.Context) (Record, error) {
	stream.mu.Lock()
	stream.nextCalls++
	block := stream.block
	if stream.index < len(stream.records) {
		current := stream.records[stream.index]
		stream.index++
		stream.mu.Unlock()
		return current, nil
	}
	err := stream.nextErr
	stream.mu.Unlock()
	if block {
		if stream.started != nil {
			stream.startOnce.Do(func() { close(stream.started) })
		}
		<-ctx.Done()
		return nil, ctx.Err()
	}
	if err != nil {
		return nil, err
	}
	return nil, io.EOF
}

func (stream *fakeStream) Close(context.Context) error {
	stream.mu.Lock()
	defer stream.mu.Unlock()
	stream.closeCalls++
	return stream.closeErr
}

type fakeClient struct {
	mu         sync.Mutex
	streams    []RecordStream
	queryErr   error
	closeErr   error
	queries    []string
	parameters []map[string]any
	closeCalls int
}

func (client *fakeClient) Query(
	_ context.Context,
	query string,
	parameters map[string]any,
) (RecordStream, error) {
	client.mu.Lock()
	defer client.mu.Unlock()
	client.queries = append(client.queries, query)
	copyParameters := make(map[string]any, len(parameters))
	for name, value := range parameters {
		copyParameters[name] = value
	}
	client.parameters = append(client.parameters, copyParameters)
	if client.queryErr != nil {
		return nil, client.queryErr
	}
	if len(client.streams) == 0 {
		return nil, errors.New("fake has no stream")
	}
	stream := client.streams[0]
	client.streams = client.streams[1:]
	return stream, nil
}

func (client *fakeClient) Close() error {
	client.mu.Lock()
	defer client.mu.Unlock()
	client.closeCalls++
	return client.closeErr
}
