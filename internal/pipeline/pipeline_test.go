package pipeline

import (
	"context"
	"errors"
	"io"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/internal/checkpoint"
	"github.com/rioriost/agefreighter/internal/sink"
	"github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
)

func testOptions() Options {
	return Options{
		MemoryLimitBytes:      200,
		MaxBatchRows:          2,
		MaxBatchBytes:         25,
		RecordChannelCapacity: 2,
		BatchChannelCapacity:  1,
		OperationTimeout:      time.Second,
	}
}

func testItem(offset, bytes int64) source.Item {
	return source.Item{
		Record: model.VertexRecord(model.Vertex{
			Label:      "Person",
			Namespace:  "test",
			ExternalID: model.ExternalID(strconv.FormatInt(offset, 10)),
			Position: model.SourcePosition{
				Connector: "fake",
				Offset:    offset,
			},
		}),
		SizeBytes: bytes,
	}
}

type fakeIterator struct {
	items       []source.Item
	index       int
	nextError   error
	errorAt     int
	closeErr    error
	beforeError <-chan struct{}
	closed      atomic.Bool
	block       bool
	nextDone    chan struct{}
	nextCalls   atomic.Uint64
}

func (iterator *fakeIterator) Next(ctx context.Context) (source.Item, error) {
	iterator.nextCalls.Add(1)
	if iterator.block && iterator.index >= len(iterator.items) {
		if iterator.nextDone != nil {
			defer close(iterator.nextDone)
		}
		<-ctx.Done()
		return source.Item{}, ctx.Err()
	}
	if iterator.nextError != nil && iterator.index == iterator.errorAt {
		if iterator.beforeError != nil {
			select {
			case <-iterator.beforeError:
			case <-ctx.Done():
				return source.Item{}, ctx.Err()
			}
		}
		return source.Item{}, iterator.nextError
	}
	if iterator.index >= len(iterator.items) {
		return source.Item{}, io.EOF
	}
	item := iterator.items[iterator.index]
	iterator.index++
	return item, nil
}

func (iterator *fakeIterator) Close() error {
	iterator.closed.Store(true)
	return iterator.closeErr
}

type fakeSink struct {
	mu            sync.Mutex
	failAt        string
	failure       error
	rollbackError error
	rollbackBlock bool
	metadata      []sink.BatchMetadata
	transactions  []*fakeTransaction
	writeStarted  chan struct{}
	releaseWrite  chan struct{}
	commitDone    chan struct{}
}

func (target *fakeSink) Begin(
	_ context.Context,
	metadata sink.BatchMetadata,
) (sink.Transaction, error) {
	target.mu.Lock()
	defer target.mu.Unlock()
	target.metadata = append(target.metadata, metadata)
	if target.failAt == "begin" {
		return nil, target.failure
	}
	transaction := &fakeTransaction{target: target}
	target.transactions = append(target.transactions, transaction)
	return transaction, nil
}

type fakeTransaction struct {
	target     *fakeSink
	records    []model.Record
	checkpoint checkpoint.State
	committed  bool
	rolledBack bool
}

func (transaction *fakeTransaction) Write(
	ctx context.Context,
	records []model.Record,
) error {
	if transaction.target.writeStarted != nil {
		select {
		case <-transaction.target.writeStarted:
		default:
			close(transaction.target.writeStarted)
		}
	}
	if transaction.target.releaseWrite != nil {
		select {
		case <-transaction.target.releaseWrite:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if transaction.target.failAt == "write" {
		return transaction.target.failure
	}
	transaction.records = append(transaction.records, records...)
	return nil
}

func (transaction *fakeTransaction) Commit(
	_ context.Context,
	state checkpoint.State,
) error {
	if transaction.target.failAt == "commit" {
		return transaction.target.failure
	}
	transaction.checkpoint = state
	transaction.committed = true
	if transaction.target.commitDone != nil {
		select {
		case <-transaction.target.commitDone:
		default:
			close(transaction.target.commitDone)
		}
	}
	return nil
}

func (transaction *fakeTransaction) Rollback(ctx context.Context) error {
	transaction.rolledBack = true
	if transaction.target.rollbackBlock {
		<-ctx.Done()
		return ctx.Err()
	}
	return transaction.target.rollbackError
}

func TestRunBatchesByRowsAndBytes(t *testing.T) {
	runner, err := New(testOptions())
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	iterator := &fakeIterator{
		items: []source.Item{
			testItem(1, 10),
			testItem(2, 10),
			testItem(3, 10),
			testItem(4, 30),
		},
	}
	target := &fakeSink{}

	if err := runner.Run(context.Background(), iterator, target); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	target.mu.Lock()
	defer target.mu.Unlock()
	if len(target.transactions) != 3 {
		t.Fatalf("transactions = %d, want 3", len(target.transactions))
	}
	wantRows := []int{2, 1, 1}
	wantFirst := []int64{1, 3, 4}
	wantLast := []int64{2, 3, 4}
	for index, transaction := range target.transactions {
		if len(transaction.records) != wantRows[index] {
			t.Errorf("batch %d rows = %d, want %d", index+1, len(transaction.records), wantRows[index])
		}
		if target.metadata[index].FirstPosition.Offset != wantFirst[index] ||
			target.metadata[index].LastPosition.Offset != wantLast[index] {
			t.Errorf("batch %d metadata = %#v", index+1, target.metadata[index])
		}
		if !transaction.committed ||
			transaction.checkpoint.Phase != checkpoint.PhaseCommitted ||
			transaction.checkpoint.BatchID != uint64(index+1) {
			t.Errorf("batch %d transaction = %#v", index+1, transaction)
		}
	}
	if !iterator.closed.Load() {
		t.Fatal("source was not closed")
	}
	snapshot := runner.Snapshot()
	if snapshot.RecordsRead != 4 ||
		snapshot.RecordsCommitted != 4 ||
		snapshot.BytesRead != 60 ||
		snapshot.BytesCommitted != 60 ||
		snapshot.BatchesStarted != 3 ||
		snapshot.BatchesCommitted != 3 ||
		snapshot.BatchesFailed != 0 ||
		snapshot.Memory.Used != 0 ||
		snapshot.Memory.Peak > snapshot.Memory.Limit {
		t.Fatalf("Snapshot() = %#v", snapshot)
	}
}

func TestRunAppliesMemoryBackpressure(t *testing.T) {
	options := testOptions()
	options.MaxBatchRows = 1
	options.MaxBatchBytes = 100
	runner, err := New(options)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	iterator := &fakeIterator{
		items: []source.Item{testItem(1, 60), testItem(2, 60)},
	}
	target := &fakeSink{
		writeStarted: make(chan struct{}),
		releaseWrite: make(chan struct{}),
	}
	done := make(chan error, 1)
	go func() {
		done <- runner.Run(context.Background(), iterator, target)
	}()

	select {
	case <-target.writeStarted:
	case <-time.After(time.Second):
		t.Fatal("first write did not start")
	}
	time.Sleep(10 * time.Millisecond)
	if calls := iterator.nextCalls.Load(); calls != 2 {
		t.Fatalf("source Next() calls = %d, want 2 while permit is blocked", calls)
	}
	if snapshot := runner.Snapshot(); snapshot.Memory.Used != 60+recordAccountingOverhead {
		t.Fatalf("memory while blocked = %#v", snapshot.Memory)
	}
	close(target.releaseWrite)
	if err := <-done; err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if snapshot := runner.Snapshot(); snapshot.Memory.Used != 0 ||
		snapshot.Memory.Peak != 60+recordAccountingOverhead {
		t.Fatalf("final memory = %#v", snapshot.Memory)
	}
}

func TestRunClassifiesFailuresAndReleasesMemory(t *testing.T) {
	failure := errors.New("injected failure")
	tests := []struct {
		name           string
		iterator       *fakeIterator
		target         *fakeSink
		wantClass      ErrorClass
		outcomeUnknown bool
		rolledBack     bool
	}{
		{
			name:      "source",
			iterator:  &fakeIterator{nextError: failure, errorAt: 0},
			target:    &fakeSink{},
			wantClass: ErrorSource,
		},
		{
			name:      "close source",
			iterator:  &fakeIterator{closeErr: failure},
			target:    &fakeSink{},
			wantClass: ErrorSource,
		},
		{
			name:      "begin",
			iterator:  &fakeIterator{items: []source.Item{testItem(1, 10)}},
			target:    &fakeSink{failAt: "begin", failure: failure},
			wantClass: ErrorSinkBegin,
		},
		{
			name:       "write",
			iterator:   &fakeIterator{items: []source.Item{testItem(1, 10)}},
			target:     &fakeSink{failAt: "write", failure: failure},
			wantClass:  ErrorSinkWrite,
			rolledBack: true,
		},
		{
			name:           "commit",
			iterator:       &fakeIterator{items: []source.Item{testItem(1, 10)}},
			target:         &fakeSink{failAt: "commit", failure: failure},
			wantClass:      ErrorSinkCommit,
			outcomeUnknown: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runner, err := New(testOptions())
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			err = runner.Run(context.Background(), test.iterator, test.target)
			var pipelineError *Error
			if !errors.As(err, &pipelineError) {
				t.Fatalf("Run() error = %v", err)
			}
			if pipelineError.Class != test.wantClass ||
				pipelineError.OutcomeUnknown != test.outcomeUnknown {
				t.Fatalf("Run() error = %#v", pipelineError)
			}
			if snapshot := runner.Snapshot(); snapshot.Memory.Used != 0 {
				t.Fatalf("memory after failure = %#v", snapshot.Memory)
			}
			if len(test.target.transactions) > 0 &&
				test.target.transactions[0].rolledBack != test.rolledBack {
				t.Fatalf(
					"rolledBack = %v, want %v",
					test.target.transactions[0].rolledBack,
					test.rolledBack,
				)
			}
		})
	}
}

func TestRunReportsRollbackFailure(t *testing.T) {
	writeFailure := errors.New("write failed")
	rollbackFailure := errors.New("rollback failed")
	runner, err := New(testOptions())
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	err = runner.Run(
		context.Background(),
		&fakeIterator{items: []source.Item{testItem(1, 10)}},
		&fakeSink{
			failAt:        "write",
			failure:       writeFailure,
			rollbackError: rollbackFailure,
		},
	)
	if !errors.Is(err, writeFailure) || !errors.Is(err, rollbackFailure) {
		t.Fatalf("Run() error = %v", err)
	}
}

func TestRunDrainsCommittedInputBeforeReportingCloseError(t *testing.T) {
	closeFailure := errors.New("close failed")
	options := testOptions()
	options.MaxBatchRows = 1
	runner, err := New(options)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	target := &fakeSink{
		writeStarted: make(chan struct{}),
		releaseWrite: make(chan struct{}),
	}
	done := make(chan error, 1)
	go func() {
		done <- runner.Run(
			context.Background(),
			&fakeIterator{
				items:    []source.Item{testItem(1, 10)},
				closeErr: closeFailure,
			},
			target,
		)
	}()

	select {
	case <-target.writeStarted:
	case <-time.After(time.Second):
		t.Fatal("write did not start")
	}
	close(target.releaseWrite)
	err = <-done
	var pipelineError *Error
	if !errors.As(err, &pipelineError) ||
		pipelineError.Class != ErrorSource ||
		!errors.Is(err, closeFailure) {
		t.Fatalf("Run() error = %v", err)
	}
	snapshot := runner.Snapshot()
	if snapshot.RecordsCommitted != 1 || snapshot.Memory.Used != 0 {
		t.Fatalf("Snapshot() = %#v", snapshot)
	}
}

func TestRunReleasesInFlightMemoryAfterMidstreamSourceError(t *testing.T) {
	sourceFailure := errors.New("source failed")
	options := testOptions()
	options.MaxBatchRows = 1
	runner, err := New(options)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	firstCommit := make(chan struct{})
	iterator := &fakeIterator{
		items:       []source.Item{testItem(1, 10), testItem(2, 10)},
		nextError:   sourceFailure,
		errorAt:     2,
		beforeError: firstCommit,
	}
	err = runner.Run(
		context.Background(),
		iterator,
		&fakeSink{commitDone: firstCommit},
	)
	var pipelineError *Error
	if !errors.As(err, &pipelineError) ||
		pipelineError.Class != ErrorSource ||
		!errors.Is(err, sourceFailure) {
		t.Fatalf("Run() error = %v", err)
	}
	snapshot := runner.Snapshot()
	if snapshot.RecordsCommitted < 1 ||
		snapshot.RecordsCommitted > 2 ||
		snapshot.Memory.Used != 0 {
		t.Fatalf("Snapshot() = %#v", snapshot)
	}
}

func TestRunEnforcesOperationAndRollbackTimeouts(t *testing.T) {
	tests := []struct {
		name      string
		target    *fakeSink
		wantClass ErrorClass
		wantText  string
	}{
		{
			name: "write timeout",
			target: &fakeSink{
				releaseWrite: make(chan struct{}),
			},
			wantClass: ErrorSinkWrite,
		},
		{
			name: "rollback timeout",
			target: &fakeSink{
				failAt:        "write",
				failure:       errors.New("write failed"),
				rollbackBlock: true,
			},
			wantClass: ErrorSinkWrite,
			wantText:  string(ErrorSinkRollback),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			options := testOptions()
			options.OperationTimeout = 10 * time.Millisecond
			runner, err := New(options)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			err = runner.Run(
				context.Background(),
				&fakeIterator{items: []source.Item{testItem(1, 10)}},
				test.target,
			)
			var pipelineError *Error
			if !errors.As(err, &pipelineError) || pipelineError.Class != test.wantClass {
				t.Fatalf("Run() error = %v", err)
			}
			if test.wantText != "" && !strings.Contains(err.Error(), test.wantText) {
				t.Fatalf("Run() error = %v, want %q", err, test.wantText)
			}
			if snapshot := runner.Snapshot(); snapshot.Memory.Used != 0 {
				t.Fatalf("memory after timeout = %#v", snapshot.Memory)
			}
		})
	}
}

func TestRunCancellationDoesNotLeakWorkers(t *testing.T) {
	runner, err := New(testOptions())
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	iterator := &fakeIterator{
		block:    true,
		nextDone: make(chan struct{}),
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- runner.Run(ctx, iterator, &fakeSink{})
	}()
	cancel()

	select {
	case err := <-done:
		var pipelineError *Error
		if !errors.As(err, &pipelineError) || pipelineError.Class != ErrorCancelled {
			t.Fatalf("Run() error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Run() did not return after cancellation")
	}
	select {
	case <-iterator.nextDone:
	default:
		t.Fatal("source worker remained active after Run returned")
	}
	if !iterator.closed.Load() {
		t.Fatal("source was not closed")
	}
}

func TestRunCancellationDuringSinkWriteReleasesAllPermits(t *testing.T) {
	options := testOptions()
	options.MaxBatchRows = 1
	options.MaxBatchBytes = 100
	runner, err := New(options)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	iterator := &fakeIterator{
		items: []source.Item{testItem(1, 60), testItem(2, 60)},
		block: true,
	}
	target := &fakeSink{
		writeStarted: make(chan struct{}),
		releaseWrite: make(chan struct{}),
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- runner.Run(ctx, iterator, target)
	}()

	select {
	case <-target.writeStarted:
	case <-time.After(time.Second):
		t.Fatal("write did not start")
	}
	cancel()
	select {
	case err := <-done:
		var pipelineError *Error
		if !errors.As(err, &pipelineError) || pipelineError.Class != ErrorCancelled {
			t.Fatalf("Run() error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Run() did not return after cancellation")
	}
	if snapshot := runner.Snapshot(); snapshot.Memory.Used != 0 {
		t.Fatalf("memory after cancellation = %#v", snapshot.Memory)
	}
}

func TestRunRejectsSourceContractViolations(t *testing.T) {
	tests := []struct {
		name string
		item source.Item
	}{
		{name: "invalid record", item: source.Item{SizeBytes: 1}},
		{name: "zero bytes", item: testItem(1, 0)},
		{name: "oversized", item: testItem(1, 201)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runner, err := New(testOptions())
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			err = runner.Run(
				context.Background(),
				&fakeIterator{items: []source.Item{test.item}},
				&fakeSink{},
			)
			var pipelineError *Error
			if !errors.As(err, &pipelineError) || pipelineError.Class != ErrorContract {
				t.Fatalf("Run() error = %v", err)
			}
			if snapshot := runner.Snapshot(); snapshot.Memory.Used != 0 {
				t.Fatalf("memory after contract failure = %#v", snapshot.Memory)
			}
		})
	}
}

func TestRunnerContracts(t *testing.T) {
	options := testOptions()
	invalid := []Options{
		{},
		func() Options { value := options; value.MaxBatchRows = 0; return value }(),
		func() Options { value := options; value.MaxBatchBytes = 0; return value }(),
		func() Options { value := options; value.MaxBatchBytes = 201; return value }(),
		func() Options {
			value := options
			value.MemoryLimitBytes = 100
			return value
		}(),
		func() Options { value := options; value.RecordChannelCapacity = 0; return value }(),
		func() Options { value := options; value.BatchChannelCapacity = 0; return value }(),
		func() Options { value := options; value.OperationTimeout = 0; return value }(),
	}
	for index, value := range invalid {
		if _, err := New(value); err == nil {
			t.Errorf("New(invalid[%d]) succeeded", index)
		}
	}

	runner, err := New(options)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := runner.Run(context.Background(), nil, &fakeSink{}); err == nil {
		t.Fatal("Run(nil source) succeeded")
	}
	var typedNilSource *fakeIterator
	if err := runner.Run(context.Background(), typedNilSource, &fakeSink{}); err == nil {
		t.Fatal("Run(typed nil source) succeeded")
	}
	if err := runner.Run(context.Background(), &fakeIterator{}, nil); err == nil {
		t.Fatal("Run(nil sink) succeeded")
	}
	var typedNilSink *fakeSink
	if err := runner.Run(context.Background(), &fakeIterator{}, typedNilSink); err == nil {
		t.Fatal("Run(typed nil sink) succeeded")
	}
	if err := runner.Run(context.Background(), &fakeIterator{}, &fakeSink{}); err != nil {
		t.Fatalf("first valid Run() error = %v", err)
	}
	if err := runner.Run(context.Background(), &fakeIterator{}, &fakeSink{}); err == nil {
		t.Fatal("second Run() succeeded")
	}
}

func TestRunRejectsNilTransaction(t *testing.T) {
	targets := []sink.Sink{nilTransactionSink{}, typedNilTransactionSink{}}
	for _, target := range targets {
		runner, err := New(testOptions())
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		err = runner.Run(
			context.Background(),
			&fakeIterator{items: []source.Item{testItem(1, 10)}},
			target,
		)
		var pipelineError *Error
		if !errors.As(err, &pipelineError) || pipelineError.Class != ErrorContract {
			t.Fatalf("Run() error = %v", err)
		}
	}
}

type nilTransactionSink struct{}

func (nilTransactionSink) Begin(
	context.Context,
	sink.BatchMetadata,
) (sink.Transaction, error) {
	return nil, nil
}

type typedNilTransactionSink struct{}

func (typedNilTransactionSink) Begin(
	context.Context,
	sink.BatchMetadata,
) (sink.Transaction, error) {
	var transaction *fakeTransaction
	return transaction, nil
}

func BenchmarkRunner(b *testing.B) {
	items := make([]source.Item, 1000)
	for index := range items {
		items[index] = testItem(int64(index+1), 128)
	}
	options := testOptions()
	options.MemoryLimitBytes = 1 << 20
	options.MaxBatchRows = 1000
	options.MaxBatchBytes = 128 * 1000

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		runner, err := New(options)
		if err != nil {
			b.Fatal(err)
		}
		iterator := &fakeIterator{items: items}
		if err := runner.Run(context.Background(), iterator, &fakeSink{}); err != nil {
			b.Fatal(err)
		}
	}
}
