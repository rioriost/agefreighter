package pipeline

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rioriost/agefreighter/internal/checkpoint"
	"github.com/rioriost/agefreighter/internal/sink"
	"github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
)

type Options struct {
	MemoryLimitBytes      int64
	MaxBatchRows          int
	MaxBatchBytes         int64
	RecordChannelCapacity int
	BatchChannelCapacity  int
	OperationTimeout      time.Duration
	InitialBatchID        uint64
	InitialAttempt        uint32
}

// recordAccountingOverhead covers the leased-record slot retained by the
// batcher and the model.Record slot materialized for a sink write.
const recordAccountingOverhead int64 = 64

func (options Options) validate() error {
	switch {
	case options.MemoryLimitBytes <= 0:
		return errors.New("memory limit must be positive")
	case options.MaxBatchRows <= 0:
		return errors.New("maximum batch rows must be positive")
	case options.MaxBatchBytes <= 0:
		return errors.New("maximum batch bytes must be positive")
	case options.MaxBatchBytes > options.MemoryLimitBytes:
		return errors.New("maximum batch bytes cannot exceed memory limit")
	case int64(options.MaxBatchRows) >
		(options.MemoryLimitBytes-options.MaxBatchBytes)/recordAccountingOverhead:
		return errors.New("maximum batch payload and record overhead exceed memory limit")
	case options.RecordChannelCapacity <= 0:
		return errors.New("record channel capacity must be positive")
	case options.BatchChannelCapacity <= 0:
		return errors.New("batch channel capacity must be positive")
	case options.OperationTimeout <= 0:
		return errors.New("operation timeout must be positive")
	default:
		return nil
	}
}

type Runner struct {
	options Options
	limiter *MemoryLimiter
	metrics counters
	started atomic.Bool
}

func New(options Options) (*Runner, error) {
	if options.InitialBatchID == 0 {
		options.InitialBatchID = 1
	}
	if options.InitialAttempt == 0 {
		options.InitialAttempt = 1
	}
	if err := options.validate(); err != nil {
		return nil, err
	}
	limiter, err := NewMemoryLimiter(options.MemoryLimitBytes)
	if err != nil {
		return nil, err
	}
	return &Runner{
		options: options,
		limiter: limiter,
	}, nil
}

func (runner *Runner) Snapshot() MetricsSnapshot {
	return runner.metrics.snapshot(runner.limiter.Snapshot())
}

func (runner *Runner) Run(
	ctx context.Context,
	iterator source.Iterator,
	target sink.Sink,
) error {
	if isNil(iterator) {
		return classifiedError(
			ErrorContract,
			"start pipeline",
			0,
			false,
			false,
			errors.New("source iterator is required"),
		)
	}
	if isNil(target) {
		return classifiedError(
			ErrorContract,
			"start pipeline",
			0,
			false,
			false,
			errors.New("target sink is required"),
		)
	}
	if !runner.started.CompareAndSwap(false, true) {
		return classifiedError(
			ErrorContract,
			"start pipeline",
			0,
			false,
			false,
			errors.New("runner can only be used once"),
		)
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	records := make(chan leasedRecord, runner.options.RecordChannelCapacity)
	batches := make(chan *batch, runner.options.BatchChannelCapacity)
	firstError := make(chan error, 1)
	closeError := make(chan error, 1)
	report := func(err error) {
		if err == nil {
			return
		}
		select {
		case firstError <- err:
			cancel()
		default:
		}
	}

	var workers sync.WaitGroup
	workers.Add(2)
	go func() {
		defer workers.Done()
		runner.produce(runCtx, iterator, records, closeError, report)
	}()
	go func() {
		defer workers.Done()
		runner.batch(runCtx, records, batches, report)
	}()

	for current := range batches {
		if runCtx.Err() == nil {
			if err := runner.writeBatch(runCtx, target, current); err != nil {
				report(err)
			}
		}
		if err := current.release(); err != nil {
			report(classifiedError(
				ErrorInternal,
				"release batch memory",
				current.id,
				false,
				false,
				err,
			))
		}
	}
	workers.Wait()

	select {
	case err := <-firstError:
		return err
	default:
	}
	select {
	case err := <-closeError:
		return err
	default:
	}
	if err := ctx.Err(); err != nil {
		return classifiedError(
			ErrorCancelled,
			"run pipeline",
			0,
			false,
			false,
			err,
		)
	}
	return nil
}

type leasedRecord struct {
	item     source.Item
	limiter  *MemoryLimiter
	weight   int64
	released bool
}

func (record *leasedRecord) release() error {
	if record.released {
		return nil
	}
	record.released = true
	return record.limiter.Release(record.weight)
}

func (runner *Runner) produce(
	ctx context.Context,
	iterator source.Iterator,
	output chan<- leasedRecord,
	closeError chan<- error,
	report func(error),
) {
	defer close(output)
	defer func() {
		if err := iterator.Close(); err != nil && ctx.Err() == nil {
			closeError <- classifiedError(
				ErrorSource,
				"close source",
				0,
				false,
				false,
				err,
			)
		}
	}()

	for {
		item, err := iterator.Next(ctx)
		if errors.Is(err, io.EOF) {
			return
		}
		if err != nil {
			if ctx.Err() == nil {
				report(classifiedError(
					ErrorSource,
					"read source",
					0,
					true,
					false,
					err,
				))
			}
			return
		}
		if item.Record.Kind() == model.RecordInvalid {
			report(classifiedError(
				ErrorContract,
				"read source",
				0,
				false,
				false,
				errors.New("source returned an invalid record"),
			))
			return
		}
		if item.SizeBytes <= 0 ||
			item.SizeBytes > runner.options.MemoryLimitBytes-recordAccountingOverhead {
			report(classifiedError(
				ErrorContract,
				"read source",
				0,
				false,
				false,
				fmt.Errorf(
					"source returned record size %d with memory limit %d",
					item.SizeBytes,
					runner.options.MemoryLimitBytes,
				),
			))
			return
		}
		weight := item.SizeBytes + recordAccountingOverhead
		if err := runner.limiter.Acquire(ctx, weight); err != nil {
			if ctx.Err() == nil {
				report(classifiedError(
					ErrorInternal,
					"acquire record memory",
					0,
					false,
					false,
					err,
				))
			}
			return
		}

		record := leasedRecord{
			item:    item,
			limiter: runner.limiter,
			weight:  weight,
		}
		runner.metrics.recordsRead.Add(1)
		runner.metrics.bytesRead.Add(uint64(item.SizeBytes))
		select {
		case output <- record:
		case <-ctx.Done():
			if err := record.release(); err != nil {
				report(classifiedError(
					ErrorInternal,
					"release record memory",
					0,
					false,
					false,
					err,
				))
			}
			return
		}
	}
}

type batch struct {
	id      uint64
	attempt uint32
	records []leasedRecord
	bytes   int64
}

func (current *batch) release() error {
	var releaseErrors []error
	for index := range current.records {
		if err := current.records[index].release(); err != nil {
			releaseErrors = append(releaseErrors, err)
		}
	}
	return errors.Join(releaseErrors...)
}

func (current *batch) positions() (model.SourcePosition, model.SourcePosition) {
	first, _ := current.records[0].item.Record.SourcePosition()
	last, _ := current.records[len(current.records)-1].item.Record.SourcePosition()
	return first, last
}

func (runner *Runner) batch(
	ctx context.Context,
	input <-chan leasedRecord,
	output chan<- *batch,
	report func(error),
) {
	defer close(output)

	nextID := runner.options.InitialBatchID
	nextAttempt := runner.options.InitialAttempt
	current := &batch{id: nextID, attempt: nextAttempt}
	exhausted := false
	flush := func() {
		if len(current.records) == 0 {
			return
		}
		select {
		case output <- current:
		case <-ctx.Done():
			if err := current.release(); err != nil {
				report(classifiedError(
					ErrorInternal,
					"release cancelled batch memory",
					current.id,
					false,
					false,
					err,
				))
			}
		}
		if nextID == math.MaxUint64 {
			exhausted = true
			current = &batch{}
			return
		}
		nextID++
		nextAttempt = 1
		current = &batch{id: nextID, attempt: nextAttempt}
	}

	for record := range input {
		if ctx.Err() != nil {
			if err := record.release(); err != nil {
				report(classifiedError(
					ErrorInternal,
					"release cancelled record memory",
					0,
					false,
					false,
					err,
				))
			}
			continue
		}
		exceedsRows := len(current.records) >= runner.options.MaxBatchRows
		exceedsBytes := len(current.records) > 0 &&
			record.item.SizeBytes > runner.options.MaxBatchBytes-current.bytes
		if exceedsRows || exceedsBytes {
			flush()
		}
		if exhausted {
			if err := record.release(); err != nil {
				report(classifiedError(
					ErrorInternal,
					"release batch-overflow record memory",
					0,
					false,
					false,
					err,
				))
			}
			report(classifiedError(
				ErrorContract,
				"allocate batch",
				math.MaxUint64,
				false,
				false,
				errors.New("batch ID counter is exhausted"),
			))
			continue
		}
		current.records = append(current.records, record)
		current.bytes += record.item.SizeBytes
		if len(current.records) >= runner.options.MaxBatchRows ||
			current.bytes >= runner.options.MaxBatchBytes {
			flush()
		}
	}
	if ctx.Err() == nil {
		flush()
	} else {
		if err := current.release(); err != nil {
			report(classifiedError(
				ErrorInternal,
				"release final batch memory",
				current.id,
				false,
				false,
				err,
			))
		}
	}
}

func (runner *Runner) writeBatch(
	ctx context.Context,
	target sink.Sink,
	current *batch,
) error {
	firstPosition, lastPosition := current.positions()
	state, err := checkpoint.NewRunning(
		current.id,
		current.attempt,
		lastPosition,
	)
	if err != nil {
		return classifiedError(
			ErrorInternal,
			"create checkpoint",
			current.id,
			false,
			false,
			err,
		)
	}
	runner.metrics.batchesStarted.Add(1)

	metadata := sink.BatchMetadata{
		ID:            current.id,
		Attempt:       state.Attempt,
		Rows:          len(current.records),
		Bytes:         current.bytes,
		FirstPosition: firstPosition,
		LastPosition:  lastPosition,
	}
	operationCtx, cancel := context.WithTimeout(ctx, runner.options.OperationTimeout)
	transaction, err := target.Begin(operationCtx, metadata)
	cancel()
	if err != nil {
		runner.metrics.batchesFailed.Add(1)
		return operationError(
			ctx,
			ErrorSinkBegin,
			"begin transaction",
			current.id,
			true,
			false,
			err,
		)
	}
	if isNil(transaction) {
		runner.metrics.batchesFailed.Add(1)
		return classifiedError(
			ErrorContract,
			"begin transaction",
			current.id,
			false,
			false,
			errors.New("sink returned a nil transaction"),
		)
	}

	records := make([]model.Record, len(current.records))
	for index, record := range current.records {
		records[index] = record.item.Record
	}
	operationCtx, cancel = context.WithTimeout(ctx, runner.options.OperationTimeout)
	err = transaction.Write(operationCtx, records)
	cancel()
	if err != nil {
		runner.metrics.batchesFailed.Add(1)
		return rollbackAfter(
			ctx,
			transaction,
			current.id,
			runner.options.OperationTimeout,
			operationError(
				ctx,
				ErrorSinkWrite,
				"write records",
				current.id,
				true,
				false,
				err,
			),
		)
	}

	committed, err := state.Transition(checkpoint.EventCommit)
	if err != nil {
		runner.metrics.batchesFailed.Add(1)
		return rollbackAfter(
			ctx,
			transaction,
			current.id,
			runner.options.OperationTimeout,
			classifiedError(
				ErrorInternal,
				"commit checkpoint",
				current.id,
				false,
				false,
				err,
			),
		)
	}
	operationCtx, cancel = context.WithTimeout(ctx, runner.options.OperationTimeout)
	err = transaction.Commit(operationCtx, committed)
	cancel()
	if err != nil {
		runner.metrics.batchesFailed.Add(1)
		return operationError(
			ctx,
			ErrorSinkCommit,
			"commit transaction",
			current.id,
			false,
			true,
			err,
		)
	}

	runner.metrics.recordsCommitted.Add(uint64(len(current.records)))
	runner.metrics.bytesCommitted.Add(uint64(current.bytes))
	runner.metrics.batchesCommitted.Add(1)
	return nil
}

func isNil(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map,
		reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	default:
		return false
	}
}

func operationError(
	ctx context.Context,
	class ErrorClass,
	operation string,
	batchID uint64,
	retryable bool,
	outcomeUnknown bool,
	err error,
) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return classifiedError(
			ErrorCancelled,
			operation,
			batchID,
			false,
			false,
			ctxErr,
		)
	}
	return classifiedError(
		class,
		operation,
		batchID,
		retryable,
		outcomeUnknown,
		err,
	)
}

func rollbackAfter(
	ctx context.Context,
	transaction sink.Transaction,
	batchID uint64,
	timeout time.Duration,
	original error,
) error {
	rollbackCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), timeout)
	defer cancel()
	if err := transaction.Rollback(rollbackCtx); err != nil {
		rollbackError := classifiedError(
			ErrorSinkRollback,
			"rollback transaction",
			batchID,
			false,
			false,
			err,
		)
		return errors.Join(original, rollbackError)
	}
	return original
}
