package pipeline

import (
	"fmt"
)

type ErrorClass string

const (
	ErrorCancelled    ErrorClass = "cancelled"
	ErrorSource       ErrorClass = "source"
	ErrorContract     ErrorClass = "contract"
	ErrorSinkBegin    ErrorClass = "sink_begin"
	ErrorSinkWrite    ErrorClass = "sink_write"
	ErrorSinkCommit   ErrorClass = "sink_commit"
	ErrorSinkRollback ErrorClass = "sink_rollback"
	ErrorInternal     ErrorClass = "internal"
)

type Error struct {
	Class          ErrorClass
	Operation      string
	BatchID        uint64
	Retryable      bool
	OutcomeUnknown bool
	Err            error
}

func (err *Error) Error() string {
	if err.BatchID == 0 {
		return fmt.Sprintf("%s %s: %v", err.Class, err.Operation, err.Err)
	}
	return fmt.Sprintf(
		"%s %s for batch %d: %v",
		err.Class,
		err.Operation,
		err.BatchID,
		err.Err,
	)
}

func (err *Error) Unwrap() error {
	return err.Err
}

func classifiedError(
	class ErrorClass,
	operation string,
	batchID uint64,
	retryable bool,
	outcomeUnknown bool,
	err error,
) error {
	if err == nil {
		return nil
	}
	return &Error{
		Class:          class,
		Operation:      operation,
		BatchID:        batchID,
		Retryable:      retryable,
		OutcomeUnknown: outcomeUnknown,
		Err:            err,
	}
}
