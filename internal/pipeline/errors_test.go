package pipeline

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestClassifiedError(t *testing.T) {
	cause := errors.New("broken")
	err := classifiedError(ErrorSinkWrite, "write", 3, true, false, cause)
	var pipelineError *Error
	if !errors.As(err, &pipelineError) {
		t.Fatalf("classifiedError() type = %T", err)
	}
	if pipelineError.Class != ErrorSinkWrite ||
		!pipelineError.Retryable ||
		pipelineError.OutcomeUnknown ||
		!errors.Is(err, cause) ||
		!strings.Contains(err.Error(), "batch 3") {
		t.Fatalf("classifiedError() = %#v", pipelineError)
	}

	err = classifiedError(ErrorCancelled, "read", 0, false, false, context.Canceled)
	if !errors.As(err, &pipelineError) ||
		pipelineError.Class != ErrorCancelled ||
		pipelineError.Retryable ||
		pipelineError.OutcomeUnknown ||
		!strings.Contains(err.Error(), "cancelled read") {
		t.Fatalf("cancelled error = %#v", err)
	}
	if classifiedError(ErrorInternal, "noop", 0, false, false, nil) != nil {
		t.Fatal("classifiedError(nil) is not nil")
	}
}
