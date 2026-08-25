package checkpoint

import (
	"errors"
	"fmt"

	"github.com/rioriost/agefreighter/pkg/model"
)

type Phase uint8

const (
	PhasePending Phase = iota
	PhaseRunning
	PhaseCommitted
	PhaseFailed
)

func (phase Phase) String() string {
	switch phase {
	case PhasePending:
		return "pending"
	case PhaseRunning:
		return "running"
	case PhaseCommitted:
		return "committed"
	case PhaseFailed:
		return "failed"
	default:
		return "unknown"
	}
}

type Event uint8

const (
	EventStart Event = iota
	EventCommit
	EventFail
	EventRetry
)

func (event Event) String() string {
	switch event {
	case EventStart:
		return "start"
	case EventCommit:
		return "commit"
	case EventFail:
		return "fail"
	case EventRetry:
		return "retry"
	default:
		return "unknown"
	}
}

type State struct {
	BatchID  uint64
	Attempt  uint32
	Phase    Phase
	Position model.SourcePosition
}

var ErrInvalidTransition = errors.New("invalid checkpoint transition")

func New(batchID uint64, position model.SourcePosition) (State, error) {
	if batchID == 0 {
		return State{}, fmt.Errorf("%w: batch ID must be positive", ErrInvalidTransition)
	}
	return State{
		BatchID:  batchID,
		Phase:    PhasePending,
		Position: position,
	}, nil
}

func NewRunning(
	batchID uint64,
	attempt uint32,
	position model.SourcePosition,
) (State, error) {
	if batchID == 0 {
		return State{}, fmt.Errorf("%w: batch ID must be positive", ErrInvalidTransition)
	}
	if attempt == 0 {
		return State{}, fmt.Errorf("%w: attempt must be positive", ErrInvalidTransition)
	}
	return State{
		BatchID:  batchID,
		Attempt:  attempt,
		Phase:    PhaseRunning,
		Position: position,
	}, nil
}

func (state State) Transition(event Event) (State, error) {
	next := state
	switch {
	case state.Phase == PhasePending && event == EventStart:
		next.Phase = PhaseRunning
		next.Attempt = 1
	case state.Phase == PhaseRunning && event == EventCommit:
		next.Phase = PhaseCommitted
	case state.Phase == PhaseRunning && event == EventFail:
		next.Phase = PhaseFailed
	case state.Phase == PhaseFailed && event == EventRetry:
		if state.Attempt == ^uint32(0) {
			return State{}, fmt.Errorf("%w: attempt counter overflow", ErrInvalidTransition)
		}
		next.Phase = PhaseRunning
		next.Attempt++
	default:
		return State{}, fmt.Errorf(
			"%w: cannot apply %s to %s",
			ErrInvalidTransition,
			event,
			state.Phase,
		)
	}
	return next, nil
}
