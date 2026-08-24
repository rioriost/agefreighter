package checkpoint

import (
	"errors"
	"math"
	"testing"

	"github.com/rioriost/agefreighter/pkg/model"
)

func TestCheckpointLifecycle(t *testing.T) {
	position := model.SourcePosition{Connector: "csv", Offset: 42}
	state, err := New(7, position)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	state, err = state.Transition(EventStart)
	if err != nil {
		t.Fatalf("Transition(start) error = %v", err)
	}
	if state.Phase != PhaseRunning || state.Attempt != 1 {
		t.Fatalf("started state = %#v", state)
	}
	state, err = state.Transition(EventFail)
	if err != nil {
		t.Fatalf("Transition(fail) error = %v", err)
	}
	state, err = state.Transition(EventRetry)
	if err != nil {
		t.Fatalf("Transition(retry) error = %v", err)
	}
	if state.Phase != PhaseRunning || state.Attempt != 2 {
		t.Fatalf("retried state = %#v", state)
	}
	state, err = state.Transition(EventCommit)
	if err != nil {
		t.Fatalf("Transition(commit) error = %v", err)
	}
	if state.Phase != PhaseCommitted || state.Position != position {
		t.Fatalf("committed state = %#v", state)
	}
}

func TestCheckpointTransitionMatrix(t *testing.T) {
	valid := map[Phase]map[Event]bool{
		PhasePending: {EventStart: true},
		PhaseRunning: {EventCommit: true, EventFail: true},
		PhaseFailed:  {EventRetry: true},
	}
	phases := []Phase{PhasePending, PhaseRunning, PhaseCommitted, PhaseFailed, Phase(99)}
	events := []Event{EventStart, EventCommit, EventFail, EventRetry, Event(99)}

	for _, phase := range phases {
		for _, event := range events {
			state := State{BatchID: 1, Attempt: 1, Phase: phase}
			next, err := state.Transition(event)
			wantValid := valid[phase][event]
			if wantValid && err != nil {
				t.Errorf("%s + %s returned error %v", phase, event, err)
			}
			if !wantValid {
				if !errors.Is(err, ErrInvalidTransition) {
					t.Errorf("%s + %s error = %v", phase, event, err)
				}
				if next != (State{}) {
					t.Errorf("%s + %s changed state to %#v", phase, event, next)
				}
			}
		}
	}
}

func TestCheckpointRejectsInvalidValues(t *testing.T) {
	if _, err := New(0, model.SourcePosition{}); !errors.Is(err, ErrInvalidTransition) {
		t.Fatalf("New(0) error = %v", err)
	}
	state := State{
		BatchID: 1,
		Attempt: math.MaxUint32,
		Phase:   PhaseFailed,
	}
	if _, err := state.Transition(EventRetry); !errors.Is(err, ErrInvalidTransition) {
		t.Fatalf("Transition(retry overflow) error = %v", err)
	}
}

func TestPhaseAndEventStrings(t *testing.T) {
	if PhaseCommitted.String() != "committed" || Phase(99).String() != "unknown" {
		t.Fatal("unexpected phase strings")
	}
	if EventRetry.String() != "retry" || Event(99).String() != "unknown" {
		t.Fatal("unexpected event strings")
	}
}
