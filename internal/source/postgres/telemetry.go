package postgres

import (
	"sync"

	sourcecontract "github.com/rioriost/agefreighter/internal/source"
)

type Telemetry = sourcecontract.Telemetry

type telemetryState struct {
	mu       sync.Mutex
	pages    int64
	failures int64
}

func (state *telemetryState) page() {
	state.mu.Lock()
	state.pages++
	state.mu.Unlock()
}

func (state *telemetryState) failure() {
	state.mu.Lock()
	state.failures++
	state.mu.Unlock()
}

func (state *telemetryState) snapshot() Telemetry {
	state.mu.Lock()
	defer state.mu.Unlock()
	return Telemetry{
		Connector:             "postgresql",
		Pages:                 state.pages,
		FailedRequestAttempts: state.failures,
	}
}
