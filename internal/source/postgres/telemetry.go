package postgres

import (
	"sync"

	sourcecontract "github.com/rioriost/agefreighter/internal/source"
)

type Telemetry = sourcecontract.Telemetry

type telemetryState struct {
	mu                sync.Mutex
	pages             int64
	failures          int64
	rawInputBytes     int64
	decodedInputBytes int64
	profileBudget     *sourcecontract.ProfileBudget
}

func (state *telemetryState) canFetchPage() error {
	return state.profileBudget.Full()
}

func (state *telemetryState) page() error {
	state.mu.Lock()
	state.pages++
	state.mu.Unlock()
	return state.profileBudget.Charge(sourcecontract.ProfileBudgetUsage{Pages: 1})
}

func (state *telemetryState) input(raw, decoded int64) error {
	state.mu.Lock()
	state.rawInputBytes += raw
	state.decodedInputBytes += decoded
	state.mu.Unlock()
	return state.profileBudget.Charge(sourcecontract.ProfileBudgetUsage{
		Rows: 1, RawInputBytes: raw, DecodedInputBytes: decoded,
	})
}

func (state *telemetryState) raw(bytes int64) error {
	state.mu.Lock()
	state.rawInputBytes += bytes
	state.mu.Unlock()
	return state.profileBudget.Charge(sourcecontract.ProfileBudgetUsage{
		RawInputBytes: bytes,
	})
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
		RawInputBytes:         state.rawInputBytes,
		DecodedInputBytes:     state.decodedInputBytes,
		FailedRequestAttempts: state.failures,
	}
}
