package neo4j

import (
	"sync"

	sourcecontract "github.com/rioriost/agefreighter/internal/source"
)

type Telemetry = sourcecontract.Telemetry

type DetailedTelemetry struct {
	Queries  int64
	Records  int64
	Failures int64
}

type telemetryState struct {
	mu       sync.Mutex
	queries  int64
	records  int64
	failures int64
}

func (state *telemetryState) query() {
	state.mu.Lock()
	state.queries++
	state.mu.Unlock()
}

func (state *telemetryState) record() {
	state.mu.Lock()
	state.records++
	state.mu.Unlock()
}

func (state *telemetryState) failure() {
	state.mu.Lock()
	state.failures++
	state.mu.Unlock()
}

func (state *telemetryState) detailed() DetailedTelemetry {
	state.mu.Lock()
	defer state.mu.Unlock()
	return DetailedTelemetry{
		Queries: state.queries, Records: state.records, Failures: state.failures,
	}
}

func (state *telemetryState) snapshot() Telemetry {
	detailed := state.detailed()
	return Telemetry{
		Connector: "neo4j", Pages: detailed.Queries,
		FailedRequestAttempts: detailed.Failures,
	}
}
