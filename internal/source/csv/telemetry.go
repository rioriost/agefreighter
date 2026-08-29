package csv

import (
	"sync"

	sourcecontract "github.com/rioriost/agefreighter/internal/source"
)

type telemetryState struct {
	mu                sync.Mutex
	pages             int64
	rawInputBytes     int64
	decodedInputBytes int64
}

func (state *telemetryState) page() {
	state.mu.Lock()
	state.pages++
	state.mu.Unlock()
}

func (state *telemetryState) raw(bytes int64) {
	state.mu.Lock()
	state.rawInputBytes += bytes
	state.mu.Unlock()
}

func (state *telemetryState) decoded(bytes int64) {
	state.mu.Lock()
	state.decodedInputBytes += bytes
	state.mu.Unlock()
}

func (state *telemetryState) snapshot() sourcecontract.Telemetry {
	state.mu.Lock()
	defer state.mu.Unlock()
	return sourcecontract.Telemetry{
		Connector:         "csv",
		Pages:             state.pages,
		RawInputBytes:     state.rawInputBytes,
		DecodedInputBytes: state.decodedInputBytes,
	}
}
