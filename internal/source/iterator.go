package source

import (
	"context"
	"errors"
	"sync"

	"github.com/rioriost/agefreighter/pkg/model"
)

type Item struct {
	Record model.Record

	// SampleBytes is the estimated logical size used by trial migration
	// budgets. Zero uses SizeBytes. Connectors set it when SizeBytes includes
	// shared buffers that are intentionally charged to multiple records.
	SampleBytes int64

	// SizeBytes is the retained heap size of the record and any source buffers
	// that remain live until the target commits it. The pipeline adds its own
	// per-record container overhead to this value.
	SizeBytes int64
}

// Iterator must return promptly after ctx is cancelled. io.EOF marks a
// successful end of input; a record returned with a non-nil error is ignored.
type Iterator interface {
	Next(ctx context.Context) (Item, error)
	Close() error
}

type RejectionCheckpointer interface {
	RejectionCheckpoint() (int64, model.SourcePosition)
}

type Telemetry struct {
	Connector             string  `json:"connector"`
	Pages                 int64   `json:"pages"`
	RawInputBytes         int64   `json:"rawInputBytes"`
	DecodedInputBytes     int64   `json:"decodedInputBytes"`
	RequestCharge         float64 `json:"requestCharge"`
	FailedRequestAttempts int64   `json:"failedRequestAttempts"`
	ThrottledRequests     int64   `json:"throttledRequests"`
	ContinuationDigest    string  `json:"continuationDigest,omitempty"`
}

type TelemetryProvider interface {
	Telemetry() Telemetry
}

var ErrProfileBudget = errors.New("source profile budget exhausted")

type ProfileBudgetLimits struct {
	Rows, Pages, RawInputBytes, DecodedInputBytes int64
	RequestCharge                                 float64
	Labels, Properties                            int
}

type ProfileBudgetUsage struct {
	Rows, Pages, RawInputBytes, DecodedInputBytes int64
	RequestCharge                                 float64
	Labels, Properties                            int
	FailedRequestAttempts, ThrottledRequests      int64
}

type ProfileBudgetDimension uint8

const (
	ProfileBudgetLabels ProfileBudgetDimension = 1 << iota
	ProfileBudgetProperties
)

// ProfileBudget is shared by discovery and iteration so every source read is
// charged to one cumulative set of limits.
type ProfileBudget struct {
	mu     sync.Mutex
	limits ProfileBudgetLimits
	usage  ProfileBudgetUsage
	limit  string
	over   bool
}

func NewProfileBudget(limits ProfileBudgetLimits) *ProfileBudget {
	return &ProfileBudget{limits: limits}
}

func (budget *ProfileBudget) Charge(delta ProfileBudgetUsage) error {
	if budget == nil {
		return nil
	}
	budget.mu.Lock()
	defer budget.mu.Unlock()
	if budget.over {
		return ErrProfileBudget
	}
	budget.usage.Rows += delta.Rows
	budget.usage.Pages += delta.Pages
	budget.usage.RawInputBytes += delta.RawInputBytes
	budget.usage.DecodedInputBytes += delta.DecodedInputBytes
	budget.usage.RequestCharge += delta.RequestCharge
	budget.usage.Labels += delta.Labels
	budget.usage.Properties += delta.Properties
	budget.usage.FailedRequestAttempts += delta.FailedRequestAttempts
	budget.usage.ThrottledRequests += delta.ThrottledRequests
	exceeded := ""
	switch {
	case budget.limits.Rows > 0 && budget.usage.Rows > budget.limits.Rows:
		exceeded = "rows"
	case budget.limits.Pages > 0 && budget.usage.Pages > budget.limits.Pages:
		exceeded = "pages"
	case budget.limits.RawInputBytes > 0 &&
		budget.usage.RawInputBytes > budget.limits.RawInputBytes:
		exceeded = "bytes"
	case budget.limits.DecodedInputBytes > 0 &&
		budget.usage.DecodedInputBytes > budget.limits.DecodedInputBytes:
		exceeded = "bytes"
	case budget.limits.RequestCharge > 0 &&
		budget.usage.RequestCharge > budget.limits.RequestCharge:
		exceeded = "request-charge"
	case budget.limits.Labels > 0 && budget.usage.Labels > budget.limits.Labels:
		exceeded = "labels"
	case budget.limits.Properties > 0 &&
		budget.usage.Properties > budget.limits.Properties:
		exceeded = "properties"
	}
	if exceeded != "" {
		budget.limit = exceeded
		budget.over = true
		return ErrProfileBudget
	}
	return nil
}

// Full reports whether another source request can start. Catalog dimensions
// are opt-in so exhausting discovery metadata does not stop row iteration or
// unrelated catalog queries.
func (budget *ProfileBudget) Full(dimensions ...ProfileBudgetDimension) error {
	if budget == nil {
		return nil
	}

	budget.mu.Lock()
	defer budget.mu.Unlock()
	var selected ProfileBudgetDimension
	for _, dimension := range dimensions {
		selected |= dimension
	}
	switch {
	case budget.over:
	case budget.limits.Rows > 0 && budget.usage.Rows >= budget.limits.Rows:
		budget.limit = "rows"
	case budget.limits.Pages > 0 && budget.usage.Pages >= budget.limits.Pages:
		budget.limit = "pages"
	case budget.limits.RawInputBytes > 0 &&
		budget.usage.RawInputBytes >= budget.limits.RawInputBytes:
		budget.limit = "bytes"
	case budget.limits.DecodedInputBytes > 0 &&
		budget.usage.DecodedInputBytes >= budget.limits.DecodedInputBytes:
		budget.limit = "bytes"
	case budget.limits.RequestCharge > 0 &&
		budget.usage.RequestCharge >= budget.limits.RequestCharge:
		budget.limit = "request-charge"
	case selected&ProfileBudgetLabels != 0 &&
		budget.limits.Labels > 0 &&
		budget.usage.Labels >= budget.limits.Labels:
		budget.limit = "labels"
	case selected&ProfileBudgetProperties != 0 &&
		budget.limits.Properties > 0 &&
		budget.usage.Properties >= budget.limits.Properties:
		budget.limit = "properties"
	default:
		return nil
	}
	return ErrProfileBudget
}

// CanProcess reports whether another already-fetched logical input item may be
// decoded. Page and raw-byte limits are checked before fetching, not between
// items in a page that has already been charged.
func (budget *ProfileBudget) CanProcess() error {
	if budget == nil {
		return nil
	}
	budget.mu.Lock()
	defer budget.mu.Unlock()
	switch {
	case budget.over:
	case budget.limits.Rows > 0 && budget.usage.Rows >= budget.limits.Rows:
		budget.limit = "rows"
	case budget.limits.DecodedInputBytes > 0 &&
		budget.usage.DecodedInputBytes >= budget.limits.DecodedInputBytes:
		budget.limit = "bytes"
	default:
		return nil
	}
	return ErrProfileBudget
}

func (budget *ProfileBudget) Snapshot() (ProfileBudgetUsage, string) {
	if budget == nil {
		return ProfileBudgetUsage{}, ""
	}
	budget.mu.Lock()
	defer budget.mu.Unlock()
	return budget.usage, budget.limit
}
