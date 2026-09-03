package rangedigest

const (
	ManifestVersion  = 1
	CanonicalVersion = "agefreighter-production-simulation-v1"
)

type Leaf struct {
	Kind       string `json:"kind"`
	Name       string `json:"name"`
	RangeIndex int    `json:"rangeIndex"`
	StartKey   int64  `json:"startKey"`
	EndKey     int64  `json:"endKey"`
	Rows       int64  `json:"rows"`
	SHA256     string `json:"sha256"`
}

type Manifest struct {
	Version          int    `json:"version"`
	CanonicalVersion string `json:"canonicalVersion"`
	Source           string `json:"source"`
	FixtureRoot      string `json:"fixtureRootSha256"`
	Graph            string `json:"graph,omitempty"`
	JobID            string `json:"jobId,omitempty"`
	RangeRows        int64  `json:"rangeRows"`
	RecordCount      int64  `json:"recordCount"`
	Leaves           []Leaf `json:"leaves"`
	RootSHA256       string `json:"rootSha256"`
}

type Comparison struct {
	Status       string `json:"status"`
	FixtureRoot  string `json:"fixtureRootSha256"`
	ExpectedRoot string `json:"expectedRootSha256"`
	ActualRoot   string `json:"actualRootSha256"`
	ExpectedRows int64  `json:"expectedRows"`
	ActualRows   int64  `json:"actualRows"`
	Leaves       int    `json:"leaves"`
	Mismatch     string `json:"mismatch,omitempty"`
}
