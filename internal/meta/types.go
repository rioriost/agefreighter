package meta

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"
)

var (
	ErrConflict            = errors.New("metadata state conflict")
	ErrGenerationMismatch  = errors.New("target generation mismatch")
	ErrIncrementalConflict = errors.New(
		"AF_INCREMENTAL_CONFLICT: incremental target is busy",
	)
	ErrNotFound = errors.New("metadata record not found")
)

type JobStatus string

const (
	JobPending   JobStatus = "pending"
	JobRunning   JobStatus = "running"
	JobCommitted JobStatus = "committed"
	JobFailed    JobStatus = "failed"
)

type Job struct {
	ID                       string
	Name                     string
	SourceType               string
	LoadMode                 string
	TargetGraph              string
	BackupGraphName          string
	ConfigFingerprint        string
	Status                   JobStatus
	GraphGenerationID        int64
	NextBatchID              uint64
	ResumeToken              string
	CommittedRows            int64
	CommittedBytes           int64
	RejectedRows             int64
	SourceRejectedRows       int64
	ErrorMessage             string
	CreatedAt                time.Time
	StartedAt                *time.Time
	UpdatedAt                time.Time
	CompletedAt              *time.Time
	BackupCleanedAt          *time.Time
	VerificationSourceAccess bool   `json:",omitempty"`
	VerificationEvidence     string `json:",omitempty"`
}

type GenerationState string

const (
	GenerationLoading GenerationState = "loading"
	GenerationActive  GenerationState = "active"
	GenerationRetired GenerationState = "retired"
)

type GraphGeneration struct {
	ID               int64
	JobID            string
	GraphName        string
	GraphOID         uint32
	NamespaceOID     uint32
	ReplacesGraphOID uint32
	Generation       uint64
	State            GenerationState
	CreatedAt        time.Time
	UpdatedAt        time.Time
}

type LabelKind byte

const (
	VertexLabel LabelKind = 'v'
	EdgeLabel   LabelKind = 'e'
)

func (kind *LabelKind) Scan(source any) error {
	if kind == nil {
		return errors.New("label kind scan target is nil")
	}
	var value string
	switch source := source.(type) {
	case string:
		value = source
	case []byte:
		value = string(source)
	default:
		return fmt.Errorf("cannot scan label kind from %T", source)
	}
	if len(value) != 1 || (value[0] != byte(VertexLabel) && value[0] != byte(EdgeLabel)) {
		return fmt.Errorf("stored label kind %q is invalid", value)
	}
	*kind = LabelKind(value[0])
	return nil
}

type LabelGeneration struct {
	ID                int64
	GraphGenerationID int64
	LabelName         string
	Kind              LabelKind
	GraphNamespaceOID uint32
	LabelID           uint16
	RelationOID       uint32
	SequenceOID       uint32
	MappingGeneration uint64
	CreatedAt         time.Time
	UpdatedAt         time.Time
}

type Position struct {
	Resource   string
	Line       int64
	ByteOffset int64
	Token      string
}

type BatchStatus string

const (
	BatchRunning   BatchStatus = "running"
	BatchCommitted BatchStatus = "committed"
	BatchFailed    BatchStatus = "failed"
)

type BatchAttempt struct {
	JobID        string
	BatchID      uint64
	Attempt      uint32
	Status       BatchStatus
	Rows         int64
	Bytes        int64
	RejectedRows int64
	First        Position
	Last         Position
	ErrorMessage string
	StartedAt    time.Time
	FinishedAt   *time.Time
}

type JobVerification struct {
	JobID                      string
	SubmittedConfigFingerprint string
	ResolvedMappingFingerprint string
	ResolvedMappingSummary     json.RawMessage
}

type CounterCompleteness string

const (
	CounterComplete   CounterCompleteness = "complete"
	CounterIncomplete CounterCompleteness = "incomplete"
)

type CounterProvenance string

const (
	CounterProvenanceLifecycle           CounterProvenance = "v17-lifecycle"
	CounterProvenanceLegacyResume        CounterProvenance = "legacy-resume"
	CounterProvenanceBaselineUnavailable CounterProvenance = "baseline-unavailable"
)

type LabelCounter struct {
	JobID             string
	LabelGenerationID int64
	Kind              LabelKind
	Completeness      CounterCompleteness
	Provenance        CounterProvenance
	AcceptedRows      *int64
	CommittedRows     *int64
	CommittedBytes    *int64
	RejectedRows      *int64
}

type BatchLabelCounter struct {
	LabelGenerationID int64
	Kind              LabelKind
	AcceptedRows      int64
	CommittedRows     int64
	CommittedBytes    *int64
	RejectedRows      int64
}

type RejectRecord struct {
	JobID        string
	BatchID      uint64
	Attempt      uint32
	Position     Position
	ErrorClass   string
	ErrorMessage string
	Record       json.RawMessage
}

func validateJob(job Job) error {
	if err := validateJobID(job.ID); err != nil {
		return err
	}
	if strings.TrimSpace(job.Name) == "" {
		return errors.New("job name is required")
	}
	switch job.SourceType {
	case "csv", "postgresql", "neo4j", "cosmos-nosql":
	default:
		return fmt.Errorf("unsupported source type %q", job.SourceType)
	}
	switch job.LoadMode {
	case "create", "replace", "append", "upsert":
	default:
		return fmt.Errorf("unsupported load mode %q", job.LoadMode)
	}
	if strings.TrimSpace(job.TargetGraph) == "" {
		return errors.New("target graph is required")
	}
	if err := validateFingerprint(job.ConfigFingerprint); err != nil {
		return err
	}
	if job.Status != "" && job.Status != JobPending {
		return errors.New("new job status must be pending")
	}
	return nil
}

func validateJobID(value string) error {
	if len(value) != 36 ||
		value != strings.ToLower(value) ||
		value[8] != '-' || value[13] != '-' ||
		value[18] != '-' || value[23] != '-' {
		return errors.New("job ID must be a canonical UUID")
	}
	encoded := strings.ReplaceAll(value, "-", "")
	if _, err := hex.DecodeString(encoded); err != nil {
		return errors.New("job ID must be a canonical UUID")
	}
	return nil
}

func ValidateJobID(value string) error {
	return validateJobID(value)
}

func validateFingerprint(value string) error {
	if len(value) != 64 || strings.ToLower(value) != value {
		return errors.New("config fingerprint must be 64 lowercase hexadecimal characters")
	}
	if _, err := hex.DecodeString(value); err != nil {
		return errors.New("config fingerprint must be 64 lowercase hexadecimal characters")
	}
	return nil
}

func validateGraphGeneration(value GraphGeneration) error {
	if err := validateJobID(value.JobID); err != nil {
		return err
	}
	if strings.TrimSpace(value.GraphName) == "" {
		return errors.New("graph generation name is required")
	}
	if value.GraphOID == 0 || value.NamespaceOID == 0 ||
		value.GraphOID != value.NamespaceOID {
		return errors.New("graph and namespace OIDs must be equal and positive")
	}
	if value.ReplacesGraphOID == value.GraphOID {
		return errors.New("replacement and shadow graph OIDs must differ")
	}
	if value.Generation == 0 || value.Generation > math.MaxInt64 {
		return errors.New("graph generation must be within 1..MaxInt64")
	}
	switch value.State {
	case GenerationLoading, GenerationActive, GenerationRetired:
	default:
		return fmt.Errorf("unsupported graph generation state %q", value.State)
	}
	return nil
}

func validateLabelGeneration(value LabelGeneration) error {
	if value.GraphGenerationID <= 0 {
		return errors.New("graph generation ID must be positive")
	}
	if strings.TrimSpace(value.LabelName) == "" {
		return errors.New("label generation name is required")
	}
	if value.Kind != VertexLabel && value.Kind != EdgeLabel {
		return fmt.Errorf("unsupported label kind %q", value.Kind)
	}
	if value.GraphNamespaceOID == 0 || value.LabelID == 0 ||
		value.RelationOID == 0 || value.SequenceOID == 0 {
		return errors.New("label catalog identifiers must be positive")
	}
	if value.MappingGeneration == 0 || value.MappingGeneration > math.MaxInt64 {
		return errors.New("mapping generation must be within 1..MaxInt64")
	}
	return nil
}

func validateBatch(value BatchAttempt) error {
	if err := validateJobID(value.JobID); err != nil {
		return err
	}
	if value.BatchID == 0 || value.BatchID > math.MaxInt64 {
		return errors.New("batch ID must be within 1..MaxInt64")
	}
	if value.Attempt == 0 || uint64(value.Attempt) > math.MaxInt32 {
		return errors.New("batch attempt must be within 1..MaxInt32")
	}
	if value.Rows < 0 || value.Bytes < 0 || value.RejectedRows < 0 {
		return errors.New("batch rows, bytes, and rejected rows cannot be negative")
	}
	if err := validatePosition(value.First); err != nil {
		return fmt.Errorf("first position: %w", err)
	}
	return nil
}

func validatePosition(value Position) error {
	if value.Line < 0 || value.ByteOffset < 0 {
		return errors.New("source line and byte offset cannot be negative")
	}
	return nil
}

func validateReject(value RejectRecord) error {
	if err := validateJobID(value.JobID); err != nil {
		return err
	}
	if value.BatchID == 0 || value.BatchID > math.MaxInt64 || value.Attempt == 0 {
		return errors.New("reject batch ID and attempt must be positive")
	}
	if strings.TrimSpace(value.Position.Token) == "" {
		return errors.New("reject resume token is required")
	}
	if err := validatePosition(value.Position); err != nil {
		return err
	}
	if strings.TrimSpace(value.ErrorClass) == "" ||
		strings.TrimSpace(value.ErrorMessage) == "" {
		return errors.New("reject error class and message are required")
	}
	if len(value.Record) > 0 && !json.Valid(value.Record) {
		return errors.New("reject record must contain valid JSON")
	}
	return nil
}
