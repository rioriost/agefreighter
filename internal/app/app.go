package app

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/pipeline"
	"github.com/rioriost/agefreighter/internal/reject"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

const maxSecretBytes = 1 << 20

type LoadResult struct {
	JobID           string                       `json:"jobId"`
	Status          meta.JobStatus               `json:"status"`
	Metrics         pipeline.MetricsSnapshot     `json:"metrics"`
	SourceTelemetry *sourcecontract.Telemetry    `json:"sourceTelemetry,omitempty"`
	Trial           *sourcecontract.TrialSummary `json:"trial,omitempty"`
}

func Load(ctx context.Context, path string) (LoadResult, error) {
	job, err := config.Load(path)
	if err != nil {
		return LoadResult{}, fmt.Errorf("load job configuration: %w", err)
	}
	jobID, err := newJobID()
	if err != nil {
		return LoadResult{}, err
	}
	return execute(ctx, job, jobID, false)
}

func Resume(ctx context.Context, path, jobID string) (LoadResult, error) {
	job, err := config.Load(path)
	if err != nil {
		return LoadResult{}, fmt.Errorf("load job configuration: %w", err)
	}
	return execute(ctx, job, jobID, true)
}

func Status(ctx context.Context, path, jobID string) (meta.Job, error) {
	job, err := config.Load(path)
	if err != nil {
		return meta.Job{}, fmt.Errorf("load target configuration: %w", err)
	}
	adapter, store, err := openCurrentTarget(ctx, job)
	if err != nil {
		return meta.Job{}, err
	}
	defer adapter.Close()
	return store.GetJob(ctx, jobID)
}

func Verify(ctx context.Context, path, jobID string) (meta.Job, error) {
	job, err := config.Load(path)
	if err != nil {
		return meta.Job{}, err
	}
	job, err = resolveSource(ctx, job)
	if err != nil {
		return meta.Job{}, err
	}
	adapter, store, err := openCurrentTarget(ctx, job)
	if err != nil {
		return meta.Job{}, err
	}
	defer adapter.Close()
	stored, err := store.GetJob(ctx, jobID)
	if err != nil {
		return meta.Job{}, err
	}
	if stored.Status != meta.JobCommitted {
		return meta.Job{}, fmt.Errorf("load job %q is %s, not committed", jobID, stored.Status)
	}
	fingerprint, err := jobFingerprint(job)
	if err != nil {
		return meta.Job{}, err
	}
	if fingerprint != stored.ConfigFingerprint {
		return meta.Job{}, errors.New("load job configuration fingerprint changed")
	}
	graph, err := store.GraphGenerationForJob(ctx, jobID)
	if err != nil {
		return meta.Job{}, err
	}
	if graph.State != meta.GenerationActive ||
		graph.GraphName != job.Target.Graph {
		return meta.Job{}, fmt.Errorf(
			"%w: committed graph generation is not active at target %q",
			meta.ErrGenerationMismatch,
			job.Target.Graph,
		)
	}
	if err := adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		transactionStore, err := transaction.Metadata()
		if err != nil {
			return err
		}
		return verifyGenerationTransaction(
			ctx,
			transaction,
			transactionStore,
			job,
			graph,
		)
	}); err != nil {
		return meta.Job{}, err
	}
	return stored, nil
}

func execute(
	ctx context.Context,
	job config.LoadJob,
	jobID string,
	resume bool,
) (result LoadResult, resultErr error) {
	result.JobID = jobID
	submittedFingerprint, err := jobFingerprint(job)
	if err != nil {
		return result, fmt.Errorf("fingerprint submitted job: %w", err)
	}
	tracer := trace.SpanFromContext(ctx).TracerProvider().Tracer(
		"github.com/rioriost/agefreighter/internal/app",
	)
	ctx, span := tracer.Start(
		ctx,
		"load.execute",
		trace.WithAttributes(
			attribute.String("source.type", string(job.Source.Type)),
			attribute.String("target.type", string(job.Target.Type)),
			attribute.String("load.mode", string(job.Target.Mode)),
			attribute.Bool("load.resume", resume),
			attribute.Bool("load.trial", job.Trial != nil && job.Trial.Enabled),
		),
	)
	defer func() {
		if resultErr != nil {
			span.SetStatus(codes.Error, "load failed")
		} else {
			span.SetStatus(codes.Ok, "")
		}
		span.End()
	}()
	resolvedJob, err := resolveSource(ctx, job)
	if err != nil {
		return result, err
	}
	job = resolvedJob
	if err := validateImplementedSource(job); err != nil {
		return result, err
	}
	switch job.Target.Mode {
	case config.LoadCreate, config.LoadReplace,
		config.LoadAppend, config.LoadUpsert:
	default:
		return result, fmt.Errorf(
			"load mode %q is not implemented",
			job.Target.Mode,
		)
	}
	if resume && job.Trial != nil && job.Trial.Enabled {
		return result, errors.New(
			"trial jobs cannot be resumed; start a new load instead",
		)
	}
	if _, err := newPipelineRunner(job, 1, 1); err != nil {
		return result, fmt.Errorf("validate load pipeline: %w", err)
	}
	adapter, store, err := openMutatingTarget(ctx, job)
	if err != nil {
		return result, err
	}
	defer adapter.Close()
	preserveRunningJob := false
	recordFailure := func(cause error) error {
		if resume && (preserveRunningJob || errors.Is(cause, meta.ErrConflict)) {
			return cause
		}
		return failJob(ctx, store, jobID, cause)
	}

	fingerprint, err := jobFingerprint(job)
	if err != nil {
		return result, err
	}
	var storedJob meta.Job
	var graph meta.GraphGeneration
	var labels []age.LoadLabel
	initialAttempt := uint32(1)
	if resume {
		storedJob, err = store.GetJob(ctx, jobID)
		if err != nil {
			return result, err
		}
		preserveRunningJob = storedJob.Status == meta.JobRunning
		if storedJob.ConfigFingerprint != fingerprint {
			return result, errors.New("load job configuration fingerprint changed")
		}
		if storedJob.Status != meta.JobPending &&
			storedJob.Status != meta.JobFailed &&
			storedJob.Status != meta.JobRunning {
			return result, fmt.Errorf("load job %q is %s, not failed", jobID, storedJob.Status)
		}
		if storedJob.Status == meta.JobPending ||
			storedJob.Status == meta.JobFailed {
			if err := store.StartJob(ctx, jobID); err != nil {
				return result, err
			}
		}
		if storedJob.GraphGenerationID == 0 {
			if incrementalMode(job.Target.Mode) {
				graph, labels, err = admitIncrementalCatalog(
					ctx,
					adapter,
					job,
					jobID,
				)
			} else {
				graph, labels, err = createCatalog(ctx, adapter, job, jobID)
			}
		} else {
			graph, labels, err = admitCatalog(ctx, adapter, store, job, storedJob)
		}
		if err != nil {
			return result, recordFailure(err)
		}
		latest, latestErr := store.LatestBatch(ctx, jobID)
		if latestErr == nil && latest.BatchID == storedJob.NextBatchID {
			switch latest.Status {
			case meta.BatchRunning:
				initialAttempt = latest.Attempt
			case meta.BatchFailed:
				if latest.Attempt == math.MaxUint32 {
					return result, recordFailure(errors.New("load batch attempt counter is exhausted"))
				}
				initialAttempt = latest.Attempt + 1
			}
		} else if latestErr != nil && !errors.Is(latestErr, meta.ErrNotFound) {
			return result, recordFailure(latestErr)
		}
	} else {
		if err := store.CreateJob(ctx, meta.Job{
			ID: jobID, Name: job.Metadata.Name,
			SourceType: string(job.Source.Type), LoadMode: string(job.Target.Mode),
			TargetGraph: job.Target.Graph, ConfigFingerprint: fingerprint,
		}); err != nil {
			return result, err
		}
		if err := store.StartJob(ctx, jobID); err != nil {
			return result, err
		}

		if incrementalMode(job.Target.Mode) {
			graph, labels, err = admitIncrementalCatalog(
				ctx,
				adapter,
				job,
				jobID,
			)
		} else {
			graph, labels, err = createCatalog(ctx, adapter, job, jobID)
		}
		if err != nil {
			return result, recordFailure(err)
		}
		storedJob, err = store.GetJob(ctx, jobID)
		if err != nil {
			return result, recordFailure(err)
		}
	}
	if err := putJobVerification(
		ctx, store, job, jobID, submittedFingerprint, labels, resume,
	); err != nil {
		return result, recordFailure(err)
	}

	var quarantine *reject.JSONLWriter
	if job.Errors.QuarantinePath != "" {
		quarantine, err = reject.NewJSONLWriter(job.Errors.QuarantinePath)
		if err != nil {
			return result, recordFailure(err)
		}
		defer func() {
			resultErr = errors.Join(resultErr, quarantine.Close())
		}()
	}
	baseIterator, err := newSourceIterator(
		ctx,
		job,
		storedJob.ResumeToken,
		quarantine,
	)
	if err != nil {
		return result, recordFailure(err)
	}
	iterator := baseIterator
	var trialIterator *sourcecontract.TrialIterator
	if job.Trial != nil && job.Trial.Enabled {
		trialIterator, err = sourcecontract.NewTrialIterator(
			baseIterator,
			sourcecontract.TrialOptions{
				MaxVerticesPerLabel: int64(job.Trial.MaxVerticesPerLabel),
				MaxVertices:         int64(job.Trial.MaxVertices),
				MaxEdges:            int64(job.Trial.MaxEdges),
				MaxBytes:            int64(job.Trial.MaxBytes),
				IncludeLabels:       trialLabels(job.Trial.IncludeLabels),
			},
		)
		if err != nil {
			_ = baseIterator.Close()
			return result, recordFailure(err)
		}
		iterator = trialIterator
	}
	runner, err := newPipelineRunner(job, storedJob.NextBatchID, initialAttempt)
	if err != nil {
		_ = iterator.Close()
		return result, recordFailure(err)
	}
	sinkOptions := age.LoadSinkOptions{
		JobID: jobID, Graph: graph, Labels: labels,
		Mode:             job.Target.Mode,
		AppendDuplicate:  job.Target.AppendDuplicate,
		PropertyMode:     job.Target.PropertyMode,
		MissingEndpoint:  job.Errors.MissingEndpoint,
		MaxDeferredEdges: job.Errors.MaxDeferredEdges,
	}
	if quarantine != nil {
		sinkOptions.Quarantine = quarantine
	}
	target, err := age.NewLoadSink(ctx, adapter, sinkOptions)
	if err != nil {
		_ = iterator.Close()
		return result, recordFailure(err)
	}
	if err := runner.Run(ctx, iterator, target); err != nil {
		return result, recordFailure(err)
	}
	sourceRejected, sourcePosition := sourceRejectionCheckpoint(baseIterator)
	if err := store.SetSourceRejections(
		ctx,
		jobID,
		sourceRejected,
		meta.Position{
			Resource: sourcePosition.Resource, Line: sourcePosition.Line,
			ByteOffset: sourcePosition.Offset, Token: sourcePosition.Token,
		},
	); err != nil {
		return result, recordFailure(err)
	}
	if quarantine != nil {
		if err := quarantine.Close(); err != nil {
			return result, recordFailure(err)
		}
		quarantine = nil
	}
	var completeErr error
	telemetry := completionTelemetry(job.Source.Type, baseIterator)
	if job.Target.Mode == config.LoadReplace {
		completeErr = promoteReplace(ctx, adapter, job, jobID, graph, telemetry)
	} else if incrementalMode(job.Target.Mode) {
		completeErr = completeIncremental(ctx, adapter, jobID, graph, telemetry)
	} else {
		completeErr = store.CompleteJobGenerationWithTelemetry(
			ctx,
			jobID,
			graph.ID,
			telemetry,
		)
	}

	if completeErr != nil {
		current, currentErr := store.GetJob(ctx, jobID)
		if currentErr == nil && current.Status == meta.JobCommitted {
			result := LoadResult{
				JobID: jobID, Status: meta.JobCommitted, Metrics: runner.Snapshot(),
				SourceTelemetry: sourceTelemetry(baseIterator),
			}
			setTrialSummary(&result, trialIterator)
			return result, nil
		}
		return result, recordFailure(completeErr)
	}
	result = LoadResult{
		JobID: jobID, Status: meta.JobCommitted, Metrics: runner.Snapshot(),
		SourceTelemetry: sourceTelemetry(baseIterator),
	}
	setTrialSummary(&result, trialIterator)
	return result, nil
}

const (
	legacyResolvedMappingSummaryVersion = 1
	resolvedMappingSummaryVersion       = 2
)

type identityCoverage string

const (
	identityCoverageUnknown  identityCoverage = ""
	identityCoverageFull     identityCoverage = "full"
	identityCoverageOptional identityCoverage = "optional"
)

type resolvedMappingSnapshot struct {
	SchemaVersion int                     `json:"schemaVersion"`
	SourceType    string                  `json:"sourceType"`
	Labels        []resolvedLabelSnapshot `json:"labels"`
}

type resolvedLabelSnapshot struct {
	ID                int64            `json:"labelGenerationId"`
	GraphGenerationID int64            `json:"graphGenerationId"`
	Name              string           `json:"name"`
	Kind              string           `json:"kind"`
	GraphNamespaceOID uint32           `json:"graphNamespaceOid"`
	LabelID           uint16           `json:"labelId"`
	RelationOID       uint32           `json:"relationOid"`
	SequenceOID       uint32           `json:"sequenceOid"`
	MappingGeneration uint64           `json:"mappingGeneration"`
	IdentityCoverage  identityCoverage `json:"identityCoverage,omitempty"`
}

func putJobVerification(
	ctx context.Context,
	store *meta.Store,
	job config.LoadJob,
	jobID string,
	submittedFingerprint string,
	labels []age.LoadLabel,
	allowLegacy bool,
) error {
	summary, err := resolvedMappingSummary(job, labels)
	if err != nil {
		return err
	}
	digest := sha256.Sum256(summary)
	value := meta.JobVerification{
		JobID:                      jobID,
		SubmittedConfigFingerprint: submittedFingerprint,
		ResolvedMappingFingerprint: hex.EncodeToString(digest[:]),
		ResolvedMappingSummary:     summary,
	}
	err = store.PutJobVerification(ctx, value)
	if err == nil || !allowLegacy || !errors.Is(err, meta.ErrConflict) {
		return err
	}
	stored, readErr := store.GetJobVerification(ctx, jobID)
	if readErr != nil {
		return errors.Join(err, readErr)
	}
	if legacyErr := validateLegacyJobVerification(
		stored, job, submittedFingerprint, labels,
	); legacyErr != nil {
		return errors.Join(err, legacyErr)
	}
	return nil
}

func validateLegacyJobVerification(
	stored meta.JobVerification,
	job config.LoadJob,
	submittedFingerprint string,
	loadLabels []age.LoadLabel,
) error {
	if stored.SubmittedConfigFingerprint != submittedFingerprint {
		return errors.New("legacy submitted configuration fingerprint changed")
	}
	snapshot, labels, _, err := parseResolvedMappingSummary(
		stored.ResolvedMappingSummary,
	)
	if err != nil {
		return fmt.Errorf("validate legacy resolved mapping: %w", err)
	}
	if snapshot.SchemaVersion != legacyResolvedMappingSummaryVersion ||
		snapshot.SourceType != string(job.Source.Type) {
		return errors.New("stored resolved mapping is not a compatible legacy snapshot")
	}
	canonical, err := json.Marshal(snapshot)
	if err != nil {
		return fmt.Errorf("canonicalize legacy resolved mapping: %w", err)
	}
	digest := sha256.Sum256(canonical)
	if hex.EncodeToString(digest[:]) != stored.ResolvedMappingFingerprint {
		return errors.New("legacy resolved mapping fingerprint changed")
	}
	if len(labels) != len(loadLabels) {
		return errors.New("legacy resolved label set changed")
	}
	byID := make(map[int64]meta.LabelGeneration, len(labels))
	for _, label := range labels {
		byID[label.ID] = label
	}
	for _, loadLabel := range loadLabels {
		storedLabel, ok := byID[loadLabel.Generation.ID]
		if !ok || !sameResolvedLabel(storedLabel, loadLabel.Generation) {
			return errors.New("legacy resolved label set changed")
		}
	}
	return nil
}

func resolvedMappingSummary(
	job config.LoadJob,
	loadLabels []age.LoadLabel,
) (json.RawMessage, error) {
	kinds, err := configuredLabels(job)
	if err != nil {
		return nil, fmt.Errorf("summarize resolved mappings: %w", err)
	}
	coverage, err := resolvedIdentityCoverage(job)
	if err != nil {
		return nil, fmt.Errorf("summarize resolved identity coverage: %w", err)
	}
	labels := make([]resolvedLabelSnapshot, 0, len(loadLabels))
	seen := make(map[string]age.LabelKind, len(loadLabels))
	for _, loadLabel := range loadLabels {
		generation := loadLabel.Generation
		if err := validateResolvedLabelSnapshot(generation); err != nil {
			return nil, fmt.Errorf("summarize resolved mapping %q: %w", generation.LabelName, err)
		}
		if _, exists := seen[generation.LabelName]; exists {
			return nil, fmt.Errorf("duplicate resolved label %q", generation.LabelName)
		}
		generationKind := age.VertexLabel
		if generation.Kind == meta.EdgeLabel {
			generationKind = age.EdgeLabel
		}
		seen[generation.LabelName] = generationKind
		labels = append(labels, resolvedLabelSnapshot{
			ID:                generation.ID,
			GraphGenerationID: generation.GraphGenerationID,
			Name:              generation.LabelName,
			Kind:              string(byte(generation.Kind)),
			GraphNamespaceOID: generation.GraphNamespaceOID,
			LabelID:           generation.LabelID,
			RelationOID:       generation.RelationOID,
			SequenceOID:       generation.SequenceOID,
			MappingGeneration: generation.MappingGeneration,
			IdentityCoverage:  coverage[generation.LabelName],
		})
	}

	if len(seen) != len(kinds) {
		return nil, errors.New("resolved label set does not match configured mappings")
	}
	for name, kind := range kinds {
		if seen[name] != kind {
			return nil, fmt.Errorf("resolved label %q kind does not match configured mapping", name)
		}
	}
	slices.SortFunc(labels, func(left, right resolvedLabelSnapshot) int {
		if left.Name != right.Name {
			return strings.Compare(left.Name, right.Name)
		}
		if left.Kind != right.Kind {
			return strings.Compare(left.Kind, right.Kind)
		}
		if left.ID < right.ID {
			return -1
		}
		if left.ID > right.ID {
			return 1
		}
		return 0
	})
	return json.Marshal(resolvedMappingSnapshot{
		SchemaVersion: resolvedMappingSummaryVersion,
		SourceType:    string(job.Source.Type),
		Labels:        labels,
	})
}

func resolvedIdentityCoverage(
	job config.LoadJob,
) (map[string]identityCoverage, error) {
	kinds, err := configuredLabels(job)
	if err != nil {
		return nil, err
	}
	coverage := make(map[string]identityCoverage, len(kinds))
	for name, kind := range kinds {
		if kind == age.VertexLabel {
			coverage[name] = identityCoverageFull
		}
	}
	recordEdge := func(name string, hasExternalIdentity bool) {
		value := identityCoverageOptional
		if hasExternalIdentity {
			value = identityCoverageFull
		}
		if coverage[name] == identityCoverageOptional ||
			value == identityCoverageOptional {
			coverage[name] = identityCoverageOptional
			return
		}
		coverage[name] = identityCoverageFull
	}
	switch job.Source.Type {
	case config.SourceCSV:
		for _, edge := range job.Source.CSV.Edges {
			recordEdge(edge.Label, edge.ExternalIDColumn != "")
		}
	case config.SourcePostgreSQL:
		for _, edge := range job.Source.PostgreSQL.Edges {
			recordEdge(edge.Label, edge.ExternalIDField != "")
		}
	case config.SourceNeo4j:
		for _, edge := range job.Source.Neo4j.Edges {
			recordEdge(edge.Label, edge.ExternalIDField != "")
		}
	case config.SourceCosmos:
		for _, edge := range job.Source.Cosmos.Edges {
			recordEdge(edge.Label, edge.ExternalIDField != "")
		}
	default:
		return nil, fmt.Errorf("source type %q is not implemented", job.Source.Type)
	}
	for name := range kinds {
		if coverage[name] == identityCoverageUnknown {
			return nil, fmt.Errorf("label %q identity coverage is unresolved", name)
		}
	}
	return coverage, nil
}

func validateResolvedLabelSnapshot(label meta.LabelGeneration) error {
	if label.ID <= 0 || label.GraphGenerationID <= 0 ||
		label.LabelName == "" || label.GraphNamespaceOID == 0 ||
		label.LabelID == 0 || label.RelationOID == 0 ||
		label.SequenceOID == 0 || label.MappingGeneration == 0 {
		return errors.New("resolved label catalog identity is incomplete")
	}
	if label.Kind != meta.VertexLabel && label.Kind != meta.EdgeLabel {
		return errors.New("resolved label kind is invalid")
	}
	return nil
}

func trialLabels(labels []string) []model.Label {
	result := make([]model.Label, len(labels))
	for index, label := range labels {
		result[index] = model.Label(label)
	}
	return result
}

func setTrialSummary(
	result *LoadResult,
	iterator *sourcecontract.TrialIterator,
) {
	if iterator == nil {
		return
	}
	summary := iterator.Summary()
	result.Trial = &summary
}

func incrementalMode(mode config.LoadMode) bool {
	return mode == config.LoadAppend || mode == config.LoadUpsert
}

func completeIncremental(
	ctx context.Context,
	adapter *age.Adapter,
	jobID string,
	graph meta.GraphGeneration,
	telemetry ...meta.ConnectorTelemetry,
) error {
	if len(telemetry) > 1 {
		return errors.New("at most one connector telemetry summary is allowed")
	}
	return adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		locked, err := transaction.TryLockGraphLifecycle(ctx, graph.GraphName)
		if err != nil {
			return err
		}
		if !locked {
			return meta.ErrIncrementalConflict
		}
		catalog, err := transaction.LookupGraph(ctx, graph.GraphName)
		if err != nil {
			return err
		}
		transactionStore, err := transaction.Metadata()
		if err != nil {
			return err
		}
		stored, err := transactionStore.GraphGenerationForJob(ctx, jobID)
		if err != nil {
			return err
		}
		if stored.ID != graph.ID ||
			stored.State != meta.GenerationActive ||
			stored.GraphName != catalog.Name ||
			stored.GraphOID != catalog.GraphOID ||
			stored.NamespaceOID != catalog.NamespaceOID {
			return fmt.Errorf(
				"%w: incremental graph generation is no longer active at %q",
				meta.ErrGenerationMismatch,
				graph.GraphName,
			)
		}
		if len(telemetry) == 1 {
			return transactionStore.CompleteJobWithTelemetry(
				ctx,
				jobID,
				telemetry[0],
			)
		}
		return transactionStore.CompleteJob(ctx, jobID)
	})
}

func completionTelemetry(
	sourceType config.SourceType,
	iterator sourcecontract.Iterator,
) meta.ConnectorTelemetry {
	value := meta.ConnectorTelemetry{Connector: string(sourceType)}
	if telemetry := sourceTelemetry(iterator); telemetry != nil {
		value.Connector = telemetry.Connector
		value.Pages = telemetry.Pages
		value.RequestCharge = telemetry.RequestCharge
		value.FailedRequestAttempts = telemetry.FailedRequestAttempts
		value.ThrottledRequests = telemetry.ThrottledRequests
		value.ContinuationDigest = telemetry.ContinuationDigest
	}
	return value
}

func newPipelineRunner(
	job config.LoadJob,
	initialBatchID uint64,
	initialAttempt uint32,
) (*pipeline.Runner, error) {
	return pipeline.New(pipeline.Options{
		MemoryLimitBytes: int64(job.Runtime.MemoryLimit),
		MaxBatchRows:     job.Runtime.BatchRows, MaxBatchBytes: int64(job.Runtime.BatchBytes),
		RecordChannelCapacity: job.Runtime.MaxSourceConcurrency,
		BatchChannelCapacity:  1, OperationTimeout: time.Duration(job.Runtime.OperationTimeout),
		InitialBatchID: initialBatchID, InitialAttempt: initialAttempt,
	})
}

func openMutatingTarget(
	ctx context.Context,
	job config.LoadJob,
) (*age.Adapter, *meta.Store, error) {
	adapter, store, err := openAGEStore(ctx, job)
	if err != nil {
		return nil, nil, err
	}
	migrationCtx, cancel := context.WithTimeout(
		ctx,
		time.Duration(job.Runtime.OperationTimeout),
	)
	defer cancel()
	if err := store.Migrate(migrationCtx); err != nil {
		adapter.Close()
		return nil, nil, err
	}
	return adapter, store, nil
}

// openTarget retains the 2.0 mutating target-open contract for load/resume and
// existing internal callers.
func openTarget(
	ctx context.Context,
	job config.LoadJob,
) (*age.Adapter, *meta.Store, error) {
	return openMutatingTarget(ctx, job)
}

type readOnlyTarget struct {
	Adapter  *age.Adapter
	Store    *meta.Store
	Metadata meta.SchemaInspection
}

func openReadOnlyTarget(
	ctx context.Context,
	job config.LoadJob,
) (readOnlyTarget, error) {
	adapter, store, err := openAGEStore(ctx, job)
	if err != nil {
		return readOnlyTarget{}, err
	}
	inspection, err := store.InspectSchema(ctx)
	if err != nil {
		adapter.Close()
		return readOnlyTarget{}, err
	}
	return readOnlyTarget{
		Adapter: adapter, Store: store, Metadata: inspection,
	}, nil
}

func openCurrentTarget(
	ctx context.Context,
	job config.LoadJob,
) (*age.Adapter, *meta.Store, error) {
	target, err := openReadOnlyTarget(ctx, job)
	if err != nil {
		return nil, nil, err
	}
	if err := target.Metadata.RequireReadCompatible(); err != nil {
		target.Adapter.Close()
		return nil, nil, err
	}
	return target.Adapter, target.Store, nil
}

func probeTarget(
	ctx context.Context,
	job config.LoadJob,
) (age.DegradedProbe, error) {
	dsn, err := resolveSecret(job.Target.Connection)
	if err != nil {
		return age.DegradedProbe{}, fmt.Errorf("resolve target connection: %w", err)
	}
	return age.ProbeDegraded(ctx, dsn, age.ProbeOptions{
		ConnectTimeout:   time.Duration(job.Runtime.OperationTimeout),
		OperationTimeout: time.Duration(job.Runtime.OperationTimeout),
	})
}

func openAGEStore(
	ctx context.Context,
	job config.LoadJob,
) (*age.Adapter, *meta.Store, error) {
	dsn, err := resolveSecret(job.Target.Connection)
	if err != nil {
		return nil, nil, fmt.Errorf("resolve target connection: %w", err)
	}
	adapter, err := age.Open(ctx, dsn, age.PoolOptions{
		MinConnections: 1, MaxConnections: int32(job.Runtime.MaxTargetConnections),
		ConnectTimeout:   time.Duration(job.Runtime.OperationTimeout),
		OperationTimeout: time.Duration(job.Runtime.OperationTimeout),
	})
	if err != nil {
		return nil, nil, err
	}
	store, err := adapter.Metadata()
	if err != nil {
		adapter.Close()
		return nil, nil, err
	}
	return adapter, store, nil
}

func resolveSecret(reference config.SecretRef) (string, error) {
	if reference.Env != "" {
		value, exists := os.LookupEnv(reference.Env)
		if !exists || value == "" {
			return "", fmt.Errorf("environment variable %q is empty or unset", reference.Env)
		}
		return value, nil
	}
	file, err := os.Open(reference.File)
	if err != nil {
		return "", fmt.Errorf("open secret file: %w", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return "", fmt.Errorf("stat secret file: %w", err)
	}
	if info.Size() > maxSecretBytes {
		return "", errors.New("secret file exceeds 1 MiB")
	}
	data, err := io.ReadAll(io.LimitReader(file, maxSecretBytes+1))
	if err != nil {
		return "", fmt.Errorf("read secret file: %w", err)
	}
	if len(data) > maxSecretBytes {
		return "", errors.New("secret file exceeds 1 MiB")
	}
	value := strings.TrimSuffix(strings.TrimSuffix(string(data), "\n"), "\r")
	if value == "" {
		return "", errors.New("secret file is empty")
	}
	return value, nil
}

func jobFingerprint(job config.LoadJob) (string, error) {
	encoded, err := json.Marshal(job)
	if err != nil {
		return "", fmt.Errorf("encode load job fingerprint: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

func newJobID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", fmt.Errorf("generate load job ID: %w", err)
	}
	value[6] = value[6]&0x0f | 0x40
	value[8] = value[8]&0x3f | 0x80
	encoded := hex.EncodeToString(value[:])
	return fmt.Sprintf("%s-%s-%s-%s-%s",
		encoded[:8], encoded[8:12], encoded[12:16], encoded[16:20], encoded[20:]), nil
}
