package app

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/pggraph"
	"github.com/rioriost/agefreighter/internal/reject"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	targetruntime "github.com/rioriost/agefreighter/internal/target"
)

func executePostgreSQLPropertyGraph(
	ctx context.Context,
	job config.LoadJob,
	jobID string,
	resume bool,
	submittedFingerprint string,
) (result LoadResult, resultErr error) {
	result.JobID = jobID
	var (
		runtime targetruntime.Runtime
		err     error
	)
	if resume {
		runtime, err = openMutatingTarget(ctx, job)
	} else {
		runtime, err = openRuntime(ctx, job)
	}
	if err != nil {
		return result, err
	}
	defer runtime.Close()
	pgRuntime, err := targetruntime.RequirePGGraph(runtime)
	if err != nil {
		return result, err
	}
	adapter := pgRuntime.PGGraphAdapter()
	store := runtime.Metadata()
	if err := adapter.LockTarget(ctx, job.Target.Schema, job.Target.Graph); err != nil {
		return result, err
	}
	jobCreated := false
	preserveRunningJob := false
	recordFailure := func(cause error) error {
		if !jobCreated || (resume && (preserveRunningJob || errors.Is(cause, meta.ErrConflict))) {
			return cause
		}
		return failJob(ctx, store, jobID, cause)
	}

	fingerprint, err := jobFingerprint(job)
	if err != nil {
		return result, err
	}
	storedJob := meta.Job{NextBatchID: 1}
	initialAttempt := uint32(1)
	if resume {
		storedJob, err = store.GetJob(ctx, jobID)
		if err != nil {
			return result, err
		}
		jobCreated = true
		if err := validateStoredTargetIdentity(job, storedJob); err != nil {
			return result, err
		}
		preserveRunningJob = storedJob.Status == meta.JobRunning
		if storedJob.ConfigFingerprint != fingerprint {
			return result, errors.New("load job configuration fingerprint changed")
		}
		if storedJob.Status != meta.JobPending && storedJob.Status != meta.JobFailed &&
			storedJob.Status != meta.JobRunning {
			return result, fmt.Errorf("load job %q is %s, not resumable", jobID, storedJob.Status)
		}
		if storedJob.Status == meta.JobPending || storedJob.Status == meta.JobFailed {
			if err := store.StartJob(ctx, jobID); err != nil {
				return result, err
			}
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
		newJob := meta.Job{
			ID: jobID, Name: job.Metadata.Name, SourceType: string(job.Source.Type),
			LoadMode:      string(job.Target.Mode),
			TargetBackend: meta.TargetBackendPostgreSQLPropertyGraph,
			TargetSchema:  job.Target.Schema, TargetGraph: job.Target.Graph,
			ConfigFingerprint: fingerprint,
		}
		created, createErr := store.CreateRunningJobIfCurrent(ctx, newJob)
		if createErr != nil {
			return result, createErr
		}
		if !created {
			migrationCtx, cancel := context.WithTimeout(ctx, time.Duration(job.Runtime.OperationTimeout))
			migrateErr := store.Migrate(migrationCtx)
			cancel()
			if migrateErr != nil {
				return result, migrateErr
			}
			if err := store.CreateRunningJob(ctx, newJob); err != nil {
				return result, err
			}
		}
		jobCreated = true
	}

	definition, err := propertyGraphDefinition(job)
	if err != nil {
		return result, recordFailure(err)
	}
	loadDefinition := definition
	switch job.Target.Mode {
	case config.LoadAppend, config.LoadUpsert:
		if _, err := adapter.PrepareExisting(ctx, jobID, definition); err != nil {
			return result, recordFailure(err)
		}
	case config.LoadReplace:
		loadDefinition, err = adapter.PrepareReplace(ctx, jobID, definition)
		if err != nil {
			return result, recordFailure(err)
		}
	default:
		if _, err := adapter.Prepare(ctx, jobID, definition); err != nil {
			return result, recordFailure(err)
		}
	}
	verification, err := propertyGraphJobVerification(
		jobID, submittedFingerprint, definition,
	)
	if err != nil {
		return result, recordFailure(err)
	}
	if resume {
		if err := store.PutJobVerification(ctx, verification); err != nil {
			return result, recordFailure(err)
		}
	}

	var quarantine *reject.JSONLWriter
	if job.Errors.QuarantinePath != "" {
		quarantine, err = reject.NewJSONLWriter(job.Errors.QuarantinePath)
		if err != nil {
			return result, recordFailure(err)
		}
		defer func() { resultErr = errors.Join(resultErr, quarantine.Close()) }()
	}
	baseIterator, err := newSourceIterator(ctx, job, storedJob.ResumeToken, quarantine)
	if err != nil {
		return result, recordFailure(err)
	}
	iterator := sourcecontract.Iterator(baseIterator)
	var trialIterator *sourcecontract.TrialIterator
	if job.Trial != nil && job.Trial.Enabled {
		trialIterator, err = sourcecontract.NewTrialIterator(baseIterator, sourcecontract.TrialOptions{
			MaxVerticesPerLabel: int64(job.Trial.MaxVerticesPerLabel),
			MaxVertices:         int64(job.Trial.MaxVertices), MaxEdges: int64(job.Trial.MaxEdges),
			MaxBytes: int64(job.Trial.MaxBytes), IncludeLabels: trialLabels(job.Trial.IncludeLabels),
		})
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
	sinkOptions := pggraph.LoadSinkOptions{
		JobID: jobID, Definition: loadDefinition,
		Mode: job.Target.Mode, AppendDuplicate: job.Target.AppendDuplicate,
		PropertyMode: job.Target.PropertyMode,
	}
	if !resume {
		sinkOptions.JobVerification = &verification
	}
	target, err := pggraph.NewLoadSink(adapter, sinkOptions)
	if err != nil {
		_ = iterator.Close()
		return result, recordFailure(err)
	}
	if err := runner.Run(ctx, iterator, target); err != nil {
		return result, recordFailure(err)
	}
	snapshot := runner.Snapshot()
	if snapshot.BatchesCommitted == 0 && !resume {
		if err := store.PutJobVerification(ctx, verification); err != nil {
			return result, recordFailure(err)
		}
	}
	sourceRejected, sourcePosition := sourceRejectionCheckpoint(baseIterator)
	if sourceRejected != 0 || snapshot.BatchesCommitted == 0 {
		if err := store.SetSourceRejections(ctx, jobID, sourceRejected, meta.Position{
			Resource: sourcePosition.Resource, Line: sourcePosition.Line,
			ByteOffset: sourcePosition.Offset, Token: sourcePosition.Token,
		}); err != nil {
			return result, recordFailure(err)
		}
	}
	if quarantine != nil {
		if err := quarantine.Close(); err != nil {
			return result, recordFailure(err)
		}
		quarantine = nil
	}
	telemetry := completionTelemetry(job.Source.Type, baseIterator)
	telemetry.JobID = jobID
	var finalizeErr error
	if job.Target.Mode == config.LoadReplace {
		finalizeErr = adapter.PromoteReplace(
			ctx, jobID, definition, loadDefinition, telemetry)
	} else {
		finalizeErr = adapter.Finalize(ctx, jobID, definition, telemetry)
	}
	if finalizeErr != nil {
		current, currentErr := store.GetJob(ctx, jobID)
		if currentErr == nil && current.Status == meta.JobCommitted {
			result = LoadResult{JobID: jobID, Status: meta.JobCommitted,
				Metrics: snapshot, SourceTelemetry: sourceTelemetry(baseIterator)}
			setTrialSummary(&result, trialIterator)
			return result, nil
		}
		return result, recordFailure(finalizeErr)
	}
	result = LoadResult{JobID: jobID, Status: meta.JobCommitted,
		Metrics: snapshot, SourceTelemetry: sourceTelemetry(baseIterator)}
	setTrialSummary(&result, trialIterator)
	return result, nil
}

type configuredPropertyEdge struct {
	label string
	start string
	end   string
}

func propertyGraphDefinition(job config.LoadJob) (pggraph.Definition, error) {
	var vertices []string
	var edges []configuredPropertyEdge
	addEdges := func(configured []config.EdgeQuery) {
		for _, edge := range configured {
			edges = append(edges, configuredPropertyEdge{
				label: edge.Label, start: edge.Start.Label, end: edge.End.Label,
			})
		}
	}
	switch job.Source.Type {
	case config.SourceCSV:
		if job.Source.CSV == nil {
			return pggraph.Definition{}, errors.New("CSV source configuration is required")
		}
		for _, vertex := range job.Source.CSV.Vertices {
			vertices = append(vertices, vertex.Label)
		}
		for _, edge := range job.Source.CSV.Edges {
			edges = append(edges, configuredPropertyEdge{
				label: edge.Label, start: edge.Start.Label, end: edge.End.Label,
			})
		}
	case config.SourcePostgreSQL:
		if job.Source.PostgreSQL == nil {
			return pggraph.Definition{}, errors.New("PostgreSQL source configuration is required")
		}
		for _, vertex := range job.Source.PostgreSQL.Vertices {
			vertices = append(vertices, vertex.Label)
		}
		addEdges(job.Source.PostgreSQL.Edges)
	case config.SourceNeo4j:
		if job.Source.Neo4j == nil {
			return pggraph.Definition{}, errors.New("Neo4j source configuration is required")
		}
		for _, vertex := range job.Source.Neo4j.Vertices {
			vertices = append(vertices, vertex.Label)
		}
		addEdges(job.Source.Neo4j.Edges)
	case config.SourceCosmos:
		if job.Source.Cosmos == nil {
			return pggraph.Definition{}, errors.New("Cosmos source configuration is required")
		}
		for _, vertex := range job.Source.Cosmos.Vertices {
			vertices = append(vertices, vertex.Label)
		}
		for _, edge := range job.Source.Cosmos.Edges {
			edges = append(edges, configuredPropertyEdge{
				label: edge.Label, start: edge.Start.Label, end: edge.End.Label,
			})
		}
	default:
		return pggraph.Definition{}, fmt.Errorf("source type %q is not implemented", job.Source.Type)
	}
	slices.Sort(vertices)
	vertices = slices.Compact(vertices)
	vertexTables := make(map[string]string, len(vertices))
	definition := pggraph.Definition{Schema: job.Target.Schema, Graph: job.Target.Graph}
	for _, label := range vertices {
		table := pggraph.PhysicalName("v_", label)
		vertexTables[label] = table
		definition.Vertices = append(definition.Vertices,
			pggraph.VertexDefinition{Table: table, Label: label})
	}
	slices.SortFunc(edges, func(left, right configuredPropertyEdge) int {
		if left.label != right.label {
			return strings.Compare(left.label, right.label)
		}
		if left.start != right.start {
			return strings.Compare(left.start, right.start)
		}
		return strings.Compare(left.end, right.end)
	})
	for index, edge := range edges {
		if index > 0 && edges[index-1].label == edge.label {
			if edges[index-1] != edge {
				return pggraph.Definition{}, fmt.Errorf("edge label %q has conflicting endpoints", edge.label)
			}
			continue
		}
		start, startOK := vertexTables[edge.start]
		end, endOK := vertexTables[edge.end]
		if !startOK || !endOK {
			return pggraph.Definition{}, fmt.Errorf("edge label %q has an unmapped endpoint", edge.label)
		}
		definition.Edges = append(definition.Edges, pggraph.EdgeDefinition{
			Table: pggraph.PhysicalName("e_", edge.label), Label: edge.label,
			SourceTable: start, DestinationTable: end,
		})
	}
	if _, err := definition.Fingerprint(); err != nil {
		return pggraph.Definition{}, err
	}
	return definition, nil
}

func propertyGraphJobVerification(
	jobID string,
	submittedFingerprint string,
	definition pggraph.Definition,
) (meta.JobVerification, error) {
	definitionFingerprint, err := definition.Fingerprint()
	if err != nil {
		return meta.JobVerification{}, err
	}
	summary, err := json.Marshal(propertyGraphMappingSnapshot{
		SchemaVersion:         propertyGraphMappingSummaryVersion,
		TargetBackend:         meta.TargetBackendPostgreSQLPropertyGraph,
		DefinitionFingerprint: definitionFingerprint, Definition: definition,
	})
	if err != nil {
		return meta.JobVerification{}, fmt.Errorf("encode PostgreSQL property graph mapping: %w", err)
	}
	digest := sha256.Sum256(summary)
	return meta.JobVerification{
		JobID: jobID, SubmittedConfigFingerprint: submittedFingerprint,
		ResolvedMappingFingerprint: hex.EncodeToString(digest[:]),
		ResolvedMappingSummary:     summary,
	}, nil
}

const propertyGraphMappingSummaryVersion = 1

type propertyGraphMappingSnapshot struct {
	SchemaVersion         int                `json:"schemaVersion"`
	TargetBackend         meta.TargetBackend `json:"targetBackend"`
	DefinitionFingerprint string             `json:"definitionFingerprint"`
	Definition            pggraph.Definition `json:"definition"`
}

func persistedPropertyGraphDefinition(
	ctx context.Context,
	store *meta.Store,
	job config.LoadJob,
	stored meta.Job,
) (pggraph.Definition, error) {
	submittedFingerprint, err := jobFingerprint(job)
	if err != nil {
		return pggraph.Definition{}, err
	}
	verification, err := store.GetJobVerification(ctx, stored.ID)
	if err != nil {
		return pggraph.Definition{}, err
	}
	if verification.SubmittedConfigFingerprint != submittedFingerprint {
		return pggraph.Definition{}, errors.New(
			"submitted PostgreSQL property graph configuration fingerprint changed",
		)
	}
	decoder := json.NewDecoder(bytes.NewReader(verification.ResolvedMappingSummary))
	decoder.DisallowUnknownFields()
	var snapshot propertyGraphMappingSnapshot
	if err := decoder.Decode(&snapshot); err != nil {
		return pggraph.Definition{}, fmt.Errorf(
			"decode persisted PostgreSQL property graph definition: %w", err,
		)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return pggraph.Definition{}, errors.New(
			"persisted PostgreSQL property graph definition must contain one JSON object",
		)
	}
	if snapshot.SchemaVersion != propertyGraphMappingSummaryVersion ||
		snapshot.TargetBackend != meta.TargetBackendPostgreSQLPropertyGraph {
		return pggraph.Definition{}, fmt.Errorf(
			"%w: persisted PostgreSQL property graph definition identity is invalid",
			meta.ErrGenerationMismatch,
		)
	}
	canonical, err := json.Marshal(snapshot)
	if err != nil {
		return pggraph.Definition{}, fmt.Errorf(
			"encode persisted PostgreSQL property graph definition: %w", err,
		)
	}
	digest := sha256.Sum256(canonical)
	if hex.EncodeToString(digest[:]) != verification.ResolvedMappingFingerprint {
		return pggraph.Definition{}, fmt.Errorf(
			"%w: persisted PostgreSQL property graph definition digest changed",
			meta.ErrGenerationMismatch,
		)
	}
	definitionFingerprint, err := snapshot.Definition.Fingerprint()
	if err != nil {
		return pggraph.Definition{}, err
	}
	if definitionFingerprint != snapshot.DefinitionFingerprint {
		return pggraph.Definition{}, fmt.Errorf(
			"%w: persisted PostgreSQL property graph definition fingerprint changed",
			meta.ErrGenerationMismatch,
		)
	}
	if snapshot.Definition.Schema != job.Target.Schema ||
		snapshot.Definition.Graph != job.Target.Graph {
		return pggraph.Definition{}, fmt.Errorf(
			"%w: persisted PostgreSQL property graph target identity changed",
			meta.ErrGenerationMismatch,
		)
	}
	return snapshot.Definition, nil
}
