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
	"strings"
	"time"

	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/pipeline"
	"github.com/rioriost/agefreighter/internal/reject"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
)

const maxSecretBytes = 1 << 20

type LoadResult struct {
	JobID           string                    `json:"jobId"`
	Status          meta.JobStatus            `json:"status"`
	Metrics         pipeline.MetricsSnapshot  `json:"metrics"`
	SourceTelemetry *sourcecontract.Telemetry `json:"sourceTelemetry,omitempty"`
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
	adapter, store, err := openTarget(ctx, job)
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
	adapter, store, err := openTarget(ctx, job)
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
	if _, err := newPipelineRunner(job, 1, 1); err != nil {
		return result, fmt.Errorf("validate load pipeline: %w", err)
	}
	adapter, store, err := openTarget(ctx, job)
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
	iterator, err := newSourceIterator(ctx, job, storedJob.ResumeToken, quarantine)
	if err != nil {
		return result, recordFailure(err)
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
	sourceRejected, sourcePosition := sourceRejectionCheckpoint(iterator)
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
	if job.Target.Mode == config.LoadReplace {
		completeErr = promoteReplace(ctx, adapter, job, jobID, graph)
	} else if incrementalMode(job.Target.Mode) {
		completeErr = completeIncremental(ctx, adapter, jobID, graph)
	} else {
		completeErr = store.CompleteJobGeneration(ctx, jobID, graph.ID)
	}

	if completeErr != nil {
		current, currentErr := store.GetJob(ctx, jobID)
		if currentErr == nil && current.Status == meta.JobCommitted {
			return LoadResult{
				JobID: jobID, Status: meta.JobCommitted, Metrics: runner.Snapshot(),
				SourceTelemetry: sourceTelemetry(iterator),
			}, nil
		}
		return result, recordFailure(completeErr)
	}
	return LoadResult{
		JobID: jobID, Status: meta.JobCommitted, Metrics: runner.Snapshot(),
		SourceTelemetry: sourceTelemetry(iterator),
	}, nil
}

func incrementalMode(mode config.LoadMode) bool {
	return mode == config.LoadAppend || mode == config.LoadUpsert
}

func completeIncremental(
	ctx context.Context,
	adapter *age.Adapter,
	jobID string,
	graph meta.GraphGeneration,
) error {
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
		return transactionStore.CompleteJob(ctx, jobID)
	})
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

func openTarget(
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
	if err := store.Migrate(ctx); err != nil {
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
