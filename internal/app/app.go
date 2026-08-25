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
	sourcecsv "github.com/rioriost/agefreighter/internal/source/csv"
)

const maxSecretBytes = 1 << 20

type LoadResult struct {
	JobID   string                   `json:"jobId"`
	Status  meta.JobStatus           `json:"status"`
	Metrics pipeline.MetricsSnapshot `json:"metrics"`
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
	graphCatalog, err := adapter.LookupGraph(ctx, job.Target.Graph)
	if err != nil {
		return meta.Job{}, fmt.Errorf("verify graph catalog: %w", err)
	}
	graph, err := store.AdmitGraphGeneration(ctx, jobID, meta.GraphGeneration{
		ID: stored.GraphGenerationID, JobID: jobID,
		GraphName: graphCatalog.Name, GraphOID: graphCatalog.GraphOID,
		NamespaceOID: graphCatalog.NamespaceOID, Generation: 1,
		State: meta.GenerationActive,
	})
	if err != nil {
		return meta.Job{}, fmt.Errorf("verify graph generation: %w", err)
	}
	kinds, err := configuredLabels(job)
	if err != nil {
		return meta.Job{}, err
	}
	for name, kind := range kinds {
		catalog, err := adapter.LookupLabel(ctx, job.Target.Graph, name)
		if err != nil {
			return meta.Job{}, fmt.Errorf("verify label catalog %q: %w", name, err)
		}
		generation, err := store.AdmitLabelGeneration(ctx, graph.ID, meta.LabelGeneration{
			ID: 1, GraphGenerationID: graph.ID, LabelName: name,
			Kind: meta.LabelKind(kind), GraphNamespaceOID: catalog.NamespaceOID,
			LabelID: catalog.LabelID, RelationOID: catalog.RelationOID,
			SequenceOID: catalog.SequenceOID, MappingGeneration: 1,
		})
		if err != nil {
			return meta.Job{}, fmt.Errorf("verify label generation %q: %w", name, err)
		}
		expected, err := store.CountLabelIdentities(
			ctx, graph.ID, generation.ID, generation.Kind,
		)
		if err != nil {
			return meta.Job{}, err
		}
		if err := adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
			return transaction.VerifyLabelRows(ctx, catalog, expected)
		}); err != nil {
			return meta.Job{}, err
		}
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
	if job.Source.Type != config.SourceCSV || job.Source.CSV == nil {
		return result, errors.New("only CSV sources are implemented")
	}
	if job.Target.Mode != config.LoadCreate {
		return result, errors.New("only create load mode is implemented")
	}
	if job.Errors.MissingEndpoint == config.MissingEndpointDefer {
		return result, errors.New("deferred missing endpoints are not implemented")
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
			graph, labels, err = createCatalog(ctx, adapter, job, jobID)
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
		graph, labels, err = createCatalog(ctx, adapter, job, jobID)
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
	iteratorOptions := sourcecsv.IteratorOptions{
		Namespace: job.Source.Namespace,
		Source:    *job.Source.CSV, AfterToken: storedJob.ResumeToken,
		RejectLimit: job.Errors.RejectLimit,
	}
	if job.Errors.MalformedRecord == config.MalformedQuarantine {
		iteratorOptions.OnMalformed = func(ctx context.Context, malformed sourcecsv.MalformedRecord) error {
			return quarantine.Write(ctx, reject.Rejection{
				Fields: malformed.Fields, Position: malformed.Position,
				Code: "malformed-record", Message: malformed.Err.Error(),
			})
		}
	}
	iterator, err := sourcecsv.NewIterator(ctx, iteratorOptions)
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
		MissingEndpoint: job.Errors.MissingEndpoint,
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
	sourceRejected, sourcePosition := iterator.RejectionCheckpoint()
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
	if err := store.CompleteJobGeneration(ctx, jobID, graph.ID); err != nil {
		current, currentErr := store.GetJob(ctx, jobID)
		if currentErr == nil && current.Status == meta.JobCommitted {
			return LoadResult{
				JobID: jobID, Status: meta.JobCommitted, Metrics: runner.Snapshot(),
			}, nil
		}
		return result, recordFailure(err)
	}
	return LoadResult{
		JobID: jobID, Status: meta.JobCommitted, Metrics: runner.Snapshot(),
	}, nil
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
		MinConnections: 2, MaxConnections: int32(job.Runtime.MaxTargetConnections),
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
