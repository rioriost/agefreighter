package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/Azure/azure-sdk-for-go/sdk/data/azcosmos"
	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
)

type cosmosFixtureRef struct {
	container *azcosmos.ContainerClient
	id        string
	partition string
}

func TestCosmosLiveIntegration(t *testing.T) {
	endpoint := os.Getenv("AGEFREIGHTER_COSMOS_TEST_ENDPOINT")
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if endpoint == "" || dsn == "" {
		t.Skip("set AGEFREIGHTER_COSMOS_TEST_ENDPOINT and AGEFREIGHTER_AGE_TEST_DSN")
	}
	database := envOrDefault("AGEFREIGHTER_COSMOS_TEST_DATABASE", "agefreighter")
	vertexContainer := envOrDefault(
		"AGEFREIGHTER_COSMOS_TEST_VERTEX_CONTAINER",
		"vertices",
	)
	edgeContainer := envOrDefault(
		"AGEFREIGHTER_COSMOS_TEST_EDGE_CONTAINER",
		"edges",
	)

	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Minute)
	defer cancel()
	runID, err := newJobID()
	if err != nil {
		t.Fatalf("newJobID: %v", err)
	}
	seedCosmosFixture(
		t,
		ctx,
		endpoint,
		database,
		vertexContainer,
		edgeContainer,
		runID,
	)

	graph := fmt.Sprintf("af_cosmos_%d", time.Now().UnixNano())
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", dsn)
	job := cosmosLiveJob(
		t,
		graph,
		endpoint,
		database,
		vertexContainer,
		edgeContainer,
		runID,
	)
	dir := t.TempDir()
	jobPath := writeLoadJob(t, dir, "cosmos-create.yaml", job)
	jobID, err := newJobID()
	if err != nil {
		t.Fatalf("newJobID: %v", err)
	}
	registerCleanup(t, dsn, graph, jobID)

	runPartialCosmosLoad(t, ctx, job, jobID)
	stored, err := Status(ctx, jobPath, jobID)
	if err != nil || stored.Status != meta.JobRunning ||
		stored.CommittedRows != 2 || stored.ResumeToken == "" {
		t.Fatalf("partial Cosmos job = %#v, %v", stored, err)
	}

	resumed, err := Resume(ctx, jobPath, jobID)
	if err != nil {
		t.Fatalf("Resume: %v", err)
	}
	if resumed.Status != meta.JobCommitted ||
		resumed.Metrics.RecordsCommitted != 9 ||
		resumed.SourceTelemetry == nil ||
		resumed.SourceTelemetry.Pages < 2 ||
		resumed.SourceTelemetry.RequestCharge <= 0 {
		t.Fatalf("Resume = %#v", resumed)
	}
	if _, err := Verify(ctx, jobPath, jobID); err != nil {
		t.Fatalf("Verify(create): %v", err)
	}

	replaceJob := job
	replaceJob.Metadata.Name = "cosmos-live-replace"
	replaceJob.Target.Mode = config.LoadReplace
	replacePath := writeLoadJob(t, dir, "cosmos-replace.yaml", replaceJob)
	replaced, err := Load(ctx, replacePath)
	if replaced.JobID != "" {
		backupName, deriveErr := age.DeriveGraphName(
			graph,
			age.BackupName,
			replaced.JobID,
		)
		if deriveErr != nil {
			t.Fatalf("derive replacement backup: %v", deriveErr)
		}
		shadowName, deriveErr := age.DeriveGraphName(
			graph,
			age.ShadowName,
			replaced.JobID,
		)
		if deriveErr != nil {
			t.Fatalf("derive replacement shadow: %v", deriveErr)
		}
		registerReplaceCleanup(
			t,
			dsn,
			replaced.JobID,
			graph,
			shadowName,
			backupName,
		)
	}
	if err != nil {
		t.Fatalf("Load(replace): %v", err)
	}
	if replaced.Status != meta.JobCommitted ||
		replaced.Metrics.RecordsCommitted != 11 ||
		replaced.SourceTelemetry == nil ||
		replaced.SourceTelemetry.Pages < 2 {
		t.Fatalf("Load(replace) = %#v", replaced)
	}
	if _, err := Verify(ctx, replacePath, replaced.JobID); err != nil {
		t.Fatalf("Verify(replace): %v", err)
	}
	if _, err := Cleanup(ctx, replacePath, replaced.JobID); err != nil {
		t.Fatalf("Cleanup(replace backup): %v", err)
	}

	connection, err := pgx.Connect(ctx, dsn)
	if err != nil {
		t.Fatalf("connect AGE for verification: %v", err)
	}
	defer connection.Close(context.Background())
	if _, err := connection.Exec(ctx, "LOAD 'age'"); err != nil {
		t.Fatalf("load AGE extension: %v", err)
	}
	if _, err := connection.Exec(
		ctx,
		`SET search_path = ag_catalog, "$user", public`,
	); err != nil {
		t.Fatalf("set AGE search path: %v", err)
	}
	assertCypherCount(t, connection, graph, "MATCH (n:Person) RETURN count(n)", 6)
	assertCypherCount(
		t,
		connection,
		graph,
		"MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r)",
		5,
	)
}

func cosmosLiveJob(
	t *testing.T,
	graph string,
	endpoint string,
	database string,
	vertexContainer string,
	edgeContainer string,
	runID string,
) config.LoadJob {
	t.Helper()
	runValue, err := config.NewCosmosParamValue(runID)
	if err != nil {
		t.Fatalf("NewCosmosParamValue: %v", err)
	}
	job := testLoadJob(graph, "unused.csv", "unused.csv")
	job.Metadata.Name = "cosmos-live-create"
	job.Source = config.Source{
		Type: config.SourceCosmos, Namespace: "cosmos-live",
		Cosmos: &config.CosmosSource{
			Endpoint: endpoint, Credential: "default-azure",
			Database: database, PageSize: 2,
			Vertices: []config.CosmosVertexQuery{{
				Container: vertexContainer,
				Label:     "Person",
				Query:     "SELECT * FROM c WHERE c.runId = @runId",
				Parameters: []config.CosmosQueryParameter{{
					Name: "@runId", Value: runValue,
				}},
				IDField: "/id",
				Properties: map[string]string{
					"name": "/profile/name",
					"rank": "/rank",
					"tags": "/tags",
				},
			}},
			Edges: []config.CosmosEdgeQuery{{
				Container: edgeContainer,
				Label:     "KNOWS",
				Query:     "SELECT * FROM c WHERE c.runId = @runId",
				Parameters: []config.CosmosQueryParameter{{
					Name: "@runId", Value: runValue,
				}},
				ExternalIDField: "/id",
				Start: config.EndpointMapping{
					Label: "Person", Field: "/endpoints/from",
				},
				End: config.EndpointMapping{
					Label: "Person", Field: "/endpoints/to",
				},
				Properties: map[string]string{"weight": "/weight"},
			}},
		},
	}
	job.Runtime.BatchRows = 2
	job.Runtime.BatchBytes = 1 << 20
	job.Runtime.MemoryLimit = 64 << 20
	return job
}

func runPartialCosmosLoad(
	t *testing.T,
	ctx context.Context,
	job config.LoadJob,
	jobID string,
) {
	t.Helper()
	adapter, store, err := openTarget(ctx, job)
	if err != nil {
		t.Fatalf("openTarget: %v", err)
	}
	defer adapter.Close()
	fingerprint, err := jobFingerprint(job)
	if err != nil {
		t.Fatalf("jobFingerprint: %v", err)
	}
	if err := store.CreateJob(ctx, meta.Job{
		ID: jobID, Name: job.Metadata.Name,
		SourceType: string(job.Source.Type), LoadMode: string(job.Target.Mode),
		TargetGraph: job.Target.Graph, ConfigFingerprint: fingerprint,
	}); err != nil {
		t.Fatalf("CreateJob: %v", err)
	}
	if err := store.StartJob(ctx, jobID); err != nil {
		t.Fatalf("StartJob: %v", err)
	}
	graph, labels, err := createCatalog(ctx, adapter, job, jobID)
	if err != nil {
		t.Fatalf("createCatalog: %v", err)
	}
	iterator, err := newSourceIterator(ctx, job, "", nil)
	if err != nil {
		t.Fatalf("newSourceIterator: %v", err)
	}
	runner, err := newPipelineRunner(job, 1, 1)
	if err != nil {
		_ = iterator.Close()
		t.Fatalf("newPipelineRunner: %v", err)
	}
	target, err := age.NewLoadSink(ctx, adapter, age.LoadSinkOptions{
		JobID: jobID, Graph: graph, Labels: labels,
		MissingEndpoint: job.Errors.MissingEndpoint,
	})
	if err != nil {
		_ = iterator.Close()
		t.Fatalf("NewLoadSink: %v", err)
	}
	if err := runner.Run(ctx, &limitedIterator{
		Iterator: iterator, remaining: 2,
	}, target); err != nil {
		t.Fatalf("partial pipeline: %v", err)
	}
}

func seedCosmosFixture(
	t *testing.T,
	ctx context.Context,
	endpoint string,
	database string,
	vertexContainerName string,
	edgeContainerName string,
	runID string,
) {
	t.Helper()
	credential, err := azidentity.NewDefaultAzureCredential(nil)
	if err != nil {
		t.Fatalf("NewDefaultAzureCredential: %v", err)
	}
	client, err := azcosmos.NewClient(endpoint, credential, nil)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() {
		client.Close()
	})
	vertexContainer, err := client.NewContainer(database, vertexContainerName)
	if err != nil {
		t.Fatalf("NewContainer(vertices): %v", err)
	}
	edgeContainer, err := client.NewContainer(database, edgeContainerName)
	if err != nil {
		t.Fatalf("NewContainer(edges): %v", err)
	}

	var refs []cosmosFixtureRef
	t.Cleanup(func() {
		deleteCosmosFixture(t, refs)
	})
	vertexIDs := make([]string, 6)
	for index := range vertexIDs {
		vertexIDs[index] = fmt.Sprintf("%s-v%d", runID, index)
		partition := fmt.Sprintf("vertex-%d", index%3)
		item := map[string]any{
			"id": vertexIDs[index], "partitionKey": partition,
			"runId": runID, "kind": "vertex",
			"profile": map[string]any{"name": fmt.Sprintf("Person %d", index)},
			"rank":    int64(index),
			"tags":    []any{"cosmos", index},
		}
		refs = append(refs, cosmosFixtureRef{
			container: vertexContainer, id: vertexIDs[index], partition: partition,
		})
		upsertCosmosFixture(t, ctx, vertexContainer, partition, item)
	}
	for index := range 5 {
		id := fmt.Sprintf("%s-e%d", runID, index)
		partition := fmt.Sprintf("edge-%d", index%3)
		item := map[string]any{
			"id": id, "partitionKey": partition,
			"runId": runID, "kind": "edge",
			"endpoints": map[string]any{
				"from": vertexIDs[index],
				"to":   vertexIDs[index+1],
			},
			"weight": float64(index) + 0.5,
		}
		refs = append(refs, cosmosFixtureRef{
			container: edgeContainer, id: id, partition: partition,
		})
		upsertCosmosFixture(t, ctx, edgeContainer, partition, item)
	}
}

func upsertCosmosFixture(
	t *testing.T,
	ctx context.Context,
	container *azcosmos.ContainerClient,
	partition string,
	item map[string]any,
) {
	t.Helper()
	encoded, err := json.Marshal(item)
	if err != nil {
		t.Fatalf("marshal Cosmos fixture: %v", err)
	}
	deadline := time.Now().Add(10 * time.Minute)
	for {
		_, err = container.UpsertItem(
			ctx,
			azcosmos.NewPartitionKeyString(partition),
			encoded,
			nil,
		)
		if err == nil {
			return
		}
		var responseErr *azcore.ResponseError
		if !errors.As(err, &responseErr) ||
			(responseErr.StatusCode != 403 && responseErr.StatusCode != 404) ||
			time.Now().After(deadline) {
			t.Fatalf("upsert Cosmos fixture: %v", err)
		}
		select {
		case <-ctx.Done():
			t.Fatalf("upsert Cosmos fixture: %v", ctx.Err())
		case <-time.After(10 * time.Second):
		}
	}
}

func deleteCosmosFixture(t *testing.T, refs []cosmosFixtureRef) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	for _, ref := range refs {
		_, err := ref.container.DeleteItem(
			ctx,
			azcosmos.NewPartitionKeyString(ref.partition),
			ref.id,
			nil,
		)
		if err == nil {
			continue
		}
		var responseErr *azcore.ResponseError
		if errors.As(err, &responseErr) && responseErr.StatusCode == 404 {
			continue
		}
		t.Errorf(
			"delete Cosmos fixture %q from partition %q: %v",
			ref.id,
			ref.partition,
			err,
		)
	}
}

func envOrDefault(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}
