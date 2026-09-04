package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
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
	gremlinLabel, gremlinRelationship := seedCosmosGremlinFixture(
		t,
		ctx,
		endpoint,
		database,
		vertexContainer,
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

	gremlinGraph := fmt.Sprintf("af_cosmos_gremlin_%d", time.Now().UnixNano())
	gremlinJob := testLoadJob(gremlinGraph, "unused.csv", "unused.csv")
	gremlinJob.Metadata.Name = "cosmos-gremlin-live"
	gremlinJob.Source = config.Source{
		Type: config.SourceCosmos, Namespace: "cosmos-gremlin-live",
		Cosmos: &config.CosmosSource{
			Endpoint: endpoint, Credential: "default-azure",
			Database: database, PageSize: 2,
			Gremlin: &config.CosmosGremlin{
				Enabled:                true,
				Container:              vertexContainer,
				PartitionKeyProperty:   "partitionKey",
				LabelPrefix:            gremlinLabel,
				RelationshipTypePrefix: gremlinRelationship,
				MaxLabels:              10,
				MaxProperties:          20,
			},
		},
	}
	gremlinPath := writeLoadJob(
		t,
		dir,
		"cosmos-gremlin-create.yaml",
		gremlinJob,
	)
	gremlinResult, err := Load(ctx, gremlinPath)
	if err != nil {
		t.Fatalf("Load(Cosmos Gremlin) error = %v", err)
	}
	registerCleanup(t, dsn, gremlinGraph, gremlinResult.JobID)
	if gremlinResult.Status != meta.JobCommitted ||
		gremlinResult.Metrics.RecordsCommitted != 3 {
		t.Fatalf("Load(Cosmos Gremlin) = %#v", gremlinResult)
	}
	if _, err := Verify(ctx, gremlinPath, gremlinResult.JobID); err != nil {
		t.Fatalf("Verify(Cosmos Gremlin): %v", err)
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
	assertCypherCount(
		t,
		connection,
		gremlinGraph,
		fmt.Sprintf("MATCH (n:%s) RETURN count(n)", gremlinLabel),
		2,
	)
	assertCypherCount(
		t,
		connection,
		gremlinGraph,
		fmt.Sprintf(
			"MATCH (:%s)-[r:%s]->(:%s) RETURN count(r)",
			gremlinLabel,
			gremlinRelationship,
			gremlinLabel,
		),
		1,
	)
	assertCypherValueContains(
		t,
		connection,
		gremlinGraph,
		fmt.Sprintf(
			"MATCH (n:%s) WHERE n.name = 'Ada' RETURN n.skills",
			gremlinLabel,
		),
		`"math"`,
		`"code"`,
	)
	assertCosmosGremlinIdentities(
		t,
		connection,
		gremlinResult.JobID,
		runID,
	)
}

func TestCosmosSourceModeMatrixIntegration(t *testing.T) {
	endpoint := os.Getenv("AGEFREIGHTER_COSMOS_TEST_ENDPOINT")
	targetDSN := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if endpoint == "" || targetDSN == "" {
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
	seedCosmosSourceModeFixture(
		t,
		ctx,
		endpoint,
		database,
		vertexContainer,
		edgeContainer,
		runID,
	)

	graph := fmt.Sprintf("af_cosmos_mode_matrix_%d", time.Now().UnixNano())
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", targetDSN)
	runSourceModeMatrix(
		t,
		targetDSN,
		t.TempDir(),
		graph,
		"cosmos",
		func(mode config.LoadMode, dataset string) config.LoadJob {
			runParam, err := config.NewCosmosParamValue(runID)
			if err != nil {
				t.Fatalf("build Cosmos run parameter: %v", err)
			}
			datasetParam, err := config.NewCosmosParamValue(dataset)
			if err != nil {
				t.Fatalf("build Cosmos dataset parameter: %v", err)
			}
			parameters := []config.CosmosQueryParameter{
				{Name: "@runId", Value: runParam},
				{Name: "@dataset", Value: datasetParam},
			}
			job := testLoadJob(graph, "unused-vertices", "unused-edges")
			job.Metadata.Name = "cosmos-matrix-" + dataset
			job.Target.Mode = mode
			job.Source = config.Source{
				Type: config.SourceCosmos, Namespace: "crm",
				Cosmos: &config.CosmosSource{
					Endpoint: endpoint, Credential: "default-azure",
					Database: database, PageSize: 2,
					Vertices: []config.CosmosVertexQuery{{
						Container: vertexContainer, Label: "Person",
						Query: "SELECT * FROM c WHERE c.runId = @runId " +
							"AND c.dataset = @dataset AND c.kind = 'vertex'",
						Parameters: parameters, IDField: "/personId",
						Properties: map[string]string{
							"name": "/name", "active": "/active", "score": "/score",
							"tags": "/tags", "profile": "/profile",
						},
					}},
					Edges: []config.CosmosEdgeQuery{{
						Container: edgeContainer, Label: "KNOWS",
						Query: "SELECT * FROM c WHERE c.runId = @runId " +
							"AND c.dataset = @dataset AND c.kind = 'edge'",
						Parameters: parameters, ExternalIDField: "/relationshipId",
						Start: config.EndpointMapping{
							Label: "Person", Field: "/fromId",
						},
						End: config.EndpointMapping{
							Label: "Person", Field: "/toId",
						},
						Properties: map[string]string{"weight": "/weight"},
					}},
				},
			}
			return job
		},
		sourceModeTypedExpectations{
			predicates: []string{
				"n.active = true",
				"n.score = 9.5",
				"n.tags[0] = 'math'",
				"n.profile.city = 'London'",
			},
		},
	)
}

func TestCosmosPostgreSQLPropertyGraphIntegration(t *testing.T) {
	endpoint := os.Getenv("AGEFREIGHTER_COSMOS_TEST_ENDPOINT")
	targetDSN := os.Getenv("AGEFREIGHTER_PGGRAPH_TEST_DSN")
	if endpoint == "" || targetDSN == "" {
		t.Skip("set AGEFREIGHTER_COSMOS_TEST_ENDPOINT and AGEFREIGHTER_PGGRAPH_TEST_DSN")
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
	t.Setenv("AGEFREIGHTER_PGGRAPH_APP_TEST_DSN", targetDSN)

	t.Run("NoSQL mode matrix", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(t.Context(), 20*time.Minute)
		defer cancel()
		runID, err := newJobID()
		if err != nil {
			t.Fatalf("newJobID: %v", err)
		}
		seedCosmosSourceModeFixture(
			t, ctx, endpoint, database, vertexContainer, edgeContainer, runID,
		)

		graph := fmt.Sprintf("af_cosmos_pgq_%d", time.Now().UnixNano())
		schema := graph
		defer dropPropertyGraphSchema(t, targetDSN, schema)
		var jobIDs []string
		defer func() {
			for index := len(jobIDs) - 1; index >= 0; index-- {
				cleanupPropertyGraphJob(t, targetDSN, jobIDs[index])
			}
		}()

		phases := []struct {
			mode     config.LoadMode
			dataset  string
			records  uint64
			vertices int64
			edges    int64
		}{
			{config.LoadCreate, "create", 3, 2, 1},
			{config.LoadReplace, "replace", 3, 2, 1},
			{config.LoadAppend, "append", 2, 3, 2},
			{config.LoadUpsert, "upsert", 4, 4, 3},
		}
		for _, phase := range phases {
			t.Run(string(phase.mode), func(t *testing.T) {
				job := cosmosPostgreSQLPropertyGraphJob(
					t, graph, schema, endpoint, database, vertexContainer,
					edgeContainer, runID, phase.mode, phase.dataset,
				)
				path := writeLoadJob(
					t, t.TempDir(), "cosmos-pggraph-"+phase.dataset+".yaml", job,
				)
				if phase.mode == config.LoadCreate {
					profile, profileErr := SourceProfile(
						t.Context(), path,
						ProfileOptions{Mode: ProfileSample, SampleSize: 100},
					)
					if profileErr != nil || profile.Command != "profile" ||
						len(profile.Sections) == 0 {
						t.Fatalf("SourceProfile(Cosmos to PG19) = %#v, %v", profile, profileErr)
					}
				}
				result, loadErr := Load(t.Context(), path)
				if result.JobID != "" {
					jobIDs = append(jobIDs, result.JobID)
				}
				if loadErr != nil {
					t.Fatalf("Load(%s): %v", phase.mode, loadErr)
				}
				if result.Status != meta.JobCommitted ||
					result.Metrics.RecordsCommitted != phase.records ||
					result.SourceTelemetry == nil ||
					result.SourceTelemetry.RequestCharge <= 0 {
					t.Fatalf("Load(%s) = %#v", phase.mode, result)
				}
				if _, err := Verify(t.Context(), path, result.JobID); err != nil {
					t.Fatalf("Verify(%s): %v", phase.mode, err)
				}
				assertPropertyGraphLoad(
					t, targetDSN, job, result.JobID, phase.vertices, phase.edges,
				)
				if phase.mode == config.LoadReplace {
					if _, err := Cleanup(t.Context(), path, result.JobID); err != nil {
						t.Fatalf("Cleanup(replace backup): %v", err)
					}
				}
				if phase.mode == config.LoadUpsert {
					assertCosmosPostgreSQLPropertyGraphProperties(t, targetDSN, job)
				}
			})
		}
	})

	t.Run("Gremlin backing documents", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(t.Context(), 20*time.Minute)
		defer cancel()
		runID, err := newJobID()
		if err != nil {
			t.Fatalf("newJobID: %v", err)
		}
		label, relationship := seedCosmosGremlinFixture(
			t, ctx, endpoint, database, vertexContainer, runID,
		)
		graph := fmt.Sprintf("af_cosmos_gremlin_pgq_%d", time.Now().UnixNano())
		job := testLoadJob(graph, "unused.csv", "unused.csv")
		job.Metadata.Name = "cosmos-gremlin-pg19"
		job.Target.Type = config.TargetPostgreSQLPropertyGraph
		job.Target.Schema = graph
		job.Target.AppendDuplicate = ""
		job.Target.Connection = config.SecretRef{Env: "AGEFREIGHTER_PGGRAPH_APP_TEST_DSN"}
		job.Source = config.Source{
			Type: config.SourceCosmos, Namespace: "cosmos-gremlin-pg19",
			Cosmos: &config.CosmosSource{
				Endpoint: endpoint, Credential: "default-azure",
				Database: database, PageSize: 2,
				Gremlin: &config.CosmosGremlin{
					Enabled: true, Container: vertexContainer,
					PartitionKeyProperty: "partitionKey",
					LabelPrefix:          label, RelationshipTypePrefix: relationship,
					MaxLabels: 10, MaxProperties: 20,
				},
			},
		}
		job.Runtime.BatchRows = 2
		path := writeLoadJob(t, t.TempDir(), "cosmos-gremlin-pggraph.yaml", job)
		parsed, err := config.Load(path)
		if err != nil {
			t.Fatalf("load Cosmos Gremlin configuration: %v", err)
		}
		resolved, err := resolveSource(ctx, parsed)
		if err != nil {
			t.Fatalf("resolve Cosmos Gremlin source: %v", err)
		}
		result, err := Load(ctx, path)
		if err != nil {
			t.Fatalf("Load(Cosmos Gremlin to PG19): %v", err)
		}
		defer cleanupPropertyGraphJob(t, targetDSN, result.JobID)
		defer dropPropertyGraphSchema(t, targetDSN, graph)
		if result.Status != meta.JobCommitted ||
			result.Metrics.RecordsCommitted != 3 ||
			result.SourceTelemetry == nil ||
			result.SourceTelemetry.RequestCharge <= 0 {
			t.Fatalf("Load(Cosmos Gremlin to PG19) = %#v", result)
		}
		if _, err := Verify(ctx, path, result.JobID); err != nil {
			t.Fatalf("Verify(Cosmos Gremlin to PG19): %v", err)
		}
		assertPropertyGraphLoad(t, targetDSN, resolved, result.JobID, 2, 1)
	})
}

func cosmosPostgreSQLPropertyGraphJob(
	t *testing.T,
	graph string,
	schema string,
	endpoint string,
	database string,
	vertexContainer string,
	edgeContainer string,
	runID string,
	mode config.LoadMode,
	dataset string,
) config.LoadJob {
	t.Helper()
	runParam, err := config.NewCosmosParamValue(runID)
	if err != nil {
		t.Fatalf("build Cosmos run parameter: %v", err)
	}
	datasetParam, err := config.NewCosmosParamValue(dataset)
	if err != nil {
		t.Fatalf("build Cosmos dataset parameter: %v", err)
	}
	parameters := []config.CosmosQueryParameter{
		{Name: "@runId", Value: runParam},
		{Name: "@dataset", Value: datasetParam},
	}
	job := testLoadJob(graph, "unused-vertices", "unused-edges")
	job.Metadata.Name = "cosmos-pg19-" + dataset
	job.Target.Type = config.TargetPostgreSQLPropertyGraph
	job.Target.Schema = schema
	job.Target.Mode = mode
	job.Target.Connection = config.SecretRef{Env: "AGEFREIGHTER_PGGRAPH_APP_TEST_DSN"}
	job.Target.AppendDuplicate = ""
	if mode == config.LoadAppend {
		job.Target.AppendDuplicate = config.AppendDuplicateError
	}
	job.Source = config.Source{
		Type: config.SourceCosmos, Namespace: "crm",
		Cosmos: &config.CosmosSource{
			Endpoint: endpoint, Credential: "default-azure",
			Database: database, PageSize: 2,
			Vertices: []config.CosmosVertexQuery{{
				Container: vertexContainer, Label: "Person",
				Query: "SELECT * FROM c WHERE c.runId = @runId " +
					"AND c.dataset = @dataset AND c.kind = 'vertex'",
				Parameters: parameters, IDField: "/personId",
				Properties: map[string]string{
					"name": "/name", "active": "/active", "score": "/score",
					"tags": "/tags", "profile": "/profile",
				},
			}},
			Edges: []config.CosmosEdgeQuery{{
				Container: edgeContainer, Label: "KNOWS",
				Query: "SELECT * FROM c WHERE c.runId = @runId " +
					"AND c.dataset = @dataset AND c.kind = 'edge'",
				Parameters: parameters, ExternalIDField: "/relationshipId",
				Start:      config.EndpointMapping{Label: "Person", Field: "/fromId"},
				End:        config.EndpointMapping{Label: "Person", Field: "/toId"},
				Properties: map[string]string{"weight": "/weight"},
			}},
		},
	}
	return job
}

func assertCosmosPostgreSQLPropertyGraphProperties(
	t *testing.T,
	dsn string,
	job config.LoadJob,
) {
	t.Helper()
	definition, err := propertyGraphDefinition(job)
	if err != nil {
		t.Fatal(err)
	}
	connection, err := pgx.Connect(t.Context(), dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close(context.Background())
	var encoded []byte
	if err := connection.QueryRow(t.Context(),
		"SELECT properties FROM "+propertyGraphTable(job, definition.Vertices[0].Table)+
			" WHERE source_namespace = 'crm' AND external_id = 'p1'",
	).Scan(&encoded); err != nil {
		t.Fatal(err)
	}
	var properties map[string]any
	if err := json.Unmarshal(encoded, &properties); err != nil {
		t.Fatal(err)
	}
	tags, tagsOK := properties["tags"].([]any)
	profile, profileOK := properties["profile"].(map[string]any)
	if properties["name"] != "Ada Upsert" || properties["active"] != true ||
		properties["score"] != float64(9.5) || !tagsOK || len(tags) != 1 ||
		tags[0] != "math" || !profileOK || profile["city"] != "London" {
		t.Fatalf("Cosmos JSONB properties = %#v", properties)
	}
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

func seedCosmosSourceModeFixture(
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
	t.Cleanup(client.Close)
	vertexContainer, err := client.NewContainer(database, vertexContainerName)
	if err != nil {
		t.Fatalf("NewContainer(vertices): %v", err)
	}
	edgeContainer, err := client.NewContainer(database, edgeContainerName)
	if err != nil {
		t.Fatalf("NewContainer(edges): %v", err)
	}

	vertices := []map[string]any{
		{
			"dataset": "create", "personId": "p1", "name": "Ada Create",
			"active": true, "score": 1.5, "tags": []any{"math"},
			"profile": map[string]any{"city": "London"},
		},
		{
			"dataset": "create", "personId": "p2", "name": "Grace Create",
			"active": false, "score": 2.0, "tags": []any{"compiler"},
			"profile": map[string]any{"city": "New York"},
		},
		{
			"dataset": "replace", "personId": "p1", "name": "Ada Replace",
			"active": true, "score": 2.5, "tags": []any{"math"},
			"profile": map[string]any{"city": "London"},
		},
		{
			"dataset": "replace", "personId": "p2", "name": "Grace Replace",
			"active": false, "score": 3.0, "tags": []any{"compiler"},
			"profile": map[string]any{"city": "New York"},
		},
		{
			"dataset": "append", "personId": "p3", "name": "Katherine Append",
			"active": true, "score": 4.5, "tags": []any{"space"},
			"profile": map[string]any{"city": "Washington"},
		},
		{
			"dataset": "upsert", "personId": "p1", "name": "Ada Upsert",
			"active": true, "score": 9.5, "tags": []any{"math"},
			"profile": map[string]any{"city": "London"},
		},
		{
			"dataset": "upsert", "personId": "p4", "name": "Dorothy Upsert",
			"active": true, "score": 5.5, "tags": []any{"navy"},
			"profile": map[string]any{"city": "Arlington"},
		},
	}
	edges := []map[string]any{
		{
			"dataset": "create", "relationshipId": "e1",
			"fromId": "p1", "toId": "p2", "weight": int64(1),
		},
		{
			"dataset": "replace", "relationshipId": "e1",
			"fromId": "p1", "toId": "p2", "weight": int64(2),
		},
		{
			"dataset": "append", "relationshipId": "e2",
			"fromId": "p2", "toId": "p3", "weight": int64(3),
		},
		{
			"dataset": "upsert", "relationshipId": "e1",
			"fromId": "p1", "toId": "p4", "weight": int64(9),
		},
		{
			"dataset": "upsert", "relationshipId": "e3",
			"fromId": "p3", "toId": "p4", "weight": int64(4),
		},
	}
	refs := make([]cosmosFixtureRef, 0, len(vertices)+len(edges))
	t.Cleanup(func() {
		deleteCosmosFixture(t, refs)
	})
	for index, item := range vertices {
		id := fmt.Sprintf("%s-mv-%d", runID, index)
		partition := fmt.Sprintf("%s-mvp-%d", runID, index%3)
		item["id"] = id
		item["partitionKey"] = partition
		item["runId"] = runID
		item["kind"] = "vertex"
		refs = append(refs, cosmosFixtureRef{
			container: vertexContainer, id: id, partition: partition,
		})
		upsertCosmosFixture(t, ctx, vertexContainer, partition, item)
	}
	for index, item := range edges {
		id := fmt.Sprintf("%s-me-%d", runID, index)
		partition := fmt.Sprintf("%s-mep-%d", runID, index%3)
		item["id"] = id
		item["partitionKey"] = partition
		item["runId"] = runID
		item["kind"] = "edge"
		refs = append(refs, cosmosFixtureRef{
			container: edgeContainer, id: id, partition: partition,
		})
		upsertCosmosFixture(t, ctx, edgeContainer, partition, item)
	}
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

func seedCosmosGremlinFixture(
	t *testing.T,
	ctx context.Context,
	endpoint string,
	database string,
	containerName string,
	runID string,
) (string, string) {
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
	container, err := client.NewContainer(database, containerName)
	if err != nil {
		t.Fatalf("NewContainer(Gremlin): %v", err)
	}
	suffix := strings.ReplaceAll(runID, "-", "_")
	label := "GV_" + suffix
	relationship := "GE_" + suffix
	vertexIDs := []string{runID + "-gv1", runID + "-gv2"}
	partitions := []string{runID + "-gp1", runID + "-gp2"}
	items := []map[string]any{
		{
			"id": vertexIDs[0], "partitionKey": partitions[0],
			"label": label,
			"name": []any{map[string]any{
				"id": runID + "-vp1", "_value": "Ada", "_meta": map[string]any{},
			}},
			"skills": []any{
				map[string]any{"id": runID + "-vp2", "_value": "math"},
				map[string]any{"id": runID + "-vp3", "_value": "code"},
			},
		},
		{
			"id": vertexIDs[1], "partitionKey": partitions[1],
			"label": label,
			"name": []any{map[string]any{
				"id": runID + "-vp4", "_value": "Grace", "_meta": map[string]any{},
			}},
		},
		{
			"id": runID + "-ge1", "partitionKey": partitions[0],
			"label": relationship, "_isEdge": true,
			"_vertexId": vertexIDs[0], "_vertexLabel": label,
			"_sink": vertexIDs[1], "_sinkLabel": label,
			"_sinkPartition": partitions[1],
			"weight":         7,
		},
	}
	refs := make([]cosmosFixtureRef, 0, len(items))
	for _, item := range items {
		id := item["id"].(string)
		partition := item["partitionKey"].(string)
		refs = append(refs, cosmosFixtureRef{
			container: container,
			id:        id,
			partition: partition,
		})
		upsertCosmosFixture(t, ctx, container, partition, item)
	}
	t.Cleanup(func() {
		deleteCosmosFixture(t, refs)
	})
	return label, relationship
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

func assertCypherValueContains(
	t *testing.T,
	connection *pgx.Conn,
	graph string,
	query string,
	want ...string,
) {
	t.Helper()
	statement := fmt.Sprintf(
		`SELECT value::text
		 FROM ag_catalog.cypher('%s', $$%s$$)
		 AS result(value ag_catalog.agtype)`,
		graph,
		query,
	)
	var value string
	if err := connection.QueryRow(t.Context(), statement).Scan(&value); err != nil {
		t.Fatalf("Cypher %q error = %v", query, err)
	}
	for _, part := range want {
		if !strings.Contains(value, part) {
			t.Fatalf("Cypher %q value = %q, want substring %q", query, value, part)
		}
	}
}

func assertCosmosGremlinIdentities(
	t *testing.T,
	connection *pgx.Conn,
	jobID string,
	runID string,
) {
	t.Helper()
	query := `
		SELECT edge.external_id, source.external_id, sink.external_id
		FROM agefreighter_meta.edge_identity edge
		JOIN agefreighter_meta.graph_generation generation
		  USING (graph_generation_id)
		JOIN agefreighter_meta.vertex_identity source
		  ON source.graph_generation_id = edge.graph_generation_id
		 AND source.graph_id = edge.start_graph_id
		JOIN agefreighter_meta.vertex_identity sink
		  ON sink.graph_generation_id = edge.graph_generation_id
		 AND sink.graph_id = edge.end_graph_id
		WHERE generation.job_id = $1::uuid`
	var edge, source, sink string
	if err := connection.QueryRow(t.Context(), query, jobID).Scan(
		&edge,
		&source,
		&sink,
	); err != nil {
		t.Fatalf("query Cosmos Gremlin identities: %v", err)
	}
	want := []string{
		fmt.Sprintf("[%q,%q]", runID+"-gp1", runID+"-ge1"),
		fmt.Sprintf("[%q,%q]", runID+"-gp1", runID+"-gv1"),
		fmt.Sprintf("[%q,%q]", runID+"-gp2", runID+"-gv2"),
	}
	got := []string{edge, source, sink}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf(
				"Cosmos Gremlin identity[%d] = %q, want %q",
				index,
				got[index],
				want[index],
			)
		}
	}
}

func envOrDefault(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}
