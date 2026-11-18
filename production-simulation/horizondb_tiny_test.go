package simulation_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/Azure/azure-sdk-for-go/sdk/data/azcosmos"
	"github.com/jackc/pgx/v5"
	"github.com/neo4j/neo4j-go-driver/v6/neo4j"
	"github.com/rioriost/agefreighter/internal/app"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
	"github.com/rioriost/agefreighter/production-simulation/internal/rangedigest"
)

func TestHorizonDBTinyIntegration(t *testing.T) {
	if os.Getenv("AGEFREIGHTER_REQUIRE_HORIZON_TINY") != "1" {
		t.Skip("explicit HorizonDB tiny qualification required")
	}
	required := []string{
		"AGEFREIGHTER_HORIZON_TINY_OUTPUT", "AGEFREIGHTER_HORIZON_PG17_HOST",
		"AGEFREIGHTER_HORIZON_PG18_HOST", "AGEFREIGHTER_HORIZON_USER",
		"AGEFREIGHTER_POSTGRES_TEST_DSN", "AGEFREIGHTER_NEO4J44_URI",
		"AGEFREIGHTER_NEO4J526_URI", "AGEFREIGHTER_NEO4J_PASSWORD",
		"AGEFREIGHTER_COSMOS_TEST_ENDPOINT",
	}
	for _, name := range required {
		if os.Getenv(name) == "" {
			t.Fatalf("required tiny qualification environment variable %s is missing", name)
		}
	}
	if err := validateTinySources(
		os.Getenv("AGEFREIGHTER_POSTGRES_TEST_DSN"),
		os.Getenv("AGEFREIGHTER_NEO4J44_URI"),
		os.Getenv("AGEFREIGHTER_NEO4J526_URI"),
		os.Getenv("AGEFREIGHTER_COSMOS_TEST_ENDPOINT"),
	); err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	root := os.Getenv(required[0])
	if err := os.Mkdir(root, 0700); err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(root, "fixture", "manifest.json")
	f, err := fixture.Generate(ctx, fixture.GenerateConfig{
		Phase: fixture.PhaseTiny, Output: filepath.Dir(manifestPath),
		Shards: 1, Workers: 1, Seed: 20260829,
	})
	if err != nil {
		t.Fatal(err)
	}
	if f.Plan.VertexTotal != 160 || f.Plan.EdgeTotal != 400 {
		t.Fatal("fixture exceeds the approved tiny envelope")
	}
	explicitDir, gremlinDir := filepath.Join(root, "explicit"), filepath.Join(root, "gremlin")
	exported, err := portable.Export(ctx, manifestPath, explicitDir)
	if err != nil {
		t.Fatal(err)
	}
	gremlin, err := portable.ExportGremlin(ctx, manifestPath, gremlinDir)
	if err != nil {
		t.Fatal(err)
	}
	reuse := os.Getenv("AGEFREIGHTER_HORIZON_TINY_REUSE_FIXTURE")
	if reuse != "" {
		previous, err := fixture.Verify(filepath.Join(reuse, "fixture", "manifest.json"))
		if err != nil || previous.RootSHA256 != f.RootSHA256 {
			t.Fatalf("immutable retry fixture differs: %v", err)
		}
		t.Log("reusing unchanged source fixtures; no source writes or cleanup")
	} else {
		seedTinyPostgres(t, ctx, exported, explicitDir)
		for _, name := range []string{"AGEFREIGHTER_NEO4J44_URI", "AGEFREIGHTER_NEO4J526_URI"} {
			seedTinyNeo4j(t, ctx, exported, explicitDir, name)
		}
	}
	credential, err := azidentity.NewManagedIdentityCredential(nil)
	if err != nil {
		t.Fatal(err)
	}
	if reuse == "" {
		client, err := azcosmos.NewClient(os.Getenv("AGEFREIGHTER_COSMOS_TEST_ENDPOINT"), credential, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer client.Close()
		database, err := client.NewDatabase("agefreighter")
		if err != nil {
			t.Fatal(err)
		}
		for _, input := range []struct {
			name, path string
			manifest   portable.Manifest
		}{{"tiny-explicit", explicitDir, exported}, {"tiny-gremlin", gremlinDir, gremlin}} {
			container, err := database.NewContainer(input.name)
			if err != nil {
				t.Fatal(err)
			}
			pager := container.NewQueryItemsPager("SELECT VALUE 1 FROM c", azcosmos.NewPartitionKey(), &azcosmos.QueryOptions{PageSizeHint: 1})
			for pager.More() {
				page, err := pager.NextPage(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if len(page.Items) != 0 {
					t.Fatal("refusing to overwrite an existing Cosmos fixture")
				}
			}
			for _, table := range input.manifest.Tables {
				for _, doc := range tinyDocuments(t, input.path, table) {
					body, err := json.Marshal(doc)
					if err != nil {
						t.Fatal(err)
					}
					pk, ok := doc["partitionKey"].(string)
					if !ok || pk == "" {
						t.Fatal("fixture partition key missing")
					}
					if _, err := container.CreateItem(ctx, azcosmos.NewPartitionKeyString(pk), body, nil); err != nil {
						t.Fatal(err)
					}
				}
			}
		}
	}
	expected, err := rangedigest.FixtureManifest(ctx, manifestPath, 100000)
	if err != nil {
		t.Fatal(err)
	}
	expectedGremlin, err := rangedigest.GremlinFixtureManifest(ctx, manifestPath, 100000)
	if err != nil {
		t.Fatal(err)
	}
	runSuffix := strconv.FormatInt(time.Now().UnixNano(), 36)
	for _, major := range []int{17, 18} {
		t.Run(fmt.Sprintf("pg%d", major), func(t *testing.T) {
			host := os.Getenv(fmt.Sprintf("AGEFREIGHTER_HORIZON_PG%d_HOST", major))
			if !strings.HasPrefix(host, "afht-") || !strings.HasSuffix(host, ".japaneast.horizondb.azure.com") {
				t.Fatal("actual Japan East HorizonDB hostname required")
			}
			token, err := credential.GetToken(ctx, policy.TokenRequestOptions{
				Scopes: []string{"https://ossrdbms-aad.database.windows.net/.default"},
			})
			if err != nil {
				t.Fatal(err)
			}
			u := url.URL{Scheme: "postgresql", Host: host + ":5432", Path: "/postgres",
				User: url.UserPassword(os.Getenv("AGEFREIGHTER_HORIZON_USER"), token.Token)}
			q := u.Query()
			q.Set("sslmode", "verify-full")
			u.RawQuery = q.Encode()
			t.Setenv("AGEFREIGHTER_HORIZON_TARGET_DSN", u.String())
			conn, err := pgx.Connect(ctx, u.String())
			if err != nil {
				t.Fatal(err)
			}
			defer conn.Close(context.Background())
			if _, err := conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS age"); err != nil {
				t.Fatal(err)
			}
			var pgVersion, ageVersion, preload string
			var versionNumber int
			if err := conn.QueryRow(ctx, `SELECT current_setting('server_version_num')::integer,
				version(), extversion, current_setting('shared_preload_libraries')
				FROM pg_extension WHERE extname='age'`).Scan(&versionNumber, &pgVersion, &ageVersion, &preload); err != nil {
				t.Fatal(err)
			}
			if versionNumber/10000 != major || !strings.HasPrefix(ageVersion, "1.8.") || !strings.Contains(preload, "age") {
				t.Fatalf("unexpected target versions: PostgreSQL=%d AGE=%s preload=%s", versionNumber, ageVersion, preload)
			}
			t.Logf("target: %s; AGE %s; preload %s", pgVersion, ageVersion, preload)
			for _, source := range []string{"csv", "postgresql", "neo4j44", "neo4j526", "cosmos", "gremlin"} {
				t.Run(source, func(t *testing.T) {
					job := tinyJob(exported, explicitDir, source, major)
					job.Metadata.Name += "-" + runSuffix
					job.Target.Graph += "_" + runSuffix
					path := filepath.Join(root, job.Metadata.Name+".json")
					writeTinyJSON(t, path, job)
					if _, err := config.Load(path); err != nil {
						t.Fatal(err)
					}
					result, err := app.Load(ctx, path)
					writeTinyJSON(t, filepath.Join(root, job.Metadata.Name+"-load.json"), result)
					if err != nil {
						t.Fatal(err)
					}
					if result.Status != meta.JobCommitted || result.Metrics.RecordsCommitted != 560 || result.Metrics.BatchesFailed != 0 {
						t.Fatalf("unexpected tiny load: %#v", result)
					}
					if _, err := app.Verify(ctx, path, result.JobID); err != nil {
						t.Fatal(err)
					}
					targetDigest := rangedigest.TinyTargetManifest
					want := expected
					if source == "gremlin" {
						targetDigest = rangedigest.TinyGremlinTargetManifest
						want = expectedGremlin
					}
					actual, err := targetDigest(ctx, u.String(), manifestPath, result.JobID, 100000)
					if err != nil {
						t.Fatal(err)
					}
					compare := rangedigest.Compare
					if source == "gremlin" {
						compare = rangedigest.CompareGremlinTarget
					}
					comparison, err := compare(want, actual)
					writeTinyJSON(t, filepath.Join(root, job.Metadata.Name+"-digest.json"), map[string]any{
						"expected": want, "actual": actual, "comparison": comparison,
						"postgresql": pgVersion, "age": ageVersion, "jobId": result.JobID,
					})
					if err != nil || actual.RecordCount != 560 {
						t.Fatalf("tiny canonical verification failed: %v", err)
					}
					t.Logf("committed %d records; canonical root %s", actual.RecordCount, actual.RootSHA256)
				})
			}
		})
	}
}

func validateTinySources(postgres, neo44, neo526, cosmos string) error {
	pg, err := url.Parse(postgres)
	if err != nil || pg.Scheme != "postgresql" || pg.Host != "127.0.0.1:5432" || pg.Path != "/source" {
		return fmt.Errorf("dedicated local tiny PostgreSQL source required")
	}
	if neo44 != "bolt://127.0.0.1:7687" || neo526 != "bolt://127.0.0.1:7688" {
		return fmt.Errorf("dedicated local tiny Neo4j sources required")
	}
	c, err := url.Parse(cosmos)
	if err != nil || c.Scheme != "https" || !strings.HasPrefix(c.Hostname(), "afht") ||
		!strings.HasSuffix(c.Hostname(), ".documents.azure.com") {
		return fmt.Errorf("run-scoped tiny Cosmos source required")
	}
	return nil
}

func TestTinySourcesRejectUnrelatedEndpoints(t *testing.T) {
	pg := "postgresql://postgres:fixture@127.0.0.1:5432/source?sslmode=disable"
	n44, n526 := "bolt://127.0.0.1:7687", "bolt://127.0.0.1:7688"
	cosmos := "https://afhttiny.documents.azure.com:443/"
	if err := validateTinySources(pg, n44, n526, cosmos); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][4]string{
		{"postgresql://database.example/source", n44, n526, cosmos},
		{pg, "bolt://database.example:7687", n526, cosmos},
		{pg, n44, n526, "https://unrelated.documents.azure.com"},
	} {
		if err := validateTinySources(args[0], args[1], args[2], args[3]); err == nil {
			t.Fatal("accepted an unrelated source endpoint")
		}
	}
}

func TestHorizonTinySourceConfigurations(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "fixture")
	if _, err := fixture.Generate(t.Context(), fixture.GenerateConfig{
		Phase: fixture.PhaseTiny, Output: path, Shards: 1, Workers: 1, Seed: 20260829,
	}); err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(dir, "portable")
	exported, err := portable.Export(t.Context(), filepath.Join(path, "manifest.json"), output)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("AGEFREIGHTER_NEO4J44_URI", "bolt://127.0.0.1:7687")
	t.Setenv("AGEFREIGHTER_NEO4J526_URI", "bolt://127.0.0.1:7688")
	t.Setenv("AGEFREIGHTER_COSMOS_TEST_ENDPOINT", "https://tiny.documents.azure.com")
	for _, source := range []string{"csv", "postgresql", "neo4j44", "neo4j526", "cosmos", "gremlin"} {
		t.Run(source, func(t *testing.T) {
			path := filepath.Join(dir, source+".json")
			writeTinyJSON(t, path, tinyJob(exported, output, source, 17))
			if _, err := config.Load(path); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func tinyJob(manifest portable.Manifest, dir, source string, major int) config.LoadJob {
	job := config.LoadJob{
		APIVersion: config.APIVersion, Kind: config.KindLoadJob,
		Metadata: config.Metadata{Name: fmt.Sprintf("horizon-tiny-pg%d-%s", major, source)},
		Source:   config.Source{Type: config.SourceCSV, Namespace: "production-simulation"},
		Target: config.Target{Type: config.TargetApacheAGE, Graph: fmt.Sprintf("ht_pg%d_%s", major, source),
			Mode: config.LoadCreate, PropertyMode: config.PropertiesReplace,
			Connection: config.SecretRef{Env: "AGEFREIGHTER_HORIZON_TARGET_DSN"}},
		Runtime: config.Runtime{MemoryLimit: 1 << 30, BatchRows: 32, BatchBytes: 1 << 20,
			MaxSourceConcurrency: 1, MaxTransformConcurrency: 1, MaxTargetConnections: 2,
			OperationTimeout: config.Duration(5 * time.Minute)},
		Errors: config.ErrorPolicies{MalformedRecord: config.MalformedFail, MissingEndpoint: config.MissingEndpointError},
	}
	switch source {
	case "csv":
		csv := manifest.CSVSource()
		for i := range csv.Vertices {
			csv.Vertices[i].Path = filepath.Join(dir, csv.Vertices[i].Path)
		}
		for i := range csv.Edges {
			csv.Edges[i].Path = filepath.Join(dir, csv.Edges[i].Path)
		}
		job.Source.CSV = &csv
	case "postgresql":
		job.Source.Type = config.SourcePostgreSQL
		job.Source.PostgreSQL = &config.PostgreSQLSource{
			Connection: config.SecretRef{Env: "AGEFREIGHTER_POSTGRES_TEST_DSN"},
			ReadMode:   config.PostgreSQLReadCopy, FetchRows: 17,
		}
	case "neo4j44", "neo4j526":
		env := "AGEFREIGHTER_NEO4J44_URI"
		if source == "neo4j526" {
			env = "AGEFREIGHTER_NEO4J526_URI"
		}
		job.Source.Type = config.SourceNeo4j
		job.Source.Neo4j = &config.Neo4jSource{
			URI: os.Getenv(env), Database: "neo4j", SourceID: "horizon-tiny-" + source,
			Username: "neo4j", Password: &config.SecretRef{Env: "AGEFREIGHTER_NEO4J_PASSWORD"},
			FetchRows: 17, MultiLabelPolicy: config.Neo4jMultiLabelConfigured,
			Discovery: &config.Neo4jDiscovery{Enabled: true, VertexKeyProperty: "source_key",
				VertexIDProperty: "external_id", EdgeKeyProperty: "source_key", EdgeIDProperty: "relationship_id",
				MaxLabels: 64, MaxProperties: 64},
		}
	case "cosmos", "gremlin":
		job.Source.Type = config.SourceCosmos
		job.Source.Cosmos = &config.CosmosSource{Endpoint: os.Getenv("AGEFREIGHTER_COSMOS_TEST_ENDPOINT"),
			Credential: "default-azure", Database: "agefreighter", PageSize: 17}
		if source == "gremlin" {
			job.Source.Cosmos.Gremlin = &config.CosmosGremlin{Enabled: true, Container: "tiny-gremlin",
				PartitionKeyProperty: "partitionKey", MaxLabels: 64, MaxProperties: 64, MaxDiscoveryDocuments: 560,
				PropertyTypes: map[string]string{"source_key": "int64", "quantity": "int64",
					"score": "float64", "distance_km": "float64", "active": "boolean", "tags": "string[]", "quantities": "int64[]"}}
		}
	}
	for _, table := range manifest.Tables {
		props, pointers := map[string]string{}, map[string]string{}
		for _, name := range table.Columns {
			if name != "start_id" && name != "end_id" {
				props[name], pointers[name] = name, "/"+name
			}
		}
		if source == "postgresql" {
			query := fmt.Sprintf("SELECT * FROM %s ORDER BY source_key", pgx.Identifier{"tiny", table.Name}.Sanitize())
			if table.Kind == "node" {
				job.Source.PostgreSQL.Vertices = append(job.Source.PostgreSQL.Vertices,
					config.VertexQuery{Label: table.Name, Query: query, IDField: "external_id", Properties: props})
			} else {
				job.Source.PostgreSQL.Edges = append(job.Source.PostgreSQL.Edges,
					config.EdgeQuery{Label: table.Name, Query: query, ExternalIDField: "relationship_id",
						Start: config.EndpointMapping{Label: table.StartLabel, Field: "start_id"},
						End:   config.EndpointMapping{Label: table.EndLabel, Field: "end_id"}, Properties: props})
			}
		} else if source == "cosmos" {
			query := fmt.Sprintf("SELECT * FROM c WHERE c.label = '%s'", table.Name)
			if table.Kind == "node" {
				job.Source.Cosmos.Vertices = append(job.Source.Cosmos.Vertices,
					config.CosmosVertexQuery{Container: "tiny-explicit", Label: table.Name, Query: query,
						IDField: "/external_id", Properties: pointers, PropertyTypes: table.Types})
			} else {
				job.Source.Cosmos.Edges = append(job.Source.Cosmos.Edges,
					config.CosmosEdgeQuery{Container: "tiny-explicit", Label: table.Name, Query: query,
						ExternalIDField: "/relationship_id", Start: config.EndpointMapping{Label: table.StartLabel, Field: "/start_id"},
						End: config.EndpointMapping{Label: table.EndLabel, Field: "/end_id"}, Properties: pointers, PropertyTypes: table.Types})
			}
		}
	}
	return job
}

func tinyDocuments(t *testing.T, dir string, table portable.Table) []map[string]any {
	t.Helper()
	f, err := os.Open(filepath.Join(dir, table.Documents))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	decoder := json.NewDecoder(f)
	decoder.UseNumber()
	var docs []map[string]any
	for {
		var doc map[string]any
		err := decoder.Decode(&doc)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		docs = append(docs, doc)
	}
	if int64(len(docs)) != table.Rows || len(docs) > 400 {
		t.Fatal("unexpected tiny fixture table size")
	}
	return docs
}

func tinyNativeValue(t *testing.T, value any) any {
	t.Helper()
	switch v := value.(type) {
	case json.Number:
		if strings.ContainsAny(string(v), ".eE") {
			n, err := v.Float64()
			if err != nil {
				t.Fatal(err)
			}
			return n
		}
		n, err := strconv.ParseInt(string(v), 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		return n
	case []any:
		for i := range v {
			v[i] = tinyNativeValue(t, v[i])
		}
	}
	return value
}

func seedTinyPostgres(t *testing.T, ctx context.Context, manifest portable.Manifest, dir string) {
	t.Helper()
	conn, err := pgx.Connect(ctx, os.Getenv("AGEFREIGHTER_POSTGRES_TEST_DSN"))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close(context.Background())
	var version int
	if err := conn.QueryRow(ctx, "SELECT current_setting('server_version_num')::integer").Scan(&version); err != nil || version/10000 != 17 {
		t.Fatalf("expected PostgreSQL 17 source: %d, %v", version, err)
	}
	if _, err := conn.Exec(ctx, "CREATE SCHEMA tiny"); err != nil {
		t.Fatal(err)
	}
	for _, table := range manifest.Tables {
		var columns []string
		for _, name := range table.Columns {
			sqlType := "text"
			switch table.Types[name] {
			case "int64":
				sqlType = "bigint"
			case "float64":
				sqlType = "double precision"
			case "boolean":
				sqlType = "boolean"
			case "string[]", "int64[]":
				sqlType = "jsonb"
			}
			columns = append(columns, pgx.Identifier{name}.Sanitize()+" "+sqlType)
		}
		id := pgx.Identifier{"tiny", table.Name}
		if _, err := conn.Exec(ctx, "CREATE TABLE "+id.Sanitize()+" ("+strings.Join(columns, ",")+")"); err != nil {
			t.Fatal(err)
		}
		var rows [][]any
		for _, doc := range tinyDocuments(t, dir, table) {
			row := make([]any, len(table.Columns))
			for i, name := range table.Columns {
				row[i] = tinyNativeValue(t, doc[name])
			}
			rows = append(rows, row)
		}
		if copied, err := conn.CopyFrom(ctx, id, table.Columns, pgx.CopyFromRows(rows)); err != nil || copied != table.Rows {
			t.Fatalf("seed PostgreSQL table %s: %d, %v", table.Name, copied, err)
		}
	}
}

func seedTinyNeo4j(t *testing.T, ctx context.Context, manifest portable.Manifest, dir, env string) {
	t.Helper()
	driver, err := neo4j.NewDriver(os.Getenv(env), neo4j.BasicAuth("neo4j", os.Getenv("AGEFREIGHTER_NEO4J_PASSWORD"), ""))
	if err != nil {
		t.Fatal(err)
	}
	defer driver.Close(context.Background())
	session := driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(context.Background())
	result, err := session.Run(ctx, "MATCH (n) RETURN count(n) AS count", nil)
	if err != nil {
		t.Fatal(err)
	}
	row, err := result.Single(ctx)
	if err != nil || row.Values[0] != int64(0) {
		t.Fatal("Neo4j fixture database must be empty")
	}
	result, err = session.Run(ctx, "CALL dbms.components() YIELD versions RETURN versions[0]", nil)
	if err != nil {
		t.Fatal(err)
	}
	row, err = result.Single(ctx)
	want := "4.4.48"
	if env == "AGEFREIGHTER_NEO4J526_URI" {
		want = "5.26.30"
	}
	if err != nil || row.Values[0] != want {
		t.Fatalf("expected pinned Neo4j %s source", want)
	}
	for _, table := range manifest.Tables {
		var rows []map[string]any
		for _, doc := range tinyDocuments(t, dir, table) {
			props := map[string]any{}
			for _, name := range table.Columns {
				if name != "start_id" && name != "end_id" {
					props[name] = tinyNativeValue(t, doc[name])
				}
			}
			rows = append(rows, map[string]any{"props": props, "start": doc["start_id"], "end": doc["end_id"]})
		}
		query := fmt.Sprintf("UNWIND $rows AS row CREATE (n:`%s`) SET n=row.props", table.Name)
		if table.Kind == "edge" {
			query = fmt.Sprintf("UNWIND $rows AS row MATCH (s:`%s` {external_id:row.start}) MATCH (e:`%s` {external_id:row.end}) CREATE (s)-[r:`%s`]->(e) SET r=row.props", table.StartLabel, table.EndLabel, table.Name)
		}
		result, err := session.Run(ctx, query, map[string]any{"rows": rows})
		if err != nil {
			t.Fatal(err)
		}
		summary, err := result.Consume(ctx)
		if err != nil {
			t.Fatal(err)
		}
		created := summary.Counters().NodesCreated()
		if table.Kind == "edge" {
			created = summary.Counters().RelationshipsCreated()
		}
		if int64(created) != table.Rows {
			t.Fatal("Neo4j fixture count mismatch")
		}
	}
}

func writeTinyJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(append(data, '\n')); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}
