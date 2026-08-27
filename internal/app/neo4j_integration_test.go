package app

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/neo4j/neo4j-go-driver/v6/neo4j"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"go.yaml.in/yaml/v3"
)

func TestNeo4jCreateIntegration(t *testing.T) {
	uri := os.Getenv("AGEFREIGHTER_NEO4J_TEST_URI")
	username := os.Getenv("AGEFREIGHTER_NEO4J_TEST_USERNAME")
	password := os.Getenv("AGEFREIGHTER_NEO4J_TEST_PASSWORD")
	database := os.Getenv("AGEFREIGHTER_NEO4J_TEST_DATABASE")
	targetDSN := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if uri == "" || database == "" || targetDSN == "" {
		t.Skip("set Neo4j source and AGE target test settings")
	}
	ctx := t.Context()
	auth := neo4j.NoAuth()
	if username != "" {
		auth = neo4j.BasicAuth(username, password, "")
	}
	driver, err := neo4j.NewDriver(uri, auth)
	if err != nil {
		t.Fatalf("create Neo4j driver: %v", err)
	}
	t.Cleanup(func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = driver.Close(closeCtx)
	})
	if err := driver.VerifyConnectivity(ctx); err != nil {
		t.Fatalf("verify Neo4j connectivity: %v", err)
	}

	suffix := time.Now().UnixNano()
	label := fmt.Sprintf("AgefreighterE2E_%d", suffix)
	relationshipType := fmt.Sprintf("AGEFREIGHTER_KNOWS_%d", suffix)
	runNeo4jStatement(t, ctx, driver, database, fmt.Sprintf(`
		CREATE (ada:%s:Scientist {
			source_key: 1, person_id: 'p1', name: 'Ada',
			born: date('1815-12-10'), location: point({longitude: -0.12, latitude: 51.5})
		})
		CREATE (grace:%s {
			source_key: 2, person_id: 'p2', name: 'Grace',
			born: date('1906-12-09'), location: point({longitude: -74.0, latitude: 40.7})
		})
		CREATE (ada)-[:%s {
			source_key: 1, relationship_id: 'e1', weight: 7
		}]->(grace)
	`, label, label, relationshipType))
	t.Cleanup(func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		session := driver.NewSession(cleanupCtx, neo4j.SessionConfig{
			AccessMode: neo4j.AccessModeWrite, DatabaseName: database,
		})
		defer session.Close(cleanupCtx)
		result, cleanupErr := session.Run(
			cleanupCtx,
			fmt.Sprintf("MATCH (n:%s) DETACH DELETE n", label),
			nil,
		)
		if cleanupErr == nil {
			_, _ = result.Consume(cleanupCtx)
		}
	})

	graph := fmt.Sprintf("neo4j_e2e_%d", suffix)
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", targetDSN)
	t.Setenv("AGEFREIGHTER_NEO4J_APP_TEST_PASSWORD", password)
	job := testLoadJob(graph, "unused-vertices", "unused-edges")
	job.Metadata.Name = "neo4j-create"
	job.Source = config.Source{
		Type: config.SourceNeo4j, Namespace: "crm",
		Neo4j: &config.Neo4jSource{
			URI: uri, Database: database,
			SourceID: fmt.Sprintf("e2e-%d", suffix),
			Username: username,
			Password: &config.SecretRef{
				Env: "AGEFREIGHTER_NEO4J_APP_TEST_PASSWORD",
			},
			FetchRows: 1, MultiLabelPolicy: config.Neo4jMultiLabelConfigured,
			Vertices: []config.VertexQuery{{
				Label: "Person",
				Query: fmt.Sprintf(
					"MATCH (n:%s) WHERE $afterKey IS NULL OR n.source_key > $afterKey "+
						"RETURN n.source_key AS source_key, n.person_id AS person_id, "+
						"n.name AS name, n.born AS born, n.location AS location ORDER BY source_key",
					label,
				),
				KeyField: "source_key", IDField: "person_id",
				Properties: map[string]string{
					"name": "name", "born": "born", "location": "location",
				},
			}},
			Edges: []config.EdgeQuery{{
				Label: "KNOWS",
				Query: fmt.Sprintf(
					"MATCH (a:%s)-[r:%s]->(b:%s) "+
						"WHERE $afterKey IS NULL OR r.source_key > $afterKey "+
						"RETURN r.source_key AS source_key, "+
						"r.relationship_id AS relationship_id, "+
						"a.person_id AS from_id, b.person_id AS to_id, "+
						"r.weight AS weight ORDER BY source_key",
					label,
					relationshipType,
					label,
				),
				KeyField: "source_key", ExternalIDField: "relationship_id",
				Start:      config.EndpointMapping{Label: "Person", Field: "from_id"},
				End:        config.EndpointMapping{Label: "Person", Field: "to_id"},
				Properties: map[string]string{"weight": "weight"},
			}},
		},
	}
	if username == "" {
		job.Source.Neo4j.Password = nil
	}
	data, err := yaml.Marshal(job)
	if err != nil {
		t.Fatalf("marshal Neo4j job: %v", err)
	}
	path := filepath.Join(t.TempDir(), "neo4j.yaml")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write Neo4j job: %v", err)
	}

	result, err := Load(ctx, path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	registerCleanup(t, targetDSN, graph, result.JobID)
	if result.Status != meta.JobCommitted ||
		result.Metrics.RecordsCommitted != 3 {
		t.Fatalf("Load() = %#v", result)
	}
	targetPool, err := pgxpool.New(ctx, targetDSN)
	if err != nil {
		t.Fatalf("open AGE target: %v", err)
	}
	defer targetPool.Close()
	connection, err := targetPool.Acquire(ctx)
	if err != nil {
		t.Fatalf("acquire AGE target: %v", err)
	}
	defer connection.Release()
	if _, err := connection.Exec(ctx, "LOAD 'age'"); err != nil {
		t.Fatalf("load AGE extension: %v", err)
	}
	if _, err := connection.Exec(
		ctx,
		`SET search_path = ag_catalog, "$user", public`,
	); err != nil {
		t.Fatalf("set AGE search path: %v", err)
	}
	assertCypherCount(t, connection.Conn(), graph, "MATCH (n:Person) RETURN count(n)", 2)
	assertCypherCount(
		t,
		connection.Conn(),
		graph,
		"MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r)",
		1,
	)

	discoveryGraph := fmt.Sprintf("neo4j_discovery_e2e_%d", suffix)
	discoveryJob := job
	discoveryJob.Metadata.Name = "neo4j-discovery-create"
	discoveryJob.Target.Graph = discoveryGraph
	discoveryJob.Source.Neo4j = &config.Neo4jSource{
		URI: uri, Database: database,
		SourceID: fmt.Sprintf("discovery-e2e-%d", suffix),
		Username: username,
		Password: &config.SecretRef{
			Env: "AGEFREIGHTER_NEO4J_APP_TEST_PASSWORD",
		},
		FetchRows:        1,
		MultiLabelPolicy: config.Neo4jMultiLabelConfigured,
		Discovery: &config.Neo4jDiscovery{
			Enabled:                true,
			LabelPrefix:            label,
			RelationshipTypePrefix: relationshipType,
			VertexKeyProperty:      "source_key",
			VertexIDProperty:       "person_id",
			EdgeKeyProperty:        "source_key",
			EdgeIDProperty:         "relationship_id",
			MaxLabels:              10,
			MaxProperties:          20,
		},
	}
	if username == "" {
		discoveryJob.Source.Neo4j.Password = nil
	}
	discoveryData, err := yaml.Marshal(discoveryJob)
	if err != nil {
		t.Fatalf("marshal Neo4j discovery job: %v", err)
	}
	discoveryPath := filepath.Join(t.TempDir(), "neo4j-discovery.yaml")
	if err := os.WriteFile(discoveryPath, discoveryData, 0o600); err != nil {
		t.Fatalf("write Neo4j discovery job: %v", err)
	}
	discoveryResult, err := Load(ctx, discoveryPath)
	if err != nil {
		t.Fatalf("Load(discovery) error = %v", err)
	}
	registerCleanup(t, targetDSN, discoveryGraph, discoveryResult.JobID)
	if discoveryResult.Status != meta.JobCommitted ||
		discoveryResult.Metrics.RecordsCommitted != 3 {
		t.Fatalf("Load(discovery) = %#v", discoveryResult)
	}
	if _, err := Verify(ctx, discoveryPath, discoveryResult.JobID); err != nil {
		t.Fatalf("Verify(discovery) error = %v", err)
	}
	assertCypherCount(
		t,
		connection.Conn(),
		discoveryGraph,
		fmt.Sprintf("MATCH (n:%s) RETURN count(n)", label),
		2,
	)
	assertCypherCount(
		t,
		connection.Conn(),
		discoveryGraph,
		fmt.Sprintf(
			"MATCH (:%s)-[r:%s]->(:%s) RETURN count(r)",
			label,
			relationshipType,
			label,
		),
		1,
	)
}

func TestNeo4jSourceModeMatrixIntegration(t *testing.T) {
	uri := os.Getenv("AGEFREIGHTER_NEO4J_TEST_URI")
	username := os.Getenv("AGEFREIGHTER_NEO4J_TEST_USERNAME")
	password := os.Getenv("AGEFREIGHTER_NEO4J_TEST_PASSWORD")
	database := os.Getenv("AGEFREIGHTER_NEO4J_TEST_DATABASE")
	targetDSN := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if uri == "" || database == "" || targetDSN == "" {
		t.Skip("set Neo4j source and AGE target test settings")
	}
	ctx := t.Context()
	auth := neo4j.NoAuth()
	if username != "" {
		auth = neo4j.BasicAuth(username, password, "")
	}
	driver, err := neo4j.NewDriver(uri, auth)
	if err != nil {
		t.Fatalf("create Neo4j driver: %v", err)
	}
	t.Cleanup(func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = driver.Close(closeCtx)
	})
	if err := driver.VerifyConnectivity(ctx); err != nil {
		t.Fatalf("verify Neo4j connectivity: %v", err)
	}

	suffix := time.Now().UnixNano()
	label := fmt.Sprintf("AgefreighterMatrix_%d", suffix)
	relationshipType := fmt.Sprintf("AGEFREIGHTER_MATRIX_KNOWS_%d", suffix)
	runNeo4jStatement(t, ctx, driver, database, fmt.Sprintf(`
		CREATE (c1:%s {
			dataset: 'create', source_key: 1, person_id: 'p1',
			name: 'Ada Create', active: true, score: 1.5,
			tags: ['math'], profile: 'London'
		})
		CREATE (c2:%s {
			dataset: 'create', source_key: 2, person_id: 'p2',
			name: 'Grace Create', active: false, score: 2.0,
			tags: ['compiler'], profile: 'New York'
		})
		CREATE (c1)-[:%s {
			dataset: 'create', source_key: 1,
			relationship_id: 'e1', weight: 1
		}]->(c2)
		CREATE (r1:%s {
			dataset: 'replace', source_key: 1, person_id: 'p1',
			name: 'Ada Replace', active: true, score: 2.5,
			tags: ['math'], profile: 'London'
		})
		CREATE (r2:%s {
			dataset: 'replace', source_key: 2, person_id: 'p2',
			name: 'Grace Replace', active: false, score: 3.0,
			tags: ['compiler'], profile: 'New York'
		})
		CREATE (r1)-[:%s {
			dataset: 'replace', source_key: 1,
			relationship_id: 'e1', weight: 2
		}]->(r2)
		CREATE (a3:%s {
			dataset: 'append', source_key: 1, person_id: 'p3',
			name: 'Katherine Append', active: true, score: 4.5,
			tags: ['space'], profile: 'Washington'
		})
		CREATE (a2:%s {
			dataset: 'append-endpoint', source_key: 2, person_id: 'p2'
		})
		CREATE (a2)-[:%s {
			dataset: 'append', source_key: 1,
			relationship_id: 'e2', weight: 3
		}]->(a3)
		CREATE (u1:%s {
			dataset: 'upsert', source_key: 1, person_id: 'p1',
			name: 'Ada Upsert', active: true, score: 9.5,
			tags: ['math'], profile: 'London'
		})
		CREATE (u4:%s {
			dataset: 'upsert', source_key: 2, person_id: 'p4',
			name: 'Dorothy Upsert', active: true, score: 5.5,
			tags: ['navy'], profile: 'Arlington'
		})
		CREATE (u3:%s {
			dataset: 'upsert-endpoint', source_key: 3, person_id: 'p3'
		})
		CREATE (u1)-[:%s {
			dataset: 'upsert', source_key: 1,
			relationship_id: 'e1', weight: 9
		}]->(u4)
		CREATE (u3)-[:%s {
			dataset: 'upsert', source_key: 2,
			relationship_id: 'e3', weight: 4
		}]->(u4)
	`,
		label, label, relationshipType,
		label, label, relationshipType,
		label, label, relationshipType,
		label, label, label, relationshipType, relationshipType,
	))
	t.Cleanup(func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		session := driver.NewSession(cleanupCtx, neo4j.SessionConfig{
			AccessMode: neo4j.AccessModeWrite, DatabaseName: database,
		})
		defer session.Close(cleanupCtx)
		result, cleanupErr := session.Run(
			cleanupCtx,
			fmt.Sprintf("MATCH (n:%s) DETACH DELETE n", label),
			nil,
		)
		if cleanupErr == nil {
			_, _ = result.Consume(cleanupCtx)
		}
	})

	graph := fmt.Sprintf("neo4j_mode_matrix_%d", suffix)
	t.Setenv("AGEFREIGHTER_APP_TEST_DSN", targetDSN)
	t.Setenv("AGEFREIGHTER_NEO4J_APP_TEST_PASSWORD", password)
	runSourceModeMatrix(
		t,
		targetDSN,
		t.TempDir(),
		graph,
		"neo4j",
		func(mode config.LoadMode, dataset string) config.LoadJob {
			job := testLoadJob(graph, "unused-vertices", "unused-edges")
			job.Metadata.Name = "neo4j-matrix-" + dataset
			job.Target.Mode = mode
			job.Source = config.Source{
				Type: config.SourceNeo4j, Namespace: "crm",
				Neo4j: &config.Neo4jSource{
					URI: uri, Database: database,
					SourceID: fmt.Sprintf("matrix-e2e-%d", suffix),
					Username: username,
					Password: &config.SecretRef{
						Env: "AGEFREIGHTER_NEO4J_APP_TEST_PASSWORD",
					},
					FetchRows: 1, MultiLabelPolicy: config.Neo4jMultiLabelConfigured,
					Vertices: []config.VertexQuery{{
						Label: "Person",
						Query: fmt.Sprintf(
							"MATCH (n:%s) WHERE n.dataset = '%s' "+
								"AND ($afterKey IS NULL OR n.source_key > $afterKey) "+
								"RETURN n.source_key AS source_key, "+
								"n.person_id AS person_id, n.name AS name, "+
								"n.active AS active, n.score AS score, "+
								"n.tags AS tags, n.profile AS profile "+
								"ORDER BY source_key",
							label,
							dataset,
						),
						KeyField: "source_key", IDField: "person_id",
						Properties: map[string]string{
							"name": "name", "active": "active", "score": "score",
							"tags": "tags", "profile": "profile",
						},
					}},
					Edges: []config.EdgeQuery{{
						Label: "KNOWS",
						Query: fmt.Sprintf(
							"MATCH (a:%s)-[r:%s]->(b:%s) "+
								"WHERE r.dataset = '%s' "+
								"AND ($afterKey IS NULL OR r.source_key > $afterKey) "+
								"RETURN r.source_key AS source_key, "+
								"r.relationship_id AS relationship_id, "+
								"a.person_id AS from_id, b.person_id AS to_id, "+
								"r.weight AS weight ORDER BY source_key",
							label,
							relationshipType,
							label,
							dataset,
						),
						KeyField: "source_key", ExternalIDField: "relationship_id",
						Start: config.EndpointMapping{
							Label: "Person", Field: "from_id",
						},
						End: config.EndpointMapping{
							Label: "Person", Field: "to_id",
						},
						Properties: map[string]string{"weight": "weight"},
					}},
				},
			}
			if username == "" {
				job.Source.Neo4j.Password = nil
			}
			return job
		},
		sourceModeTypedExpectations{
			predicates: []string{
				"n.active = true",
				"n.score = 9.5",
				"n.tags[0] = 'math'",
				"n.profile = 'London'",
			},
		},
	)
}

func runNeo4jStatement(
	t *testing.T,
	ctx context.Context,
	driver neo4j.Driver,
	database string,
	query string,
) {
	t.Helper()
	session := driver.NewSession(ctx, neo4j.SessionConfig{
		AccessMode: neo4j.AccessModeWrite, DatabaseName: database,
	})
	defer session.Close(ctx)
	result, err := session.Run(ctx, query, nil)
	if err != nil {
		t.Fatalf("run Neo4j fixture statement: %v", err)
	}
	if _, err := result.Consume(ctx); err != nil {
		t.Fatalf("consume Neo4j fixture statement: %v", err)
	}
}
