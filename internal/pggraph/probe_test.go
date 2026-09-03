package pggraph

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/checkpoint"
	"github.com/rioriost/agefreighter/internal/meta"
	sinkcontract "github.com/rioriost/agefreighter/internal/sink"
	"github.com/rioriost/agefreighter/pkg/model"
)

const integrationDSNEnvironment = "AGEFREIGHTER_PGGRAPH_TEST_DSN"

func TestValidateServerVersion(t *testing.T) {
	if err := ValidateServerVersion(190000); err != nil {
		t.Fatalf("ValidateServerVersion(190000) error = %v", err)
	}
	if err := ValidateServerVersion(180006); err == nil ||
		!strings.Contains(err.Error(), ">= 190000") {
		t.Fatalf("ValidateServerVersion(180006) error = %v", err)
	}
}

func TestOpenValidation(t *testing.T) {
	valid := PoolOptions{MinConnections: 0, MaxConnections: 1,
		ConnectTimeout: time.Second, OperationTimeout: time.Second}
	for _, test := range []struct {
		name    string
		dsn     string
		options PoolOptions
	}{
		{"missing DSN", "", valid},
		{"negative minimum", "postgres://invalid", PoolOptions{
			MinConnections: -1, MaxConnections: 1,
			ConnectTimeout: time.Second, OperationTimeout: time.Second,
		}},
		{"zero maximum", "postgres://invalid", PoolOptions{
			MaxConnections: 0, ConnectTimeout: time.Second, OperationTimeout: time.Second,
		}},
		{"reversed limits", "postgres://invalid", PoolOptions{
			MinConnections: 2, MaxConnections: 1,
			ConnectTimeout: time.Second, OperationTimeout: time.Second,
		}},
		{"zero connect timeout", "postgres://invalid", PoolOptions{
			MaxConnections: 1, OperationTimeout: time.Second,
		}},
		{"zero operation timeout", "postgres://invalid", PoolOptions{
			MaxConnections: 1, ConnectTimeout: time.Second,
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if adapter, err := Open(t.Context(), test.dsn, test.options); err == nil {
				adapter.Close()
				t.Fatal("Open() accepted invalid configuration")
			}
		})
	}
	if _, err := (&Adapter{}).Prepare(t.Context(), "bad", Definition{}); err == nil {
		t.Fatal("Prepare() accepted an invalid definition")
	}
	if _, err := definitionMapping("job", Definition{
		Vertices: []VertexDefinition{{Table: "person", Label: "Person"}},
		Edges: []EdgeDefinition{{
			Table: "knows", Label: "KNOWS",
			SourceTable: "missing", DestinationTable: "person",
		}},
	}, strings.Repeat("a", 64)); err == nil {
		t.Fatal("definitionMapping() accepted an unknown endpoint table")
	}
}

func TestPropertyGraphSinkReplayAndAbortIntegration(t *testing.T) {
	dsn := strings.TrimSpace(os.Getenv(integrationDSNEnvironment))
	if dsn == "" {
		t.Skip("set " + integrationDSNEnvironment + " to run PostgreSQL property graph integration tests")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	adapter, err := Open(ctx, dsn, PoolOptions{
		MinConnections: 1, MaxConnections: 2,
		ConnectTimeout: 5 * time.Second, OperationTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()
	if err := adapter.Metadata().Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	jobID := fmt.Sprintf("%08x-0000-4000-8000-%012x", time.Now().Unix(), time.Now().UnixNano()&0xffffffffffff)
	schema := fmt.Sprintf("af_pgq_sink_%d", time.Now().UnixNano())
	definition := Definition{Schema: schema, Graph: schema,
		Vertices: []VertexDefinition{{Table: "person", Label: "Person"}},
		Edges: []EdgeDefinition{{
			Table: "knows", Label: "KNOWS",
			SourceTable: "person", DestinationTable: "person",
		}}}
	job := meta.Job{
		ID: jobID, Name: "sink-state", SourceType: "csv", LoadMode: "create",
		TargetBackend: meta.TargetBackendPostgreSQLPropertyGraph,
		TargetSchema:  schema, TargetGraph: schema,
		ConfigFingerprint: strings.Repeat("a", 64),
	}
	if err := adapter.Metadata().CreateRunningJob(ctx, job); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		connection, connectErr := pgx.Connect(cleanupCtx, dsn)
		if connectErr != nil {
			return
		}
		defer connection.Close(context.Background())
		_, _ = connection.Exec(cleanupCtx, "DROP SCHEMA IF EXISTS "+QuoteIdentifier(schema)+" CASCADE")
		_, _ = connection.Exec(cleanupCtx,
			`DELETE FROM agefreighter_meta.load_job WHERE job_id = $1::uuid`, jobID)
	})
	if _, err := adapter.Prepare(ctx, jobID, definition); err != nil {
		t.Fatal(err)
	}
	sink, err := NewLoadSink(adapter, LoadSinkOptions{JobID: jobID, Definition: definition})
	if err != nil {
		t.Fatal(err)
	}
	first := model.SourcePosition{Resource: "vertices", Line: 1, Token: "1"}
	last := model.SourcePosition{Resource: "vertices", Line: 1, Token: "2"}
	batch := sinkcontract.BatchMetadata{
		ID: 1, Attempt: 1, Rows: 1, Bytes: 10,
		FirstPosition: first, LastPosition: last,
	}
	records := []model.Record{{Vertex: &model.Vertex{
		Label: "Person", Namespace: "crm", ExternalID: "p1",
		EncodedProperties: []byte(`{"name":"Ada"}`),
	}}}
	transaction, err := sink.Begin(ctx, batch)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sink.Begin(ctx, batch); err == nil {
		t.Fatal("second active Begin() succeeded")
	}
	if err := transaction.Write(ctx, records); err != nil {
		t.Fatal(err)
	}
	state := checkpoint.State{BatchID: 1, Attempt: 1,
		Phase: checkpoint.PhaseCommitted, Position: last}
	if err := transaction.Commit(ctx, state); err != nil {
		t.Fatal(err)
	}

	replay, err := sink.Begin(ctx, batch)
	if err != nil {
		t.Fatal(err)
	}
	if err := replay.Write(ctx, records); err != nil {
		t.Fatal(err)
	}
	if err := replay.Commit(ctx, state); err != nil {
		t.Fatal(err)
	}
	replay, err = sink.Begin(ctx, batch)
	if err != nil {
		t.Fatal(err)
	}
	if err := replay.Write(ctx, records); err != nil {
		t.Fatal(err)
	}
	changed := state
	changed.Position.Token = "changed"
	if err := replay.Commit(ctx, changed); err == nil {
		t.Fatal("changed replay checkpoint succeeded")
	}

	abortBatch := batch
	abortBatch.ID = 2
	abortBatch.Attempt = 1
	abortBatch.FirstPosition.Token = "3"
	abortBatch.LastPosition.Token = "4"
	abortTransaction, err := sink.Begin(ctx, abortBatch)
	if err != nil {
		t.Fatal(err)
	}
	if err := abortTransaction.Commit(ctx, checkpoint.State{}); err == nil {
		t.Fatal("Commit() before Write() succeeded")
	}
	if err := abortTransaction.Rollback(ctx); err != nil {
		t.Fatal(err)
	}

}

func TestPropertyGraphSinkFailureIntegration(t *testing.T) {
	dsn := strings.TrimSpace(os.Getenv(integrationDSNEnvironment))
	if dsn == "" {
		t.Skip("set " + integrationDSNEnvironment + " to run PostgreSQL property graph integration tests")
	}
	newFixture := func(t *testing.T) (*Adapter, *LoadSink, Definition, string) {
		t.Helper()
		ctx := t.Context()
		adapter, err := Open(ctx, dsn, PoolOptions{
			MinConnections: 1, MaxConnections: 2,
			ConnectTimeout: 5 * time.Second, OperationTimeout: 5 * time.Second,
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(adapter.Close)
		if err := adapter.Metadata().Migrate(ctx); err != nil {
			t.Fatal(err)
		}
		jobID := fmt.Sprintf("%08x-0000-4000-8000-%012x", time.Now().Unix(), time.Now().UnixNano()&0xffffffffffff)
		schema := fmt.Sprintf("af_pgq_failure_%d", time.Now().UnixNano())
		definition := Definition{Schema: schema, Graph: schema,
			Vertices: []VertexDefinition{{Table: "person", Label: "Person"}},
			Edges: []EdgeDefinition{{
				Table: "knows", Label: "KNOWS",
				SourceTable: "person", DestinationTable: "person",
			}}}
		job := meta.Job{
			ID: jobID, Name: "sink-failure", SourceType: "csv", LoadMode: "create",
			TargetBackend: meta.TargetBackendPostgreSQLPropertyGraph,
			TargetSchema:  schema, TargetGraph: schema, ConfigFingerprint: strings.Repeat("b", 64),
		}
		if err := adapter.Metadata().CreateRunningJob(ctx, job); err != nil {
			t.Fatal(err)
		}
		if _, err := adapter.Prepare(ctx, jobID, definition); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			connection, err := pgx.Connect(cleanupCtx, dsn)
			if err != nil {
				return
			}
			defer connection.Close(context.Background())
			_, _ = connection.Exec(cleanupCtx, "DROP SCHEMA IF EXISTS "+QuoteIdentifier(schema)+" CASCADE")
			_, _ = connection.Exec(cleanupCtx,
				`DELETE FROM agefreighter_meta.load_job WHERE job_id = $1::uuid`, jobID)
		})
		sink, err := NewLoadSink(adapter, LoadSinkOptions{JobID: jobID, Definition: definition})
		if err != nil {
			t.Fatal(err)
		}
		return adapter, sink, definition, jobID
	}
	batch := func(id uint64, rows int) sinkcontract.BatchMetadata {
		return sinkcontract.BatchMetadata{
			ID: id, Attempt: 1, Rows: rows, Bytes: int64(rows),
			FirstPosition: model.SourcePosition{Resource: "fixture", Token: fmt.Sprintf("%d-first", id)},
			LastPosition:  model.SourcePosition{Resource: "fixture", Token: fmt.Sprintf("%d-last", id)},
		}
	}
	vertex := model.Record{Vertex: &model.Vertex{
		Label: "Person", Namespace: "crm", ExternalID: "p1",
		EncodedProperties: []byte(`{"name":"Ada"}`),
	}}
	validEdge := func() model.Record {
		return model.Record{Edge: &model.Edge{
			Label: "KNOWS", Namespace: "crm", ExternalID: "e1",
			Start:             model.Endpoint{Label: "Person", Namespace: "crm", ExternalID: "p1"},
			End:               model.Endpoint{Label: "Person", Namespace: "crm", ExternalID: "p2"},
			EncodedProperties: []byte(`{"since":2024}`),
		}}
	}
	for _, test := range []struct {
		name   string
		record model.Record
	}{
		{"empty edge identity", func() model.Record { value := validEdge(); value.Edge.Namespace = ""; return value }()},
		{"wrong endpoint label", func() model.Record { value := validEdge(); value.Edge.Start.Label = "Other"; return value }()},
		{"invalid edge properties", func() model.Record { value := validEdge(); value.Edge.EncodedProperties = []byte(`bad`); return value }()},
		{"unresolved endpoints", validEdge()},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, sink, _, _ := newFixture(t)
			tx, err := sink.Begin(t.Context(), batch(1, 1))
			if err != nil {
				t.Fatal(err)
			}
			if err := tx.Write(t.Context(), []model.Record{test.record}); err == nil {
				t.Fatal("invalid edge write succeeded")
			}
			if err := tx.Rollback(t.Context()); err != nil {
				t.Fatal(err)
			}
		})
	}

	t.Run("checkpoint mismatch", func(t *testing.T) {
		_, sink, _, _ := newFixture(t)
		metadata := batch(1, 1)
		tx, err := sink.Begin(t.Context(), metadata)
		if err != nil {
			t.Fatal(err)
		}
		if err := tx.Write(t.Context(), []model.Record{vertex}); err != nil {
			t.Fatal(err)
		}
		if err := tx.Commit(t.Context(), checkpoint.State{
			BatchID: 1, Attempt: 1, Phase: checkpoint.PhaseRunning,
			Position: metadata.LastPosition,
		}); err == nil {
			t.Fatal("mismatched checkpoint succeeded")
		}
	})

	t.Run("closed adapter", func(t *testing.T) {
		adapter, sink, definition, jobID := newFixture(t)
		if _, err := adapter.GraphExists(t.Context(), "", definition.Graph); err == nil {
			t.Fatal("GraphExists() accepted an invalid schema")
		}
		if _, err := adapter.ComputeDigests(t.Context(), "bad", definition); err == nil {
			t.Fatal("ComputeDigests() accepted an invalid job ID")
		}
		if _, err := adapter.ComputeDigests(t.Context(), jobID, Definition{}); err == nil {
			t.Fatal("ComputeDigests() accepted an invalid definition")
		}
		adapter.Close()
		if _, err := adapter.GraphExists(t.Context(), definition.Schema, definition.Graph); err == nil {
			t.Fatal("GraphExists() succeeded after Close()")
		}
		if _, err := adapter.ComputeDigests(t.Context(), jobID, definition); err == nil {
			t.Fatal("ComputeDigests() succeeded after Close()")
		}
		if _, err := adapter.Inspect(t.Context(), jobID, definition); err == nil {
			t.Fatal("Inspect() succeeded after Close()")
		}
		if _, err := adapter.Prepare(t.Context(), jobID, definition); err == nil {
			t.Fatal("Prepare() succeeded after Close()")
		}
		if err := adapter.Finalize(t.Context(), jobID, definition, meta.ConnectorTelemetry{}); err == nil {
			t.Fatal("Finalize() succeeded after Close()")
		}
		if _, err := sink.Begin(t.Context(), batch(1, 1)); err == nil {
			t.Fatal("Begin() succeeded after Close()")
		}
	})

	t.Run("stored definition changed", func(t *testing.T) {
		adapter, _, definition, jobID := newFixture(t)
		changed := definition
		changed.Graph += "_changed"
		if _, err := adapter.Prepare(t.Context(), jobID, changed); !errors.Is(err, meta.ErrGenerationMismatch) {
			t.Fatalf("Prepare(changed definition) error = %v", err)
		}
		if _, err := adapter.Inspect(t.Context(), jobID, changed); !errors.Is(err, meta.ErrGenerationMismatch) {
			t.Fatalf("Inspect(changed definition) error = %v", err)
		}
	})

	t.Run("missing graph object", func(t *testing.T) {
		adapter, _, definition, jobID := newFixture(t)
		if _, err := adapter.pool.Exec(t.Context(),
			"DROP SCHEMA "+QuoteIdentifier(definition.Schema)+" CASCADE"); err != nil {
			t.Fatal(err)
		}
		if _, err := adapter.Prepare(t.Context(), jobID, definition); !errors.Is(err, meta.ErrGenerationMismatch) {
			t.Fatalf("Prepare(missing graph) error = %v", err)
		}
		if _, err := adapter.ComputeDigests(t.Context(), jobID, definition); err == nil {
			t.Fatal("ComputeDigests() accepted a missing vertex table")
		}
	})

	t.Run("renamed graph table", func(t *testing.T) {
		adapter, _, definition, jobID := newFixture(t)
		if _, err := adapter.pool.Exec(t.Context(), "ALTER TABLE "+
			qualifiedName(definition.Schema, definition.Vertices[0].Table)+
			" RENAME TO "+QuoteIdentifier("renamed_person")); err != nil {
			t.Fatal(err)
		}
		if _, err := adapter.Prepare(t.Context(), jobID, definition); !errors.Is(err, meta.ErrGenerationMismatch) {
			t.Fatalf("Prepare(renamed table) error = %v", err)
		}
	})

	t.Run("finalization guards", func(t *testing.T) {
		adapter, _, definition, jobID := newFixture(t)
		if err := adapter.Finalize(t.Context(), jobID, Definition{}, meta.ConnectorTelemetry{}); err == nil {
			t.Fatal("Finalize() accepted an invalid definition")
		}
		missingJobID := "11111111-2222-4333-8444-555555555555"
		if err := adapter.Finalize(t.Context(), missingJobID, definition, meta.ConnectorTelemetry{}); err == nil {
			t.Fatal("Finalize() accepted a missing job")
		}
		if _, err := adapter.pool.Exec(t.Context(), "INSERT INTO "+
			qualifiedName(definition.Schema, definition.Vertices[0].Table)+
			` (source_namespace, external_id, properties, digest_range, source_digest)
			 VALUES ('test', 'extra', '{}'::jsonb, 0, repeat('0', 64))`); err != nil {
			t.Fatal(err)
		}
		if err := adapter.Finalize(t.Context(), jobID, definition, meta.ConnectorTelemetry{}); err == nil ||
			!strings.Contains(err.Error(), "row count") {
			t.Fatalf("Finalize(extra row) error = %v", err)
		}
	})

	t.Run("PostgreSQL numeric canonicalization", func(t *testing.T) {
		adapter, sink, definition, jobID := newFixture(t)
		metadata := batch(1, 1)
		tx, err := sink.Begin(t.Context(), metadata)
		if err != nil {
			t.Fatal(err)
		}
		record := vertex
		record.Vertex = &model.Vertex{
			Label: "Person", Namespace: "crm", ExternalID: "numeric",
			EncodedProperties: []byte(`{"exponent":1e0,"negativeZero":-0.0,"scaled":1.2300e2}`),
		}
		if err := tx.Write(t.Context(), []model.Record{record}); err != nil {
			t.Fatal(err)
		}
		if err := tx.Commit(t.Context(), checkpoint.State{
			BatchID: 1, Attempt: 1, Phase: checkpoint.PhaseCommitted,
			Position: metadata.LastPosition,
		}); err != nil {
			t.Fatal(err)
		}
		if err := adapter.Finalize(t.Context(), jobID, definition, meta.ConnectorTelemetry{
			JobID: jobID, Connector: "csv",
		}); err != nil {
			t.Fatalf("Finalize(numeric properties): %v", err)
		}
		inspection, err := adapter.Inspect(t.Context(), jobID, definition)
		if err != nil {
			t.Fatal(err)
		}
		if err := inspection.Validate(); err != nil || inspection.Rows != 1 {
			t.Fatalf("numeric inspection = %#v, %v", inspection, err)
		}
	})
}

func TestPropertyGraphIntegration(t *testing.T) {
	dsn := strings.TrimSpace(os.Getenv(integrationDSNEnvironment))
	if dsn == "" {
		t.Skip("set " + integrationDSNEnvironment + " to run PostgreSQL property graph integration tests")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	capabilities, err := Probe(ctx, dsn)
	if err != nil {
		t.Fatalf("Probe() error = %v", err)
	}
	if capabilities.ServerVersionNumber < minimumServerVersion ||
		!capabilities.PropertyGraph {
		t.Fatalf("Probe() capabilities = %#v", capabilities)
	}

	connection, err := pgx.Connect(ctx, dsn)
	if err != nil {
		t.Fatalf("pgx.Connect() error = %v", err)
	}
	t.Cleanup(func() { connection.Close(context.Background()) })

	const schema = "af pggraph integration"
	if _, err := connection.Exec(ctx,
		"DROP SCHEMA IF EXISTS "+QuoteIdentifier(schema)+" CASCADE",
	); err != nil {
		t.Fatalf("pre-test cleanup: %v", err)
	}
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		connection.Exec(cleanupCtx,
			"DROP SCHEMA IF EXISTS "+QuoteIdentifier(schema)+" CASCADE")
	})

	definition := Definition{
		Schema: schema,
		Graph:  "Supply Graph",
		Vertices: []VertexDefinition{
			{Table: "person", Label: "Person"},
		},
		Edges: []EdgeDefinition{
			{
				Table:            "knows",
				Label:            "KNOWS",
				SourceTable:      "person",
				DestinationTable: "person",
			},
		},
	}
	statements, err := definition.DDL()
	if err != nil {
		t.Fatalf("DDL() error = %v", err)
	}
	transaction, err := connection.Begin(ctx)
	if err != nil {
		t.Fatalf("Begin() error = %v", err)
	}
	defer transaction.Rollback(context.Background())
	for _, statement := range statements {
		if _, err := transaction.Exec(ctx, statement); err != nil {
			t.Fatalf("execute %q: %v", statement, err)
		}
	}
	aliceRange, aliceDigest, err := vertexRecordDigest(
		"Person", "neo4j", "alice", []byte(`{"role":"buyer"}`))
	if err != nil {
		t.Fatal(err)
	}
	bobRange, bobDigest, err := vertexRecordDigest(
		"Person", "neo4j", "bob", []byte(`{"role":"supplier"}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := transaction.Exec(ctx,
		`INSERT INTO "af pggraph integration"."person" `+
			`(source_namespace, external_id, properties, digest_range, source_digest) VALUES `+
			`('neo4j', 'alice', '{"role":"buyer"}', $1, $2), `+
			`('neo4j', 'bob', '{"role":"supplier"}', $3, $4)`,
		int(aliceRange), aliceDigest, int(bobRange), bobDigest,
	); err != nil {
		t.Fatalf("insert vertices: %v", err)
	}
	edgeRange, edgeDigest, err := edgeRecordDigest(
		"KNOWS", "neo4j", "10", "Person", "neo4j", "alice",
		"Person", "neo4j", "bob", []byte(`{"since":2024}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := transaction.Exec(ctx,
		`INSERT INTO "af pggraph integration"."knows" `+
			`(source_namespace, external_id, start_id, end_id, properties, digest_range, source_digest) `+
			`SELECT 'neo4j', '10', a.id, b.id, '{"since":2024}', $1, $2 `+
			`FROM "af pggraph integration"."person" a, `+
			`"af pggraph integration"."person" b `+
			`WHERE a.external_id = 'alice' AND b.external_id = 'bob'`,
		int(edgeRange), edgeDigest,
	); err != nil {
		t.Fatalf("insert edge: %v", err)
	}
	var source, target, role string
	err = transaction.QueryRow(ctx,
		`SELECT source, target, properties->>'role' FROM GRAPH_TABLE (`+
			`"Supply Graph" MATCH (a IS "Person")-[k IS "KNOWS"]->(b IS "Person") `+
			`COLUMNS (a.external_id AS source, b.external_id AS target, `+
			`b.properties AS properties))`,
	).Scan(&source, &target, &role)
	if err != nil {
		t.Fatalf("GRAPH_TABLE query: %v", err)
	}
	if source != "alice" || target != "bob" || role != "supplier" {
		t.Fatalf("GRAPH_TABLE row = %q, %q, %q", source, target, role)
	}
	if err := transaction.Commit(ctx); err != nil {
		t.Fatalf("Commit() error = %v", err)
	}

	var relationKind string
	if err := connection.QueryRow(ctx,
		`SELECT c.relkind::text FROM pg_class c `+
			`JOIN pg_namespace n ON n.oid = c.relnamespace `+
			`WHERE n.nspname = $1 AND c.relname = $2`,
		schema, definition.Graph,
	).Scan(&relationKind); err != nil {
		t.Fatalf("read property graph catalog: %v", err)
	}
	if relationKind != "g" {
		t.Fatalf("property graph relkind = %q, want g", relationKind)
	}
}
