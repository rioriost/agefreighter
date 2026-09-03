package pggraph

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
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
	if _, err := transaction.Exec(ctx,
		`INSERT INTO "af pggraph integration"."person" `+
			`(source_namespace, external_id, properties) VALUES `+
			`('neo4j', 'alice', '{"role":"buyer"}'), `+
			`('neo4j', 'bob', '{"role":"supplier"}')`,
	); err != nil {
		t.Fatalf("insert vertices: %v", err)
	}
	if _, err := transaction.Exec(ctx,
		`INSERT INTO "af pggraph integration"."knows" `+
			`(source_namespace, external_id, start_id, end_id, properties) `+
			`SELECT 'neo4j', '10', a.id, b.id, '{"since":2024}' `+
			`FROM "af pggraph integration"."person" a, `+
			`"af pggraph integration"."person" b `+
			`WHERE a.external_id = 'alice' AND b.external_id = 'bob'`,
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
