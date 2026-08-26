package neo4j

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"testing"
	"time"

	neodriver "github.com/neo4j/neo4j-go-driver/v6/neo4j"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

func TestNeo4jIntegration(t *testing.T) {
	uri := os.Getenv("AGEFREIGHTER_NEO4J_TEST_URI")
	if uri == "" {
		t.Skip("AGEFREIGHTER_NEO4J_TEST_URI is not set")
	}
	username := os.Getenv("AGEFREIGHTER_NEO4J_TEST_USERNAME")
	password := os.Getenv("AGEFREIGHTER_NEO4J_TEST_PASSWORD")
	database := os.Getenv("AGEFREIGHTER_NEO4J_TEST_DATABASE")
	if database == "" {
		database = "neo4j"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	auth := neodriver.NoAuth()
	if username != "" {
		auth = neodriver.BasicAuth(username, password, "")
	}
	fixtureDriver, err := neodriver.NewDriverWithContext(uri, auth)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := fixtureDriver.Close(context.Background()); err != nil {
			t.Errorf("close fixture driver: %v", err)
		}
	}()
	label := "AgefreighterM9_" + strconv.FormatInt(time.Now().UnixNano(), 36)
	extraLabel := label + "_Extra"
	writeCypher(t, ctx, fixtureDriver, database, fmt.Sprintf(`
		CREATE (a:%s {k: 1, external_id: 'one', name: 'Alice',
			born: date('2026-01-02'),
			location: point({longitude: 1.5, latitude: 2.5})})
		CREATE (b:%s {k: 2, external_id: 'two', name: 'Bob',
			born: localdatetime('2026-01-02T03:04:05'),
			location: point({x: 1.0, y: 2.0, z: 3.0, srid: 9157})})
		CREATE (c:%s:%s {k: 3, external_id: 'three', name: 'Carol',
			born: time('03:04:05+01:00'),
			location: point({longitude: 3.5, latitude: 4.5})})
		CREATE (a)-[:KNOWS {k: 10, external_id: 'edge-one'}]->(b)
	`, label, label, label, extraLabel), nil)
	defer writeCypher(t, context.Background(), fixtureDriver, database,
		fmt.Sprintf("MATCH (n:%s) DETACH DELETE n", label), nil)

	query := fmt.Sprintf(`
		MATCH (n:%s)
		WHERE $afterKey IS NULL OR n.k > $afterKey
		RETURN n.k AS k, n.external_id AS external_id, n AS node,
			n.born AS temporal, n.location AS spatial
		ORDER BY k`, label)
	source := config.Neo4jSource{
		URI: uri, Database: database, SourceID: label, Username: username,
		FetchRows: 1, MultiLabelPolicy: config.Neo4jMultiLabelConfigured,
		Vertices: []config.VertexQuery{{
			Label: "Person", Query: query, KeyField: "k", IDField: "external_id",
			Properties: map[string]string{
				"node": "node", "temporal": "temporal", "spatial": "spatial",
			},
		}},
	}

	t.Run("streaming temporal spatial and resume", func(t *testing.T) {
		client, err := NewSDKClient(ctx, uri, database, username, password, 1)
		if err != nil {
			t.Fatal(err)
		}
		iterator := newIntegrationIterator(t, ctx, source, client, "", nil)
		first, err := iterator.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if first.Record.Kind() != model.RecordVertex ||
			first.Record.Vertex.ExternalID != "one" {
			t.Fatalf("first = %#v", first.Record)
		}
		properties := first.Record.Vertex.Properties
		if properties["temporal"].Kind != model.ValueString ||
			properties["spatial"].Kind != model.ValueObject ||
			properties["spatial"].Object["srid"].Integer == 0 {
			t.Fatalf("converted properties = %#v", properties)
		}
		token := first.Record.Vertex.Position.Token
		if err := iterator.Close(); err != nil {
			t.Fatal(err)
		}

		resumeClient, err := NewSDKClient(ctx, uri, database, username, password, 1)
		if err != nil {
			t.Fatal(err)
		}
		resumed := newIntegrationIterator(t, ctx, source, resumeClient, token, nil)
		var ids []model.ExternalID
		for {
			item, err := resumed.Next(ctx)
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			ids = append(ids, item.Record.Vertex.ExternalID)
		}
		if len(ids) != 2 || ids[0] != "two" || ids[1] != "three" {
			t.Fatalf("resumed IDs = %v", ids)
		}
		if telemetry := resumed.DetailedTelemetry(); telemetry.Queries != 1 ||
			telemetry.Records != 2 {
			t.Fatalf("telemetry = %#v", telemetry)
		}
		if err := resumed.Close(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("multi-label rejection", func(t *testing.T) {
		rejectSource := source
		rejectSource.MultiLabelPolicy = config.Neo4jMultiLabelReject
		rejectSource.Vertices = append([]config.VertexQuery(nil), source.Vertices...)
		rejectSource.Vertices[0].Query = fmt.Sprintf(`
			MATCH (n:%s:%s)
			WHERE $afterKey IS NULL OR n.k > $afterKey
			RETURN n.k AS k, n.external_id AS external_id, n AS node,
				n.born AS temporal, n.location AS spatial
			ORDER BY k`, label, extraLabel)
		client, err := NewSDKClient(ctx, uri, database, username, password, 2)
		if err != nil {
			t.Fatal(err)
		}
		rejected := 0
		iterator := newIntegrationIterator(
			t, ctx, rejectSource, client, "",
			func(context.Context, MalformedRecord) error {
				rejected++
				return nil
			},
		)
		if _, err := iterator.Next(ctx); !errors.Is(err, io.EOF) {
			t.Fatalf("end after quarantine = %v", err)
		}
		if count, _ := iterator.RejectionCheckpoint(); count != 1 || rejected != 1 {
			t.Fatalf("rejections = %d, handled = %d", count, rejected)
		}
		if err := iterator.Close(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("mapping transactions observe intervening mutation", func(t *testing.T) {
		mutationSource := source
		mutationSource.Vertices = []config.VertexQuery{
			source.Vertices[0], source.Vertices[0],
		}
		mutationSource.Vertices[1].Label = "PersonAgain"
		client, err := NewSDKClient(ctx, uri, database, username, password, 1)
		if err != nil {
			t.Fatal(err)
		}
		iterator := newIntegrationIterator(t, ctx, mutationSource, client, "", nil)
		for range 3 {
			if _, err := iterator.Next(ctx); err != nil {
				t.Fatal(err)
			}
		}
		writeCypher(t, ctx, fixtureDriver, database, fmt.Sprintf(
			"CREATE (n:%s {k: 4, external_id: 'four', name: 'Dora'})", label,
		), nil)
		seen := 0
		for {
			item, err := iterator.Next(ctx)
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			seen++
			if seen == 4 && item.Record.Vertex.ExternalID != "four" {
				t.Fatalf("fourth second-mapping record = %#v", item.Record.Vertex)
			}
		}
		if seen != 4 {
			t.Fatalf("second mapping saw %d records", seen)
		}
		if err := iterator.Close(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("cancellation and cleanup", func(t *testing.T) {
		client, err := NewSDKClient(ctx, uri, database, username, password, 1)
		if err != nil {
			t.Fatal(err)
		}
		iterator := newIntegrationIterator(t, ctx, source, client, "", nil)
		if _, err := iterator.Next(ctx); err != nil {
			t.Fatal(err)
		}
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		if _, err := iterator.Next(cancelled); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation = %v", err)
		}
		if err := iterator.Close(); err != nil {
			t.Fatal(err)
		}
		if err := iterator.Close(); err != nil {
			t.Fatalf("second close: %v", err)
		}
	})
}

func newIntegrationIterator(
	t *testing.T,
	ctx context.Context,
	source config.Neo4jSource,
	client Client,
	token string,
	handler MalformedHandler,
) *Iterator {
	t.Helper()
	rejectLimit := 0
	if handler != nil {
		rejectLimit = 10
	}
	iterator, err := NewIterator(ctx, IteratorOptions{
		Namespace: "integration", Source: source, Client: client,
		AfterToken: token, RejectLimit: rejectLimit, OnMalformed: handler,
		MaxRecordBytes: 1 << 20, MaxProperties: 100,
	})
	if err != nil {
		_ = client.Close()
		t.Fatal(err)
	}
	return iterator
}

func writeCypher(
	t *testing.T,
	ctx context.Context,
	driver neodriver.Driver,
	database, query string,
	parameters map[string]any,
) {
	t.Helper()
	session := driver.NewSession(ctx, neodriver.SessionConfig{
		AccessMode: neodriver.AccessModeWrite, DatabaseName: database,
	})
	defer func() {
		if err := session.Close(context.Background()); err != nil {
			t.Errorf("close fixture session: %v", err)
		}
	}()
	result, err := session.Run(ctx, query, parameters)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := result.Consume(ctx); err != nil {
		t.Fatal(err)
	}
}
