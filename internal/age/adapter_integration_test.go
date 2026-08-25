package age

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

const (
	integrationDSNEnvironment = "AGEFREIGHTER_AGE_TEST_DSN"
	integrationPassword       = "agefreighter_dev_only"
	restrictedRole            = "agefreighter_it_restricted"
)

func TestAdapterIntegration(t *testing.T) {
	dsn := os.Getenv(integrationDSNEnvironment)
	if dsn == "" {
		t.Skip("set " + integrationDSNEnvironment + " to run Apache AGE integration tests")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	adapter := openIntegrationAdapter(t, ctx, dsn, 3)
	t.Cleanup(adapter.Close)
	if adapter.Capabilities().PostgreSQLMajor != 17 ||
		adapter.Capabilities().AGEVersion.Major != 1 ||
		adapter.Capabilities().AGEVersion.Minor != 6 {
		t.Fatalf("unexpected capabilities: %#v", adapter.Capabilities())
	}
	if adapter.Capabilities().AGEPreloadStatus != PreloadConfigured {
		t.Fatalf("pinned integration image preload status = %q", adapter.Capabilities().AGEPreloadStatus)
	}
	assertPoolSearchPath(t, ctx, adapter, 3)

	graphName := "af_it_graph"
	dropGraphIfPresent(t, ctx, adapter, graphName)
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		dropGraphIfPresent(t, cleanupCtx, adapter, graphName)
	})

	var person, knows LabelCatalog
	err := adapter.InTransaction(ctx, func(transaction *Transaction) error {
		if err := transaction.CreateGraph(ctx, graphName); err != nil {
			return err
		}
		if err := transaction.CreateLabel(ctx, graphName, "Person", VertexLabel); err != nil {
			return err
		}
		if err := transaction.CreateLabel(ctx, graphName, "KNOWS", EdgeLabel); err != nil {
			return err
		}
		var err error
		person, err = transaction.LookupLabel(ctx, graphName, "Person")
		if err != nil {
			return err
		}
		knows, err = transaction.LookupLabel(ctx, graphName, "KNOWS")
		return err
	})
	if err != nil {
		t.Fatalf("create graph and labels: %v", err)
	}
	if person.Kind != VertexLabel || knows.Kind != EdgeLabel ||
		person.SequenceName == "" || knows.SequenceName == "" {
		t.Fatalf("invalid label catalogs: %#v %#v", person, knows)
	}
	if _, err := adapter.LookupGraph(ctx, graphName); err != nil {
		t.Fatalf("adapter LookupGraph() error = %v", err)
	}
	if _, err := adapter.LookupLabel(ctx, graphName, "Person"); err != nil {
		t.Fatalf("adapter LookupLabel() error = %v", err)
	}
	if _, err := adapter.LookupGraph(ctx, "missing_graph"); !errors.Is(err, ErrCatalogEntryNotFound) {
		t.Fatalf("missing LookupGraph() error = %v", err)
	}
	assertLabelDropLifecycle(t, ctx, adapter, graphName)
	assertSequenceCatalogChangeRollsBack(t, ctx, adapter, person)

	var vertexIDs []GraphID
	err = adapter.InTransaction(ctx, func(transaction *Transaction) error {
		block, err := transaction.ReserveIDs(ctx, person, 3)
		if err != nil {
			return err
		}
		vertexIDs = make([]GraphID, 3)
		for index := range vertexIDs {
			vertexIDs[index], err = block.GraphID(uint64(index))
			if err != nil {
				return err
			}
		}
		if _, err := transaction.CopyVertices(ctx, person, []VertexRow{
			{ID: vertexIDs[0], Properties: []byte(`{"name":"Ada"}`)},
			{ID: vertexIDs[1], Properties: []byte(`{"name":"Grace"}`)},
		}, DirectTextCopy); err != nil {
			return err
		}
		if _, err := transaction.CopyVertices(ctx, person, []VertexRow{
			{ID: vertexIDs[2], Properties: []byte(`{"name":"Linus"}`)},
		}, StagedBinaryCopy); err != nil {
			return err
		}
		return transaction.VerifyLabelRows(ctx, person, 3)
	})
	if err != nil {
		t.Fatalf("copy vertices: %v", err)
	}
	assertVerificationFailures(t, ctx, adapter, person)

	err = adapter.InTransaction(ctx, func(transaction *Transaction) error {
		block, err := transaction.ReserveIDs(ctx, knows, 2)
		if err != nil {
			return err
		}
		first, err := block.GraphID(0)
		if err != nil {
			return err
		}
		second, err := block.GraphID(1)
		if err != nil {
			return err
		}
		if _, err := transaction.CopyEdges(ctx, knows, []EdgeRow{{
			ID: first, StartID: vertexIDs[0], EndID: vertexIDs[1],
			Properties: []byte(`{"since":1959}`),
		}}, DirectTextCopy); err != nil {
			return err
		}
		if _, err := transaction.CopyEdges(ctx, knows, []EdgeRow{{
			ID: second, StartID: vertexIDs[1], EndID: vertexIDs[2],
			Properties: []byte(`{"since":1991}`),
		}}, StagedBinaryCopy); err != nil {
			return err
		}
		if err := transaction.AnalyzeLabel(ctx, person); err != nil {
			return err
		}
		if err := transaction.AnalyzeLabel(ctx, knows); err != nil {
			return err
		}
		return transaction.VerifyLabelRows(ctx, knows, 2)
	})
	if err != nil {
		t.Fatalf("copy edges and verify labels: %v", err)
	}

	failureID, err := MakeGraphID(person.LabelID, 1000)
	if err != nil {
		t.Fatalf("make failure-injection graphid: %v", err)
	}
	assertCopyFailureRollsBack(t, ctx, adapter, person, failureID)
	assertCypherSequenceContinues(t, ctx, adapter, graphName, person)
	assertGraphRenameRoundTrip(t, ctx, adapter, graphName)
	assertRenameFailureRollsBack(t, ctx, adapter, graphName)
	testRestrictedRole(t, ctx, adapter, dsn)
}

func assertLabelDropLifecycle(
	t *testing.T,
	ctx context.Context,
	adapter *Adapter,
	graphName string,
) {
	t.Helper()
	err := adapter.InTransaction(ctx, func(transaction *Transaction) error {
		if err := transaction.CreateLabel(ctx, graphName, "TemporaryVertex", VertexLabel); err != nil {
			return err
		}
		if err := transaction.CreateLabel(ctx, graphName, "TemporaryEdge", EdgeLabel); err != nil {
			return err
		}
		if err := transaction.DropLabel(ctx, graphName, "TemporaryEdge", false); err != nil {
			return err
		}
		return transaction.DropLabel(ctx, graphName, "TemporaryVertex", false)
	})
	if err != nil {
		t.Fatalf("label drop lifecycle: %v", err)
	}
	if _, err := adapter.LookupLabel(
		ctx,
		graphName,
		"TemporaryVertex",
	); !errors.Is(err, ErrCatalogEntryNotFound) {
		t.Fatalf("dropped LookupLabel() error = %v", err)
	}
}

func assertSequenceCatalogChangeRollsBack(
	t *testing.T,
	ctx context.Context,
	adapter *Adapter,
	person LabelCatalog,
) {
	t.Helper()
	changed := person
	changed.SequenceName += "_changed"
	err := adapter.InTransaction(ctx, func(transaction *Transaction) error {
		_, err := transaction.ReserveIDs(ctx, changed, 1)
		return err
	})
	if err == nil {
		t.Fatal("ReserveIDs() accepted changed catalog")
	}
}

func openIntegrationAdapter(
	t *testing.T,
	ctx context.Context,
	dsn string,
	maxConnections int32,
) *Adapter {
	t.Helper()
	adapter, err := Open(ctx, dsn, PoolOptions{
		MinConnections:   0,
		MaxConnections:   maxConnections,
		ConnectTimeout:   5 * time.Second,
		OperationTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	return adapter
}

func assertPoolSearchPath(
	t *testing.T,
	ctx context.Context,
	adapter *Adapter,
	count int,
) {
	t.Helper()
	connections := make([]*pgx.Conn, 0, count)
	acquired := make([]interface{ Release() }, 0, count)
	for range count {
		connection, err := adapter.pool.Acquire(ctx)
		if err != nil {
			t.Fatalf("Acquire() error = %v", err)
		}
		acquired = append(acquired, connection)
		connections = append(connections, connection.Conn())
	}
	defer func() {
		for _, connection := range acquired {
			connection.Release()
		}
	}()
	for _, connection := range connections {
		var searchPath string
		if err := connection.QueryRow(ctx, "SHOW search_path").Scan(&searchPath); err != nil {
			t.Fatalf("SHOW search_path error = %v", err)
		}
		if searchPath != `ag_catalog, "$user", public` {
			t.Fatalf("search_path = %q", searchPath)
		}
	}
}

func assertCopyFailureRollsBack(
	t *testing.T,
	ctx context.Context,
	adapter *Adapter,
	person LabelCatalog,
	failureID GraphID,
) {
	t.Helper()
	err := adapter.InTransaction(ctx, func(transaction *Transaction) error {
		_, err := transaction.CopyVertices(ctx, person, []VertexRow{{
			ID: failureID, Properties: []byte(`{`),
		}}, DirectTextCopy)
		return err
	})
	if err == nil {
		t.Fatal("invalid agtype COPY succeeded")
	}
	err = adapter.InTransaction(ctx, func(transaction *Transaction) error {
		return transaction.VerifyLabelRows(ctx, person, 3)
	})
	if err != nil {
		t.Fatalf("failed COPY was not rolled back: %v", err)
	}
}

func assertVerificationFailures(
	t *testing.T,
	ctx context.Context,
	adapter *Adapter,
	person LabelCatalog,
) {
	t.Helper()
	err := adapter.InTransaction(ctx, func(transaction *Transaction) error {
		return transaction.VerifyLabelRows(ctx, person, 999)
	})
	if err == nil {
		t.Fatal("VerifyLabelRows() accepted an incorrect row count")
	}

	changed := person
	changed.SequenceName += "_changed"
	err = adapter.InTransaction(ctx, func(transaction *Transaction) error {
		return transaction.AnalyzeLabel(ctx, changed)
	})
	if err == nil {
		t.Fatal("AnalyzeLabel() accepted a changed catalog")
	}

	var wrongIDLabel LabelCatalog
	err = adapter.InTransaction(ctx, func(transaction *Transaction) error {
		if err := transaction.CreateLabel(
			ctx,
			person.GraphName,
			"WrongID",
			VertexLabel,
		); err != nil {
			return err
		}
		var err error
		wrongIDLabel, err = transaction.LookupLabel(ctx, person.GraphName, "WrongID")
		return err
	})
	if err != nil {
		t.Fatalf("create wrong-ID verification label: %v", err)
	}
	err = adapter.InTransaction(ctx, func(transaction *Transaction) error {
		wrongID, err := MakeGraphID(person.LabelID, 10_000)
		if err != nil {
			return err
		}
		table := pgx.Identifier{
			wrongIDLabel.GraphName,
			wrongIDLabel.LabelName,
		}.Sanitize()
		if _, err := transaction.tx.Exec(
			ctx,
			fmt.Sprintf(
				"INSERT INTO %s (id, properties) VALUES ($1::text::graphid, '{}'::agtype)",
				table,
			),
			fmt.Sprint(int64(wrongID)),
		); err != nil {
			return err
		}
		return transaction.VerifyLabelRows(ctx, wrongIDLabel, 1)
	})
	if err == nil {
		t.Fatal("VerifyLabelRows() accepted an ID from another label")
	}
	if err := adapter.InTransaction(ctx, func(transaction *Transaction) error {
		return transaction.DropLabel(ctx, person.GraphName, "WrongID", false)
	}); err != nil {
		t.Fatalf("drop wrong-ID verification label: %v", err)
	}
}

func assertCypherSequenceContinues(
	t *testing.T,
	ctx context.Context,
	adapter *Adapter,
	graphName string,
	person LabelCatalog,
) {
	t.Helper()
	err := adapter.InTransaction(ctx, func(transaction *Transaction) error {
		var created string
		query := fmt.Sprintf(
			`SELECT value::text
			 FROM ag_catalog.cypher(
				'%s',
				$$CREATE (person:Person {name: "Cypher"}) RETURN id(person)$$
			 ) AS result(value ag_catalog.agtype)`,
			graphName,
		)
		if err := transaction.tx.QueryRow(
			ctx,
			query,
		).Scan(&created); err != nil {
			return fmt.Errorf("Cypher CREATE after COPY: %w", err)
		}
		if created == "" {
			return fmt.Errorf("Cypher CREATE returned an empty ID")
		}
		return transaction.VerifyLabelRows(ctx, person, 4)
	})
	if err != nil {
		t.Fatal(err)
	}
}

func assertGraphRenameRoundTrip(
	t *testing.T,
	ctx context.Context,
	adapter *Adapter,
	graphName string,
) {
	t.Helper()
	renamed := graphName + "_renamed"
	err := adapter.InTransaction(ctx, func(transaction *Transaction) error {
		if err := transaction.RenameGraph(ctx, graphName, renamed); err != nil {
			return err
		}
		if _, err := transaction.LookupGraph(ctx, renamed); err != nil {
			return err
		}
		return transaction.RenameGraph(ctx, renamed, graphName)
	})
	if err != nil {
		t.Fatalf("graph rename round trip: %v", err)
	}
}

func assertRenameFailureRollsBack(
	t *testing.T,
	ctx context.Context,
	adapter *Adapter,
	graphName string,
) {
	t.Helper()
	const collision = "af_it_collision"
	dropGraphIfPresent(t, ctx, adapter, collision)
	if err := adapter.InTransaction(ctx, func(transaction *Transaction) error {
		return transaction.CreateGraph(ctx, collision)
	}); err != nil {
		t.Fatalf("create rename collision graph: %v", err)
	}
	defer dropGraphIfPresent(t, ctx, adapter, collision)

	err := adapter.InTransaction(ctx, func(transaction *Transaction) error {
		return transaction.RenameGraph(ctx, graphName, collision)
	})
	if err == nil {
		t.Fatal("RenameGraph() replaced an existing graph")
	}
	if _, err := adapter.LookupGraph(ctx, graphName); err != nil {
		t.Fatalf("failed rename did not preserve source graph: %v", err)
	}
	if _, err := adapter.LookupGraph(ctx, collision); err != nil {
		t.Fatalf("failed rename did not preserve destination graph: %v", err)
	}
}

func testRestrictedRole(
	t *testing.T,
	ctx context.Context,
	admin *Adapter,
	adminDSN string,
) {
	t.Helper()
	const graphName = "af_restricted_graph"
	dropGraphIfPresent(t, ctx, admin, graphName)
	removeRestrictedRole(t, ctx, admin)
	if _, err := admin.pool.Exec(ctx, fmt.Sprintf(
		"CREATE ROLE %s LOGIN PASSWORD '%s'",
		pgx.Identifier{restrictedRole}.Sanitize(),
		integrationPassword,
	)); err != nil {
		t.Fatalf("create restricted role: %v", err)
	}
	t.Cleanup(func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		dropGraphIfPresent(t, cleanupCtx, admin, graphName)
		removeRestrictedRole(t, cleanupCtx, admin)
	})
	database := pgx.Identifier{admin.pool.Config().ConnConfig.Database}.Sanitize()
	grants := []string{
		fmt.Sprintf("GRANT CONNECT, CREATE ON DATABASE %s TO %s", database, pgx.Identifier{restrictedRole}.Sanitize()),
		fmt.Sprintf("GRANT USAGE ON SCHEMA ag_catalog TO %s", pgx.Identifier{restrictedRole}.Sanitize()),
		fmt.Sprintf("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ag_catalog TO %s", pgx.Identifier{restrictedRole}.Sanitize()),
		fmt.Sprintf("GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA ag_catalog TO %s", pgx.Identifier{restrictedRole}.Sanitize()),
		fmt.Sprintf("GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA ag_catalog TO %s", pgx.Identifier{restrictedRole}.Sanitize()),
	}
	for _, grant := range grants {
		if _, err := admin.pool.Exec(ctx, grant); err != nil {
			t.Fatalf("grant restricted role privileges: %v", err)
		}
	}

	restrictedDSN, err := dsnForRole(adminDSN, restrictedRole, integrationPassword)
	if err != nil {
		t.Fatalf("build restricted DSN: %v", err)
	}
	restricted := openIntegrationAdapter(t, ctx, restrictedDSN, 1)
	if restricted.Capabilities().CurrentUserSuperuser {
		restricted.Close()
		t.Fatal("restricted integration role is a superuser")
	}
	if restricted.Capabilities().AGEPreloadStatus != PreloadUnknown {
		restricted.Close()
		t.Fatalf(
			"restricted preload status = %q, want unknown",
			restricted.Capabilities().AGEPreloadStatus,
		)
	}
	err = restricted.InTransaction(ctx, func(transaction *Transaction) error {
		if err := transaction.CreateGraph(ctx, graphName); err != nil {
			return err
		}
		return transaction.CreateLabel(ctx, graphName, "Person", VertexLabel)
	})
	if err == nil {
		err = restricted.InTransaction(ctx, func(transaction *Transaction) error {
			return transaction.DropGraph(ctx, graphName, true)
		})
	}
	restricted.Close()
	if err != nil {
		t.Fatalf("restricted AGE lifecycle: %v", err)
	}
	removeRestrictedRole(t, ctx, admin)
}

func dsnForRole(dsn, role, password string) (string, error) {
	parsed, err := url.Parse(dsn)
	if err != nil {
		return "", err
	}
	parsed.User = url.UserPassword(role, password)
	return parsed.String(), nil
}

func dropGraphIfPresent(
	t *testing.T,
	ctx context.Context,
	adapter *Adapter,
	graphName string,
) {
	t.Helper()
	err := adapter.InTransaction(ctx, func(transaction *Transaction) error {
		_, err := transaction.LookupGraph(ctx, graphName)
		if err != nil {
			if strings.Contains(err.Error(), ErrCatalogEntryNotFound.Error()) {
				return nil
			}
			return err
		}
		return transaction.DropGraph(ctx, graphName, true)
	})
	if err != nil {
		t.Fatalf("drop graph %q: %v", graphName, err)
	}
}

func removeRestrictedRole(t *testing.T, ctx context.Context, admin *Adapter) {
	t.Helper()
	role := pgx.Identifier{restrictedRole}.Sanitize()
	var exists bool
	if err := admin.pool.QueryRow(
		ctx,
		"SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = $1)",
		restrictedRole,
	).Scan(&exists); err != nil {
		t.Fatalf("check restricted role: %v", err)
	}
	if !exists {
		return
	}
	if _, err := admin.pool.Exec(ctx, "DROP OWNED BY "+role); err != nil {
		t.Fatalf("drop restricted role objects: %v", err)
	}
	if _, err := admin.pool.Exec(ctx, "DROP ROLE IF EXISTS "+role); err != nil {
		t.Fatalf("drop restricted role: %v", err)
	}
}
