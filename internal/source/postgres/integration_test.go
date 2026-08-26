package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

func postgresIntegrationDSN(t *testing.T) string {
	t.Helper()
	dsn := os.Getenv("AGEFREIGHTER_POSTGRES_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_POSTGRES_TEST_DSN")
	}
	return dsn
}

func integrationConnection(t *testing.T) (*pgx.Conn, string) {
	t.Helper()
	dsn := postgresIntegrationDSN(t)
	conn, err := pgx.Connect(t.Context(), dsn)
	if err != nil {
		t.Fatalf("connect PostgreSQL integration database: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close(context.Background()) })
	return conn, dsn
}

func integrationTable(t *testing.T, conn *pgx.Conn) string {
	t.Helper()
	name := fmt.Sprintf("agefreighter_source_%d", time.Now().UnixNano())
	identifier := pgx.Identifier{name}.Sanitize()
	if _, err := conn.Exec(
		t.Context(),
		"CREATE TABLE "+identifier+
			" (seq bigint PRIMARY KEY, graph_id text, payload jsonb NOT NULL)",
	); err != nil {
		t.Fatalf("create fixture: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, _ = conn.Exec(ctx, "DROP TABLE IF EXISTS "+identifier)
	})
	return identifier
}

func integrationSource(
	mode config.PostgreSQLReadMode,
	table string,
) config.PostgreSQLSource {
	query := "SELECT seq, graph_id, payload FROM " + table + " ORDER BY seq"
	keyField := ""
	if mode == config.PostgreSQLReadKeyset {
		query = "SELECT seq, graph_id, payload FROM " + table +
			" WHERE ($1::bigint IS NULL OR seq > $1) ORDER BY seq LIMIT $2"
		keyField = "seq"
	}
	query += " -- stable ordering"
	return config.PostgreSQLSource{
		ReadMode: mode, FetchRows: 1,
		Vertices: []config.VertexQuery{{
			Label: "Person", Query: query, IDField: "graph_id",
			KeyField: keyField, Properties: map[string]string{"payload": "payload"},
		}},
	}
}

func collectIntegrationIDs(
	t *testing.T,
	iterator *Iterator,
) []model.ExternalID {
	t.Helper()
	var ids []model.ExternalID
	for {
		item, err := iterator.Next(t.Context())
		if errors.Is(err, io.EOF) {
			return ids
		}
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		ids = append(ids, item.Record.Vertex.ExternalID)
	}
}

func TestPostgreSQLModesSnapshotAndResumeIntegration(t *testing.T) {
	conn, dsn := integrationConnection(t)
	for _, mode := range []config.PostgreSQLReadMode{
		config.PostgreSQLReadCopy,
		config.PostgreSQLReadCursor,
		config.PostgreSQLReadKeyset,
	} {
		t.Run(string(mode), func(t *testing.T) {
			table := integrationTable(t, conn)
			if _, err := conn.Exec(
				t.Context(),
				"INSERT INTO "+table+` VALUES
					(1, 'p1', '{"rank":1}'), (2, 'p2', '{"rank":2}')`,
			); err != nil {
				t.Fatal(err)
			}
			source := integrationSource(mode, table)
			iterator, err := NewIterator(t.Context(), IteratorOptions{
				Namespace: "crm", Source: source, DSN: dsn,
				MaxReaders: 2, PreencodeProperties: true,
			})
			if err != nil {
				t.Fatalf("NewIterator() error = %v", err)
			}
			first, err := iterator.Next(t.Context())
			if err != nil {
				t.Fatalf("first Next() error = %v", err)
			}
			position, _ := first.Record.SourcePosition()
			if _, err := conn.Exec(
				t.Context(),
				"INSERT INTO "+table+" VALUES (3, 'late', '{}')",
			); err != nil {
				t.Fatal(err)
			}
			remaining := collectIntegrationIDs(t, iterator)
			if len(remaining) != 1 || remaining[0] != "p2" {
				t.Fatalf("snapshot remaining IDs = %v", remaining)
			}
			if err := iterator.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}
			if err := iterator.Close(); err != nil {
				t.Fatalf("second Close() error = %v", err)
			}

			resumed, err := NewIterator(t.Context(), IteratorOptions{
				Namespace: "crm", Source: source, DSN: dsn,
				AfterToken: position.Token, MaxReaders: 1,
			})
			if err != nil {
				t.Fatalf("resumed NewIterator() error = %v", err)
			}
			t.Cleanup(func() { _ = resumed.Close() })
			resumedIDs := collectIntegrationIDs(t, resumed)
			if len(resumedIDs) != 2 ||
				resumedIDs[0] != "p2" ||
				resumedIDs[1] != "late" {
				t.Fatalf("resumed IDs = %v", resumedIDs)
			}
		})
	}
}

func TestPostgreSQLMalformedIntegration(t *testing.T) {
	conn, dsn := integrationConnection(t)
	table := integrationTable(t, conn)
	if _, err := conn.Exec(
		t.Context(),
		"INSERT INTO "+table+` VALUES
			(1, 'p1', '{}'), (2, NULL, '{}'), (3, 'p3', '{}')`,
	); err != nil {
		t.Fatal(err)
	}
	var malformed []MalformedRecord
	iterator, err := NewIterator(t.Context(), IteratorOptions{
		Namespace: "crm", Source: integrationSource(config.PostgreSQLReadCursor, table),
		DSN: dsn, RejectLimit: 1,
		OnMalformed: func(_ context.Context, record MalformedRecord) error {
			malformed = append(malformed, record)
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = iterator.Close() })
	ids := collectIntegrationIDs(t, iterator)
	if len(ids) != 2 || ids[0] != "p1" || ids[1] != "p3" {
		t.Fatalf("IDs = %v", ids)
	}
	if len(malformed) != 1 ||
		malformed[0].Position.Line != 2 ||
		malformed[0].Position.Resource != "vertex[0]:Person" {
		t.Fatalf("malformed = %#v", malformed)
	}
	rejected, position := iterator.RejectionCheckpoint()
	if rejected != 1 || position.Line != 3 {
		t.Fatalf("checkpoint = %d, %#v", rejected, position)
	}
}

func TestSnapshotCoordinatorConcurrentReadersIntegration(t *testing.T) {
	conn, dsn := integrationConnection(t)
	table := integrationTable(t, conn)
	if _, err := conn.Exec(
		t.Context(),
		"INSERT INTO "+table+" VALUES (1, 'p1', '{}'), (2, 'p2', '{}')",
	); err != nil {
		t.Fatal(err)
	}
	coordinator, err := NewSnapshotCoordinator(t.Context(), dsn, 2)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = coordinator.Close() })
	if !validSnapshotID(coordinator.SnapshotID()) {
		t.Fatalf("SnapshotID() = %q", coordinator.SnapshotID())
	}
	if _, err := conn.Exec(
		t.Context(),
		"INSERT INTO "+table+" VALUES (3, 'late', '{}')",
	); err != nil {
		t.Fatal(err)
	}

	const readers = 6
	results := make(chan int, readers)
	failures := make(chan error, readers)
	var wait sync.WaitGroup
	for range readers {
		wait.Add(1)
		go func() {
			defer wait.Done()
			reader, err := coordinator.OpenReader(t.Context())
			if err != nil {
				failures <- err
				return
			}
			defer reader.Close()
			var count int
			if err := reader.Tx().QueryRow(
				t.Context(), "SELECT count(*) FROM "+table,
			).Scan(&count); err != nil {
				failures <- err
				return
			}
			results <- count
		}()
	}
	wait.Wait()
	close(results)
	close(failures)
	for err := range failures {
		t.Errorf("reader error = %v", err)
	}
	for count := range results {
		if count != 2 {
			t.Errorf("snapshot count = %d, want 2", count)
		}
	}
}

func TestSnapshotCoordinatorBoundedReaderIntegration(t *testing.T) {
	_, dsn := integrationConnection(t)
	coordinator, err := NewSnapshotCoordinator(t.Context(), dsn, 1)
	if err != nil {
		t.Fatal(err)
	}
	first, err := coordinator.OpenReader(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	blocked, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	if _, err := coordinator.OpenReader(blocked); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("bounded OpenReader() error = %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("first reader Close() error = %v", err)
	}
	second, err := coordinator.OpenReader(t.Context())
	if err != nil {
		t.Fatalf("OpenReader() after release error = %v", err)
	}
	if err := second.Close(); err != nil {
		t.Fatalf("second reader Close() error = %v", err)
	}
	if err := coordinator.Close(); err != nil {
		t.Fatalf("coordinator Close() error = %v", err)
	}
	if err := coordinator.Close(); err != nil {
		t.Fatalf("second coordinator Close() error = %v", err)
	}
	if _, err := coordinator.OpenReader(t.Context()); err == nil {
		t.Fatal("OpenReader() succeeded after Close()")
	}
}

func TestPostgreSQLReaderFailuresIntegration(t *testing.T) {
	_, dsn := integrationConnection(t)
	coordinator, err := NewSnapshotCoordinator(t.Context(), dsn, 2)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = coordinator.Close() })
	mapping := compiledMapping{
		kind: vertexMapping, label: "Person", namespace: "crm",
		idField: "id", query: "SELECT 1 AS id ORDER BY id",
	}
	var telemetry telemetryState
	if _, err := openRecordReader(
		t.Context(), coordinator, mapping, "invalid", 1, 1024, nil, &telemetry,
	); err == nil {
		t.Fatal("openRecordReader() accepted invalid mode")
	}

	reader, err := coordinator.OpenReader(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	badMapping := mapping
	badMapping.query = "SELECT missing FROM agefreighter_missing_relation ORDER BY missing"
	if _, err := newCursorReader(t.Context(), reader, badMapping, 1, &telemetry); err == nil {
		t.Fatal("newCursorReader() accepted invalid query")
	}
	_ = reader.Close()

	reader, err = coordinator.OpenReader(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	keyset := newKeysetReader(reader, badMapping, 1, nil, &telemetry)
	if _, err := keyset.Next(t.Context()); err == nil {
		t.Fatal("keyset query failure was ignored")
	}
	_ = keyset.Close()
	if telemetry.snapshot().FailedRequestAttempts < 2 {
		t.Fatalf("telemetry = %#v", telemetry.snapshot())
	}
}

func TestPostgreSQLResumeSkipBoundsIntegration(t *testing.T) {
	conn, dsn := integrationConnection(t)
	table := integrationTable(t, conn)
	if _, err := conn.Exec(
		t.Context(),
		"INSERT INTO "+table+" VALUES (1, 'p1', '{}'), (2, 'p2', '{}')",
	); err != nil {
		t.Fatal(err)
	}
	options := IteratorOptions{
		Namespace: "crm",
		Source:    integrationSource(config.PostgreSQLReadCursor, table),
		DSN:       dsn,
	}
	options.AfterToken = resumeForOptions(t, options, func(state *resumeState) {
		state.consumed = 3
	})
	iterator, err := NewIterator(t.Context(), options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = iterator.Close() })
	if _, err := iterator.Next(t.Context()); err == nil ||
		!strings.Contains(err.Error(), "consumed count exceeds") {
		t.Fatalf("Next() error = %v", err)
	}
}

func TestPostgreSQLCopyCancellationIntegration(t *testing.T) {
	_, dsn := integrationConnection(t)
	source := config.PostgreSQLSource{
		ReadMode: config.PostgreSQLReadCopy, FetchRows: 1,
		Vertices: []config.VertexQuery{{
			Label: "Slow", Query: "SELECT 1 AS id FROM pg_sleep(30) ORDER BY id",
			IDField: "id",
		}},
	}
	iterator, err := NewIterator(t.Context(), IteratorOptions{
		Namespace: "test", Source: source, DSN: dsn,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = iterator.Close() })
	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err = iterator.Next(ctx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Next() error = %v", err)
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Fatalf("cancellation took %s", elapsed)
	}
}

func TestPostgreSQLCopyEscapedStringsIntegration(t *testing.T) {
	conn, dsn := integrationConnection(t)
	table := integrationTable(t, conn)
	want := "quote \" backslash \\ tab\tnewline\n"
	payload, err := json.Marshal(map[string]string{"value": want})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := conn.Exec(
		t.Context(),
		"INSERT INTO "+table+" VALUES (1, 'p1', $1::jsonb)",
		string(payload),
	); err != nil {
		t.Fatal(err)
	}
	iterator, err := NewIterator(t.Context(), IteratorOptions{
		Namespace: "crm",
		Source:    integrationSource(config.PostgreSQLReadCopy, table),
		DSN:       dsn,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = iterator.Close() })
	item, err := iterator.Next(t.Context())
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	payloadValue := item.Record.Vertex.Properties["payload"]
	got := payloadValue.Object["value"]
	if got.Kind != model.ValueString || got.String != want {
		t.Fatalf("COPY escaped value = %#v, want %q", got, want)
	}
}

func TestPostgreSQLConnectionLossIntegration(t *testing.T) {
	conn, dsn := integrationConnection(t)
	table := integrationTable(t, conn)
	if _, err := conn.Exec(
		t.Context(),
		"INSERT INTO "+table+" VALUES (1, 'p1', '{}'), (2, 'p2', '{}')",
	); err != nil {
		t.Fatal(err)
	}
	iterator, err := NewIterator(t.Context(), IteratorOptions{
		Namespace: "crm",
		Source:    integrationSource(config.PostgreSQLReadCursor, table),
		DSN:       dsn,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = iterator.Close() })
	if _, err := iterator.Next(t.Context()); err != nil {
		t.Fatalf("first Next() error = %v", err)
	}
	current, ok := iterator.current.(*cursorReader)
	if !ok {
		t.Fatalf("current reader = %T", iterator.current)
	}
	var terminated bool
	if err := conn.QueryRow(
		t.Context(),
		"SELECT pg_terminate_backend($1)",
		current.snapshot.conn.PgConn().PID(),
	).Scan(&terminated); err != nil {
		t.Fatalf("terminate source backend: %v", err)
	}
	if !terminated {
		t.Skip("database role cannot terminate its source backend")
	}
	if _, err := iterator.Next(t.Context()); err == nil {
		t.Fatal("Next() succeeded after source connection termination")
	}
	if telemetry := iterator.Telemetry(); telemetry.FailedRequestAttempts == 0 {
		t.Fatalf("telemetry = %#v", telemetry)
	}
}
