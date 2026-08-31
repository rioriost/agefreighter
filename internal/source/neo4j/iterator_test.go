package neo4j

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
	sourcecontract "github.com/rioriost/agefreighter/internal/source"
	"github.com/rioriost/agefreighter/pkg/model"
)

func testSource() config.Neo4jSource {
	return config.Neo4jSource{
		URI: "neo4j://example.invalid", Database: "neo4j", SourceID: "fixture",
		Username: "reader", FetchRows: 2,
		MultiLabelPolicy: config.Neo4jMultiLabelConfigured,
		Vertices: []config.VertexQuery{{
			Label: "Person", KeyField: "k", IDField: "id",
			Query: "MATCH (n) WHERE $afterKey IS NULL OR n.k > $afterKey " +
				"RETURN n.k AS k, n.id AS id, n.name AS name ORDER BY k",
			Properties: map[string]string{"name": "name"},
		}},
	}

}

func newTestIterator(t *testing.T, source config.Neo4jSource, client Client) *Iterator {
	t.Helper()
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "people", Source: source, Client: client,
		MaxRecordBytes: 1 << 20, MaxProperties: 20,
	})
	if err != nil {
		t.Fatal(err)
	}
	return iterator
}

func TestIteratorStreamsAndResumes(t *testing.T) {
	firstStream := &fakeStream{records: []Record{
		record(map[string]any{"k": int64(1), "id": "a", "name": "Alice"}, "k", "id", "name"),
		record(map[string]any{"k": int64(2), "id": "b", "name": "Bob"}, "k", "id", "name"),
	}}
	client := &fakeClient{streams: []RecordStream{firstStream}}
	iterator := newTestIterator(t, testSource(), client)

	first, err := iterator.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if first.Record.Vertex.ExternalID != "a" || first.Record.Vertex.Properties["name"].String != "Alice" {
		t.Fatalf("unexpected first record: %#v", first.Record.Vertex)
	}
	if firstStream.nextCalls != 1 {
		t.Fatalf("stream read ahead: Next called %d times", firstStream.nextCalls)
	}
	token := first.Record.Vertex.Position.Token
	if token == "" || first.Record.Vertex.Position.Line != 1 {
		t.Fatalf("unexpected position: %#v", first.Record.Vertex.Position)
	}
	if err := iterator.Close(); err != nil {
		t.Fatal(err)
	}

	resumeStream := &fakeStream{records: []Record{
		record(map[string]any{"k": int64(2), "id": "b", "name": "Bob"}, "k", "id", "name"),
	}}
	resumeClient := &fakeClient{streams: []RecordStream{resumeStream}}
	resumed, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "people", Source: testSource(), Client: resumeClient,
		AfterToken: token, MaxRecordBytes: 1 << 20, MaxProperties: 20,
	})
	if err != nil {
		t.Fatal(err)
	}
	item, err := resumed.Next(context.Background())
	if err != nil || item.Record.Vertex.ExternalID != "b" {
		t.Fatalf("resumed Next = %#v, %v", item, err)
	}
	if got := resumeClient.parameters[0]["afterKey"]; got != int64(1) {
		t.Fatalf("afterKey = %#v", got)
	}
	if _, err := resumed.Next(context.Background()); !errors.Is(err, io.EOF) {
		t.Fatalf("end error = %v", err)
	}
	if resumeStream.closeCalls != 1 {
		t.Fatalf("stream close calls = %d", resumeStream.closeCalls)
	}
	if err := resumed.Close(); err != nil {
		t.Fatal(err)
	}
	if err := resumed.Close(); err != nil || resumeClient.closeCalls != 1 {
		t.Fatalf("idempotent Close = %v, calls %d", err, resumeClient.closeCalls)
	}
	detailed := resumed.DetailedTelemetry()
	if detailed.Queries != 1 || detailed.Records != 1 || detailed.Failures != 0 {
		t.Fatalf("telemetry = %#v", detailed)
	}
	if resumed.Telemetry().DecodedInputBytes == 0 {
		t.Fatalf("input byte telemetry = %#v", resumed.Telemetry())
	}
}

func TestIteratorPagesBoundedKeysetQueries(t *testing.T) {
	source := testSource()
	source.Vertices[0].Query += " LIMIT $pageRows"
	client := &fakeClient{streams: []RecordStream{
		&fakeStream{records: []Record{
			record(map[string]any{"k": int64(1), "id": "a", "name": "A"}, "k", "id", "name"),
			record(map[string]any{"k": int64(2), "id": "b", "name": "B"}, "k", "id", "name"),
		}},
		&fakeStream{records: []Record{
			record(map[string]any{"k": int64(3), "id": "c", "name": "C"}, "k", "id", "name"),
		}},
	}}
	iterator := newTestIterator(t, source, client)
	var ids []model.ExternalID
	for {
		item, err := iterator.Next(context.Background())
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, item.Record.Vertex.ExternalID)
	}
	if len(ids) != 3 {
		t.Fatalf("ids = %#v", ids)
	}
	if got := strings.Join([]string{string(ids[0]), string(ids[1]), string(ids[2])}, ","); got != "a,b,c" {
		t.Fatalf("ids = %q", got)
	}
	if len(client.parameters) != 2 ||
		client.parameters[0]["afterKey"] != nil ||
		client.parameters[0]["pageRows"] != 2 ||
		client.parameters[1]["afterKey"] != int64(2) ||
		client.parameters[1]["pageRows"] != 2 {
		t.Fatalf("parameters = %#v", client.parameters)
	}
}

func TestIteratorPreservesVertexThenEdgeOrder(t *testing.T) {
	source := testSource()
	source.Vertices = append(source.Vertices, config.VertexQuery{
		Label: "Team", KeyField: "k", IDField: "id",
		Query: "MATCH (n) WHERE $afterKey IS NULL OR n.k > $afterKey " +
			"RETURN n.k AS k, n.id AS id ORDER BY k",
	})
	source.Edges = []config.EdgeQuery{{
		Label: "MEMBER_OF", KeyField: "k", ExternalIDField: "edgeID",
		Query: "MATCH ()-[r]->() WHERE $afterKey IS NULL OR r.k > $afterKey " +
			"RETURN r.k AS k, r.id AS edgeID, r.start AS start, r.end AS end ORDER BY k",
		Start: config.EndpointMapping{Label: "Person", Field: "start"},
		End:   config.EndpointMapping{Label: "Team", Field: "end"},
	}}
	client := &fakeClient{streams: []RecordStream{
		&fakeStream{records: []Record{
			record(map[string]any{"k": int64(1), "id": "person", "name": "P"}, "k", "id", "name"),
		}},
		&fakeStream{records: []Record{
			record(map[string]any{"k": int64(10), "id": "team"}, "k", "id"),
		}},
		&fakeStream{records: []Record{
			record(map[string]any{
				"k": int64(20), "edgeID": 4.5, "start": "person", "end": int64(10),
			}, "k", "edgeID", "start", "end"),
		}},
	}}
	iterator := newTestIterator(t, source, client)
	var kinds []model.RecordKind
	for {
		item, err := iterator.Next(context.Background())
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		kinds = append(kinds, item.Record.Kind())
		if item.Record.Kind() == model.RecordEdge {
			if item.Record.Edge.ExternalID != "4.5" ||
				item.Record.Edge.End.ExternalID != "10" {
				t.Fatalf("edge IDs = %#v", item.Record.Edge)
			}
		}
	}
	if len(kinds) != 3 || kinds[0] != model.RecordVertex ||
		kinds[1] != model.RecordVertex || kinds[2] != model.RecordEdge {
		t.Fatalf("record kinds = %v", kinds)
	}
	if len(client.parameters) != 3 {
		t.Fatalf("queries = %d", len(client.parameters))
	}
	for index, parameters := range client.parameters {
		if value, ok := parameters["afterKey"]; !ok || value != nil {
			t.Fatalf("query %d afterKey = %#v", index, parameters)
		}
	}
}

func TestIteratorQuarantinesOnlyAfterValidKey(t *testing.T) {
	stream := &fakeStream{records: []Record{
		record(map[string]any{"k": int64(1), "name": "bad"}, "k", "name"),
		record(map[string]any{"k": int64(2), "id": "good", "name": "ok"}, "k", "id", "name"),
	}}
	var malformed []MalformedRecord
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "people", Source: testSource(),
		Client:      &fakeClient{streams: []RecordStream{stream}},
		RejectLimit: 1, MaxRecordBytes: 1 << 20, MaxProperties: 20,
		OnMalformed: func(_ context.Context, record MalformedRecord) error {
			malformed = append(malformed, record)
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	item, err := iterator.Next(context.Background())
	if err != nil || item.Record.Vertex.ExternalID != "good" {
		t.Fatalf("Next = %#v, %v", item, err)
	}
	if len(malformed) != 1 || malformed[0].Position.Line != 1 {
		t.Fatalf("malformed = %#v", malformed)
	}
	state, err := parseResumeToken(malformed[0].Position.Token)
	if err != nil || state.lastKey == nil || *state.lastKey != 1 || state.rejected != 1 {
		t.Fatalf("quarantine token = %#v, %v", state, err)
	}
	count, position := iterator.RejectionCheckpoint()
	if count != 1 || position.Line != 2 {
		t.Fatalf("checkpoint = %d %#v", count, position)
	}
}

func TestIteratorRejectsInvalidAndNonMonotonicKeys(t *testing.T) {
	tests := []struct {
		name    string
		records []Record
		want    string
	}{
		{
			name: "invalid", want: "signed 64-bit",
			records: []Record{record(map[string]any{"k": 1.0, "id": "a", "name": "A"}, "k", "id", "name")},
		},
		{
			name: "duplicate", want: "strictly increasing",
			records: []Record{
				record(map[string]any{"k": int64(1), "id": "a", "name": "A"}, "k", "id", "name"),
				record(map[string]any{"k": int64(1), "id": "b", "name": "B"}, "k", "id", "name"),
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			called := false
			budget := sourcecontract.NewProfileBudget(
				sourcecontract.ProfileBudgetLimits{
					Rows: 10, Pages: 10, DecodedInputBytes: 1 << 20,
				},
			)
			iterator, err := NewIterator(context.Background(), IteratorOptions{
				Namespace: "people", Source: testSource(),
				Client:      &fakeClient{streams: []RecordStream{&fakeStream{records: test.records}}},
				RejectLimit: 5, MaxRecordBytes: 1 << 20, MaxProperties: 20,
				ProfileBudget: budget,
				OnMalformed: func(context.Context, MalformedRecord) error {
					called = true
					return nil
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			var nextErr error
			for nextErr == nil {
				_, nextErr = iterator.Next(context.Background())
			}
			if !strings.Contains(nextErr.Error(), test.want) {
				t.Fatalf("error = %v", nextErr)
			}
			if _, repeated := iterator.Next(context.Background()); repeated == nil ||
				repeated.Error() != nextErr.Error() {
				t.Fatalf("fatal error was not sticky: first=%v repeated=%v", nextErr, repeated)
			}
			if called {
				t.Fatal("invalid key was quarantined")
			}
			usage, _ := budget.Snapshot()
			if usage.Rows != int64(len(test.records)) || usage.DecodedInputBytes == 0 {
				t.Fatalf("invalid-key usage = %#v", usage)
			}
		})
	}
}

func TestIteratorCancellationAndSanitizedFailures(t *testing.T) {
	t.Run("cancellation", func(t *testing.T) {
		iterator := newTestIterator(t, testSource(), &fakeClient{
			streams: []RecordStream{&fakeStream{block: true}},
		})
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := iterator.Next(ctx); !errors.Is(err, context.Canceled) {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("query error", func(t *testing.T) {
		secret := "password=do-not-leak"
		iterator := newTestIterator(t, testSource(), &fakeClient{
			queryErr: errors.New(secret),
		})
		_, err := iterator.Next(context.Background())
		if err == nil || strings.Contains(err.Error(), secret) {
			t.Fatalf("unsafe error = %v", err)
		}
		if iterator.Telemetry().FailedRequestAttempts != 1 {
			t.Fatalf("telemetry = %#v", iterator.Telemetry())
		}
		if _, repeated := iterator.Next(context.Background()); repeated == nil ||
			repeated.Error() != err.Error() {
			t.Fatalf("query error was not sticky: %v", repeated)
		}
	})
	t.Run("stream error", func(t *testing.T) {
		secret := "record-value-secret"
		iterator := newTestIterator(t, testSource(), &fakeClient{
			streams: []RecordStream{&fakeStream{nextErr: errors.New(secret)}},
		})
		_, err := iterator.Next(context.Background())
		if err == nil || strings.Contains(err.Error(), secret) {
			t.Fatalf("unsafe error = %v", err)
		}
	})
	t.Run("partial stream error", func(t *testing.T) {
		secret := "server record secret"
		iterator := newTestIterator(t, testSource(), &fakeClient{
			streams: []RecordStream{&fakeStream{
				records: []Record{
					record(map[string]any{
						"k": int64(1), "id": "one", "name": "One",
					}, "k", "id", "name"),
				},
				nextErr: errors.New(secret),
			}},
		})
		if _, err := iterator.Next(context.Background()); err != nil {
			t.Fatal(err)
		}
		_, err := iterator.Next(context.Background())
		if err == nil || strings.Contains(err.Error(), secret) {
			t.Fatalf("partial result error = %v", err)
		}
		if _, repeated := iterator.Next(context.Background()); repeated == nil ||
			repeated.Error() != err.Error() {
			t.Fatalf("partial failure was not sticky: %v", repeated)
		}
	})
}

func TestIteratorLimitsPreencodingAndCloseErrors(t *testing.T) {
	t.Run("record bytes quarantined", func(t *testing.T) {
		handled := 0
		source := testSource()
		iterator, err := NewIterator(context.Background(), IteratorOptions{
			Namespace: "people", Source: source,
			Client: &fakeClient{streams: []RecordStream{&fakeStream{records: []Record{
				record(map[string]any{
					"k": int64(1), "id": "bad", "name": strings.Repeat("x", 100),
				}, "k", "id", "name"),
				record(map[string]any{"k": int64(2), "id": "ok", "name": "x"}, "k", "id", "name"),
			}}}},
			RejectLimit: 1, MaxRecordBytes: 150, MaxProperties: 20,
			OnMalformed: func(context.Context, MalformedRecord) error {
				handled++
				return nil
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		item, err := iterator.Next(context.Background())
		if err != nil || item.Record.Vertex.ExternalID != "ok" || handled != 1 {
			t.Fatalf("Next = %#v, %v, handled=%d", item, err, handled)
		}
	})
	t.Run("preencode", func(t *testing.T) {
		iterator, err := NewIterator(context.Background(), IteratorOptions{
			Namespace: "people", Source: testSource(),
			Client: &fakeClient{streams: []RecordStream{&fakeStream{records: []Record{
				record(map[string]any{"k": int64(1), "id": "a", "name": "A"}, "k", "id", "name"),
			}}}},
			PreencodeProperties: true, MaxRecordBytes: 1 << 20, MaxProperties: 20,
		})
		if err != nil {
			t.Fatal(err)
		}
		item, err := iterator.Next(context.Background())
		if err != nil || item.Record.Vertex.Properties != nil ||
			len(item.Record.Vertex.EncodedProperties) == 0 {
			t.Fatalf("preencoded item = %#v, %v", item, err)
		}
	})
	t.Run("close error sanitized", func(t *testing.T) {
		client := &fakeClient{closeErr: errors.New("password secret")}
		iterator := newTestIterator(t, testSource(), client)
		err := iterator.Close()
		if err == nil || strings.Contains(err.Error(), "password secret") {
			t.Fatalf("Close error = %v", err)
		}
	})
}

func TestNewIteratorValidation(t *testing.T) {
	valid := testSource()
	tests := []struct {
		name string
		edit func(*IteratorOptions)
	}{
		{"nil client", func(options *IteratorOptions) { options.Client = nil }},
		{"source id", func(options *IteratorOptions) { options.Source.SourceID = "" }},
		{"uri", func(options *IteratorOptions) { options.Source.URI = "" }},
		{"database", func(options *IteratorOptions) { options.Source.Database = "" }},
		{"fetch rows", func(options *IteratorOptions) { options.Source.FetchRows = 0 }},
		{"policy", func(options *IteratorOptions) { options.Source.MultiLabelPolicy = "first" }},
		{"reject limit", func(options *IteratorOptions) { options.RejectLimit = -1 }},
		{"handler", func(options *IteratorOptions) { options.RejectLimit = 1 }},
		{"record bytes", func(options *IteratorOptions) { options.MaxRecordBytes = -1 }},
		{"properties", func(options *IteratorOptions) { options.MaxProperties = -1 }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			options := IteratorOptions{
				Namespace: "people", Source: valid, Client: &fakeClient{},
				MaxRecordBytes: 10, MaxProperties: 10,
			}
			test.edit(&options)
			if _, err := NewIterator(context.Background(), options); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
	if _, err := NewIterator(nil, IteratorOptions{}); err == nil {
		t.Fatal("expected nil context error")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewIterator(ctx, IteratorOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled constructor error = %v", err)
	}
}

func TestIteratorCloseCancelsBlockedNext(t *testing.T) {
	started := make(chan struct{})
	stream := &fakeStream{block: true, started: started}
	iterator := newTestIterator(t, testSource(), &fakeClient{
		streams: []RecordStream{stream},
	})
	nextDone := make(chan error, 1)
	go func() {
		_, err := iterator.Next(context.Background())
		nextDone <- err
	}()
	<-started
	if err := iterator.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-nextDone:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("blocked Next() error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Close() did not unblock Next()")
	}
}
