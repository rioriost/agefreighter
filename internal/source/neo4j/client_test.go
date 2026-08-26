package neo4j

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	neodriver "github.com/neo4j/neo4j-go-driver/v6/neo4j"
)

type fakeSDKResult struct {
	records []*neodriver.Record
	index   int
	err     error
}

type blockingSDKResult struct {
	started chan struct{}
	once    sync.Once
}

type stubbornSDKResult struct {
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func (result *stubbornSDKResult) NextRecord(
	_ context.Context,
	_ **neodriver.Record,
) bool {
	result.once.Do(func() { close(result.started) })
	<-result.release
	return false
}

func (result *stubbornSDKResult) Err() error {
	return nil
}

func (result *blockingSDKResult) NextRecord(
	ctx context.Context,
	_ **neodriver.Record,
) bool {
	result.once.Do(func() { close(result.started) })
	<-ctx.Done()
	return false
}

func (result *blockingSDKResult) Err() error {
	return nil
}

func (result *fakeSDKResult) NextRecord(
	_ context.Context,
	target **neodriver.Record,
) bool {
	if result.index >= len(result.records) {
		return false
	}
	*target = result.records[result.index]
	result.index++
	return true
}

func (result *fakeSDKResult) Err() error {
	return result.err
}

type fakeSDKSession struct {
	result     sdkResult
	runErr     error
	closeErr   error
	query      string
	parameters map[string]any
	closeCalls int
}

func (session *fakeSDKSession) Run(
	_ context.Context,
	query string,
	parameters map[string]any,
	_ ...func(*neodriver.TransactionConfig),
) (sdkResult, error) {
	session.query = query
	session.parameters = parameters
	return session.result, session.runErr
}

func (session *fakeSDKSession) Close(context.Context) error {
	session.closeCalls++
	return session.closeErr
}

func fakeOfficialRecord(values map[string]any, keys ...string) *neodriver.Record {
	record := &neodriver.Record{Keys: keys, Values: make([]any, len(keys))}
	for index, key := range keys {
		record.Values[index] = values[key]
	}
	return record
}

func newFakeSDKClient(session sdkSession) (*SDKClient, *neodriver.SessionConfig, *int) {
	var gotConfig neodriver.SessionConfig
	driverCloses := 0
	lifetime, cancel := context.WithCancel(context.Background())
	client := &SDKClient{
		newSession: func(_ context.Context, config neodriver.SessionConfig) sdkSession {
			gotConfig = config
			return session
		},
		closeDriver: func(context.Context) error {
			driverCloses++
			return nil
		},
		database: "movies", fetchRows: 17,
		lifetime:  lifetime,
		cancel:    cancel,
		streams:   make(map[*sdkStream]struct{}),
		closeDone: make(chan struct{}),
	}
	return client, &gotConfig, &driverCloses
}

func TestSDKClientStreamsOfficialRecords(t *testing.T) {
	result := &fakeSDKResult{records: []*neodriver.Record{
		fakeOfficialRecord(map[string]any{"k": int64(1), "name": "one"}, "k", "name"),
		fakeOfficialRecord(map[string]any{"k": int64(2), "name": "two"}, "k", "name"),
	}}
	session := &fakeSDKSession{result: result}
	client, config, driverCloses := newFakeSDKClient(session)
	if _, err := client.Query(nil, "RETURN rows", nil); err == nil {
		t.Fatal("Query() accepted nil context")
	}
	stream, err := client.Query(context.Background(), "RETURN rows", map[string]any{"afterKey": nil})
	if err != nil {
		t.Fatal(err)
	}
	if config.AccessMode != neodriver.AccessModeRead ||
		config.DatabaseName != "movies" || config.FetchSize != 17 {
		t.Fatalf("session config = %#v", config)
	}
	if session.query != "RETURN rows" || session.parameters["afterKey"] != nil {
		t.Fatalf("run = %q %#v", session.query, session.parameters)
	}
	for index, expected := range []string{"one", "two"} {
		current, err := stream.Next(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		value, ok := current.Get("name")
		if !ok || value != expected || len(current.Keys()) != 2 {
			t.Fatalf("record %d = %#v %v", index, value, ok)
		}
	}
	if _, err := stream.Next(context.Background()); !errors.Is(err, io.EOF) {
		t.Fatalf("end error = %v", err)
	}
	if err := stream.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := stream.Close(context.Background()); err != nil || session.closeCalls != 1 {
		t.Fatalf("idempotent stream close = %v, calls %d", err, session.closeCalls)
	}
	if err := client.Close(); err != nil {
		t.Fatal(err)
	}
	if err := client.Close(); err != nil || *driverCloses != 1 {
		t.Fatalf("idempotent client close = %v, driver closes %d", err, *driverCloses)
	}
	if _, err := client.Query(context.Background(), "RETURN 1", nil); err == nil {
		t.Fatal("closed client accepted query")
	}
}

func TestSDKClientCleansActiveStreams(t *testing.T) {
	session := &fakeSDKSession{result: &fakeSDKResult{}}
	client, _, driverCloses := newFakeSDKClient(session)
	if _, err := client.Query(context.Background(), "RETURN 1", nil); err != nil {
		t.Fatal(err)
	}

	if err := client.Close(); err != nil {
		t.Fatal(err)
	}
	if session.closeCalls != 1 || *driverCloses != 1 || len(client.streams) != 0 {
		t.Fatalf("cleanup: session=%d driver=%d streams=%d",
			session.closeCalls, *driverCloses, len(client.streams))
	}
}

func TestSDKClientCloseCancelsBlockedStream(t *testing.T) {
	result := &blockingSDKResult{started: make(chan struct{})}
	session := &fakeSDKSession{result: result}
	client, _, _ := newFakeSDKClient(session)
	stream, err := client.Query(context.Background(), "RETURN 1", nil)
	if err != nil {
		t.Fatal(err)
	}

	nextDone := make(chan error, 1)
	go func() {
		_, err := stream.Next(context.Background())
		nextDone <- err
	}()
	<-result.started
	if err := client.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-nextDone:
		if err == nil {
			t.Fatal("blocked Next() returned no error after close")
		}
	case <-time.After(time.Second):
		t.Fatal("client close did not unblock Next()")
	}
}

func TestSDKStreamTimeoutCompletesDeferredCleanup(t *testing.T) {
	result := &stubbornSDKResult{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	session := &fakeSDKSession{result: result}
	client, _, _ := newFakeSDKClient(session)
	stream, err := client.Query(context.Background(), "RETURN 1", nil)
	if err != nil {
		t.Fatal(err)
	}
	nextDone := make(chan error, 1)
	go func() {
		_, err := stream.Next(context.Background())
		nextDone <- err
	}()
	<-result.started
	closeCtx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if err := stream.Close(closeCtx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Close() error = %v", err)
	}
	if session.closeCalls != 0 {
		t.Fatalf("session closed before active read ended: %d", session.closeCalls)
	}
	close(result.release)
	<-nextDone
	if err := stream.Close(context.Background()); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("deferred Close() error = %v", err)
	}
	if session.closeCalls != 1 || len(client.streams) != 0 {
		t.Fatalf("deferred cleanup: closes=%d streams=%d",
			session.closeCalls, len(client.streams))
	}
}

func TestSDKAdapterSanitizesFailuresAndCancellation(t *testing.T) {
	t.Run("run", func(t *testing.T) {
		session := &fakeSDKSession{runErr: errors.New("query param secret")}
		client, _, _ := newFakeSDKClient(session)
		_, err := client.Query(context.Background(), "RETURN 1", nil)
		if err == nil || strings.Contains(err.Error(), "secret") || session.closeCalls != 1 {
			t.Fatalf("query error = %v, closes=%d", err, session.closeCalls)
		}
	})
	t.Run("result", func(t *testing.T) {
		session := &fakeSDKSession{result: &fakeSDKResult{err: errors.New("record secret")}}
		client, _, _ := newFakeSDKClient(session)
		stream, err := client.Query(context.Background(), "RETURN 1", nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = stream.Next(context.Background())
		if err == nil || strings.Contains(err.Error(), "secret") {
			t.Fatalf("result error = %v", err)
		}
	})
	t.Run("safe code survives adapter layers", func(t *testing.T) {
		code := "Neo.TransientError.General.DatabaseUnavailable"
		session := &fakeSDKSession{result: &fakeSDKResult{
			err: &neodriver.Neo4jError{Code: code, Msg: "server secret"},
		}}
		client, _, _ := newFakeSDKClient(session)
		stream, err := client.Query(context.Background(), "RETURN 1", nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = stream.Next(context.Background())
		err = safeError(context.Background(), "outer", err)
		if err == nil || !strings.Contains(err.Error(), code) ||
			strings.Contains(err.Error(), "secret") {
			t.Fatalf("layered error = %v", err)
		}
	})
	t.Run("session close", func(t *testing.T) {
		session := &fakeSDKSession{
			result: &fakeSDKResult{}, closeErr: errors.New("close secret"),
		}
		client, _, _ := newFakeSDKClient(session)
		stream, err := client.Query(context.Background(), "RETURN 1", nil)
		if err != nil {
			t.Fatal(err)
		}
		err = stream.Close(context.Background())
		if err == nil || strings.Contains(err.Error(), "secret") {
			t.Fatalf("close error = %v", err)
		}
	})
	t.Run("driver close", func(t *testing.T) {
		client, _, _ := newFakeSDKClient(&fakeSDKSession{})
		client.closeDriver = func(context.Context) error {
			return errors.New("driver secret")
		}
		err := client.Close()
		if err == nil || strings.Contains(err.Error(), "secret") {
			t.Fatalf("driver close error = %v", err)
		}
	})
	t.Run("cancelled", func(t *testing.T) {
		session := &fakeSDKSession{result: &fakeSDKResult{}}
		client, _, _ := newFakeSDKClient(session)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := client.Query(ctx, "RETURN 1", nil); !errors.Is(err, context.Canceled) {
			t.Fatalf("query cancellation = %v", err)
		}
		stream := newSDKStream(client, session, session.result)
		if _, err := stream.Next(ctx); !errors.Is(err, context.Canceled) {
			t.Fatalf("next cancellation = %v", err)
		}
	})
}
