package neo4j

import (
	"context"
	"errors"
	"fmt"
	"io"
	"regexp"
	"sync"
	"time"

	neodriver "github.com/neo4j/neo4j-go-driver/v6/neo4j"
)

// Record is the minimal record surface needed by Iterator. Keys are returned
// column names, not expressions from the query text.
type Record interface {
	Get(string) (any, bool)
	Keys() []string
}

// RecordStream is a forward-only, bounded stream. Next must return io.EOF
// after the result is exhausted.
type RecordStream interface {
	Next(context.Context) (Record, error)
	Close(context.Context) error
}

// Client opens one auto-commit read query stream per configured mapping.
type Client interface {
	Query(context.Context, string, map[string]any) (RecordStream, error)
	Close() error
}

// SDKClient adapts the official Neo4j v6 driver to Client.
type SDKClient struct {
	newSession  func(context.Context, neodriver.SessionConfig) sdkSession
	closeDriver func(context.Context) error
	database    string
	fetchRows   int
	lifetime    context.Context
	cancel      context.CancelFunc

	mu        sync.Mutex
	streams   map[*sdkStream]struct{}
	opening   sync.WaitGroup
	closed    bool
	closeDone chan struct{}
	closeErr  error
}

// NewSDKClient creates, verifies, and owns an official Neo4j driver.
func NewSDKClient(
	ctx context.Context,
	uri, database, username, password string,
	fetchRows int,
) (Client, error) {
	if ctx == nil {
		return nil, errors.New("Neo4j client context is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if database == "" {
		return nil, errors.New("Neo4j database is required")
	}
	if fetchRows < 1 || fetchRows > 100_000 {
		return nil, errors.New("Neo4j fetch rows must be between 1 and 100000")
	}
	auth := neodriver.NoAuth()
	if username != "" {
		auth = neodriver.BasicAuth(username, password, "")
	}
	driver, err := neodriver.NewDriverWithContext(uri, auth)
	if err != nil {
		return nil, safeError(ctx, "create Neo4j driver", err)
	}
	if err := driver.VerifyConnectivity(ctx); err != nil {
		_ = driver.Close(context.Background())
		return nil, safeError(ctx, "verify Neo4j connectivity", err)
	}
	lifetime, cancel := context.WithCancel(context.Background())
	return &SDKClient{
		newSession: func(ctx context.Context, config neodriver.SessionConfig) sdkSession {
			return driverSession{Session: driver.NewSession(ctx, config)}
		},
		closeDriver: driver.Close,
		database:    database,
		fetchRows:   fetchRows,
		lifetime:    lifetime,
		cancel:      cancel,
		streams:     make(map[*sdkStream]struct{}),
		closeDone:   make(chan struct{}),
	}, nil
}

func (client *SDKClient) Query(
	ctx context.Context,
	query string,
	parameters map[string]any,
) (RecordStream, error) {
	if ctx == nil {
		return nil, errors.New("Neo4j query context is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	client.mu.Lock()
	if client.closed {
		client.mu.Unlock()
		return nil, errors.New("Neo4j client is closed")
	}
	client.opening.Add(1)
	session := client.newSession(ctx, neodriver.SessionConfig{
		AccessMode:   neodriver.AccessModeRead,
		DatabaseName: client.database,
		FetchSize:    client.fetchRows,
	})
	client.mu.Unlock()
	defer client.opening.Done()

	opCtx, cancel := context.WithCancel(ctx)
	stop := context.AfterFunc(client.lifetime, cancel)
	defer func() {
		stop()
		cancel()
	}()
	result, err := session.Run(opCtx, query, parameters)
	if err != nil {
		closeCtx, closeCancel := context.WithTimeout(
			context.Background(), 5*time.Second,
		)
		_ = session.Close(closeCtx)
		closeCancel()
		return nil, safeError(opCtx, "run Neo4j query", err)
	}
	stream := newSDKStream(client, session, result)
	client.mu.Lock()
	if client.closed {
		client.mu.Unlock()
		_ = stream.Close(context.Background())
		return nil, errors.New("Neo4j client is closed")
	}
	client.streams[stream] = struct{}{}
	client.mu.Unlock()
	return stream, nil
}

func (client *SDKClient) Close() error {
	client.mu.Lock()
	if client.closed {
		done := client.closeDone
		client.mu.Unlock()
		<-done
		client.mu.Lock()
		err := client.closeErr
		client.mu.Unlock()
		return err
	}
	client.closed = true
	client.cancel()
	client.mu.Unlock()
	client.opening.Wait()

	client.mu.Lock()
	streams := make([]*sdkStream, 0, len(client.streams))
	for stream := range client.streams {
		streams = append(streams, stream)
	}
	client.mu.Unlock()

	var closeErr error
	for _, stream := range streams {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		closeErr = errors.Join(closeErr, stream.Close(ctx))
		cancel()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := client.closeDriver(ctx); err != nil {
		closeErr = errors.Join(closeErr, safeError(nil, "close Neo4j driver", err))
	}
	client.mu.Lock()
	client.closeErr = closeErr
	close(client.closeDone)
	client.mu.Unlock()
	return closeErr
}

type sdkRecord struct {
	record *neodriver.Record
}

func (record sdkRecord) Get(name string) (any, bool) {
	return record.record.Get(name)
}

func (record sdkRecord) Keys() []string {
	return record.record.Keys
}

type sdkStream struct {
	client    *SDKClient
	session   sdkSession
	result    sdkResult
	lifetime  context.Context
	cancel    context.CancelFunc
	operation chan struct{}

	mu        sync.Mutex
	closed    bool
	closeDone chan struct{}
	closeErr  error
}

func newSDKStream(
	client *SDKClient,
	session sdkSession,
	result sdkResult,
) *sdkStream {
	lifetime, cancel := context.WithCancel(context.Background())
	operation := make(chan struct{}, 1)
	operation <- struct{}{}
	return &sdkStream{
		client: client, session: session, result: result,
		lifetime: lifetime, cancel: cancel, operation: operation,
		closeDone: make(chan struct{}),
	}
}

type sdkSession interface {
	Run(
		context.Context,
		string,
		map[string]any,
		...func(*neodriver.TransactionConfig),
	) (sdkResult, error)
	Close(context.Context) error
}

type sdkResult interface {
	NextRecord(context.Context, **neodriver.Record) bool
	Err() error
}

type driverSession struct {
	neodriver.Session
}

func (session driverSession) Run(
	ctx context.Context,
	query string,
	parameters map[string]any,
	configurers ...func(*neodriver.TransactionConfig),
) (sdkResult, error) {
	return session.Session.Run(ctx, query, parameters, configurers...)
}

func (stream *sdkStream) Next(ctx context.Context) (Record, error) {
	if ctx == nil {
		return nil, errors.New("Neo4j stream context is required")
	}
	stream.mu.Lock()
	if stream.closed {
		stream.mu.Unlock()
		return nil, errors.New("Neo4j record stream is closed")
	}
	stream.mu.Unlock()
	select {
	case <-stream.operation:
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-stream.lifetime.Done():
		return nil, errors.New("Neo4j record stream is closed")
	}
	defer func() { stream.operation <- struct{}{} }()
	stream.mu.Lock()
	if stream.closed {
		stream.mu.Unlock()
		return nil, errors.New("Neo4j record stream is closed")
	}
	stream.mu.Unlock()

	opCtx, cancel := context.WithCancel(ctx)
	stop := context.AfterFunc(stream.lifetime, cancel)
	defer func() {
		stop()
		cancel()
	}()
	var record *neodriver.Record
	if stream.result.NextRecord(opCtx, &record) {
		return sdkRecord{record: record}, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := stream.lifetime.Err(); err != nil {
		return nil, errors.New("Neo4j record stream is closed")
	}
	if err := stream.result.Err(); err != nil {
		return nil, safeError(opCtx, "read Neo4j query result", err)
	}
	return nil, io.EOF
}

func (stream *sdkStream) Close(ctx context.Context) error {
	if ctx == nil {
		return errors.New("Neo4j stream close context is required")
	}
	stream.mu.Lock()
	if stream.closed {
		done := stream.closeDone
		stream.mu.Unlock()
		select {
		case <-done:
			stream.mu.Lock()
			err := stream.closeErr
			stream.mu.Unlock()
			return err
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	stream.closed = true
	stream.cancel()
	stream.mu.Unlock()

	closeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	select {
	case <-stream.operation:
	case <-closeCtx.Done():
		err := safeError(closeCtx, "close Neo4j session", closeCtx.Err())
		go stream.finishCloseAfterOperation(err)
		return err
	}
	err := stream.closeSession(closeCtx)
	stream.operation <- struct{}{}
	stream.finishClose(err)
	return err
}

func (stream *sdkStream) finishCloseAfterOperation(prior error) {
	<-stream.operation
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err := errors.Join(prior, stream.closeSession(ctx))
	stream.operation <- struct{}{}
	stream.finishClose(err)
}

func (stream *sdkStream) closeSession(ctx context.Context) error {
	if err := stream.session.Close(ctx); err != nil {
		return safeError(ctx, "close Neo4j session", err)
	}
	return nil
}

func (stream *sdkStream) finishClose(err error) {
	stream.client.mu.Lock()
	delete(stream.client.streams, stream)
	stream.client.mu.Unlock()
	stream.mu.Lock()
	stream.closeErr = err
	close(stream.closeDone)
	stream.mu.Unlock()
}

var safeCodePattern = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,256}$`)

type sanitizedError struct {
	message string
}

func (err *sanitizedError) Error() string {
	return err.message
}

func safeError(ctx context.Context, action string, err error) error {
	if ctx != nil {
		if contextErr := ctx.Err(); contextErr != nil {
			return contextErr
		}
	}
	if errors.Is(err, context.Canceled) {
		return context.Canceled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return context.DeadlineExceeded
	}
	var alreadySafe *sanitizedError
	if errors.As(err, &alreadySafe) {
		return alreadySafe
	}
	var serverError *neodriver.Neo4jError
	if errors.As(err, &serverError) && safeCodePattern.MatchString(serverError.Code) {
		return &sanitizedError{
			message: fmt.Sprintf("%s failed (code %s)", action, serverError.Code),
		}
	}
	return &sanitizedError{message: fmt.Sprintf("%s failed", action)}
}

var _ Client = (*SDKClient)(nil)
