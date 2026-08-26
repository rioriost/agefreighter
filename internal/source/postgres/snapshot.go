package postgres

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

var snapshotIDPattern = regexp.MustCompile(`^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{8}-[0-9]+$`)

// SnapshotCoordinator owns an exported PostgreSQL snapshot and creates
// bounded reader transactions that import it before executing any query.
type SnapshotCoordinator struct {
	dsn      string
	snapshot string
	owner    *pgx.Conn
	ownerTx  pgx.Tx

	ctx    context.Context
	cancel context.CancelFunc
	slots  chan struct{}

	mu      sync.Mutex
	closed  bool
	readers map[*SnapshotReader]struct{}
	opening sync.WaitGroup
	once    sync.Once
	err     error
}

// SnapshotReader is a read-only transaction attached to a coordinator's
// exported snapshot.
type SnapshotReader struct {
	coordinator *SnapshotCoordinator
	conn        *pgx.Conn
	tx          pgx.Tx
	once        sync.Once
	err         error
}

// NewSnapshotCoordinator starts the repeatable-read, read-only transaction
// that owns the exported snapshot.
func NewSnapshotCoordinator(
	ctx context.Context,
	dsn string,
	maxReaders int,
) (*SnapshotCoordinator, error) {
	if ctx == nil {
		return nil, errors.New("PostgreSQL snapshot context is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if dsn == "" {
		return nil, errors.New("PostgreSQL DSN is required")
	}
	if maxReaders < 1 || maxReaders > 256 {
		return nil, errors.New("PostgreSQL maximum readers must be between 1 and 256")
	}
	owner, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return nil, safeDatabaseError(ctx, "connect PostgreSQL snapshot owner", err)
	}
	tx, err := owner.BeginTx(ctx, pgx.TxOptions{
		IsoLevel:   pgx.RepeatableRead,
		AccessMode: pgx.ReadOnly,
	})
	if err != nil {
		_ = owner.Close(context.Background())
		return nil, safeDatabaseError(ctx, "begin PostgreSQL snapshot owner", err)
	}
	var snapshot string
	if err := tx.QueryRow(ctx, "SELECT pg_export_snapshot()").Scan(&snapshot); err != nil {
		_ = tx.Rollback(context.Background())
		_ = owner.Close(context.Background())
		return nil, safeDatabaseError(ctx, "export PostgreSQL snapshot", err)
	}
	if !validSnapshotID(snapshot) {
		_ = tx.Rollback(context.Background())
		_ = owner.Close(context.Background())
		return nil, errors.New("PostgreSQL returned an invalid exported snapshot identifier")
	}
	lifetime, cancel := context.WithCancel(context.Background())
	return &SnapshotCoordinator{
		dsn: dsn, snapshot: snapshot, owner: owner, ownerTx: tx,
		ctx: lifetime, cancel: cancel, slots: make(chan struct{}, maxReaders),
		readers: make(map[*SnapshotReader]struct{}),
	}, nil
}

func validSnapshotID(snapshot string) bool {
	return snapshotIDPattern.MatchString(snapshot)
}

// SnapshotID returns the validated exported snapshot identifier.
func (coordinator *SnapshotCoordinator) SnapshotID() string {
	return coordinator.snapshot
}

// OpenReader creates a transaction and imports the exported snapshot as its
// first statement after BEGIN.
func (coordinator *SnapshotCoordinator) OpenReader(
	ctx context.Context,
) (*SnapshotReader, error) {
	if ctx == nil {
		return nil, errors.New("PostgreSQL reader context is required")
	}
	select {
	case coordinator.slots <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-coordinator.ctx.Done():
		return nil, errors.New("PostgreSQL snapshot coordinator is closed")
	}

	coordinator.mu.Lock()
	if coordinator.closed {
		coordinator.mu.Unlock()
		<-coordinator.slots
		return nil, errors.New("PostgreSQL snapshot coordinator is closed")
	}
	coordinator.opening.Add(1)
	coordinator.mu.Unlock()
	defer coordinator.opening.Done()

	opCtx, cancel := context.WithCancel(ctx)
	stop := context.AfterFunc(coordinator.ctx, cancel)
	defer func() {
		stop()
		cancel()
	}()

	conn, err := pgx.Connect(opCtx, coordinator.dsn)
	if err != nil {
		<-coordinator.slots
		return nil, safeDatabaseError(opCtx, "connect PostgreSQL snapshot reader", err)
	}
	tx, err := conn.BeginTx(opCtx, pgx.TxOptions{
		IsoLevel:   pgx.RepeatableRead,
		AccessMode: pgx.ReadOnly,
	})
	if err != nil {
		_ = conn.Close(context.Background())
		<-coordinator.slots
		return nil, safeDatabaseError(opCtx, "begin PostgreSQL snapshot reader", err)
	}
	statement := fmt.Sprintf("SET TRANSACTION SNAPSHOT '%s'", coordinator.snapshot)
	if _, err := tx.Exec(opCtx, statement); err != nil {
		_ = tx.Rollback(context.Background())
		_ = conn.Close(context.Background())
		<-coordinator.slots
		return nil, safeDatabaseError(opCtx, "import PostgreSQL snapshot", err)
	}
	reader := &SnapshotReader{coordinator: coordinator, conn: conn, tx: tx}
	coordinator.mu.Lock()
	if coordinator.closed {
		coordinator.mu.Unlock()
		_ = reader.close(false)
		return nil, errors.New("PostgreSQL snapshot coordinator is closed")
	}
	coordinator.readers[reader] = struct{}{}
	coordinator.mu.Unlock()
	return reader, nil
}

// Tx exposes the imported read-only transaction for integration tests and
// package readers.
func (reader *SnapshotReader) Tx() pgx.Tx {
	return reader.tx
}

// Close rolls back the reader transaction and releases its bounded slot.
func (reader *SnapshotReader) Close() error {
	return reader.close(true)
}

func (reader *SnapshotReader) close(remove bool) error {
	reader.once.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		rollbackErr := reader.tx.Rollback(ctx)
		if errors.Is(rollbackErr, pgx.ErrTxClosed) {
			rollbackErr = nil
		}
		closeErr := reader.conn.Close(ctx)
		reader.err = errors.Join(rollbackErr, closeErr)
		if remove {
			reader.coordinator.mu.Lock()
			delete(reader.coordinator.readers, reader)
			reader.coordinator.mu.Unlock()
		}
		<-reader.coordinator.slots
	})
	return reader.err
}

// Close closes every reader before releasing the snapshot owner.
func (coordinator *SnapshotCoordinator) Close() error {
	coordinator.once.Do(func() {
		coordinator.mu.Lock()
		coordinator.closed = true
		coordinator.cancel()
		readers := make([]*SnapshotReader, 0, len(coordinator.readers))
		for reader := range coordinator.readers {
			readers = append(readers, reader)
		}
		coordinator.mu.Unlock()
		for _, reader := range readers {
			_ = reader.Close()
		}
		coordinator.opening.Wait()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		rollbackErr := coordinator.ownerTx.Rollback(ctx)
		if errors.Is(rollbackErr, pgx.ErrTxClosed) {
			rollbackErr = nil
		}
		coordinator.err = errors.Join(rollbackErr, coordinator.owner.Close(ctx))
	})
	return coordinator.err
}

func safeDatabaseError(ctx context.Context, operation string, err error) error {
	if ctx != nil && ctx.Err() != nil {
		return fmt.Errorf("%s: %w", operation, ctx.Err())
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("%s: %w", operation, err)
	}
	var databaseError *pgconn.PgError
	if errors.As(err, &databaseError) {
		return fmt.Errorf(
			"%s failed (SQLSTATE %s)",
			operation,
			databaseError.Code,
		)
	}
	return fmt.Errorf("%s failed", operation)
}
