package age

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

type PoolOptions struct {
	MinConnections   int32
	MaxConnections   int32
	ConnectTimeout   time.Duration
	OperationTimeout time.Duration
}

type Capabilities struct {
	PostgreSQLVersionNumber int
	PostgreSQLMajor         int
	AGEVersion              Version
	AGEPreloadStatus        PreloadStatus
	CurrentUser             string
	CurrentUserSuperuser    bool
}

type PreloadStatus string

const (
	PreloadConfigured    PreloadStatus = "configured"
	PreloadNotConfigured PreloadStatus = "not-configured"
	PreloadUnknown       PreloadStatus = "unknown"
)

type Adapter struct {
	pool             *pgxpool.Pool
	capabilities     Capabilities
	operationTimeout time.Duration
	loadSlotOnce     sync.Once
	loadSlots        chan struct{}
}

func Open(ctx context.Context, dsn string, options PoolOptions) (*Adapter, error) {
	if strings.TrimSpace(dsn) == "" {
		return nil, errors.New("Apache AGE connection string is required")
	}
	if options.MinConnections < 0 {
		return nil, errors.New("minimum connections cannot be negative")
	}
	if options.MaxConnections <= 0 ||
		options.MinConnections > options.MaxConnections {
		return nil, errors.New("maximum connections must be positive and at least the minimum")
	}
	if options.ConnectTimeout <= 0 {
		return nil, errors.New("connect timeout must be positive")
	}
	if options.OperationTimeout <= 0 {
		return nil, errors.New("operation timeout must be positive")
	}

	config, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("parse Apache AGE connection string: %w", err)
	}
	config.MinConns = options.MinConnections
	config.MaxConns = options.MaxConnections
	config.ConnConfig.ConnectTimeout = options.ConnectTimeout
	config.AfterConnect = initializeSession

	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("create Apache AGE pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("connect to Apache AGE: %w", err)
	}
	capabilities, err := ProbeCapabilities(ctx, pool)
	if err != nil {
		pool.Close()
		return nil, err
	}
	if err := ValidateVersions(
		capabilities.PostgreSQLVersionNumber,
		capabilities.AGEVersion,
	); err != nil {
		pool.Close()
		return nil, err
	}
	return &Adapter{
		pool:             pool,
		capabilities:     capabilities,
		operationTimeout: options.OperationTimeout,
	}, nil
}

func (adapter *Adapter) acquireLoadSlot(ctx context.Context) error {
	adapter.loadSlotOnce.Do(func() {
		adapter.loadSlots = make(
			chan struct{},
			adapter.pool.Config().MaxConns-1,
		)
	})
	select {
	case adapter.loadSlots <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (adapter *Adapter) releaseLoadSlot() {
	<-adapter.loadSlots
}

func initializeSession(ctx context.Context, connection *pgx.Conn) error {
	preloadStatus, err := probePreloadStatus(ctx, connection)
	if err != nil {
		return err
	}
	switch preloadStatus {
	case PreloadConfigured:
	case PreloadNotConfigured, PreloadUnknown:
		// Invoking an installed C function loads AGE and runs its _PG_init hook
		// without requiring the superuser-only LOAD command.
		if _, err := connection.Exec(ctx, "SELECT ag_catalog.age_pi()"); err != nil {
			return fmt.Errorf("dynamically initialize Apache AGE: %w", err)
		}
	default:
		return fmt.Errorf("unsupported Apache AGE preload status %q", preloadStatus)
	}
	if _, err := connection.Exec(
		ctx,
		`SET search_path = ag_catalog, "$user", public`,
	); err != nil {
		return fmt.Errorf("set Apache AGE search path: %w", err)
	}
	return nil
}

func probePreloadStatus(
	ctx context.Context,
	database capabilityQuerier,
) (PreloadStatus, error) {
	var preloadLibraries string
	err := database.QueryRow(
		ctx,
		`SELECT setting
		 FROM pg_catalog.pg_settings
		 WHERE name = 'shared_preload_libraries'`,
	).Scan(&preloadLibraries)
	if errors.Is(err, pgx.ErrNoRows) {
		return PreloadUnknown, nil
	}
	if err != nil {
		return "", fmt.Errorf("read shared_preload_libraries: %w", err)
	}
	if listContains(preloadLibraries, "age") {
		return PreloadConfigured, nil
	}
	return PreloadNotConfigured, nil
}

func listContains(list, expected string) bool {
	for item := range strings.SplitSeq(list, ",") {
		if strings.TrimSpace(item) == expected {
			return true
		}
	}
	return false
}

type capabilityQuerier interface {
	QueryRow(context.Context, string, ...any) pgx.Row
}

func ProbeCapabilities(
	ctx context.Context,
	database capabilityQuerier,
) (Capabilities, error) {
	var (
		serverVersionText string
		ageVersionText    string
		capabilities      Capabilities
	)
	err := database.QueryRow(
		ctx,
		`SELECT
			current_setting('server_version_num'),
			extversion,
			current_user,
			r.rolsuper
		FROM pg_extension e
		CROSS JOIN pg_roles r
		WHERE e.extname = 'age'
		  AND r.rolname = current_user`,
	).Scan(
		&serverVersionText,
		&ageVersionText,
		&capabilities.CurrentUser,
		&capabilities.CurrentUserSuperuser,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return Capabilities{}, errors.New("Apache AGE extension is not installed")
		}
		return Capabilities{}, fmt.Errorf("probe Apache AGE capabilities: %w", err)
	}
	serverVersion, err := strconv.Atoi(serverVersionText)
	if err != nil {
		return Capabilities{}, fmt.Errorf(
			"parse PostgreSQL server version %q: %w",
			serverVersionText,
			err,
		)
	}
	ageVersion, err := ParseVersion(ageVersionText)
	if err != nil {
		return Capabilities{}, fmt.Errorf("parse Apache AGE version: %w", err)
	}
	capabilities.PostgreSQLVersionNumber = serverVersion
	capabilities.PostgreSQLMajor = serverVersion / 10000
	capabilities.AGEVersion = ageVersion
	capabilities.AGEPreloadStatus, err = probePreloadStatus(ctx, database)
	if err != nil {
		return Capabilities{}, err
	}
	return capabilities, nil
}

func (adapter *Adapter) Capabilities() Capabilities {
	return adapter.capabilities
}

func (adapter *Adapter) Close() {
	if adapter != nil && adapter.pool != nil {
		adapter.pool.Close()
	}
}

type Transaction struct {
	tx pgx.Tx
}

func (adapter *Adapter) InTransaction(
	ctx context.Context,
	run func(*Transaction) error,
) error {
	if run == nil {
		return errors.New("transaction callback is required")
	}
	transaction, err := adapter.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin Apache AGE transaction: %w", err)
	}
	wrapped := &Transaction{tx: transaction}
	if err := run(wrapped); err != nil {
		rollbackCtx, cancel := context.WithTimeout(
			context.WithoutCancel(ctx),
			adapter.operationTimeout,
		)
		defer cancel()
		rollbackErr := transaction.Rollback(rollbackCtx)
		if rollbackErr != nil && !errors.Is(rollbackErr, pgx.ErrTxClosed) {
			return errors.Join(err, fmt.Errorf("rollback Apache AGE transaction: %w", rollbackErr))
		}
		return err
	}
	if err := transaction.Commit(ctx); err != nil {
		return fmt.Errorf("commit Apache AGE transaction: %w", err)
	}
	return nil
}

type databaseExecutor interface {
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
	QueryRow(context.Context, string, ...any) pgx.Row
}
