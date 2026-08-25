package meta_test

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/age"
	"github.com/rioriost/agefreighter/internal/meta"
)

func TestAGEAdapterMetadataTransactions(t *testing.T) {
	dsn := os.Getenv("AGEFREIGHTER_AGE_TEST_DSN")
	if dsn == "" {
		t.Skip("set AGEFREIGHTER_AGE_TEST_DSN to run AGE metadata integration tests")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	adapter, err := age.Open(ctx, dsn, age.PoolOptions{
		MinConnections:   1,
		MaxConnections:   2,
		ConnectTimeout:   5 * time.Second,
		OperationTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("open AGE adapter: %v", err)
	}
	t.Cleanup(adapter.Close)
	store, err := adapter.Metadata()
	if err != nil {
		t.Fatalf("adapter Metadata() error = %v", err)
	}
	if err := store.Migrate(ctx); err != nil {
		t.Fatalf("Migrate() error = %v", err)
	}

	const rolledBackJobID = "33333333-4444-4555-8666-777777777777"
	const failedJobID = "44444444-5555-4666-8777-888888888888"
	cleanupPool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("open cleanup pool: %v", err)
	}
	t.Cleanup(cleanupPool.Close)
	if _, err := cleanupPool.Exec(
		ctx,
		`DELETE FROM agefreighter_meta.load_job
		 WHERE job_id IN ($1::uuid, $2::uuid)`,
		rolledBackJobID,
		failedJobID,
	); err != nil {
		t.Fatalf("initial metadata cleanup: %v", err)
	}
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cleanupCancel()
		_, _ = cleanupPool.Exec(
			cleanupCtx,
			`DELETE FROM agefreighter_meta.load_job
			 WHERE job_id IN ($1::uuid, $2::uuid)`,
			rolledBackJobID,
			failedJobID,
		)
	})

	injected := errors.New("injected outer rollback")
	err = adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		transactionStore, storeErr := transaction.Metadata()
		if storeErr != nil {
			return storeErr
		}
		if storeErr := transactionStore.CreateJob(ctx, testJob(rolledBackJobID)); storeErr != nil {
			return storeErr
		}
		_, storeErr = transactionStore.RegisterGraphGeneration(ctx, meta.GraphGeneration{
			JobID:        rolledBackJobID,
			GraphName:    "meta_rollback",
			GraphOID:     61000,
			NamespaceOID: 61000,
			Generation:   1,
			State:        meta.GenerationLoading,
		})
		if storeErr != nil {
			return storeErr
		}
		return injected
	})
	if !errors.Is(err, injected) {
		t.Fatalf("outer transaction error = %v", err)
	}
	if _, err := store.GetJob(ctx, rolledBackJobID); !errors.Is(err, meta.ErrNotFound) {
		t.Fatalf("rolled-back GetJob() error = %v", err)
	}

	if err := store.CreateJob(ctx, testJob(failedJobID)); err != nil {
		t.Fatalf("diagnostic CreateJob() error = %v", err)
	}
	if err := store.StartJob(ctx, failedJobID); err != nil {
		t.Fatalf("diagnostic StartJob() error = %v", err)
	}
	err = adapter.InTransaction(ctx, func(transaction *age.Transaction) error {
		transactionStore, storeErr := transaction.Metadata()
		if storeErr != nil {
			return storeErr
		}
		_, storeErr = transactionStore.StartBatch(ctx, meta.BatchAttempt{
			JobID:   failedJobID,
			BatchID: 1,
			Attempt: 1,
			Rows:    1,
			Bytes:   10,
			First:   meta.Position{Token: "first"},
		})
		if storeErr != nil {
			return storeErr
		}
		return injected
	})
	if !errors.Is(err, injected) {
		t.Fatalf("batch outer transaction error = %v", err)
	}
	failedBatch := meta.BatchAttempt{
		JobID:        failedJobID,
		BatchID:      1,
		Attempt:      1,
		Rows:         1,
		Bytes:        10,
		RejectedRows: 1,
		First:        meta.Position{Token: "first"},
	}
	if err := store.RecordFailedBatch(ctx, failedBatch, "copy failed"); err != nil {
		t.Fatalf("independent RecordFailedBatch() error = %v", err)
	}
	stored, err := store.GetBatch(ctx, failedJobID, 1, 1)
	if err != nil || stored.Status != meta.BatchFailed || stored.RejectedRows != 1 {
		t.Fatalf("surviving failed batch = %#v, %v", stored, err)
	}
}

func testJob(id string) meta.Job {
	return meta.Job{
		ID:                id,
		Name:              "adapter-metadata",
		SourceType:        "csv",
		LoadMode:          "create",
		TargetGraph:       "adapter_metadata",
		ConfigFingerprint: strings.Repeat("a", 64),
	}
}
