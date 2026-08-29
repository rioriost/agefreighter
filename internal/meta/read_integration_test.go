package meta

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

func TestReadAPIsIntegration(t *testing.T) {
	dsn := os.Getenv(metadataTestDSNEnvironment)
	if dsn == "" {
		t.Skip("set " + metadataTestDSNEnvironment + " to run metadata integration tests")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("open metadata test pool: %v", err)
	}
	t.Cleanup(pool.Close)
	store, err := New(pool)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	inspection, err := store.InspectSchema(ctx)
	if err != nil {
		t.Fatalf("InspectSchema() error = %v", err)
	}
	if inspection.State != SchemaCurrent {
		t.Skipf("metadata schema is %s, not current", inspection.State)
	}

	if _, err := store.ListJobs(ctx, 2); err != nil {
		t.Fatalf("ListJobs() error = %v", err)
	}
	graphs, err := store.ListGraphGenerations(ctx, 2)
	if err != nil {
		t.Fatalf("ListGraphGenerations() error = %v", err)
	}
	if len(graphs) > 0 {
		if _, err := store.ListLabelGenerations(ctx, graphs[0].ID, 2); err != nil {
			t.Fatalf("ListLabelGenerations() error = %v", err)
		}
	}
	missingJob := "ffffffff-ffff-4fff-8fff-ffffffffffff"
	if batches, err := store.ListBatches(ctx, missingJob, 2); err != nil || len(batches) != 0 {
		t.Fatalf("ListBatches() = %#v, %v", batches, err)
	}
	if summaries, err := store.ListRejectSummaries(ctx, missingJob, 2); err != nil ||
		len(summaries.Summaries) != 0 ||
		summaries.ScannedRows != 0 ||
		summaries.Truncated {
		t.Fatalf("ListRejectSummaries() = %#v, %v", summaries, err)
	}
	if _, err := store.ListRetainedBackups(ctx, 2); err != nil {
		t.Fatalf("ListRetainedBackups() error = %v", err)
	}
}
