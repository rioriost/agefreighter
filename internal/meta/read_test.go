package meta

import (
	"context"
	"strings"
	"testing"
)

func TestBoundedReadsRequireExplicitValidLimits(t *testing.T) {
	store := &Store{database: panicDatabase{}}
	validJobID := "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
	tests := []struct {
		name string
		run  func(int) error
	}{
		{
			name: "jobs",
			run: func(limit int) error {
				_, err := store.ListJobs(t.Context(), limit)
				return err
			},
		},
		{
			name: "graphs",
			run: func(limit int) error {
				_, err := store.ListGraphGenerations(t.Context(), limit)
				return err
			},
		},
		{
			name: "labels",
			run: func(limit int) error {
				_, err := store.ListLabelGenerations(t.Context(), 1, limit)
				return err
			},
		},
		{
			name: "batches",
			run: func(limit int) error {
				_, err := store.ListBatches(t.Context(), validJobID, limit)
				return err
			},
		},
		{
			name: "reject summaries",
			run: func(limit int) error {
				_, err := store.ListRejectSummaries(t.Context(), validJobID, limit)
				return err
			},
		},
		{
			name: "backups",
			run: func(limit int) error {
				_, err := store.ListRetainedBackups(t.Context(), limit)
				return err
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, limit := range []int{0, -1, MaxReadLimit + 1} {
				if err := test.run(limit); err == nil {
					t.Fatalf("limit %d succeeded", limit)
				}
			}
		})
	}
}

func TestBoundedReadsValidateIdentifiersBeforeQuery(t *testing.T) {
	store := &Store{database: panicDatabase{}}
	if _, err := store.ListLabelGenerations(t.Context(), 0, 1); err == nil {
		t.Fatal("ListLabelGenerations() accepted zero graph generation ID")
	}
	if _, err := store.ListBatches(t.Context(), "bad", 1); err == nil {
		t.Fatal("ListBatches() accepted invalid job ID")
	}
	if _, err := store.ListRejectSummaries(t.Context(), "bad", 1); err == nil {
		t.Fatal("ListRejectSummaries() accepted invalid job ID")
	}
}

func TestBoundedReadsRequireContextDeadline(t *testing.T) {
	store := &Store{database: panicDatabase{}}
	_, err := store.ListJobs(context.Background(), 1)
	if err == nil || !strings.Contains(err.Error(), "requires a deadline") {
		t.Fatalf("ListJobs() error = %v", err)
	}
}
