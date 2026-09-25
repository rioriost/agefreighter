package app

import (
	"context"
	"testing"
)

func TestLoadWithIDValidatesBeforeConfigurationOrTargetAccess(t *testing.T) {
	for _, id := range []string{"", "bad", "11111111-1111-4111-8111-11111111111G", "11111111-1111-4111-8111-111111111111\n"} {
		result, err := LoadWithID(context.Background(), "/must-not-be-read", id)
		if err == nil || result.JobID != "" {
			t.Fatalf("invalid identity accepted: %q", id)
		}
	}
	id := "11111111-1111-4111-8111-111111111111"
	result, err := LoadWithID(context.Background(), "/does-not-exist", id)
	if err == nil || result.JobID != id {
		t.Fatal("retained identity lost on configuration failure")
	}
}
