package app

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
)

func TestP3Neo4j44SnapshotUsesCorrectedResolvedJobFingerprint(t *testing.T) {
	job, err := config.Load(
		"../../production-simulation/configs/neo4j-4.4.48.yaml",
	)
	if err != nil {
		t.Fatalf("load P3 job: %v", err)
	}
	snapshotPath, err := filepath.Abs(
		"../../production-simulation/configs/neo4j-4.4.48-discovery-snapshot.json",
	)
	if err != nil {
		t.Fatalf("resolve P3 snapshot path: %v", err)
	}
	t.Setenv(neo4jDiscoverySnapshotEnvironment, snapshotPath)
	resolved, err := resolveSource(context.Background(), job)
	if err != nil {
		t.Fatalf("resolve P3 snapshot: %v", err)
	}
	fingerprint, err := jobFingerprint(resolved)
	if err != nil {
		t.Fatalf("fingerprint P3 resolved job: %v", err)
	}
	const correctedResolvedJobFingerprint = "39ee60214738cec0ecbac1df6133071ecdc585a4ce21028677a85fee2e594bed"
	if fingerprint != correctedResolvedJobFingerprint {
		t.Fatalf(
			"P3 resolved job fingerprint = %s, corrected job = %s",
			fingerprint,
			correctedResolvedJobFingerprint,
		)
	}
	const failedR6ResolvedJobFingerprint = "86648c1e1eff8a1fb8842161122761aad2a05a6c26d6f863b7769e252e665ac0"
	if fingerprint == failedR6ResolvedJobFingerprint {
		t.Fatal("corrected P3 job still matches the failed r6 query plan")
	}
}
