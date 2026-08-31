package neo4j

import (
	"context"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
)

func TestP3Neo4j44DiscoverySnapshotMatchesRetainedCheckpoint(t *testing.T) {
	job, err := config.Load(
		"../../../production-simulation/configs/neo4j-4.4.48.yaml",
	)
	if err != nil {
		t.Fatalf("load P3 job: %v", err)
	}
	snapshot, err := LoadDiscoverySnapshot(
		"../../../production-simulation/configs/neo4j-4.4.48-discovery-snapshot.json",
	)
	if err != nil {
		t.Fatalf("load P3 discovery snapshot: %v", err)
	}
	resolved, err := ResolveMappingsSnapshot(*job.Source.Neo4j, snapshot)
	if err != nil {
		t.Fatalf("resolve P3 discovery snapshot: %v", err)
	}
	mappings, err := buildMappings(
		context.Background(),
		job.Source.Namespace,
		resolved,
		1024,
	)
	if err != nil {
		t.Fatalf("compile P3 discovery snapshot: %v", err)
	}
	fingerprint, err := bindFingerprint(
		resolved,
		job.Source.Namespace,
		mappings,
	)
	if err != nil {
		t.Fatalf("fingerprint P3 discovery snapshot: %v", err)
	}
	const retainedCheckpointFingerprint = "9099a7181a7dccb1f4a381779151cc783d0f9c60edc96e27a66a9476793eb6eb"
	if fingerprint != retainedCheckpointFingerprint {
		t.Fatalf(
			"P3 discovery snapshot fingerprint = %s, retained checkpoint = %s",
			fingerprint,
			retainedCheckpointFingerprint,
		)
	}
}

func TestP3Neo4j526DiscoverySnapshotCompiles(t *testing.T) {
	job, err := config.Load(
		"../../../production-simulation/configs/neo4j-5.26.30.yaml",
	)
	if err != nil {
		t.Fatalf("load P3 job: %v", err)
	}
	snapshot, err := LoadDiscoverySnapshot(
		"../../../production-simulation/configs/neo4j-5.26.30-discovery-snapshot.json",
	)
	if err != nil {
		t.Fatalf("load P3 discovery snapshot: %v", err)
	}
	resolved, err := ResolveMappingsSnapshot(*job.Source.Neo4j, snapshot)
	if err != nil {
		t.Fatalf("resolve P3 discovery snapshot: %v", err)
	}
	mappings, err := buildMappings(
		context.Background(), job.Source.Namespace, resolved, 1024,
	)
	if err != nil {
		t.Fatalf("compile P3 discovery snapshot: %v", err)
	}
	if len(mappings) != 18 {
		t.Fatalf("P3 discovery snapshot mapping count = %d, want 18", len(mappings))
	}
}

func TestParseDiscoverySnapshotRejectsUnknownAndTrailingData(t *testing.T) {
	for _, input := range []string{
		`{"schemaVersion":1,"sourceId":"source","labels":[],"relationships":[],"extra":true}`,
		`{"schemaVersion":1,"sourceId":"source","labels":[],"relationships":[]} {}`,
	} {
		if _, err := ParseDiscoverySnapshot([]byte(input)); err == nil {
			t.Fatalf("ParseDiscoverySnapshot(%q) accepted invalid input", input)
		}
	}
}

func TestResolveMappingsSnapshotRejectsSourceMismatch(t *testing.T) {
	source := config.Neo4jSource{
		SourceID: "expected",
		Discovery: &config.Neo4jDiscovery{
			Enabled: true, MaxLabels: 1, MaxProperties: 2,
			VertexKeyProperty: "key", VertexIDProperty: "id",
			EdgeKeyProperty: "key", EdgeIDProperty: "id",
		},
	}
	_, err := ResolveMappingsSnapshot(source, DiscoverySnapshot{
		SchemaVersion: 1,
		SourceID:      "different",
	})
	if err == nil || !strings.Contains(err.Error(), "sourceId") {
		t.Fatalf("ResolveMappingsSnapshot() error = %v", err)
	}
}
