package app

import (
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
)

func TestDenseEndpointCacheBytes(t *testing.T) {
	job := config.LoadJob{
		Source: config.Source{
			Type: config.SourceNeo4j,
			Neo4j: &config.Neo4jSource{
				ResolvedVertexIdentity: config.Neo4jVertexIdentityInternalID,
			},
		},
		Target: config.Target{Mode: config.LoadCreate},
		Runtime: config.Runtime{
			MemoryLimit: 2 << 30,
			BatchBytes:  64 << 20,
		},
	}
	if got, want := denseEndpointCacheBytes(job), int64(2<<30-denseEndpointCacheReserve); got != want {
		t.Fatalf("denseEndpointCacheBytes() = %d, want %d", got, want)
	}
	job.Source.Neo4j.ResolvedVertexIdentity = config.Neo4jVertexIdentityProperty
	if got := denseEndpointCacheBytes(job); got != 0 {
		t.Fatalf("property identity cache bytes = %d, want 0", got)
	}
	job.Source.Neo4j.ResolvedVertexIdentity = config.Neo4jVertexIdentityInternalID
	job.Target.Mode = config.LoadAppend
	if got := denseEndpointCacheBytes(job); got != 0 {
		t.Fatalf("append cache bytes = %d, want 0", got)
	}
}
