package age

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/checkpoint"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	"github.com/rioriost/agefreighter/internal/sink"
	"github.com/rioriost/agefreighter/pkg/model"
)

func TestValidateLoadLabel(t *testing.T) {
	graph := meta.GraphGeneration{
		ID: 1, JobID: "11111111-2222-4333-8444-555555555555",
		GraphName: "people", GraphOID: 42, NamespaceOID: 42,
		Generation: 1, State: meta.GenerationLoading,
	}
	vertex := LoadLabel{
		Catalog: LabelCatalog{
			GraphName: "people", LabelName: "Person",
			GraphOID: 42, NamespaceOID: 42, LabelID: 1,
			Kind: VertexLabel, RelationOID: 43, SequenceOID: 44,
		},
		Generation: meta.LabelGeneration{
			ID: 1, GraphGenerationID: 1, LabelName: "Person",
			Kind: meta.VertexLabel, GraphNamespaceOID: 42, LabelID: 1,
			RelationOID: 43, SequenceOID: 44, MappingGeneration: 1,
		},
	}
	if err := validateLoadLabel(graph, vertex); err != nil {
		t.Fatalf("validateLoadLabel(vertex) error = %v", err)
	}
	edge := vertex
	edge.Catalog.LabelName = "KNOWS"
	edge.Catalog.Kind = EdgeLabel
	edge.Generation.LabelName = "KNOWS"
	edge.Generation.Kind = meta.EdgeLabel
	if err := validateLoadLabel(graph, edge); err != nil {
		t.Fatalf("validateLoadLabel(edge) error = %v", err)
	}
	tests := []LoadLabel{
		withLoadLabel(vertex, func(label *LoadLabel) { label.Catalog.GraphName = "other" }),
		withLoadLabel(vertex, func(label *LoadLabel) { label.Catalog.Kind = LabelKind('x') }),
		withLoadLabel(vertex, func(label *LoadLabel) { label.Generation.GraphGenerationID = 2 }),
		withLoadLabel(vertex, func(label *LoadLabel) { label.Generation.LabelName = "Other" }),
		withLoadLabel(vertex, func(label *LoadLabel) { label.Generation.Kind = meta.EdgeLabel }),
		withLoadLabel(vertex, func(label *LoadLabel) { label.Generation.RelationOID++ }),
	}
	for index, label := range tests {
		if err := validateLoadLabel(graph, label); err == nil {
			t.Fatalf("invalid load label %d accepted: %#v", index, label)
		}
	}
}

func TestLoadSinkValidationHelpers(t *testing.T) {
	tests := []sink.BatchMetadata{
		{},
		{ID: math.MaxInt64 + 1, Attempt: 1, Rows: 1, Bytes: 1, LastPosition: tokenPosition()},
		{ID: 1, Rows: 1, Bytes: 1, LastPosition: tokenPosition()},
		{ID: 1, Attempt: 1, Bytes: 1, LastPosition: tokenPosition()},
		{ID: 1, Attempt: 1, Rows: 1, LastPosition: tokenPosition()},
		{ID: 1, Attempt: 1, Rows: 1, Bytes: 1},
	}
	for index, batch := range tests {
		if err := validateBatchMetadata(batch); err == nil {
			t.Fatalf("invalid batch %d accepted: %#v", index, batch)
		}
	}
	valid := sink.BatchMetadata{
		ID: 1, Attempt: 1, Rows: 1, Bytes: 1, LastPosition: tokenPosition(),
	}
	if err := validateBatchMetadata(valid); err != nil {
		t.Fatalf("validateBatchMetadata() error = %v", err)
	}
	if got := missingEndpointMessage(model.Edge{
		ExternalID: "e1",
		Start: model.Endpoint{
			Namespace: "crm", Label: "Person", ExternalID: "p1",
		},
		End: model.Endpoint{
			Namespace: "crm", Label: "Person", ExternalID: "missing",
		},
	}); !strings.Contains(got, "crm/Person/p1") ||
		!strings.Contains(got, "crm/Person/missing") {
		t.Fatalf("missingEndpointMessage() = %q", got)
	}
	position := model.SourcePosition{
		Resource: "source.csv", Line: 2, Offset: 10, Token: "token",
	}
	if got := metaPosition(position); got != (meta.Position{
		Resource: "source.csv", Line: 2, ByteOffset: 10, Token: "token",
	}) {
		t.Fatalf("metaPosition() = %#v", got)
	}
}

func TestNewLoadSinkRejectsInvalidOptions(t *testing.T) {
	if _, err := NewLoadSink(nil, nil, LoadSinkOptions{}); err == nil {
		t.Fatal("NewLoadSink() accepted nil context")
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewLoadSink(cancelled, nil, LoadSinkOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("NewLoadSink(cancelled) error = %v", err)
	}
	if _, err := NewLoadSink(
		context.Background(),
		nil,
		LoadSinkOptions{},
	); err == nil {
		t.Fatal("NewLoadSink() accepted nil adapter")
	}
	adapter := &Adapter{}
	if _, err := NewLoadSink(
		context.Background(),
		adapter,
		LoadSinkOptions{},
	); err == nil {
		t.Fatal("NewLoadSink() accepted adapter without pool")
	}
	poolConfig, err := pgxpool.ParseConfig(
		"postgres://localhost/unused?connect_timeout=1",
	)
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}
	poolConfig.MaxConns = 1
	pool, err := pgxpool.NewWithConfig(context.Background(), poolConfig)
	if err != nil {
		t.Fatalf("NewWithConfig() error = %v", err)
	}
	t.Cleanup(pool.Close)
	adapter = &Adapter{pool: pool}
	if _, err := NewLoadSink(
		context.Background(),
		adapter,
		LoadSinkOptions{
			JobID:           "11111111-2222-4333-8444-555555555555",
			Graph:           meta.GraphGeneration{ID: 1},
			MissingEndpoint: config.MissingEndpointQuarantine,
		},
	); err == nil || !strings.Contains(err.Error(), "at least two target connections") {
		t.Fatalf("single-connection NewLoadSink() error = %v", err)
	}
	if _, err := NewLoadSink(
		context.Background(),
		adapter,
		LoadSinkOptions{
			MissingEndpoint: config.MissingEndpointError,
		},
	); err == nil || !strings.Contains(err.Error(), "job and graph generation") {
		t.Fatalf("identity-less NewLoadSink() error = %v", err)
	}
	if _, err := NewLoadSink(
		context.Background(),
		adapter,
		LoadSinkOptions{
			JobID:           "11111111-2222-4333-8444-555555555555",
			Graph:           meta.GraphGeneration{ID: 1},
			MissingEndpoint: config.MissingEndpointDefer,
		},
	); err == nil || !strings.Contains(err.Error(), "unsupported missing endpoint") {
		t.Fatalf("unsupported-policy NewLoadSink() error = %v", err)
	}
}

func TestLoadSinkBeginReportsConnectionAdmissionFailures(t *testing.T) {
	poolConfig, err := pgxpool.ParseConfig(
		"postgres://localhost/unused?connect_timeout=1",
	)
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}
	poolConfig.MaxConns = 2
	pool, err := pgxpool.NewWithConfig(context.Background(), poolConfig)
	if err != nil {
		t.Fatalf("NewWithConfig() error = %v", err)
	}
	adapter := &Adapter{pool: pool}
	target := &LoadSink{
		adapter: adapter,
		options: LoadSinkOptions{
			JobID: "11111111-2222-4333-8444-555555555555",
		},
	}
	batch := sink.BatchMetadata{
		ID: 1, Attempt: 1, Rows: 1, Bytes: 1,
		FirstPosition: tokenPosition(),
		LastPosition:  tokenPosition(),
	}

	if err := adapter.acquireLoadSlot(t.Context()); err != nil {
		t.Fatalf("fill load slot: %v", err)
	}
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := target.Begin(cancelled, batch); !errors.Is(err, context.Canceled) {
		t.Fatalf("Begin(blocked slot) error = %v", err)
	}
	adapter.releaseLoadSlot()

	pool.Close()
	if _, err := target.Begin(t.Context(), batch); err == nil ||
		!strings.Contains(err.Error(), "ownership connection") {
		t.Fatalf("Begin(closed pool) error = %v", err)
	}
}

func TestLoadTransactionValidation(t *testing.T) {
	transaction := &loadTransaction{finalized: true}
	if err := transaction.Write(context.Background(), nil); err == nil {
		t.Fatal("finalized Write() succeeded")
	}
	if err := transaction.Commit(context.Background(), checkpoint.State{}); err == nil {
		t.Fatal("finalized Commit() succeeded")
	}
	if err := transaction.Rollback(context.Background()); err != nil {
		t.Fatalf("finalized Rollback() error = %v", err)
	}
	transaction = &loadTransaction{wrote: true}
	if err := transaction.Write(context.Background(), nil); err == nil {
		t.Fatal("second Write() succeeded")
	}
	target := &LoadSink{active: true}
	if _, err := target.Begin(context.Background(), sink.BatchMetadata{
		ID: 1, Attempt: 1, Rows: 1, Bytes: 1, LastPosition: tokenPosition(),
	}); err == nil || !strings.Contains(err.Error(), "active transaction") {
		t.Fatalf("active Begin() error = %v", err)
	}
	target = &LoadSink{}
	if _, err := target.Begin(context.Background(), sink.BatchMetadata{}); err == nil {
		t.Fatal("Begin() accepted invalid metadata")
	}
}

func TestLoadTransactionRejectsInvalidRecordsBeforeIO(t *testing.T) {
	vertexBinding := LoadLabel{
		Catalog: LabelCatalog{LabelName: "Person", Kind: VertexLabel},
	}
	edgeBinding := LoadLabel{
		Catalog: LabelCatalog{LabelName: "KNOWS", Kind: EdgeLabel},
	}
	tests := []struct {
		name    string
		labels  map[model.Label]LoadLabel
		records []model.Record
		want    string
	}{
		{
			name: "invalid record", records: []model.Record{{}},
			want: "invalid",
		},
		{
			name: "vertex follows edge",
			records: []model.Record{
				model.EdgeRecord(model.Edge{Label: "KNOWS"}),
				model.VertexRecord(model.Vertex{Label: "Person"}),
			},
			want: "follows an edge",
		},
		{
			name: "unknown vertex",
			records: []model.Record{
				model.VertexRecord(model.Vertex{Label: "Person"}),
			},
			want: "not registered",
		},
		{
			name: "unknown edge",
			records: []model.Record{
				model.EdgeRecord(model.Edge{Label: "KNOWS"}),
			},
			want: "not registered",
		},
		{
			name:   "edge binding used by vertex",
			labels: map[model.Label]LoadLabel{"Person": edgeBinding},
			records: []model.Record{
				model.VertexRecord(model.Vertex{Label: "Person"}),
			},
			want: "not a vertex label",
		},
		{
			name:   "vertex binding used by edge",
			labels: map[model.Label]LoadLabel{"KNOWS": vertexBinding},
			records: []model.Record{
				model.EdgeRecord(model.Edge{Label: "KNOWS"}),
			},
			want: "not an edge label",
		},
		{
			name:   "empty vertex identity",
			labels: map[model.Label]LoadLabel{"Person": vertexBinding},
			records: []model.Record{
				model.VertexRecord(model.Vertex{Label: "Person"}),
			},
			want: "identity is empty",
		},
		{
			name:   "duplicate vertex identity",
			labels: map[model.Label]LoadLabel{"Person": vertexBinding},
			records: []model.Record{
				validUnitVertex("p1"),
				validUnitVertex("p1"),
			},
			want: "duplicates external identity",
		},
		{
			name:   "invalid vertex property",
			labels: map[model.Label]LoadLabel{"Person": vertexBinding},
			records: []model.Record{
				model.VertexRecord(model.Vertex{
					Label: "Person", Namespace: "crm", ExternalID: "p1",
					Properties: model.Properties{
						"bad": {Kind: model.ValueKind(255)},
					},
				}),
			},
			want: "encode vertex",
		},
		{
			name:   "empty edge identity",
			labels: map[model.Label]LoadLabel{"KNOWS": edgeBinding},
			records: []model.Record{
				model.EdgeRecord(model.Edge{Label: "KNOWS"}),
			},
			want: "identity or endpoint is empty",
		},
		{
			name:   "empty edge token",
			labels: map[model.Label]LoadLabel{"KNOWS": edgeBinding},
			records: []model.Record{
				validUnitEdge("e1", ""),
			},
			want: "token is empty",
		},
		{
			name:   "duplicate edge identity",
			labels: map[model.Label]LoadLabel{"KNOWS": edgeBinding},
			records: []model.Record{
				validUnitEdge("e1", "one"),
				validUnitEdge("e1", "two"),
			},
			want: "duplicates external identity",
		},
		{
			name:   "invalid edge property",
			labels: map[model.Label]LoadLabel{"KNOWS": edgeBinding},
			records: []model.Record{
				model.EdgeRecord(model.Edge{
					Label: "KNOWS", Namespace: "crm", ExternalID: "e1",
					Start: model.Endpoint{
						Label: "Person", Namespace: "crm", ExternalID: "p1",
					},
					End: model.Endpoint{
						Label: "Person", Namespace: "crm", ExternalID: "p2",
					},
					Properties: model.Properties{
						"bad": {Kind: model.ValueKind(255)},
					},
					Position: model.SourcePosition{Token: "token"},
				}),
			},
			want: "encode edge",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			transaction := &loadTransaction{
				sink: &LoadSink{labels: test.labels},
				metadata: sink.BatchMetadata{
					Rows: len(test.records),
				},
			}
			if err := transaction.Write(
				context.Background(),
				test.records,
			); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Write() error = %v, want %q", err, test.want)
			}
		})
	}

	transaction := &loadTransaction{metadata: sink.BatchMetadata{Rows: 2}}
	if err := transaction.Write(
		context.Background(),
		[]model.Record{validUnitVertex("p1")},
	); err == nil || !strings.Contains(err.Error(), "expected 2") {
		t.Fatalf("mismatched Write() error = %v", err)
	}
}

func TestCommittedReplayValidation(t *testing.T) {
	transaction := &committedReplayTransaction{
		sink:     &LoadSink{},
		metadata: sink.BatchMetadata{Rows: 1},
	}
	if err := transaction.Write(context.Background(), nil); err == nil {
		t.Fatal("committed replay accepted wrong record count")
	}
	transaction = &committedReplayTransaction{
		sink:     &LoadSink{},
		metadata: sink.BatchMetadata{Rows: 1},
	}
	if err := transaction.Write(
		context.Background(),
		[]model.Record{validUnitVertex("p1")},
	); err != nil {
		t.Fatalf("committed replay Write() error = %v", err)
	}
	if err := transaction.Write(
		context.Background(),
		[]model.Record{validUnitVertex("p1")},
	); err == nil {
		t.Fatal("committed replay accepted records twice")
	}
	if err := transaction.Rollback(context.Background()); err != nil {
		t.Fatalf("committed replay Rollback() error = %v", err)
	}
	if err := transaction.Rollback(context.Background()); err != nil {
		t.Fatalf("idempotent committed replay Rollback() error = %v", err)
	}

	finalized := &committedReplayTransaction{finalized: true}
	if err := finalized.Write(context.Background(), nil); err == nil {
		t.Fatal("finalized committed replay Write() succeeded")
	}
	if err := finalized.Commit(
		context.Background(),
		checkpoint.State{},
	); err == nil {
		t.Fatal("finalized committed replay Commit() succeeded")
	}
	unwrittenSink := &LoadSink{active: true}
	unwritten := &committedReplayTransaction{sink: unwrittenSink}
	if err := unwritten.Commit(
		context.Background(),
		checkpoint.State{},
	); err == nil {
		t.Fatal("unwritten committed replay Commit() succeeded")
	}
	if unwrittenSink.active || !unwritten.finalized {
		t.Fatal("unwritten committed replay did not release its sink")
	}
	mismatchedSink := &LoadSink{active: true}
	mismatched := &committedReplayTransaction{
		sink:  mismatchedSink,
		wrote: true,
		metadata: sink.BatchMetadata{
			ID: 1, Attempt: 1, LastPosition: tokenPosition(),
		},
	}
	if err := mismatched.Commit(
		context.Background(),
		checkpoint.State{BatchID: 2},
	); err == nil {
		t.Fatal("mismatched committed replay Commit() succeeded")
	}
	if mismatchedSink.active || !mismatched.finalized {
		t.Fatal("mismatched committed replay did not release its sink")
	}
}

func TestSortedLabels(t *testing.T) {
	groups := map[model.Label][]int{
		"Zebra": {1},
		"Alpha": {2},
	}
	got := sortedLabels(groups)
	if len(got) != 2 || got[0] != "Alpha" || got[1] != "Zebra" {
		t.Fatalf("sortedLabels() = %#v", got)
	}
}

func withLoadLabel(value LoadLabel, change func(*LoadLabel)) LoadLabel {
	change(&value)
	return value
}

func tokenPosition() model.SourcePosition {
	return model.SourcePosition{Token: "token"}
}

func validUnitVertex(id string) model.Record {
	return model.VertexRecord(model.Vertex{
		Label: "Person", Namespace: "crm", ExternalID: model.ExternalID(id),
	})
}

func validUnitEdge(id, token string) model.Record {
	return model.EdgeRecord(model.Edge{
		Label: "KNOWS", Namespace: "crm", ExternalID: model.ExternalID(id),
		Start: model.Endpoint{
			Label: "Person", Namespace: "crm", ExternalID: "p1",
		},
		End: model.Endpoint{
			Label: "Person", Namespace: "crm", ExternalID: "p2",
		},
		Position: model.SourcePosition{Token: token},
	})
}

func TestMissingEndpointPolicies(t *testing.T) {
	for _, policy := range []config.MissingEndpointPolicy{
		config.MissingEndpointError,
		config.MissingEndpointQuarantine,
	} {
		if policy == "" {
			t.Fatal("missing endpoint policy constant is empty")
		}
	}
}
