package pggraph

import (
	"context"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rioriost/agefreighter/internal/checkpoint"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/meta"
	sinkcontract "github.com/rioriost/agefreighter/internal/sink"
	"github.com/rioriost/agefreighter/pkg/model"
)

func TestRecordProperties(t *testing.T) {
	encoded, err := recordProperties(nil, []byte(`{"b":2,"a":1}`))
	if err != nil || string(encoded) != `{"b":2,"a":1}` {
		t.Fatalf("recordProperties(encoded) = %q, %v", encoded, err)
	}
	encoded, err = recordProperties(model.Properties{
		"name": {Kind: model.ValueString, String: "Ada"},
	}, nil)
	if err != nil || !strings.Contains(string(encoded), `"name":"Ada"`) {
		t.Fatalf("recordProperties(map) = %q, %v", encoded, err)
	}
	for _, value := range [][]byte{[]byte(`no`), []byte(`null`), []byte(`[]`)} {
		if _, err := recordProperties(nil, value); err == nil {
			t.Fatalf("recordProperties(%q) succeeded", value)
		}
	}
}

func TestPropertyUpdateExpression(t *testing.T) {
	for mode, want := range map[config.PropertyMode]string{
		config.PropertiesReplace:         "EXCLUDED.properties",
		config.PropertiesMerge:           "current.properties || EXCLUDED.properties",
		config.PropertiesMergeDeleteNull: "jsonb_strip_nulls(current.properties || EXCLUDED.properties)",
	} {
		if got := propertyUpdateExpression("current", mode); got != want {
			t.Fatalf("propertyUpdateExpression(%q) = %q, want %q", mode, got, want)
		}
	}
}

func TestValidateBatchMetadata(t *testing.T) {
	valid := sinkcontract.BatchMetadata{
		ID: 1, Attempt: 1, Rows: 1, Bytes: 1,
		LastPosition: model.SourcePosition{Token: "last"},
	}
	if err := validateBatchMetadata(valid); err != nil {
		t.Fatal(err)
	}
	for _, edit := range []func(*sinkcontract.BatchMetadata){
		func(value *sinkcontract.BatchMetadata) { value.ID = 0 },
		func(value *sinkcontract.BatchMetadata) { value.Attempt = 0 },
		func(value *sinkcontract.BatchMetadata) { value.Rows = 0 },
		func(value *sinkcontract.BatchMetadata) { value.Bytes = 0 },
		func(value *sinkcontract.BatchMetadata) { value.LastPosition.Token = "" },
	} {
		value := valid
		edit(&value)
		if err := validateBatchMetadata(value); err == nil {
			t.Fatalf("invalid metadata accepted: %#v", value)
		}
	}
}

func TestCommittedReplayTransactionStateMachine(t *testing.T) {
	sink := &LoadSink{active: true}
	metadata := sinkcontract.BatchMetadata{Rows: 2}
	transaction := &committedReplayTransaction{sink: sink, metadata: metadata}
	if err := transaction.Commit(context.Background(), checkpoint.State{}); err == nil {
		t.Fatal("commit before write succeeded")
	}
	if err := transaction.Write(context.Background(), []model.Record{{}}); err == nil {
		t.Fatal("changed replay size succeeded")
	}
	if err := transaction.Write(context.Background(), []model.Record{{}, {}}); err != nil {
		t.Fatal(err)
	}
	if err := transaction.Write(context.Background(), []model.Record{{}, {}}); err == nil {
		t.Fatal("second replay write succeeded")
	}
	if err := transaction.Rollback(context.Background()); err != nil || sink.active {
		t.Fatalf("Rollback() = %v, active=%t", err, sink.active)
	}
	if err := transaction.Rollback(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestLoadTransactionWriteValidation(t *testing.T) {
	bindings := map[model.Label]loadBinding{
		"Person": {kind: 'v', table: "person"},
		"KNOWS":  {kind: 'e', table: "knows", startLabel: "Person", endLabel: "Person"},
	}
	newTransaction := func(rows int) *loadTransaction {
		return &loadTransaction{
			sink:     &LoadSink{bindings: bindings},
			metadata: sinkcontract.BatchMetadata{Rows: rows},
		}
	}
	vertex := func(label, namespace, id string, encoded []byte) model.Record {
		return model.Record{Vertex: &model.Vertex{
			Label: model.Label(label), Namespace: model.Namespace(namespace),
			ExternalID: model.ExternalID(id), EncodedProperties: encoded,
		}}
	}
	edge := func(label string) model.Record {
		return model.Record{Edge: &model.Edge{Label: model.Label(label)}}
	}
	validEdge := func() model.Record {
		return model.Record{Edge: &model.Edge{
			Label: "KNOWS", Namespace: "crm", ExternalID: "e1",
			Start: model.Endpoint{Label: "Person", Namespace: "crm", ExternalID: "p1"},
			End:   model.Endpoint{Label: "Person", Namespace: "crm", ExternalID: "p2"},
		}}
	}
	tests := []struct {
		name    string
		prepare func(*loadTransaction)
		records []model.Record
		want    string
	}{
		{"finalized", func(tx *loadTransaction) { tx.finalized = true }, nil, "finalized"},
		{"already wrote", func(tx *loadTransaction) { tx.wrote = true }, nil, "already written"},
		{"size", nil, nil, "contains 0 records"},
		{"invalid record", nil, []model.Record{{}}, "is invalid"},
		{"vertex after edge", nil, []model.Record{edge("missing"), vertex("Person", "crm", "p1", nil)}, "follows an edge"},
		{"unknown vertex", nil, []model.Record{vertex("Missing", "crm", "p1", nil)}, "not registered"},
		{"empty vertex namespace", nil, []model.Record{vertex("Person", "", "p1", nil)}, "identity is empty"},
		{"empty vertex ID", nil, []model.Record{vertex("Person", "crm", "", nil)}, "identity is empty"},
		{"duplicate vertex", nil, []model.Record{
			vertex("Person", "crm", "p1", nil), vertex("Person", "crm", "p1", nil),
		}, "duplicates external identity"},
		{"bad vertex properties", nil, []model.Record{vertex("Person", "crm", "p1", []byte(`no`))}, "encode vertex"},
		{"unknown edge", nil, []model.Record{edge("Missing")}, "not registered"},
		{"empty edge identity", nil, []model.Record{edge("KNOWS")}, "identity or endpoint is empty"},
		{"wrong edge endpoint label", nil, []model.Record{func() model.Record {
			record := validEdge()
			record.Edge.Start.Label = "Other"
			return record
		}()}, "endpoint labels do not match"},
		{"duplicate edge", nil, []model.Record{validEdge(), validEdge()}, "duplicates external identity"},
		{"bad edge properties", nil, []model.Record{func() model.Record {
			record := validEdge()
			record.Edge.EncodedProperties = []byte(`no`)
			return record
		}()}, "encode edge"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tx := newTransaction(len(test.records))
			if test.name == "size" {
				tx.metadata.Rows = 1
			}
			if test.prepare != nil {
				test.prepare(tx)
			}
			if err := tx.Write(t.Context(), test.records); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("Write() error = %v, want %q", err, test.want)
			}
		})
	}

	tx := newTransaction(0)
	tx.finalized = true
	if err := tx.Commit(t.Context(), checkpoint.State{}); err == nil {
		t.Fatal("Commit() accepted a finalized transaction")
	}
	if err := tx.Rollback(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestNewLoadSinkValidation(t *testing.T) {
	if _, err := NewLoadSink(nil, LoadSinkOptions{}); err == nil {
		t.Fatal("NewLoadSink(nil) succeeded")
	}
	adapter := &Adapter{}
	if _, err := NewLoadSink(adapter, LoadSinkOptions{}); err == nil {
		t.Fatal("NewLoadSink(incomplete adapter) succeeded")
	}
	adapter = &Adapter{pool: &pgxpool.Pool{}, store: &meta.Store{}}
	if _, err := NewLoadSink(adapter, LoadSinkOptions{JobID: "bad"}); err == nil {
		t.Fatal("NewLoadSink(invalid job) succeeded")
	}
	if _, err := NewLoadSink(adapter, LoadSinkOptions{
		JobID: "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
	}); err == nil {
		t.Fatal("NewLoadSink(invalid definition) succeeded")
	}
	definition := Definition{
		Schema: "public", Graph: "people",
		Vertices: []VertexDefinition{{Table: "person", Label: "Person"}},
	}
	for name, options := range map[string]LoadSinkOptions{
		"load mode": {
			JobID: "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", Definition: definition,
			Mode: "future",
		},
		"property mode": {
			JobID: "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", Definition: definition,
			Mode: config.LoadUpsert, PropertyMode: "future",
		},
		"append policy": {
			JobID: "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", Definition: definition,
			Mode: config.LoadAppend, AppendDuplicate: "future",
		},
		"append policy without append": {
			JobID: "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", Definition: definition,
			Mode: config.LoadCreate, AppendDuplicate: config.AppendDuplicateError,
		},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := NewLoadSink(adapter, options); err == nil {
				t.Fatal("NewLoadSink(invalid options) succeeded")
			}
		})
	}
	appendSink, err := NewLoadSink(adapter, LoadSinkOptions{
		JobID: "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", Definition: definition,
		Mode: config.LoadAppend,
	})
	if err != nil || appendSink.options.AppendDuplicate != config.AppendDuplicateError {
		t.Fatalf("NewLoadSink append defaults = %#v, %v", appendSink, err)
	}
	target := &LoadSink{}
	if _, err := target.Begin(t.Context(), sinkcontract.BatchMetadata{}); err == nil {
		t.Fatal("Begin(invalid metadata) succeeded")
	}
}
