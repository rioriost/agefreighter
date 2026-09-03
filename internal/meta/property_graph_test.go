package meta

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestValidatePropertyGraph(t *testing.T) {
	valid := PropertyGraphGeneration{
		JobID:  "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
		Schema: "Graph Data", Graph: "Supply Graph",
		DefinitionFingerprint: strings.Repeat("a", 64),
		State:                 PropertyGraphLoading,
		Labels: []PropertyGraphLabel{
			{Name: "Person", Kind: VertexLabel, Table: "v_person"},
			{
				Name: "KNOWS", Kind: EdgeLabel, Table: "e_knows",
				StartLabel: "Person", EndLabel: "Person",
			},
		},
	}
	if err := validatePropertyGraph(valid); err != nil {
		t.Fatalf("valid property graph rejected: %v", err)
	}

	tests := []struct {
		name string
		edit func(*PropertyGraphGeneration)
		want string
	}{
		{
			name: "invalid job",
			edit: func(value *PropertyGraphGeneration) { value.JobID = "bad" },
			want: "job ID",
		},
		{
			name: "invalid schema",
			edit: func(value *PropertyGraphGeneration) { value.Schema = "" },
			want: "valid identifiers",
		},
		{
			name: "invalid fingerprint",
			edit: func(value *PropertyGraphGeneration) { value.DefinitionFingerprint = "bad" },
			want: "fingerprint",
		},
		{
			name: "invalid state",
			edit: func(value *PropertyGraphGeneration) { value.State = "retired" },
			want: "unsupported property graph state",
		},
		{
			name: "no labels",
			edit: func(value *PropertyGraphGeneration) { value.Labels = nil },
			want: "requires labels",
		},
		{
			name: "invalid label",
			edit: func(value *PropertyGraphGeneration) { value.Labels[0].Name = "" },
			want: "invalid name",
		},
		{
			name: "duplicate label",
			edit: func(value *PropertyGraphGeneration) {
				value.Labels[1].Name = value.Labels[0].Name
			},
			want: "duplicate property graph label",
		},
		{
			name: "invalid kind",
			edit: func(value *PropertyGraphGeneration) { value.Labels[0].Kind = 'x' },
			want: "unsupported property graph label kind",
		},
		{
			name: "duplicate table",
			edit: func(value *PropertyGraphGeneration) {
				value.Labels[1].Table = value.Labels[0].Table
			},
			want: "duplicate property graph table",
		},
		{
			name: "edge without start",
			edit: func(value *PropertyGraphGeneration) { value.Labels[1].StartLabel = "" },
			want: "requires endpoints",
		},
		{
			name: "unknown start endpoint",
			edit: func(value *PropertyGraphGeneration) { value.Labels[1].StartLabel = "Missing" },
			want: "unknown start label",
		},
		{
			name: "unknown endpoint",
			edit: func(value *PropertyGraphGeneration) { value.Labels[1].EndLabel = "Missing" },
			want: "unknown end label",
		},
		{
			name: "vertex endpoint",
			edit: func(value *PropertyGraphGeneration) { value.Labels[0].StartLabel = "Person" },
			want: "has edge endpoints",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			value := valid
			value.Labels = append([]PropertyGraphLabel(nil), valid.Labels...)
			test.edit(&value)
			err := validatePropertyGraph(value)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("validatePropertyGraph() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestPropertyGraphStoreDatabaseErrors(t *testing.T) {
	injected := errors.New("injected database failure")
	store := &Store{database: errorDatabase{err: injected}}
	jobID := "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
	generation := PropertyGraphGeneration{
		JobID: jobID, Schema: "graph_data", Graph: "supply_graph",
		DefinitionFingerprint: strings.Repeat("a", 64), State: PropertyGraphLoading,
		Labels: []PropertyGraphLabel{{Name: "Person", Kind: VertexLabel, Table: "person"}},
	}
	if err := store.RegisterPropertyGraph(t.Context(), generation); !errors.Is(err, injected) {
		t.Fatalf("RegisterPropertyGraph() error = %v", err)
	}
	if _, err := store.GetPropertyGraph(t.Context(), jobID); !errors.Is(err, injected) {
		t.Fatalf("GetPropertyGraph() error = %v", err)
	}
	if err := store.ReplacePropertyGraphDigests(t.Context(), jobID, nil,
		strings.Repeat("b", 64), 0, 256); !errors.Is(err, injected) {
		t.Fatalf("ReplacePropertyGraphDigests() error = %v", err)
	}
	if _, err := store.ListPropertyGraphDigests(t.Context(), jobID); !errors.Is(err, injected) {
		t.Fatalf("ListPropertyGraphDigests() error = %v", err)
	}
	if err := store.ActivatePropertyGraph(t.Context(), jobID); !errors.Is(err, injected) {
		t.Fatalf("ActivatePropertyGraph() error = %v", err)
	}
	if _, err := store.ActivePropertyGraph(t.Context(), "graph_data", "supply_graph"); !errors.Is(err, injected) {
		t.Fatalf("ActivePropertyGraph() error = %v", err)
	}
	if _, err := store.PropertyGraphByTargetState(t.Context(), "graph_data", "supply_graph",
		PropertyGraphRetainedBackup); !errors.Is(err, injected) {
		t.Fatalf("PropertyGraphByTargetState() error = %v", err)
	}
	if err := store.ActivatePropertyGraphReplacing(t.Context(), jobID, "graph_data", "supply_graph"); !errors.Is(err, injected) {
		t.Fatalf("ActivatePropertyGraphReplacing() error = %v", err)
	}
	if err := store.RelocatePropertyGraph(t.Context(), generation); !errors.Is(err, injected) {
		t.Fatalf("RelocatePropertyGraph() error = %v", err)
	}
	if err := store.RegisterPropertyGraph(t.Context(), PropertyGraphGeneration{}); err == nil {
		t.Fatal("RegisterPropertyGraph() accepted invalid input")
	}
	if _, err := store.GetPropertyGraph(t.Context(), "bad"); err == nil {
		t.Fatal("GetPropertyGraph() accepted invalid job ID")
	}
	if _, err := store.ListPropertyGraphDigests(t.Context(), "bad"); err == nil {
		t.Fatal("ListPropertyGraphDigests() accepted invalid job ID")
	}
	if err := store.ActivatePropertyGraph(t.Context(), "bad"); err == nil {
		t.Fatal("ActivatePropertyGraph() accepted invalid job ID")
	}
	if _, err := store.ActivePropertyGraph(t.Context(), "", "supply_graph"); err == nil {
		t.Fatal("ActivePropertyGraph() accepted an invalid schema")
	}
	if _, err := store.PropertyGraphByTargetState(t.Context(), "graph_data", "supply_graph",
		"future"); err == nil {
		t.Fatal("PropertyGraphByTargetState() accepted an invalid state")
	}
	if err := store.ActivatePropertyGraphReplacing(t.Context(), "bad", "graph_data",
		"supply_graph"); err == nil {
		t.Fatal("ActivatePropertyGraphReplacing() accepted an invalid job ID")
	}
	if err := store.ActivatePropertyGraphReplacing(t.Context(), jobID, "", "supply_graph"); err == nil {
		t.Fatal("ActivatePropertyGraphReplacing() accepted an invalid schema")
	}
	if err := store.RelocatePropertyGraph(t.Context(), PropertyGraphGeneration{}); err == nil {
		t.Fatal("RelocatePropertyGraph() accepted invalid input")
	}
}

func TestPropertyGraphStoredEncodingValidation(t *testing.T) {
	jobID := "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
	for _, test := range []struct {
		name   string
		labels []byte
	}{
		{"invalid JSON", []byte(`bad`)},
		{"invalid kind", []byte(`[{"name":"Person","kind":"x","table":"person","startLabel":"","endLabel":""}]`)},
	} {
		t.Run("generation "+test.name, func(t *testing.T) {
			store := &Store{database: propertyGraphRowDatabase{row: scanLifecycleRow(func(dest ...any) error {
				*(dest[0].(*string)) = jobID
				*(dest[1].(*string)) = "graph_data"
				*(dest[2].(*string)) = "supply_graph"
				*(dest[3].(*string)) = strings.Repeat("a", 64)
				*(dest[4].(*PropertyGraphState)) = PropertyGraphActive
				*(dest[5].(*string)) = strings.Repeat("b", 64)
				*(dest[6].(*int64)) = 1
				*(dest[7].(*int)) = 256
				*(dest[8].(*[]byte)) = test.labels
				return nil
			})}}
			if _, err := store.GetPropertyGraph(t.Context(), jobID); err == nil {
				t.Fatal("GetPropertyGraph() accepted invalid stored labels")
			}
		})
	}
	store := &Store{database: propertyGraphRowDatabase{row: lifecycleErrorRow{err: pgx.ErrNoRows}}}
	if _, err := store.GetPropertyGraph(t.Context(), jobID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetPropertyGraph(missing) error = %v", err)
	}

	for _, test := range []struct {
		name   string
		ranges []byte
	}{
		{"invalid JSON", []byte(`bad`)},
		{"invalid kind", []byte(`[{"labelName":"Person","kind":"x","rangeId":1,"rows":1,"digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]`)},
		{"invalid range", []byte(`[{"labelName":"Person","kind":"v","rangeId":256,"rows":1,"digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]`)},
	} {
		t.Run("ranges "+test.name, func(t *testing.T) {
			store := &Store{database: propertyGraphRowDatabase{row: scanLifecycleRow(func(dest ...any) error {
				*(dest[0].(*[]byte)) = test.ranges
				return nil
			})}}
			if _, err := store.ListPropertyGraphDigests(t.Context(), jobID); err == nil {
				t.Fatal("ListPropertyGraphDigests() accepted invalid stored ranges")
			}
		})
	}
}

func TestPropertyGraphTransactionFailures(t *testing.T) {
	injected := errors.New("injected transaction failure")
	jobID := "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
	generation := PropertyGraphGeneration{
		JobID: jobID, Schema: "graph_data", Graph: "supply_graph",
		DefinitionFingerprint: strings.Repeat("a", 64), State: PropertyGraphLoading,
		Labels: []PropertyGraphLabel{{Name: "Person", Kind: VertexLabel, Table: "person"}},
	}
	for name, tx := range map[string]*scriptedLifecycleTx{
		"generation": {exec: []scriptedLifecycleExec{{err: injected}}},
		"label": {exec: []scriptedLifecycleExec{
			{tag: pgconn.NewCommandTag("INSERT 0 1")}, {err: injected},
		}},
	} {
		t.Run("register "+name, func(t *testing.T) {
			store := &Store{database: tx}
			if err := store.RegisterPropertyGraph(t.Context(), generation); !errors.Is(err, injected) {
				t.Fatalf("RegisterPropertyGraph() error = %v", err)
			}
		})
	}
	t.Run("registration commit", func(t *testing.T) {
		tx := &scriptedLifecycleTx{
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("INSERT 0 1")},
				{tag: pgconn.NewCommandTag("INSERT 0 1")},
			},
			commitErr: injected,
		}
		store := &Store{database: scriptedLifecycleDatabase{tx: tx}}
		if err := store.RegisterPropertyGraph(t.Context(), generation); !errors.Is(err, injected) {
			t.Fatalf("RegisterPropertyGraph() commit error = %v", err)
		}
	})
	rangeValue := PropertyGraphDigestRange{
		JobID: jobID, LabelName: "Person", Kind: VertexLabel,
		RangeID: 1, Rows: 1, Digest: strings.Repeat("c", 64),
	}
	for name, tx := range map[string]*scriptedLifecycleTx{
		"clear": {exec: []scriptedLifecycleExec{{err: injected}}},
		"range": {exec: []scriptedLifecycleExec{
			{tag: pgconn.NewCommandTag("DELETE 1")}, {err: injected},
		}},
		"root": {exec: []scriptedLifecycleExec{
			{tag: pgconn.NewCommandTag("DELETE 1")},
			{tag: pgconn.NewCommandTag("INSERT 0 1")}, {err: injected},
		}},
	} {
		t.Run("digest "+name, func(t *testing.T) {
			store := &Store{database: tx}
			if err := store.ReplacePropertyGraphDigests(t.Context(), jobID,
				[]PropertyGraphDigestRange{rangeValue}, strings.Repeat("d", 64), 1, 256,
			); err == nil {
				t.Fatal("ReplacePropertyGraphDigests() succeeded")
			}
		})
	}
	for name, tx := range map[string]*scriptedLifecycleTx{
		"missing root row": {
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("DELETE 1")},
				{tag: pgconn.NewCommandTag("INSERT 0 1")},
				{tag: pgconn.NewCommandTag("UPDATE 0")},
			},
		},
		"commit": {
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("DELETE 1")},
				{tag: pgconn.NewCommandTag("INSERT 0 1")},
				{tag: pgconn.NewCommandTag("UPDATE 1")},
			},
			commitErr: injected,
		},
	} {
		t.Run("digest "+name, func(t *testing.T) {
			store := &Store{database: scriptedLifecycleDatabase{tx: tx}}
			if err := store.ReplacePropertyGraphDigests(t.Context(), jobID,
				[]PropertyGraphDigestRange{rangeValue}, strings.Repeat("d", 64), 1, 256,
			); err == nil {
				t.Fatal("ReplacePropertyGraphDigests() succeeded")
			}
		})
	}
}

func TestPropertyGraphLifecycleQueriesAndTransitions(t *testing.T) {
	jobID := "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
	generation := PropertyGraphGeneration{
		JobID: jobID, Schema: "graph_data", Graph: "supply_graph",
		DefinitionFingerprint: strings.Repeat("a", 64), State: PropertyGraphLoading,
		Labels: []PropertyGraphLabel{{Name: "Person", Kind: VertexLabel, Table: "person"}},
	}

	for name, call := range map[string]func(*Store) error{
		"active missing": func(store *Store) error {
			_, err := store.ActivePropertyGraph(t.Context(), generation.Schema, generation.Graph)
			return err
		},
		"state missing": func(store *Store) error {
			_, err := store.PropertyGraphByTargetState(t.Context(), generation.Schema,
				generation.Graph, PropertyGraphRetainedBackup)
			return err
		},
	} {
		t.Run(name, func(t *testing.T) {
			store := &Store{database: propertyGraphRowDatabase{
				row: lifecycleErrorRow{err: pgx.ErrNoRows},
			}}
			if err := call(store); !errors.Is(err, ErrNotFound) {
				t.Fatalf("missing lifecycle query error = %v", err)
			}
		})
	}
	store := &Store{database: errorDatabase{}}
	if _, err := store.PropertyGraphByTargetState(t.Context(), "", generation.Graph,
		PropertyGraphActive); err == nil {
		t.Fatal("PropertyGraphByTargetState accepted an invalid schema")
	}
	if comparePropertyGraphDigestRanges(
		PropertyGraphDigestRange{Kind: VertexLabel, LabelName: "A"},
		PropertyGraphDigestRange{Kind: VertexLabel, LabelName: "B"},
	) >= 0 {
		t.Fatal("property graph digest labels are not ordered")
	}
	if comparePropertyGraphDigestRanges(
		PropertyGraphDigestRange{Kind: VertexLabel, LabelName: "A", RangeID: 1},
		PropertyGraphDigestRange{Kind: VertexLabel, LabelName: "A", RangeID: 2},
	) >= 0 {
		t.Fatal("property graph digest ranges are not ordered")
	}

	injected := errors.New("injected transition failure")
	for name, tx := range map[string]*scriptedLifecycleTx{
		"activate second update": {
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("UPDATE 1")}, {err: injected},
			},
		},
		"activate missing loading row": {
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("UPDATE 1")},
				{tag: pgconn.NewCommandTag("UPDATE 0")},
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			store := &Store{database: tx}
			if err := store.ActivatePropertyGraphReplacing(t.Context(), jobID,
				generation.Schema, generation.Graph); err == nil {
				t.Fatal("ActivatePropertyGraphReplacing succeeded")
			}
		})
	}

	for name, tx := range map[string]*scriptedLifecycleTx{
		"label update": {
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("UPDATE 1")}, {err: injected},
			},
		},
		"missing label": {
			exec: []scriptedLifecycleExec{
				{tag: pgconn.NewCommandTag("UPDATE 1")},
				{tag: pgconn.NewCommandTag("UPDATE 0")},
			},
		},
	} {
		t.Run("relocate "+name, func(t *testing.T) {
			store := &Store{database: tx}
			if err := store.RelocatePropertyGraph(t.Context(), generation); err == nil {
				t.Fatal("RelocatePropertyGraph succeeded")
			}
		})
	}
	t.Run("relocate missing generation", func(t *testing.T) {
		store := &Store{database: &scriptedLifecycleTx{exec: []scriptedLifecycleExec{{
			tag: pgconn.NewCommandTag("UPDATE 0"),
		}}}}
		if err := store.RelocatePropertyGraph(t.Context(), generation); err == nil {
			t.Fatal("RelocatePropertyGraph accepted a missing generation")
		}
	})
}

type propertyGraphRowDatabase struct{ row pgx.Row }

func (database propertyGraphRowDatabase) Begin(context.Context) (pgx.Tx, error) {
	panic("unexpected Begin")
}

func (database propertyGraphRowDatabase) Exec(context.Context, string, ...any) (pgconn.CommandTag, error) {
	panic("unexpected Exec")
}

func (database propertyGraphRowDatabase) QueryRow(context.Context, string, ...any) pgx.Row {
	return database.row
}

func TestReplacePropertyGraphDigestsValidation(t *testing.T) {
	store := &Store{database: errorDatabase{}}
	jobID := "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
	valid := PropertyGraphDigestRange{
		JobID: jobID, LabelName: "Person", Kind: VertexLabel,
		RangeID: 7, Rows: 1, Digest: strings.Repeat("a", 64),
	}
	tests := []struct {
		name       string
		jobID      string
		ranges     []PropertyGraphDigestRange
		root       string
		rows       int64
		rangeCount int
	}{
		{"job", "bad", nil, strings.Repeat("b", 64), 0, 256},
		{"root", jobID, nil, "bad", 0, 256},
		{"rows", jobID, nil, strings.Repeat("b", 64), -1, 256},
		{"range count", jobID, nil, strings.Repeat("b", 64), 0, 0},
		{"range job", jobID, []PropertyGraphDigestRange{
			func() PropertyGraphDigestRange { value := valid; value.JobID = "other"; return value }(),
		}, strings.Repeat("b", 64), 1, 256},
		{"label", jobID, []PropertyGraphDigestRange{
			func() PropertyGraphDigestRange { value := valid; value.LabelName = ""; return value }(),
		}, strings.Repeat("b", 64), 1, 256},
		{"kind", jobID, []PropertyGraphDigestRange{
			func() PropertyGraphDigestRange { value := valid; value.Kind = 'x'; return value }(),
		}, strings.Repeat("b", 64), 1, 256},
		{"empty rows", jobID, []PropertyGraphDigestRange{
			func() PropertyGraphDigestRange { value := valid; value.Rows = 0; return value }(),
		}, strings.Repeat("b", 64), 1, 256},
		{"range digest", jobID, []PropertyGraphDigestRange{
			func() PropertyGraphDigestRange { value := valid; value.Digest = "bad"; return value }(),
		}, strings.Repeat("b", 64), 1, 256},
		{"duplicate", jobID, []PropertyGraphDigestRange{valid, valid},
			strings.Repeat("b", 64), 2, 256},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := store.ReplacePropertyGraphDigests(
				context.Background(), test.jobID, test.ranges, test.root,
				test.rows, test.rangeCount,
			); err == nil {
				t.Fatal("invalid digest replacement succeeded")
			}
		})
	}
}
