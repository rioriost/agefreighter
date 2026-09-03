package pggraph

import (
	"context"
	"errors"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestRenameDefinitionTablesRejectsMisalignedMappings(t *testing.T) {
	from := Definition{
		Schema: "graph_data", Graph: "supply_graph",
		Vertices: []VertexDefinition{{Table: "person", Label: "Person"}},
		Edges: []EdgeDefinition{{
			Table: "knows", Label: "KNOWS", SourceTable: "person", DestinationTable: "person",
		}},
	}
	for name, to := range map[string]Definition{
		"counts": {Schema: from.Schema, Graph: from.Graph},
		"vertex label": {
			Schema: from.Schema, Graph: from.Graph,
			Vertices: []VertexDefinition{{Table: "other", Label: "Other"}},
			Edges:    from.Edges,
		},
		"edge label": {
			Schema: from.Schema, Graph: from.Graph,
			Vertices: from.Vertices,
			Edges: []EdgeDefinition{{
				Table: "other", Label: "OTHER", SourceTable: "person", DestinationTable: "person",
			}},
		},
	} {
		t.Run(name, func(t *testing.T) {
			if err := renameDefinitionTables(t.Context(), nil, from, to); err == nil {
				t.Fatal("renameDefinitionTables accepted misaligned mappings")
			}
		})
	}
}

func TestInspectFinalTargetRejectsInvalidEvidence(t *testing.T) {
	injected := errors.New("injected inspection failure")
	definition := Definition{
		Schema: "graph_data", Graph: "supply_graph",
		Vertices: []VertexDefinition{{Table: "person", Label: "Person"}},
	}
	tests := map[string]*inspectionTestTx{
		"search path": {execErr: injected},
		"table count": {rows: []pgx.Row{inspectionTestRow{err: injected}}},
		"graph query": {rows: []pgx.Row{
			inspectionTestRow{value: 1}, inspectionTestRow{err: injected},
		}},
		"verification table": {rows: []pgx.Row{
			inspectionTestRow{value: 1}, inspectionTestRow{value: 1},
			inspectionTestRow{err: injected},
		}},
		"SQL PGQ mismatch": {rows: []pgx.Row{
			inspectionTestRow{value: 1}, inspectionTestRow{value: 2},
			inspectionTestRow{value: 1},
		}},
		"digest query": {
			rows: []pgx.Row{
				inspectionTestRow{value: 1}, inspectionTestRow{value: 1},
				inspectionTestRow{value: 1},
			},
			queryErr: injected,
		},
	}
	for name, tx := range tests {
		t.Run(name, func(t *testing.T) {
			if _, _, err := inspectFinalTarget(t.Context(), tx,
				"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", definition); err == nil {
				t.Fatal("inspectFinalTarget accepted invalid evidence")
			}
		})
	}
}

type inspectionTestTx struct {
	pgx.Tx
	execErr  error
	queryErr error
	rows     []pgx.Row
}

func (tx *inspectionTestTx) Query(
	context.Context, string, ...any,
) (pgx.Rows, error) {
	return nil, tx.queryErr
}

func (tx *inspectionTestTx) Exec(
	context.Context, string, ...any,
) (pgconn.CommandTag, error) {
	return pgconn.CommandTag{}, tx.execErr
}

func (tx *inspectionTestTx) QueryRow(context.Context, string, ...any) pgx.Row {
	row := tx.rows[0]
	tx.rows = tx.rows[1:]
	return row
}

type inspectionTestRow struct {
	value int64
	err   error
}

func (row inspectionTestRow) Scan(destinations ...any) error {
	if row.err != nil {
		return row.err
	}
	if len(destinations) != 1 {
		return errors.New("unexpected inspection destination count")
	}
	value, ok := destinations[0].(*int64)
	if !ok {
		return errors.New("unexpected inspection destination type")
	}
	*value = row.value
	return nil
}
