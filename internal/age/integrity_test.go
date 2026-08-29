package age

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/rioriost/agefreighter/internal/meta"
)

func TestSetStatementTimeoutValidationExecutionAndErrors(t *testing.T) {
	transaction := &Transaction{tx: &integrityTx{}}
	if err := transaction.SetStatementTimeout(t.Context(), 0); err == nil {
		t.Fatal("SetStatementTimeout accepted zero duration")
	}
	if err := transaction.SetStatementTimeout(context.Background(), time.Second); err == nil {
		t.Fatal("SetStatementTimeout accepted context without deadline")
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	tx := &integrityTx{}
	transaction.tx = tx
	if err := transaction.SetStatementTimeout(ctx, 500*time.Microsecond); err != nil {
		t.Fatalf("SetStatementTimeout() error = %v", err)
	}
	if len(tx.execArguments) != 1 || len(tx.execArguments[0]) != 1 || tx.execArguments[0][0] != "1ms" {
		t.Fatalf("statement timeout arguments = %#v", tx.execArguments)
	}
	injected := errors.New("exec failed")
	tx.execErr = injected
	if err := transaction.SetStatementTimeout(ctx, time.Second); !errors.Is(err, injected) {
		t.Fatalf("SetStatementTimeout() database error = %v", err)
	}
}

func TestVerifyBoundedIntegrityVertexAndEdgeResults(t *testing.T) {
	label := integrityLabel()
	vertexTx := &integrityTx{rows: []pgx.Row{
		integrityCatalogRow(label),
		integrityRow(func(dest ...any) error {
			*dest[0].(*int64) = 3
			*dest[1].(*bool) = true
			*dest[2].(*int64) = 1
			return nil
		}),
		integrityRow(func(dest ...any) error {
			*dest[0].(*int64) = 4
			*dest[1].(*bool) = false
			*dest[2].(*int64) = 2
			return nil
		}),
	}}
	result, err := (&Transaction{tx: vertexTx}).VerifyBoundedIntegrity(
		t.Context(), label, 1, 2, meta.VertexLabel, 10,
	)
	if err != nil || result.IdentityRows != 3 || !result.IdentityTruncated ||
		result.MissingPhysicalRows != 1 || result.PhysicalRows != 4 ||
		result.OrphanPhysicalRows != 2 || !result.PhysicalCoverageChecked {
		t.Fatalf("vertex VerifyBoundedIntegrity() = %#v, %v", result, err)
	}
	if len(vertexTx.statements) != 3 ||
		!strings.Contains(vertexTx.statements[1], "vertex_identity") ||
		!strings.Contains(vertexTx.statements[2], "vertex_identity") {
		t.Fatalf("vertex statements = %#v", vertexTx.statements)
	}

	edgeTx := &integrityTx{rows: []pgx.Row{
		integrityCatalogRow(label),
		integrityRow(func(dest ...any) error {
			*dest[0].(*int64) = 5
			*dest[1].(*bool) = false
			*dest[2].(*int64) = 1
			*dest[3].(*int64) = 2
			*dest[4].(*int64) = 3
			return nil
		}),
	}}
	result, err = (&Transaction{tx: edgeTx}).VerifyBoundedIntegrityForIdentityCoverage(
		t.Context(), label, 1, 2, meta.EdgeLabel, 10, false,
	)
	if err != nil || result.IdentityRows != 5 || result.MissingEndpointRows != 2 ||
		result.ChangedEndpointRows != 3 || result.PhysicalCoverageChecked {
		t.Fatalf("bounded edge verification = %#v, %v", result, err)
	}
	if len(edgeTx.rows) != 0 {
		t.Fatalf("edge verification left %d rows", len(edgeTx.rows))
	}

	fullEdgeTx := &integrityTx{rows: []pgx.Row{
		integrityCatalogRow(label),
		integrityRow(func(dest ...any) error {
			*dest[0].(*int64) = 1
			return nil
		}),
		integrityRow(func(dest ...any) error {
			*dest[0].(*int64) = 2
			*dest[1].(*bool) = true
			*dest[2].(*int64) = 1
			return nil
		}),
	}}
	result, err = (&Transaction{tx: fullEdgeTx}).VerifyBoundedIntegrity(
		t.Context(), label, 1, 2, meta.EdgeLabel, 10,
	)
	if err != nil || !result.PhysicalCoverageChecked || !result.PhysicalTruncated ||
		result.PhysicalRows != 2 || result.OrphanPhysicalRows != 1 {
		t.Fatalf("full edge verification = %#v, %v", result, err)
	}
	if !strings.Contains(fullEdgeTx.statements[2], "edge_identity") {
		t.Fatalf("edge physical statement = %q", fullEdgeTx.statements[2])
	}
}

func TestVerifyBoundedIntegrityValidationCatalogAndScanErrors(t *testing.T) {
	label := integrityLabel()
	for _, test := range []struct {
		name    string
		graphID int64
		labelID int64
		limit   int
		kind    meta.LabelKind
		rows    []pgx.Row
		want    string
		full    bool
	}{
		{"graph ID", 0, 1, 1, meta.VertexLabel, nil, "must be positive", true},
		{"label ID", 1, 0, 1, meta.VertexLabel, nil, "must be positive", true},
		{"limit", 1, 1, 0, meta.VertexLabel, nil, "limit must be positive", true},
		{"lookup", 1, 1, 1, meta.VertexLabel, []pgx.Row{
			integrityRow(func(...any) error { return errors.New("lookup failed") }),
		}, "lookup failed", true},
		{"changed catalog", 1, 1, 1, meta.VertexLabel, []pgx.Row{
			integrityCatalogRow(withIntegrityLabel(label, func(value *LabelCatalog) { value.LabelID++ })),
		}, "catalog changed", true},
		{"kind", 1, 1, 1, meta.LabelKind('x'), []pgx.Row{
			integrityCatalogRow(label),
		}, "unsupported label kind", true},
		{"identity scan", 1, 1, 1, meta.VertexLabel, []pgx.Row{
			integrityCatalogRow(label),
			integrityRow(func(...any) error { return errors.New("identity failed") }),
		}, "identity integrity", true},
		{"physical scan", 1, 1, 1, meta.VertexLabel, []pgx.Row{
			integrityCatalogRow(label),
			integrityRow(func(...any) error { return nil }),
			integrityRow(func(...any) error { return errors.New("physical failed") }),
		}, "physical integrity", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := (&Transaction{tx: &integrityTx{rows: test.rows}}).
				VerifyBoundedIntegrityForIdentityCoverage(
					t.Context(), label, test.graphID, test.labelID,
					test.kind, test.limit, test.full,
				)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("VerifyBoundedIntegrityForIdentityCoverage() error = %v, want %q", err, test.want)
			}
		})
	}
}

type integrityTx struct {
	pgx.Tx
	rows          []pgx.Row
	statements    []string
	execArguments [][]any
	execErr       error
}

func (tx *integrityTx) Exec(
	_ context.Context,
	statement string,
	arguments ...any,
) (pgconn.CommandTag, error) {
	tx.statements = append(tx.statements, statement)
	tx.execArguments = append(tx.execArguments, arguments)
	return pgconn.CommandTag{}, tx.execErr
}

func (tx *integrityTx) QueryRow(
	_ context.Context,
	statement string,
	_ ...any,
) pgx.Row {
	tx.statements = append(tx.statements, statement)
	row := tx.rows[0]
	tx.rows = tx.rows[1:]
	return row
}

type integrityRow func(...any) error

func (row integrityRow) Scan(dest ...any) error { return row(dest...) }

func integrityCatalogRow(label LabelCatalog) integrityRow {
	return func(dest ...any) error {
		*dest[0].(*string) = label.GraphName
		*dest[1].(*string) = label.LabelName
		*dest[2].(*uint32) = label.GraphOID
		*dest[3].(*uint32) = label.NamespaceOID
		*dest[4].(*int32) = int32(label.LabelID)
		*dest[5].(*string) = string(label.Kind)
		*dest[6].(*uint32) = label.RelationOID
		*dest[7].(*uint32) = label.NamespaceOID
		*dest[8].(*uint32) = label.SequenceOID
		*dest[9].(*string) = label.SequenceName
		return nil
	}
}

func integrityLabel() LabelCatalog {
	return LabelCatalog{
		GraphName: "graph", LabelName: "Person",
		GraphOID: 2, NamespaceOID: 2, LabelID: 3, Kind: VertexLabel,
		RelationOID: 4, SequenceOID: 5, SequenceName: "Person_id_seq",
	}
}

func withIntegrityLabel(value LabelCatalog, change func(*LabelCatalog)) LabelCatalog {
	change(&value)
	return value
}
