package age

import (
	"context"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"
)

type VertexRow struct {
	ID         GraphID
	Properties []byte
}

type EdgeRow struct {
	ID         GraphID
	StartID    GraphID
	EndID      GraphID
	Properties []byte
}

type CopyStrategy string

const (
	DirectTextCopy   CopyStrategy = "direct-text"
	StagedBinaryCopy CopyStrategy = "staged-binary"
)

func (transaction *Transaction) CopyVertices(
	ctx context.Context,
	label LabelCatalog,
	rows []VertexRow,
	strategy CopyStrategy,
) (int64, error) {
	if label.Kind != VertexLabel {
		return 0, fmt.Errorf("label %q is not a vertex label", label.LabelName)
	}
	if err := validateVertexRows(label, rows); err != nil {
		return 0, err
	}
	switch strategy {
	case DirectTextCopy:
		return transaction.copyVerticesText(ctx, label, rows)
	case StagedBinaryCopy:
		return transaction.copyVerticesStaged(ctx, label, rows)
	default:
		return 0, fmt.Errorf("unsupported COPY strategy %q", strategy)
	}
}

func (transaction *Transaction) CopyEdges(
	ctx context.Context,
	label LabelCatalog,
	rows []EdgeRow,
	strategy CopyStrategy,
) (int64, error) {
	if label.Kind != EdgeLabel {
		return 0, fmt.Errorf("label %q is not an edge label", label.LabelName)
	}
	if err := validateEdgeRows(label, rows); err != nil {
		return 0, err
	}
	switch strategy {
	case DirectTextCopy:
		return transaction.copyEdgesText(ctx, label, rows)
	case StagedBinaryCopy:
		return transaction.copyEdgesStaged(ctx, label, rows)
	default:
		return 0, fmt.Errorf("unsupported COPY strategy %q", strategy)
	}
}

func validateVertexRows(label LabelCatalog, rows []VertexRow) error {
	for index, row := range rows {
		if err := validateRowID(label, row.ID); err != nil {
			return fmt.Errorf("vertex row %d: %w", index, err)
		}
		if len(row.Properties) == 0 {
			return fmt.Errorf("vertex row %d has empty properties", index)
		}
	}
	return nil
}

func validateEdgeRows(label LabelCatalog, rows []EdgeRow) error {
	for index, row := range rows {
		if err := validateRowID(label, row.ID); err != nil {
			return fmt.Errorf("edge row %d: %w", index, err)
		}
		if err := row.StartID.Validate(); err != nil {
			return fmt.Errorf("edge row %d start ID: %w", index, err)
		}
		if err := row.EndID.Validate(); err != nil {
			return fmt.Errorf("edge row %d end ID: %w", index, err)
		}
		if len(row.Properties) == 0 {
			return fmt.Errorf("edge row %d has empty properties", index)
		}
	}
	return nil
}

func validateRowID(label LabelCatalog, id GraphID) error {
	if err := id.Validate(); err != nil {
		return err
	}
	if id.LabelID() != label.LabelID {
		return fmt.Errorf(
			"graphid label %d does not match catalog label %d",
			id.LabelID(),
			label.LabelID,
		)
	}
	return nil
}

func (transaction *Transaction) copyVerticesText(
	ctx context.Context,
	label LabelCatalog,
	rows []VertexRow,
) (int64, error) {
	reader := &copyTextReader{
		rowCount: len(rows),
		vertexAt: func(index int, output []byte) []byte {
			output = strconv.AppendInt(output, int64(rows[index].ID), 10)
			output = append(output, '\t')
			output = appendCopyText(output, rows[index].Properties)
			return append(output, '\n')
		},
	}
	return transaction.copyText(
		ctx,
		label,
		[]string{"id", "properties"},
		reader,
		len(rows),
	)
}

func (transaction *Transaction) copyEdgesText(
	ctx context.Context,
	label LabelCatalog,
	rows []EdgeRow,
) (int64, error) {
	reader := &copyTextReader{
		rowCount: len(rows),
		vertexAt: func(index int, output []byte) []byte {
			row := rows[index]
			output = strconv.AppendInt(output, int64(row.ID), 10)
			output = append(output, '\t')
			output = strconv.AppendInt(output, int64(row.StartID), 10)
			output = append(output, '\t')
			output = strconv.AppendInt(output, int64(row.EndID), 10)
			output = append(output, '\t')
			output = appendCopyText(output, row.Properties)
			return append(output, '\n')
		},
	}
	return transaction.copyText(
		ctx,
		label,
		[]string{"id", "start_id", "end_id", "properties"},
		reader,
		len(rows),
	)
}

func (transaction *Transaction) copyText(
	ctx context.Context,
	label LabelCatalog,
	columns []string,
	reader io.Reader,
	expectedRows int,
) (int64, error) {
	table := pgx.Identifier{label.GraphName, label.LabelName}.Sanitize()
	quotedColumns := make([]string, len(columns))
	for index, column := range columns {
		quotedColumns[index] = pgx.Identifier{column}.Sanitize()
	}
	command := fmt.Sprintf(
		"COPY %s (%s) FROM STDIN WITH (FORMAT text)",
		table,
		strings.Join(quotedColumns, ", "),
	)
	tag, err := transaction.tx.Conn().PgConn().CopyFrom(ctx, reader, command)
	if err != nil {
		return 0, fmt.Errorf("direct text COPY into %s: %w", table, err)
	}
	rowsAffected := tag.RowsAffected()
	if rowsAffected != int64(expectedRows) {
		return 0, fmt.Errorf(
			"direct text COPY into %s wrote %d rows, expected %d",
			table,
			rowsAffected,
			expectedRows,
		)
	}
	return rowsAffected, nil
}

type copyTextReader struct {
	rowCount int
	index    int
	line     []byte
	offset   int
	vertexAt func(int, []byte) []byte
}

func (reader *copyTextReader) Read(output []byte) (int, error) {
	if len(output) == 0 {
		return 0, nil
	}
	for reader.offset >= len(reader.line) {
		if reader.index >= reader.rowCount {
			return 0, io.EOF
		}
		reader.line = reader.vertexAt(reader.index, reader.line[:0])
		reader.index++
		reader.offset = 0
	}
	copied := copy(output, reader.line[reader.offset:])
	reader.offset += copied
	return copied, nil
}

func appendCopyText(output []byte, value []byte) []byte {
	for _, character := range value {
		switch character {
		case '\\':
			output = append(output, '\\', '\\')
		case '\t':
			output = append(output, '\\', 't')
		case '\n':
			output = append(output, '\\', 'n')
		case '\r':
			output = append(output, '\\', 'r')
		default:
			output = append(output, character)
		}
	}
	return output
}

func (transaction *Transaction) copyVerticesStaged(
	ctx context.Context,
	label LabelCatalog,
	rows []VertexRow,
) (int64, error) {
	const stage = "agefreighter_vertex_stage"
	if _, err := transaction.tx.Exec(
		ctx,
		`CREATE TEMP TABLE IF NOT EXISTS pg_temp.agefreighter_vertex_stage (
			id bigint NOT NULL,
			properties text NOT NULL
		) ON COMMIT DROP`,
	); err != nil {
		return 0, fmt.Errorf("prepare vertex staging table: %w", err)
	}
	if _, err := transaction.tx.Exec(
		ctx,
		"TRUNCATE pg_temp.agefreighter_vertex_stage",
	); err != nil {
		return 0, fmt.Errorf("truncate vertex staging table: %w", err)
	}
	copied, err := transaction.tx.CopyFrom(
		ctx,
		pgx.Identifier{"pg_temp", stage},
		[]string{"id", "properties"},
		pgx.CopyFromSlice(len(rows), func(index int) ([]any, error) {
			return []any{int64(rows[index].ID), string(rows[index].Properties)}, nil
		}),
	)
	if err != nil {
		return 0, fmt.Errorf("binary COPY vertex staging table: %w", err)
	}
	if copied != int64(len(rows)) {
		return 0, fmt.Errorf("binary COPY staged %d vertex rows, expected %d", copied, len(rows))
	}
	table := pgx.Identifier{label.GraphName, label.LabelName}.Sanitize()
	tag, err := transaction.tx.Exec(
		ctx,
		fmt.Sprintf(
			`INSERT INTO %s (id, properties)
			 SELECT id::text::ag_catalog.graphid,
			        properties::ag_catalog.agtype
			 FROM pg_temp.%s`,
			table,
			pgx.Identifier{stage}.Sanitize(),
		),
	)
	if err != nil {
		return 0, fmt.Errorf("merge staged vertices into %s: %w", table, err)
	}
	return requireAffectedRows("merge staged vertices", tag.RowsAffected(), len(rows))
}

func (transaction *Transaction) copyEdgesStaged(
	ctx context.Context,
	label LabelCatalog,
	rows []EdgeRow,
) (int64, error) {
	const stage = "agefreighter_edge_stage"
	if _, err := transaction.tx.Exec(
		ctx,
		`CREATE TEMP TABLE IF NOT EXISTS pg_temp.agefreighter_edge_stage (
			id bigint NOT NULL,
			start_id bigint NOT NULL,
			end_id bigint NOT NULL,
			properties text NOT NULL
		) ON COMMIT DROP`,
	); err != nil {
		return 0, fmt.Errorf("prepare edge staging table: %w", err)
	}
	if _, err := transaction.tx.Exec(
		ctx,
		"TRUNCATE pg_temp.agefreighter_edge_stage",
	); err != nil {
		return 0, fmt.Errorf("truncate edge staging table: %w", err)
	}
	copied, err := transaction.tx.CopyFrom(
		ctx,
		pgx.Identifier{"pg_temp", stage},
		[]string{"id", "start_id", "end_id", "properties"},
		pgx.CopyFromSlice(len(rows), func(index int) ([]any, error) {
			row := rows[index]
			return []any{
				int64(row.ID),
				int64(row.StartID),
				int64(row.EndID),
				string(row.Properties),
			}, nil
		}),
	)
	if err != nil {
		return 0, fmt.Errorf("binary COPY edge staging table: %w", err)
	}
	if copied != int64(len(rows)) {
		return 0, fmt.Errorf("binary COPY staged %d edge rows, expected %d", copied, len(rows))
	}
	table := pgx.Identifier{label.GraphName, label.LabelName}.Sanitize()
	tag, err := transaction.tx.Exec(
		ctx,
		fmt.Sprintf(
			`INSERT INTO %s (id, start_id, end_id, properties)
			 SELECT id::text::ag_catalog.graphid,
			        start_id::text::ag_catalog.graphid,
			        end_id::text::ag_catalog.graphid,
			        properties::ag_catalog.agtype
			 FROM pg_temp.%s`,
			table,
			pgx.Identifier{stage}.Sanitize(),
		),
	)
	if err != nil {
		return 0, fmt.Errorf("merge staged edges into %s: %w", table, err)
	}
	return requireAffectedRows("merge staged edges", tag.RowsAffected(), len(rows))
}

func requireAffectedRows(operation string, actual int64, expected int) (int64, error) {
	if actual != int64(expected) {
		return 0, fmt.Errorf("%s affected %d rows, expected %d", operation, actual, expected)
	}
	return actual, nil
}

var _ io.Reader = (*copyTextReader)(nil)
