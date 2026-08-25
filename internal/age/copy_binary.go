package age

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"strings"

	"github.com/jackc/pgx/v5"
)

var binaryCopyHeader = []byte("PGCOPY\n\xff\r\n\x00\x00\x00\x00\x00\x00\x00\x00\x00")

type copyBinaryReader struct {
	rowCount int
	index    int
	chunk    []byte
	offset   int
	rowAt    func(int, []byte) []byte
}

func (reader *copyBinaryReader) Read(output []byte) (int, error) {
	if len(output) == 0 {
		return 0, nil
	}
	written := 0
	for written < len(output) {
		for reader.offset >= len(reader.chunk) {
			switch {
			case reader.index == 0:
				reader.chunk = append(reader.chunk[:0], binaryCopyHeader...)
			case reader.index <= reader.rowCount:
				reader.chunk = reader.rowAt(reader.index-1, reader.chunk[:0])
			case reader.index == reader.rowCount+1:
				reader.chunk = appendBinaryInt16(reader.chunk[:0], -1)
			default:
				if written > 0 {
					return written, nil
				}
				return 0, io.EOF
			}
			reader.index++
			reader.offset = 0
		}
		copied := copy(output[written:], reader.chunk[reader.offset:])
		reader.offset += copied
		written += copied
	}
	return written, nil
}

func (transaction *Transaction) copyBinaryTable(
	ctx context.Context,
	table pgx.Identifier,
	columns []string,
	reader io.Reader,
	expectedRows int,
) (int64, error) {
	tableName := table.Sanitize()
	quotedColumns := make([]string, len(columns))
	for index, column := range columns {
		quotedColumns[index] = pgx.Identifier{column}.Sanitize()
	}
	command := fmt.Sprintf(
		"COPY %s (%s) FROM STDIN WITH (FORMAT binary)",
		tableName,
		strings.Join(quotedColumns, ", "),
	)
	tag, err := transaction.tx.Conn().PgConn().CopyFrom(ctx, reader, command)
	if err != nil {
		return 0, err
	}
	rows := tag.RowsAffected()
	if rows != int64(expectedRows) {
		return 0, fmt.Errorf(
			"COPY into %s wrote %d rows, expected %d",
			tableName,
			rows,
			expectedRows,
		)
	}
	return rows, nil
}

func appendBinaryInt16(output []byte, value int16) []byte {
	return binary.BigEndian.AppendUint16(output, uint16(value))
}

func appendBinaryInt32(output []byte, value int32) []byte {
	return binary.BigEndian.AppendUint32(output, uint32(value))
}

func appendBinaryInt64(output []byte, value int64) []byte {
	return binary.BigEndian.AppendUint64(output, uint64(value))
}

func appendBinaryInt32Field(output []byte, value int32) []byte {
	output = appendBinaryInt32(output, 4)
	return appendBinaryInt32(output, value)
}

func appendBinaryInt64Field(output []byte, value int64) []byte {
	output = appendBinaryInt32(output, 8)
	return appendBinaryInt64(output, value)
}

func appendBinaryTextField[Bytes ~[]byte | ~string](
	output []byte,
	value Bytes,
) []byte {
	output = appendBinaryInt32(output, int32(len(value)))
	return append(output, value...)
}
