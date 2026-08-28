package postgres

import (
	"bufio"
	"bytes"
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"strconv"
	"sync"

	"github.com/jackc/pgx/v5"
)

type sourceRow struct {
	raw       []byte
	key       *keyValue
	accounted bool
}

type recordReader interface {
	Next(context.Context) (sourceRow, error)
	Close() error
}

type telemetryReader struct {
	input   io.Reader
	observe func(int64) error
}

func (reader *telemetryReader) Read(output []byte) (int, error) {
	count, err := reader.input.Read(output)
	if count > 0 {
		if observeErr := reader.observe(int64(count)); observeErr != nil {
			return count, observeErr
		}
	}
	return count, err
}

func openRecordReader(
	ctx context.Context,
	coordinator *SnapshotCoordinator,
	mapping compiledMapping,
	mode string,
	fetchRows int,
	maxRecordBytes int64,
	afterKey *keyValue,
	telemetry *telemetryState,
) (recordReader, error) {
	reader, err := coordinator.OpenReader(ctx)
	if err != nil {
		telemetry.failure()
		return nil, err
	}
	var opened recordReader
	switch mode {
	case "copy":
		opened, err = newCopyReader(reader, mapping, maxRecordBytes, telemetry)
	case "cursor":
		opened, err = newCursorReader(ctx, reader, mapping, fetchRows, telemetry)
	case "keyset":
		opened = newKeysetReader(reader, mapping, fetchRows, afterKey, telemetry)
	default:
		err = errors.New("PostgreSQL read mode is invalid")
	}
	if err != nil {
		_ = reader.Close()
		return nil, err
	}
	return opened, nil
}

type copyReader struct {
	snapshot  *SnapshotReader
	pipe      *io.PipeReader
	scanner   *bufio.Scanner
	cancel    context.CancelFunc
	done      chan error
	telemetry *telemetryState
	completed bool
	once      sync.Once
	err       error
}

func newCopyReader(
	reader *SnapshotReader,
	mapping compiledMapping,
	maxRecordBytes int64,
	telemetry *telemetryState,
) (*copyReader, error) {
	pipeReader, pipeWriter := io.Pipe()
	streamCtx, cancel := context.WithCancel(context.Background())
	scanner := bufio.NewScanner(&telemetryReader{
		input: pipeReader, observe: telemetry.raw,
	})
	maxWireBytes := boundedScannerSize(maxRecordBytes)
	initial := 64 << 10
	if maxWireBytes < initial {
		initial = maxWireBytes
	}
	scanner.Buffer(make([]byte, initial), maxWireBytes)
	copy := &copyReader{
		snapshot: reader, pipe: pipeReader, scanner: scanner,
		cancel: cancel, done: make(chan error, 1), telemetry: telemetry,
	}
	sql := "COPY (" + rowJSONQuery(mapping.query) + ") TO STDOUT WITH (FORMAT csv)"
	if err := telemetry.canFetchPage(); err != nil {
		cancel()
		_ = pipeReader.Close()
		_ = reader.Close()
		return nil, err
	}
	if err := telemetry.page(); err != nil {
		cancel()
		_ = pipeReader.Close()
		_ = reader.Close()
		return nil, err
	}
	go func() {
		_, err := reader.conn.PgConn().CopyTo(streamCtx, pipeWriter, sql)
		if err != nil {
			telemetry.failure()
			err = safeDatabaseError(streamCtx, "stream PostgreSQL COPY", err)
		}
		_ = pipeWriter.CloseWithError(err)
		copy.done <- err
	}()
	return copy, nil
}

func boundedScannerSize(maxRecordBytes int64) int {
	maxInt := int64(^uint(0) >> 1)
	if maxRecordBytes > (maxInt-1024)/2 {
		return int(maxInt)
	}
	size := maxRecordBytes*2 + 1024
	if size < 1 {
		return 1
	}
	return int(size)
}

func rowJSONQuery(query string) string {
	return "SELECT row_to_json(af_row)::text FROM (\n" +
		query + "\n) AS af_row"
}

func (reader *copyReader) Next(ctx context.Context) (sourceRow, error) {
	stop := context.AfterFunc(ctx, func() {
		reader.cancel()
		_ = reader.pipe.CloseWithError(ctx.Err())
	})
	scanned := reader.scanner.Scan()
	stop()
	if err := ctx.Err(); err != nil {
		return sourceRow{}, ctx.Err()
	}
	if scanned {
		raw, err := decodeCopyCSV(reader.scanner.Bytes())
		return sourceRow{raw: raw}, err
	}

	if err := reader.scanner.Err(); err != nil {
		return sourceRow{}, err
	}
	copyErr := <-reader.done
	reader.completed = true
	if copyErr != nil {
		return sourceRow{}, copyErr
	}
	return sourceRow{}, io.EOF
}

func decodeCopyCSV(wire []byte) ([]byte, error) {
	reader := csv.NewReader(bytes.NewReader(wire))
	reader.FieldsPerRecord = 1
	record, err := reader.Read()
	if err != nil {
		return nil, errors.New("PostgreSQL COPY CSV row is invalid")
	}
	if len(record) != 1 {
		return nil, errors.New("PostgreSQL COPY CSV row must contain exactly one field")
	}
	if _, err := reader.Read(); !errors.Is(err, io.EOF) {
		return nil, errors.New("PostgreSQL COPY CSV row has trailing content")
	}
	return []byte(record[0]), nil
}

func (reader *copyReader) Close() error {
	reader.once.Do(func() {
		reader.cancel()
		_ = reader.pipe.Close()
		if !reader.completed {
			<-reader.done
			reader.completed = true
		}
		reader.err = reader.snapshot.Close()
	})
	return reader.err
}

type cursorReader struct {
	snapshot  *SnapshotReader
	fetchRows int
	rows      pgx.Rows
	query     func(context.Context, string, ...any) (pgx.Rows, error)
	pageRows  int
	done      bool
	telemetry *telemetryState
	once      sync.Once
	err       error
}

func newCursorReader(
	ctx context.Context,
	reader *SnapshotReader,
	mapping compiledMapping,
	fetchRows int,
	telemetry *telemetryState,
) (*cursorReader, error) {
	sql := "DECLARE agefreighter_cursor NO SCROLL CURSOR FOR " +
		rowJSONQuery(mapping.query)
	if _, err := reader.tx.Exec(ctx, sql); err != nil {
		telemetry.failure()
		return nil, safeDatabaseError(ctx, "declare PostgreSQL cursor", err)
	}
	return &cursorReader{
		snapshot: reader, fetchRows: fetchRows, telemetry: telemetry,
	}, nil
}

func (reader *cursorReader) Next(ctx context.Context) (sourceRow, error) {
	for {
		if reader.done {
			return sourceRow{}, io.EOF
		}
		if reader.rows == nil {
			if err := reader.telemetry.canFetchPage(); err != nil {
				return sourceRow{}, err
			}
			sql := "FETCH FORWARD " + strconv.Itoa(reader.fetchRows) +
				" FROM agefreighter_cursor"
			query := reader.query
			if query == nil {
				query = reader.snapshot.tx.Query
			}
			rows, err := query(ctx, sql)
			if err != nil {
				reader.telemetry.failure()
				return sourceRow{}, safeDatabaseError(ctx, "fetch PostgreSQL cursor", err)
			}
			if err := reader.telemetry.page(); err != nil {
				rows.Close()
				return sourceRow{}, err
			}
			reader.rows = rows
			reader.pageRows = 0
		}
		if reader.rows.Next() {
			var text string
			if err := reader.rows.Scan(&text); err != nil {
				reader.telemetry.failure()
				return sourceRow{}, safeDatabaseError(ctx, "scan PostgreSQL cursor row", err)
			}
			reader.pageRows++
			return sourceRow{raw: []byte(text)}, nil
		}
		err := reader.rows.Err()
		reader.rows.Close()
		reader.rows = nil
		if err != nil {
			reader.telemetry.failure()
			return sourceRow{}, safeDatabaseError(ctx, "read PostgreSQL cursor fetch", err)
		}
		if reader.pageRows < reader.fetchRows {
			reader.done = true
			return sourceRow{}, io.EOF
		}
		if reader.pageRows == 0 {
			reader.done = true
			return sourceRow{}, io.EOF
		}
	}
}

func (reader *cursorReader) Close() error {
	reader.once.Do(func() {
		if reader.rows != nil {
			reader.rows.Close()
		}
		reader.err = reader.snapshot.Close()
	})
	return reader.err
}

type keysetReader struct {
	snapshot  *SnapshotReader
	mapping   compiledMapping
	fetchRows int
	rows      pgx.Rows
	query     func(context.Context, string, ...any) (pgx.Rows, error)
	pageRows  int
	done      bool
	lastKey   *keyValue
	telemetry *telemetryState
	once      sync.Once
	err       error
}

func newKeysetReader(
	reader *SnapshotReader,
	mapping compiledMapping,
	fetchRows int,
	afterKey *keyValue,
	telemetry *telemetryState,
) *keysetReader {
	return &keysetReader{
		snapshot: reader, mapping: mapping, fetchRows: fetchRows,
		lastKey: afterKey, telemetry: telemetry,
	}
}

func (reader *keysetReader) Next(ctx context.Context) (sourceRow, error) {
	for {
		if reader.done {
			return sourceRow{}, io.EOF
		}
		if reader.rows == nil {
			if err := reader.telemetry.canFetchPage(); err != nil {
				return sourceRow{}, err
			}
			var prior any
			if reader.lastKey != nil {
				prior = reader.lastKey.native
			}
			sql := rowJSONQuery(reader.mapping.query) + " LIMIT $2"
			query := reader.query
			if query == nil {
				query = reader.snapshot.tx.Query
			}
			rows, err := query(ctx, sql, prior, reader.fetchRows)
			if err != nil {
				reader.telemetry.failure()
				return sourceRow{}, safeDatabaseError(ctx, "fetch PostgreSQL keyset page", err)
			}
			if err := reader.telemetry.page(); err != nil {
				rows.Close()
				return sourceRow{}, err
			}
			reader.rows = rows
			reader.pageRows = 0
		}
		if reader.rows.Next() {
			var text string
			if err := reader.rows.Scan(&text); err != nil {
				reader.telemetry.failure()
				return sourceRow{}, safeDatabaseError(ctx, "scan PostgreSQL keyset row", err)
			}
			reader.pageRows++
			if err := reader.telemetry.input(0, int64(len(text))); err != nil {
				return sourceRow{}, err
			}
			key, err := extractKey([]byte(text), reader.mapping.keyField)
			if err != nil {
				reader.telemetry.failure()
				return sourceRow{}, fmt.Errorf("validate PostgreSQL keyset page: %w", err)
			}
			if reader.lastKey != nil {
				comparison, err := compareKeys(*reader.lastKey, key)
				if err != nil {
					reader.telemetry.failure()
					return sourceRow{}, fmt.Errorf("validate PostgreSQL keyset page: %w", err)
				}
				if comparison >= 0 {
					reader.telemetry.failure()
					return sourceRow{}, errors.New(
						"PostgreSQL keyset keys are not strictly increasing",
					)
				}
			}
			reader.lastKey = &key
			return sourceRow{raw: []byte(text), key: &key, accounted: true}, nil
		}
		err := reader.rows.Err()
		reader.rows.Close()
		reader.rows = nil
		if err != nil {
			reader.telemetry.failure()
			return sourceRow{}, safeDatabaseError(ctx, "read PostgreSQL keyset page", err)
		}
		if reader.pageRows < reader.fetchRows {
			reader.done = true
			return sourceRow{}, io.EOF
		}
		if reader.pageRows == 0 {
			reader.done = true
			return sourceRow{}, io.EOF
		}
	}
}

func (reader *keysetReader) Close() error {
	reader.once.Do(func() {
		if reader.rows != nil {
			reader.rows.Close()
		}
		reader.err = reader.snapshot.Close()
	})
	return reader.err
}
