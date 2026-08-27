package reject

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/rioriost/agefreighter/pkg/model"
)

type Rejection struct {
	Record   *model.Record
	Fields   []string
	Position model.SourcePosition
	Code     string
	Message  string
}

type Writer interface {
	Write(ctx context.Context, rejection Rejection) error
	Close() error
}

type JSONLWriter struct {
	file   *os.File
	mu     sync.Mutex
	closed bool
}

const maxJSONLRecordBytes = 32 << 20

type jsonlEntry struct {
	Version  int                  `json:"version"`
	Position model.SourcePosition `json:"position"`
	Code     string               `json:"code"`
	Message  string               `json:"message"`
	Record   *model.Record        `json:"record,omitempty"`
	Fields   []string             `json:"fields,omitempty"`
}

func NewJSONLWriter(path string) (*JSONLWriter, error) {
	if strings.TrimSpace(path) == "" {
		return nil, errors.New("quarantine path is required")
	}
	file, err := os.OpenFile(
		path,
		os.O_CREATE|os.O_EXCL|os.O_RDWR|os.O_APPEND,
		0o600,
	)
	created := err == nil
	if errors.Is(err, os.ErrExist) {
		file, err = os.OpenFile(path, os.O_RDWR|os.O_APPEND, 0)
	}
	if err != nil {
		return nil, fmt.Errorf("open quarantine output %q: %w", path, err)
	}
	closeWithError := func(cause error) (*JSONLWriter, error) {
		return nil, errors.Join(cause, file.Close())
	}
	if err := file.Chmod(0o600); err != nil {
		return closeWithError(
			fmt.Errorf("restrict quarantine output permissions: %w", err),
		)
	}
	if created {
		directory, openErr := os.Open(filepath.Dir(path))
		if openErr != nil {
			return closeWithError(
				fmt.Errorf("open quarantine output directory: %w", openErr),
			)
		}
		syncErr := directory.Sync()
		closeErr := directory.Close()
		if syncErr != nil || closeErr != nil {
			if syncErr != nil {
				syncErr = fmt.Errorf(
					"sync quarantine output directory: %w",
					syncErr,
				)
			}
			return closeWithError(errors.Join(syncErr, closeErr))
		}
	}
	return &JSONLWriter{file: file}, nil
}

func (writer *JSONLWriter) Write(
	ctx context.Context,
	rejection Rejection,
) error {
	if ctx == nil {
		return errors.New("quarantine context is required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if strings.TrimSpace(rejection.Position.Token) == "" {
		return errors.New("quarantine resume token is required")
	}
	if strings.TrimSpace(rejection.Code) == "" {
		return errors.New("quarantine error code is required")
	}
	if strings.TrimSpace(rejection.Message) == "" {
		return errors.New("quarantine error message is required")
	}
	entry := jsonlEntry{
		Version:  1,
		Position: rejection.Position,
		Code:     rejection.Code,
		Message:  rejection.Message,
		Record:   rejection.Record,
		Fields:   rejection.Fields,
	}
	encoded, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("encode quarantine record: %w", err)
	}
	if len(encoded)+1 > maxJSONLRecordBytes {
		return fmt.Errorf(
			"quarantine record is %d bytes, maximum is %d",
			len(encoded)+1,
			maxJSONLRecordBytes,
		)
	}

	writer.mu.Lock()
	defer writer.mu.Unlock()
	if writer.closed || writer.file == nil {
		return errors.New("quarantine writer is closed")
	}
	return writer.withExclusiveLock(ctx, func() error {
		if err := writer.recoverTail(); err != nil {
			return err
		}
		found, err := writer.findToken(ctx, entry.Position.Token, encoded)
		if err != nil {
			return err
		}
		if !found {
			if _, err := writer.file.Write(append(encoded, '\n')); err != nil {
				return fmt.Errorf("append quarantine record: %w", err)
			}
		}
		if err := writer.file.Sync(); err != nil {
			return fmt.Errorf("sync quarantine output: %w", err)
		}
		return nil
	})
}

func (writer *JSONLWriter) findToken(
	ctx context.Context,
	token string,
	expected []byte,
) (bool, error) {
	if _, err := writer.file.Seek(0, io.SeekStart); err != nil {
		return false, fmt.Errorf("rewind quarantine output: %w", err)
	}
	scanner := bufio.NewScanner(writer.file)
	scanner.Buffer(make([]byte, 64<<10), maxJSONLRecordBytes)
	found := false
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		if err := ctx.Err(); err != nil {
			return false, err
		}
		var stored jsonlEntry
		line := scanner.Bytes()
		if len(bytes.TrimSpace(line)) == 0 {
			return false, fmt.Errorf(
				"read quarantine output: line %d is empty",
				lineNumber,
			)
		}
		if err := json.Unmarshal(line, &stored); err != nil {
			return false, fmt.Errorf(
				"read quarantine output line %d: %w",
				lineNumber,
				err,
			)
		}
		if err := validateJSONLEntry(stored); err != nil {
			return false, fmt.Errorf(
				"read quarantine output line %d: %w",
				lineNumber,
				err,
			)
		}
		if stored.Position.Token != token {
			continue
		}
		encoded, err := json.Marshal(stored)
		if err != nil {
			return false, fmt.Errorf("encode stored quarantine record: %w", err)
		}
		if string(encoded) != string(expected) {
			return false, fmt.Errorf(
				"quarantine resume token %q has conflicting content",
				token,
			)
		}
		found = true
	}
	if err := scanner.Err(); err != nil {
		return false, fmt.Errorf("read quarantine output: %w", err)
	}
	return found, nil
}

func validateJSONLEntry(entry jsonlEntry) error {
	if entry.Version != 1 {
		return fmt.Errorf("unsupported quarantine record version %d", entry.Version)
	}
	if strings.TrimSpace(entry.Position.Token) == "" {
		return errors.New("quarantine record resume token is empty")
	}
	if strings.TrimSpace(entry.Code) == "" {
		return errors.New("quarantine record error code is empty")
	}
	if strings.TrimSpace(entry.Message) == "" {
		return errors.New("quarantine record error message is empty")
	}
	return nil
}

func (writer *JSONLWriter) recoverTail() error {
	info, err := writer.file.Stat()
	if err != nil {
		return fmt.Errorf("stat quarantine output: %w", err)
	}
	if info.Size() == 0 {
		return nil
	}
	var last [1]byte
	if _, err := writer.file.ReadAt(last[:], info.Size()-1); err != nil {
		return fmt.Errorf("read quarantine output tail: %w", err)
	}
	if last[0] == '\n' {
		return nil
	}
	start, err := writer.lastCompleteBoundary(info.Size())
	if err != nil {
		return err
	}
	if info.Size()-start >= maxJSONLRecordBytes {
		return fmt.Errorf(
			"incomplete quarantine output tail exceeds %d bytes",
			maxJSONLRecordBytes,
		)
	}
	section := io.NewSectionReader(writer.file, start, info.Size()-start)
	decoder := json.NewDecoder(section)
	var entry jsonlEntry
	decodeErr := decoder.Decode(&entry)
	var extra any
	extraErr := decoder.Decode(&extra)
	if decodeErr == nil && errors.Is(extraErr, io.EOF) {
		if _, err := writer.file.Write([]byte{'\n'}); err != nil {
			return fmt.Errorf("complete quarantine output tail: %w", err)
		}
	} else {
		if err := writer.file.Truncate(start); err != nil {
			return fmt.Errorf("truncate incomplete quarantine output tail: %w", err)
		}
	}
	if err := writer.file.Sync(); err != nil {
		return fmt.Errorf("sync repaired quarantine output: %w", err)
	}
	return nil
}

func (writer *JSONLWriter) lastCompleteBoundary(size int64) (int64, error) {
	const blockSize int64 = 4096
	buffer := make([]byte, blockSize)
	lowerBound := max(int64(0), size-maxJSONLRecordBytes)
	for end := size; end > lowerBound; {
		start := max(lowerBound, end-blockSize)
		length := end - start
		if _, err := writer.file.ReadAt(buffer[:length], start); err != nil {
			return 0, fmt.Errorf("scan quarantine output tail: %w", err)
		}
		for index := length - 1; index >= 0; index-- {
			if buffer[index] == '\n' {
				return start + index + 1, nil
			}
		}
		end = start
	}
	if lowerBound > 0 {
		return 0, fmt.Errorf(
			"incomplete quarantine output tail exceeds %d bytes",
			maxJSONLRecordBytes,
		)
	}
	return 0, nil
}

func (writer *JSONLWriter) withExclusiveLock(
	ctx context.Context,
	run func() error,
) (err error) {
	if err := lockExclusive(ctx, writer.file); err != nil {
		return fmt.Errorf("lock quarantine output: %w", err)
	}
	defer func() {
		unlockErr := unlockExclusive(writer.file)
		if unlockErr != nil {
			err = errors.Join(err, fmt.Errorf("unlock quarantine output: %w", unlockErr))
		}
	}()
	return run()
}

func (writer *JSONLWriter) Close() error {
	if writer == nil {
		return nil
	}
	writer.mu.Lock()
	defer writer.mu.Unlock()
	if writer.closed {
		return nil
	}
	writer.closed = true
	if writer.file == nil {
		return nil
	}
	if err := writer.file.Close(); err != nil {
		return fmt.Errorf("close quarantine output: %w", err)
	}
	return nil
}

var _ Writer = (*JSONLWriter)(nil)
