package reject

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/pkg/model"
)

func TestJSONLWriterIsDurableAndIdempotent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rejects.jsonl")
	writer, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter() error = %v", err)
	}
	record := model.VertexRecord(model.Vertex{
		Label: "Person", Namespace: "crm", ExternalID: "p1",
	})
	rejection := Rejection{
		Record: &record,
		Position: model.SourcePosition{
			Connector: "csv", Resource: "people.csv", Line: 2, Token: "token-1",
		},
		Code: "malformed-record", Message: "invalid field",
	}
	if err := writer.Write(t.Context(), rejection); err != nil {
		t.Fatalf("Write() error = %v", err)
	}
	if err := writer.Write(t.Context(), rejection); err != nil {
		t.Fatalf("idempotent Write() error = %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("idempotent Close() error = %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 1 {
		t.Fatalf("quarantine lines = %d, want 1", len(lines))
	}
	var stored jsonlEntry
	if err := json.Unmarshal([]byte(lines[0]), &stored); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if stored.Version != 1 || stored.Position.Token != "token-1" ||
		stored.Record == nil || stored.Record.Vertex == nil {
		t.Fatalf("stored quarantine entry = %#v", stored)
	}
	if info, err := os.Stat(path); err != nil {
		t.Fatalf("Stat() error = %v", err)
	} else if info.Mode().Perm() != 0o600 {
		t.Fatalf("quarantine mode = %o, want 600", info.Mode().Perm())
	}
}

func TestJSONLWriterRejectsConflictsAndInvalidState(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rejects.jsonl")
	writer, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter() error = %v", err)
	}
	valid := Rejection{
		Fields: []string{"bad"},
		Position: model.SourcePosition{
			Token: "token-1",
		},
		Code: "malformed-record", Message: "first",
	}
	if err := writer.Write(t.Context(), valid); err != nil {
		t.Fatalf("Write() error = %v", err)
	}
	conflict := valid
	conflict.Message = "different"
	if err := writer.Write(t.Context(), conflict); err == nil ||
		!strings.Contains(err.Error(), "conflicting content") {
		t.Fatalf("conflicting Write() error = %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if err := writer.Write(t.Context(), valid); err == nil ||
		!strings.Contains(err.Error(), "closed") {
		t.Fatalf("closed Write() error = %v", err)
	}
}

func TestJSONLWriterSerializesInstances(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rejects.jsonl")
	first, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter(first) error = %v", err)
	}
	t.Cleanup(func() { _ = first.Close() })
	second, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter(second) error = %v", err)
	}
	t.Cleanup(func() { _ = second.Close() })
	rejection := validRejection()

	var wait sync.WaitGroup
	errs := make(chan error, 2)
	for _, writer := range []*JSONLWriter{first, second} {
		wait.Add(1)
		go func() {
			defer wait.Done()
			errs <- writer.Write(t.Context(), rejection)
		}()
	}
	wait.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent Write() error = %v", err)
		}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if lines := strings.Count(strings.TrimSpace(string(data)), "\n") + 1; lines != 1 {
		t.Fatalf("concurrent quarantine lines = %d, want 1", lines)
	}
}

func TestJSONLWriterRecoversIncompleteTail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rejects.jsonl")
	first := validRejection()
	first.Position.Token = "first"
	entry, err := json.Marshal(jsonlEntry{
		Version: 1, Position: first.Position,
		Code: first.Code, Message: first.Message,
	})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if err := os.WriteFile(
		path,
		append(append(entry, '\n'), []byte(`{"version":1,"position"`)...),
		0o644,
	); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	writer, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter() error = %v", err)
	}
	t.Cleanup(func() { _ = writer.Close() })
	second := validRejection()
	second.Position.Token = "second"
	if err := writer.Write(t.Context(), second); err != nil {
		t.Fatalf("Write() error = %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if lines := strings.Count(strings.TrimSpace(string(data)), "\n") + 1; lines != 2 {
		t.Fatalf("recovered quarantine lines = %d, want 2", lines)
	}
	if info, err := os.Stat(path); err != nil {
		t.Fatalf("Stat() error = %v", err)
	} else if info.Mode().Perm() != 0o600 {
		t.Fatalf("reopened quarantine mode = %o, want 600", info.Mode().Perm())
	}
}

func TestJSONLWriterCompletesValidUnterminatedTail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rejects.jsonl")
	rejection := validRejection()
	entry, err := json.Marshal(jsonlEntry{
		Version: 1, Position: rejection.Position,
		Code: rejection.Code, Message: rejection.Message,
	})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if err := os.WriteFile(path, entry, 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	writer, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter() error = %v", err)
	}
	t.Cleanup(func() { _ = writer.Close() })
	if err := writer.Write(t.Context(), rejection); err != nil {
		t.Fatalf("Write() error = %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if string(data) != string(entry)+"\n" {
		t.Fatalf("completed quarantine output = %q", data)
	}
}

func TestJSONLWriterFindsLaterConflict(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rejects.jsonl")
	valid := validRejection()
	first := jsonlEntry{
		Version: 1, Position: valid.Position,
		Code: valid.Code, Message: valid.Message,
	}
	second := first
	second.Message = "conflicting"
	firstJSON, _ := json.Marshal(first)
	secondJSON, _ := json.Marshal(second)
	content := append(append(firstJSON, '\n'), append(secondJSON, '\n')...)
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	writer, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter() error = %v", err)
	}
	t.Cleanup(func() { _ = writer.Close() })
	if err := writer.Write(t.Context(), valid); err == nil ||
		!strings.Contains(err.Error(), "conflicting content") {
		t.Fatalf("Write() later conflict error = %v", err)
	}
}

func TestJSONLWriterValidationAndCorruptOutput(t *testing.T) {
	if _, err := NewJSONLWriter(""); err == nil {
		t.Fatal("NewJSONLWriter() accepted empty path")
	}

	if _, err := NewJSONLWriter(t.TempDir()); err == nil {
		t.Fatal("NewJSONLWriter() opened directory")
	}
	path := filepath.Join(t.TempDir(), "rejects.jsonl")
	if err := os.WriteFile(path, []byte("{broken\n"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	writer, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter() error = %v", err)
	}
	t.Cleanup(func() { _ = writer.Close() })

	tests := []struct {
		name      string
		ctx       context.Context
		rejection Rejection
		want      string
	}{
		{name: "nil context", want: "context"},
		{
			name: "cancelled context", ctx: cancelledContext(),
			rejection: validRejection(), want: "canceled",
		},
		{
			name: "missing token", ctx: t.Context(),
			rejection: Rejection{Code: "code", Message: "message"}, want: "token",
		},
		{
			name: "missing code", ctx: t.Context(),
			rejection: Rejection{
				Position: model.SourcePosition{Token: "token"}, Message: "message",
			},
			want: "code",
		},
		{
			name: "missing message", ctx: t.Context(),
			rejection: Rejection{
				Position: model.SourcePosition{Token: "token"}, Code: "code",
			},
			want: "message",
		},
		{
			name: "corrupt output", ctx: t.Context(),
			rejection: validRejection(), want: "read quarantine",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := writer.Write(test.ctx, test.rejection); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("Write() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestJSONLWriterRejectsUnencodableAndOversizedRecords(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rejects.jsonl")
	writer, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter() error = %v", err)
	}
	t.Cleanup(func() { _ = writer.Close() })

	object := map[string]model.Value{}
	object["self"] = model.Value{Kind: model.ValueObject, Object: object}
	record := model.VertexRecord(model.Vertex{
		Label: "Person",
		Properties: model.Properties{
			"cycle": {Kind: model.ValueObject, Object: object},
		},
	})
	unencodable := validRejection()
	unencodable.Record = &record
	if err := writer.Write(t.Context(), unencodable); err == nil ||
		!strings.Contains(err.Error(), "encode quarantine record") {
		t.Fatalf("Write(unencodable) error = %v", err)
	}

	oversized := validRejection()
	oversized.Position.Token = "oversized"
	oversized.Fields = []string{strings.Repeat("x", maxJSONLRecordBytes)}
	if err := writer.Write(t.Context(), oversized); err == nil ||
		!strings.Contains(err.Error(), "maximum") {
		t.Fatalf("Write(oversized) error = %v", err)
	}
}

func TestJSONLWriterRejectsOversizedIncompleteTail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rejects.jsonl")
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		t.Fatalf("OpenFile() error = %v", err)
	}
	if err := file.Truncate(maxJSONLRecordBytes + 1); err != nil {
		_ = file.Close()
		t.Fatalf("Truncate() error = %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	writer, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter() error = %v", err)
	}
	t.Cleanup(func() { _ = writer.Close() })
	if err := writer.Write(t.Context(), validRejection()); err == nil ||
		!strings.Contains(err.Error(), "tail exceeds") {
		t.Fatalf("Write(oversized tail) error = %v", err)
	}
}

func TestJSONLWriterRejectsStructurallyInvalidLines(t *testing.T) {
	for name, content := range map[string]string{
		"missing fields":      "{}\n",
		"unsupported version": `{"version":2,"position":{"Token":"token"},"code":"code","message":"message"}` + "\n",
		"missing token":       `{"version":1,"position":{},"code":"code","message":"message"}` + "\n",
		"missing code":        `{"version":1,"position":{"Token":"token"},"message":"message"}` + "\n",
		"missing message":     `{"version":1,"position":{"Token":"token"},"code":"code"}` + "\n",
		"two values one line": "{} {}\n",
		"empty line":          "\n",
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "rejects.jsonl")
			if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
				t.Fatalf("WriteFile() error = %v", err)
			}
			writer, err := NewJSONLWriter(path)
			if err != nil {
				t.Fatalf("NewJSONLWriter() error = %v", err)
			}
			t.Cleanup(func() { _ = writer.Close() })
			if err := writer.Write(t.Context(), validRejection()); err == nil ||
				!strings.Contains(err.Error(), "read quarantine output") {
				t.Fatalf("Write() error = %v", err)
			}
		})
	}
}

func TestJSONLWriterLockWaitHonorsCancellation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rejects.jsonl")
	first, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter(first) error = %v", err)
	}
	t.Cleanup(func() { _ = first.Close() })
	second, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter(second) error = %v", err)
	}
	t.Cleanup(func() { _ = second.Close() })
	if err := lockExclusive(t.Context(), first.file); err != nil {
		t.Fatalf("lockExclusive() error = %v", err)
	}
	defer func() {
		_ = unlockExclusive(first.file)
	}()
	ctx, cancel := context.WithCancel(t.Context())
	time.AfterFunc(20*time.Millisecond, cancel)
	if err := second.Write(ctx, validRejection()); !errors.Is(err, context.Canceled) {
		t.Fatalf("Write(blocked lock) error = %v", err)
	}
}

func TestNilJSONLWriterClose(t *testing.T) {
	var writer *JSONLWriter
	if err := writer.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if err := (&JSONLWriter{}).Close(); err != nil {
		t.Fatalf("Close(empty writer) error = %v", err)
	}
	path := filepath.Join(t.TempDir(), "rejects.jsonl")
	writer, err := NewJSONLWriter(path)
	if err != nil {
		t.Fatalf("NewJSONLWriter() error = %v", err)
	}
	if err := writer.Write(t.Context(), validRejection()); err != nil {
		t.Fatalf("Write() error = %v", err)
	}
	if _, err := writer.findToken(
		cancelledContext(),
		"token",
		nil,
	); !errors.Is(err, context.Canceled) {
		t.Fatalf("findToken(cancelled) error = %v", err)
	}
	if err := writer.file.Close(); err != nil {
		t.Fatalf("close underlying file: %v", err)
	}
	if err := writer.recoverTail(); err == nil ||
		!strings.Contains(err.Error(), "stat quarantine") {
		t.Fatalf("recoverTail(closed file) error = %v", err)
	}
	if err := writer.withExclusiveLock(
		t.Context(),
		func() error { return nil },
	); err == nil || !strings.Contains(err.Error(), "lock quarantine") {
		t.Fatalf("withExclusiveLock(closed file) error = %v", err)
	}
	if err := writer.Close(); err == nil ||
		!strings.Contains(err.Error(), "close quarantine") {
		t.Fatalf("Close(closed file) error = %v", err)
	}
}

func validRejection() Rejection {
	return Rejection{
		Position: model.SourcePosition{Token: "token"},
		Code:     "code",
		Message:  "message",
	}
}

func cancelledContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}
