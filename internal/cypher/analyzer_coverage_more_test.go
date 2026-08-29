package cypher

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestAnalyzerRemainingBoundAndLexicalBranches(t *testing.T) {
	path := writeQuery(t, "RETURN 1")
	report, err := AnalyzeFiles(t.Context(), []string{path, path}, Options{})
	if err != nil || len(report.Files) != 2 {
		t.Fatalf("AnalyzeFiles(duplicate path) = %#v, %v", report, err)
	}

	lexicalOnly := writeQuery(t, "/* unterminated")
	report, err = AnalyzeFiles(t.Context(), []string{lexicalOnly}, Options{})
	if err != nil || len(report.Queries) != 1 ||
		report.Queries[0].Classification != Unknown {
		t.Fatalf("AnalyzeFiles(lexical only) = %#v, %v", report, err)
	}

	tooManyTokens := writeQuery(t, "RETURN "+strings.Repeat("x, ", MaxTokens)+"x")
	if _, err := AnalyzeFiles(
		t.Context(), []string{tooManyTokens}, Options{},
	); !errors.Is(err, errLimit) {
		t.Fatalf("token limit error = %v", err)
	}

	var labels strings.Builder
	labels.WriteString("MATCH ")
	for index := 0; index < 129; index++ {
		if index > 0 {
			labels.WriteString(",")
		}
		labels.WriteString("(n:L")
		labels.WriteString(strconv.Itoa(index))
		labels.WriteString(")")
	}
	labels.WriteString(" RETURN 1")
	patternHeavy := writeQuery(t, labels.String())
	if _, err := AnalyzeFiles(
		t.Context(), []string{patternHeavy}, Options{},
	); !errors.Is(err, errLimit) {
		t.Fatalf("pattern limit error = %v", err)
	}

	directory := t.TempDir()
	paths := make([]string, 9)
	block := []byte("RETURN '" + strings.Repeat("x", MaxFileBytes-20) + "'")
	for index := range paths {
		paths[index] = filepath.Join(directory, string(rune('a'+index))+".cypher")
		if err := os.WriteFile(paths[index], block, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := AnalyzeFiles(t.Context(), paths, Options{}); !errors.Is(err, errLimit) {
		t.Fatalf("total byte limit error = %v", err)
	}
}

func TestAnalyzerFileHelperBranches(t *testing.T) {
	for _, path := range []string{"", "  ", "-"} {
		if _, _, err := readRegularFile(t.Context(), path); !errors.Is(err, errInput) {
			t.Fatalf("readRegularFile(%q) error = %v", path, err)
		}
	}
	reader := &contextReader{ctx: canceledCypherContext(), reader: strings.NewReader("x")}
	if _, err := reader.Read(make([]byte, 1)); !errors.Is(err, context.Canceled) {
		t.Fatalf("contextReader.Read() error = %v", err)
	}
	if got := safeBasename("bad\nname.cypher"); !strings.Contains(got, "�") {
		t.Fatalf("safeBasename() = %q", got)
	}

	for range cap(fileOperationSlots) {
		fileOperationSlots <- struct{}{}
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if _, err := boundedFileCall(ctx, nil, func() (int, error) {
		return 1, nil
	}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("boundedFileCall(slot timeout) error = %v", err)
	}
	for range cap(fileOperationSlots) {
		<-fileOperationSlots
	}

	var cleaned atomic.Bool
	started := make(chan struct{})
	release := make(chan struct{})
	cleanupDone := make(chan struct{})
	ctx, cancel = context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := boundedFileCall(ctx, func(int) {
		cleaned.Store(true)
		close(cleanupDone)
	}, func() (int, error) {
		close(started)
		<-release
		return 1, nil
	}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("boundedFileCall(operation timeout) error = %v", err)
	}
	<-started
	close(release)
	select {
	case <-cleanupDone:
	case <-time.After(time.Second):
		t.Fatal("boundedFileCall() cleanup did not finish")
	}
	fileOperationSlots <- struct{}{}
	<-fileOperationSlots
	if !cleaned.Load() {
		t.Fatal("boundedFileCall() did not clean a late result")
	}
}

func TestAnalyzerCanonicalAndOutputBranches(t *testing.T) {
	query := Query{
		Findings: []Finding{
			{Line: 1, Column: 2, Code: "z"},
			{Line: 1, Column: 2, Code: "a"},
			{Line: 1, Column: 1, Code: "m"},
		},
		Patterns: []Pattern{
			{Kind: "z", Label: "z"},
			{Kind: "a", Label: "a"},
		},
	}
	canonicalizeQuery(&query)
	if query.Findings[0].Column != 1 || query.Findings[1].Code != "a" {
		t.Fatalf("canonical findings = %#v", query.Findings)
	}

	report := Report{Queries: []Query{{
		Findings: []Finding{{Evidence: strings.Repeat("x", MaxOutputBytes+1)}},
	}}}
	if _, err := Render(report, FormatJSON); !errors.Is(err, errLimit) {
		t.Fatalf("Render(oversized) error = %v", err)
	}
}

func canceledCypherContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}

var _ io.Reader = (*contextReader)(nil)
