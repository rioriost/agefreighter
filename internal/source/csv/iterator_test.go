package csv

import (
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/pkg/model"
)

func BenchmarkEncodeCSVProperties(b *testing.B) {
	for _, propertyCount := range []int{1, 4, 16, 64} {
		b.Run(fmt.Sprintf("properties=%d", propertyCount), func(b *testing.B) {
			properties := make([]compiledProperty, propertyCount)
			fields := make([]string, propertyCount)
			for index := range propertyCount {
				name := fmt.Sprintf("property_%02d", index)
				properties[index] = compiledProperty{
					name:        name,
					encodedName: []byte(strconv.Quote(name)),
					index:       index,
				}
				fields[index] = fmt.Sprintf("value_%02d", index)
			}
			ctx := context.Background()
			b.ReportAllocs()
			b.ReportMetric(float64(propertyCount), "properties/op")
			b.ResetTimer()
			for range b.N {
				if _, err := encodeCSVProperties(
					ctx,
					properties,
					fields,
					"",
				); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestIteratorVerticesBeforeEdgesAndResume(t *testing.T) {
	directory := t.TempDir()
	vertices := writeTestFile(t, directory, "people.csv", strings.Join([]string{
		"id,name,note",
		"p1,Ada,",
		"p2,Grace,\"line one",
		"line two\"",
		"",
	}, "\n"))
	edges := writeTestFile(t, directory, "knows.tsv", strings.Join([]string{
		"edge_id\tfrom\tto\tsince",
		"e1\tp1\tp2\t1843",
		"e2\tp1\tp1\tNULL",
		"",
	}, "\n"))
	header := true
	null := ""
	edgeNull := "NULL"
	options := IteratorOptions{
		Namespace: "crm",
		Source: config.CSVSource{
			Defaults: config.DelimitedOptions{
				Delimiter: ",",
				Quote:     `"`,
				Escape:    `"`,
				Header:    &header,
				Encoding:  "utf-8",
				NullValue: &null,
			},
			Vertices: []config.CSVVertex{{
				Label:    "Person",
				Path:     vertices,
				IDColumn: "id",
				Properties: map[string]string{
					"name": "name",
					"note": "note",
				},
			}},
			Edges: []config.CSVEdge{{
				Label:            "KNOWS",
				Path:             edges,
				ExternalIDColumn: "edge_id",
				Start: config.EndpointMapping{
					Label: "Person",
					Field: "from",
				},
				End: config.EndpointMapping{
					Label:     "Person",
					Namespace: "crm",
					Field:     "to",
				},
				Properties: map[string]string{"since": "since"},
				Format: &config.DelimitedOptions{
					Delimiter: "\t",
					NullValue: &edgeNull,
				},
			}},
		},
	}

	t.Run("empty edge identity", func(t *testing.T) {
		directory := t.TempDir()
		vertices := writeTestFile(t, directory, "vertices.csv", "id\np1\np2\n")
		edges := writeTestFile(t, directory, "edges.csv", "id,start,end\n,p1,p2\n")
		header := true
		null := ""
		iterator, err := NewIterator(context.Background(), IteratorOptions{
			Namespace: "crm",
			Source: config.CSVSource{
				Defaults: config.DelimitedOptions{
					Delimiter: ",", Quote: `"`, Escape: `"`,
					Header: &header, Encoding: "utf-8", NullValue: &null,
				},
				Vertices: []config.CSVVertex{{
					Label: "Person", Path: vertices, IDColumn: "id",
				}},
				Edges: []config.CSVEdge{{
					Label: "KNOWS", Path: edges, ExternalIDColumn: "id",
					Start: config.EndpointMapping{Label: "Person", Field: "start"},
					End:   config.EndpointMapping{Label: "Person", Field: "end"},
				}},
			},
		})
		if err != nil {
			t.Fatalf("NewIterator() error = %v", err)
		}
		defer iterator.Close()
		_ = nextItem(t, iterator)
		_ = nextItem(t, iterator)
		if _, err := iterator.Next(context.Background()); err == nil ||
			!strings.Contains(err.Error(), "edge external ID") {
			t.Fatalf("Next() error = %v", err)
		}
	})

	iterator, err := NewIterator(context.Background(), options)
	if err != nil {
		t.Fatalf("NewIterator() error = %v", err)
	}
	first := nextItem(t, iterator)
	second := nextItem(t, iterator)
	third := nextItem(t, iterator)
	fourth := nextItem(t, iterator)
	if first.Record.Kind() != model.RecordVertex ||
		second.Record.Kind() != model.RecordVertex ||
		third.Record.Kind() != model.RecordEdge ||
		fourth.Record.Kind() != model.RecordEdge {
		t.Fatal("iterator did not emit all vertices before edges")
	}
	if got := first.Record.Vertex.ExternalID; got != "p1" {
		t.Fatalf("first vertex ID = %q", got)
	}
	if value := first.Record.Vertex.Properties["note"]; value.Kind != model.ValueNull {
		t.Fatalf("null property = %#v", value)
	}
	if value := second.Record.Vertex.Properties["note"]; value.String != "line one\nline two" {
		t.Fatalf("multiline property = %#v", value)
	}
	if third.Record.Edge.Start.ExternalID != "p1" ||
		third.Record.Edge.End.ExternalID != "p2" {
		t.Fatalf("edge endpoints = %#v", third.Record.Edge)
	}
	if value := fourth.Record.Edge.Properties["since"]; value.Kind != model.ValueNull {
		t.Fatalf("edge null property = %#v", value)
	}
	if first.SizeBytes <= 0 || third.SizeBytes <= 0 {
		t.Fatalf("record sizes = vertex %d, edge %d", first.SizeBytes, third.SizeBytes)
	}
	if _, err := iterator.Next(context.Background()); !errors.Is(err, io.EOF) {
		t.Fatalf("final Next() error = %v, want EOF", err)
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("second Close() error = %v", err)
	}

	resumeOptions := options
	resumeOptions.AfterToken = first.Record.Vertex.Position.Token
	resumed, err := NewIterator(context.Background(), resumeOptions)
	if err != nil {
		t.Fatalf("resume NewIterator() error = %v", err)
	}
	defer resumed.Close()
	resumedItem := nextItem(t, resumed)
	if resumedItem.Record.Vertex.ExternalID != "p2" {
		t.Fatalf("resumed vertex ID = %q, want p2", resumedItem.Record.Vertex.ExternalID)
	}
}

func TestIteratorPreencodesCanonicalCSVProperties(t *testing.T) {
	directory := t.TempDir()
	path := writeTestFile(
		t,
		directory,
		"people.csv",
		"id,name,note\np1,\"Ada \"\"A\"\"\",NULL\n",
	)
	header := true
	nullValue := "NULL"
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "crm",
		Source: config.CSVSource{
			Defaults: config.DelimitedOptions{
				Delimiter: ",", Quote: `"`, Escape: `"`,
				Header: &header, Encoding: "utf-8", NullValue: &nullValue,
			},
			Vertices: []config.CSVVertex{{
				Label: "Person", Path: path, IDColumn: "id",
				Properties: map[string]string{"note": "note", "name": "name"},
			}},
		},
		PreencodeProperties: true,
		OptimizeRFC4180:     true,
	})
	if err != nil {
		t.Fatalf("NewIterator() error = %v", err)
	}
	defer iterator.Close()
	item := nextItem(t, iterator)
	if item.Record.Vertex.Properties != nil {
		t.Fatalf("typed properties = %#v, want nil", item.Record.Vertex.Properties)
	}
	const want = `{"name":"Ada \"A\"","note":null}`
	if got := string(item.Record.Vertex.EncodedProperties); got != want {
		t.Fatalf("encoded properties = %s, want %s", got, want)
	}
}

func TestIteratorGzipHeaderlessAndQuarantine(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "people.csv.gz")
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create gzip file: %v", err)
	}
	compressed := gzip.NewWriter(file)
	if _, err := compressed.Write([]byte("bad\np1,Ada,unused\n,Missing,unused\np2,Grace,unused\n")); err != nil {
		t.Fatalf("write gzip file: %v", err)
	}
	if err := compressed.Close(); err != nil {
		t.Fatalf("close gzip writer: %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("close gzip file: %v", err)
	}

	header := false
	null := "NULL"
	var rejected []MalformedRecord
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "crm",
		Source: config.CSVSource{
			Defaults: config.DelimitedOptions{
				Delimiter: ",",
				Quote:     `"`,
				Escape:    `"`,
				Header:    &header,
				Encoding:  "utf-8",
				NullValue: &null,
			},
			Vertices: []config.CSVVertex{{
				Label:      "Person",
				Path:       path,
				IDColumn:   "0",
				Properties: map[string]string{"name": "1"},
			}},
		},
		RejectLimit: 2,
		OnMalformed: func(_ context.Context, malformed MalformedRecord) error {
			rejected = append(rejected, malformed)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("NewIterator() error = %v", err)
	}
	defer iterator.Close()
	first := nextItem(t, iterator)
	second := nextItem(t, iterator)
	if first.Record.Vertex.ExternalID != "p1" ||
		second.Record.Vertex.ExternalID != "p2" {
		t.Fatalf("gzip IDs = %q, %q", first.Record.Vertex.ExternalID, second.Record.Vertex.ExternalID)
	}
	if len(rejected) != 2 {
		t.Fatalf("rejections = %d, want 2", len(rejected))
	}
	if len(rejected[0].Fields) != 1 || len(rejected[1].Fields) != 3 {
		t.Fatalf("rejected fields = %#v", rejected)
	}

	resumeOptions := iterator.options
	resumeOptions.AfterToken = first.Record.Vertex.Position.Token
	resumed, err := NewIterator(context.Background(), resumeOptions)
	if err != nil {
		t.Fatalf("resume NewIterator() error = %v", err)
	}
	resumedItem := nextItem(t, resumed)
	if resumedItem.Record.Vertex.ExternalID != "p2" {
		t.Fatalf("resumed gzip ID = %q", resumedItem.Record.Vertex.ExternalID)
	}
	if len(rejected) != 3 ||
		rejected[2].Position.Token != rejected[1].Position.Token {
		t.Fatalf("resumed quarantine tokens = %#v", rejected)
	}
	_ = resumed.Close()

	quarantineResumeOptions := iterator.options
	quarantineResumeOptions.AfterToken = rejected[1].Position.Token
	quarantineResumed, err := NewIterator(context.Background(), quarantineResumeOptions)
	if err != nil {
		t.Fatalf("quarantine resume NewIterator() error = %v", err)
	}
	quarantineResumedItem := nextItem(t, quarantineResumed)
	if quarantineResumedItem.Record.Vertex.ExternalID != "p2" {
		t.Fatalf(
			"quarantine resumed gzip ID = %q",
			quarantineResumedItem.Record.Vertex.ExternalID,
		)
	}
	_ = quarantineResumed.Close()

	exhaustedOptions := iterator.options
	exhaustedOptions.AfterToken = resumedItem.Record.Vertex.Position.Token
	exhausted, err := NewIterator(context.Background(), exhaustedOptions)
	if err != nil {
		t.Fatalf("exhausted resume NewIterator() error = %v", err)
	}
	if _, err := exhausted.Next(context.Background()); !errors.Is(err, io.EOF) {
		t.Fatalf("exhausted Next() error = %v, want EOF", err)
	}
	rejectedCount, checkpoint := exhausted.RejectionCheckpoint()
	if rejectedCount != 2 || checkpoint.Token != exhaustedOptions.AfterToken {
		t.Fatalf("exhausted checkpoint = %d, %#v", rejectedCount, checkpoint)
	}
	_ = exhausted.Close()

	resumeOptions.RejectLimit = 1
	overLimit, err := NewIterator(context.Background(), resumeOptions)
	if err != nil {
		t.Fatalf("over-limit NewIterator() error = %v", err)
	}
	defer overLimit.Close()
	if _, err := overLimit.Next(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "reject limit 1 exceeded") {
		t.Fatalf("over-limit Next() error = %v", err)
	}
}

func TestIteratorRejectLimitAndHandlerFailure(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "people.csv", "id\n\np1\n")
	source := singleVertexSource(path)
	var calls int
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace:   "crm",
		Source:      source,
		RejectLimit: 1,
		OnMalformed: func(context.Context, MalformedRecord) error {
			calls++
			return nil
		},
	})
	if err != nil {
		t.Fatalf("NewIterator() error = %v", err)
	}
	item := nextItem(t, iterator)
	if item.Record.Vertex.ExternalID != "p1" || calls != 1 {
		t.Fatalf("item = %#v, calls = %d", item, calls)
	}
	_ = iterator.Close()

	iterator, err = NewIterator(context.Background(), IteratorOptions{
		Namespace:   "crm",
		Source:      source,
		RejectLimit: 2,
		OnMalformed: func(context.Context, MalformedRecord) error {
			return errors.New("injected quarantine failure")
		},
	})
	if err != nil {
		t.Fatalf("handler failure NewIterator() error = %v", err)
	}
	defer iterator.Close()
	if _, err := iterator.Next(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "injected quarantine failure") {
		t.Fatalf("Next() error = %v", err)
	}
}

func TestIteratorResumeRejectsChangedSource(t *testing.T) {
	directory := t.TempDir()
	path := writeTestFile(t, directory, "people.csv", "id\np1\np2\n")
	options := IteratorOptions{Namespace: "crm", Source: singleVertexSource(path)}
	iterator, err := NewIterator(context.Background(), options)
	if err != nil {
		t.Fatalf("NewIterator() error = %v", err)
	}
	item := nextItem(t, iterator)
	_ = iterator.Close()

	if err := os.WriteFile(path, []byte("id\np1\nchanged\n"), 0o600); err != nil {
		t.Fatalf("change source: %v", err)
	}
	options.AfterToken = item.Record.Vertex.Position.Token
	resumed, err := NewIterator(context.Background(), options)
	if err != nil {
		t.Fatalf("resume NewIterator() error = %v", err)
	}
	defer resumed.Close()
	if _, err := resumed.Next(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "fingerprint changed") {
		t.Fatalf("resumed Next() error = %v", err)
	}

	mappingOptions := IteratorOptions{Namespace: "crm", Source: singleVertexSource(path)}
	mappingIterator, err := NewIterator(context.Background(), mappingOptions)
	if err != nil {
		t.Fatalf("mapping NewIterator() error = %v", err)
	}
	mappingItem := nextItem(t, mappingIterator)
	_ = mappingIterator.Close()
	mappingOptions.Source.Vertices[0].Label = "Customer"
	mappingOptions.AfterToken = mappingItem.Record.Vertex.Position.Token
	changedMapping, err := NewIterator(context.Background(), mappingOptions)
	if err != nil {
		t.Fatalf("changed mapping NewIterator() error = %v", err)
	}
	defer changedMapping.Close()
	if _, err := changedMapping.Next(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "fingerprint changed") {
		t.Fatalf("changed mapping Next() error = %v", err)
	}
}

func TestIteratorResumeBindsCompleteManifest(t *testing.T) {
	directory := t.TempDir()
	vertices := writeTestFile(t, directory, "people.csv", "id\np1\n")
	edges := writeTestFile(t, directory, "knows.csv", "id,from,to\ne1,p1,p1\n")
	source := singleVertexSource(vertices)
	source.Edges = []config.CSVEdge{{
		Label:            "KNOWS",
		Path:             edges,
		ExternalIDColumn: "id",
		Start:            config.EndpointMapping{Label: "Person", Field: "from"},
		End:              config.EndpointMapping{Label: "Person", Field: "to"},
	}}
	options := IteratorOptions{Namespace: "crm", Source: source}

	iterator, err := NewIterator(context.Background(), options)
	if err != nil {
		t.Fatalf("NewIterator() error = %v", err)
	}
	vertex := nextItem(t, iterator)
	_ = iterator.Close()
	if err := os.WriteFile(edges, []byte("id,from,to\ne2,p1,p1\n"), 0o600); err != nil {
		t.Fatalf("change later source: %v", err)
	}
	options.AfterToken = vertex.Record.Vertex.Position.Token
	changedLater, err := NewIterator(context.Background(), options)
	if err != nil {
		t.Fatalf("changed-later NewIterator() error = %v", err)
	}
	defer changedLater.Close()
	if _, err := changedLater.Next(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "manifest fingerprint changed") {
		t.Fatalf("changed-later Next() error = %v", err)
	}

	if err := os.WriteFile(edges, []byte("id,from,to\ne1,p1,p1\n"), 0o600); err != nil {
		t.Fatalf("restore later source: %v", err)
	}
	options.AfterToken = ""
	iterator, err = NewIterator(context.Background(), options)
	if err != nil {
		t.Fatalf("second NewIterator() error = %v", err)
	}
	_ = nextItem(t, iterator)
	edge := nextItem(t, iterator)
	_ = iterator.Close()
	if err := os.WriteFile(vertices, []byte("id\np2\n"), 0o600); err != nil {
		t.Fatalf("change earlier source: %v", err)
	}
	options.AfterToken = edge.Record.Edge.Position.Token
	changedEarlier, err := NewIterator(context.Background(), options)
	if err != nil {
		t.Fatalf("changed-earlier NewIterator() error = %v", err)
	}
	defer changedEarlier.Close()
	if _, err := changedEarlier.Next(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "manifest fingerprint changed") {
		t.Fatalf("changed-earlier Next() error = %v", err)
	}

	if err := os.WriteFile(vertices, []byte("id\np1\n"), 0o600); err != nil {
		t.Fatalf("restore earlier source: %v", err)
	}
	options.AfterToken = ""
	iterator, err = NewIterator(context.Background(), options)
	if err != nil {
		t.Fatalf("third NewIterator() error = %v", err)
	}
	token := nextItem(t, iterator).Record.Vertex.Position.Token
	_ = iterator.Close()
	added := options
	added.Source.Vertices = append(
		added.Source.Vertices,
		config.CSVVertex{
			Label:    "Other",
			Path:     writeTestFile(t, directory, "other.csv", "id\no1\n"),
			IDColumn: "id",
		},
	)
	added.AfterToken = token
	changedSet, err := NewIterator(context.Background(), added)
	if err != nil {
		t.Fatalf("changed-set NewIterator() error = %v", err)
	}
	defer changedSet.Close()
	if _, err := changedSet.Next(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "manifest fingerprint changed") {
		t.Fatalf("changed-set Next() error = %v", err)
	}
}

func TestIteratorSnapshotsMappingConfiguration(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "people.csv", "id,name\np1,NULL\n")
	header := true
	nullValue := "NULL"
	properties := map[string]string{"name": "name"}
	source := config.CSVSource{
		Defaults: config.DelimitedOptions{
			Header:    &header,
			NullValue: &nullValue,
		},
		Vertices: []config.CSVVertex{{
			Label:      "Person",
			Path:       path,
			IDColumn:   "id",
			Properties: properties,
		}},
	}
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "crm",
		Source:    source,
	})
	if err != nil {
		t.Fatalf("NewIterator() error = %v", err)
	}
	defer iterator.Close()

	delete(properties, "name")
	properties["renamed"] = "name"
	header = false
	nullValue = "other"

	item := nextItem(t, iterator)
	if _, ok := item.Record.Vertex.Properties["renamed"]; ok {
		t.Fatalf("iterator observed mutated property mapping: %#v", item.Record.Vertex.Properties)
	}
	if value := item.Record.Vertex.Properties["name"]; value.Kind != model.ValueNull {
		t.Fatalf("iterator observed mutated null value: %#v", value)
	}
}

func TestIteratorUsesOpenedFileSnapshot(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows does not permit replacing an open file")
	}
	directory := t.TempDir()
	path := writeTestFile(t, directory, "people.csv", "id\np1\np2\n")
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "crm",
		Source:    singleVertexSource(path),
	})
	if err != nil {
		t.Fatalf("NewIterator() error = %v", err)
	}
	first := nextItem(t, iterator)
	replacement := writeTestFile(t, directory, "replacement.csv", "id\nnew\n")
	if err := os.Rename(replacement, path); err != nil {
		t.Fatalf("replace active source: %v", err)
	}
	second := nextItem(t, iterator)
	if second.Record.Vertex.ExternalID != "p2" {
		t.Fatalf("snapshot second ID = %q, want p2", second.Record.Vertex.ExternalID)
	}
	if _, err := iterator.Next(context.Background()); !errors.Is(err, io.EOF) {
		t.Fatalf("snapshot final Next() error = %v, want EOF", err)
	}
	_ = iterator.Close()

	resumed, err := NewIterator(context.Background(), IteratorOptions{
		Namespace:  "crm",
		Source:     singleVertexSource(path),
		AfterToken: first.Record.Vertex.Position.Token,
	})
	if err != nil {
		t.Fatalf("resume NewIterator() error = %v", err)
	}
	defer resumed.Close()
	if _, err := resumed.Next(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "manifest fingerprint changed") {
		t.Fatalf("snapshot resume Next() error = %v", err)
	}
}

func TestIteratorDetectsChangesDuringIteration(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "people.csv", "id\np1\np2\n")
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "crm",
		Source:    singleVertexSource(path),
	})
	if err != nil {
		t.Fatalf("NewIterator() error = %v", err)
	}
	defer iterator.Close()
	if item := nextItem(t, iterator); item.Record.Vertex.ExternalID != "p1" {
		t.Fatalf("first ID = %q", item.Record.Vertex.ExternalID)
	}
	if err := os.WriteFile(path, []byte("id\np1\np3\n"), 0o600); err != nil {
		t.Fatalf("change iterated source: %v", err)
	}
	if _, err := iterator.Next(context.Background()); err != nil {
		t.Fatalf("buffered second Next() error = %v", err)
	}
	if _, err := iterator.Next(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "changed during iteration") {
		t.Fatalf("EOF verification error = %v", err)
	}
}

func TestFingerprintCoversMiddleContent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "large.csv")
	content := []byte(strings.Repeat("x", 256<<10))
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatalf("write large source: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat large source: %v", err)
	}
	before, err := fingerprint(context.Background(), path, []byte("mapping"))
	if err != nil {
		t.Fatalf("initial fingerprint: %v", err)
	}
	content[len(content)/2] = 'y'
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatalf("rewrite large source: %v", err)
	}
	if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
		t.Fatalf("restore large source timestamp: %v", err)
	}
	after, err := fingerprint(context.Background(), path, []byte("mapping"))
	if err != nil {
		t.Fatalf("changed fingerprint: %v", err)
	}
	if before == after {
		t.Fatal("fingerprint ignored changed middle content")
	}
}

func TestIteratorValidationAndLifecycle(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "people.csv", "id\np1\n")
	source := singleVertexSource(path)
	tests := []IteratorOptions{
		{Source: source},
		{Namespace: "crm", Source: source, RejectLimit: -1},
		{Namespace: "crm", Source: source, RejectLimit: 1},
		{Namespace: "crm"},
		{Namespace: "crm", Source: source, AfterToken: "bad"},
	}
	for _, options := range tests {
		if _, err := NewIterator(context.Background(), options); err == nil {
			t.Fatalf("NewIterator(%#v) succeeded", options)
		}
	}
	if _, err := NewIterator(nil, IteratorOptions{
		Namespace: "crm",
		Source:    source,
	}); err == nil {
		t.Fatal("NewIterator(nil context) succeeded")
	}
	cancelledContext, cancelConstruction := context.WithCancel(context.Background())
	cancelConstruction()
	if _, err := NewIterator(cancelledContext, IteratorOptions{
		Namespace: "crm",
		Source:    source,
	}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled NewIterator() error = %v", err)
	}

	iterator, err := NewIterator(
		context.Background(),
		IteratorOptions{Namespace: "crm", Source: source},
	)
	if err != nil {
		t.Fatalf("NewIterator() error = %v", err)
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if _, err := iterator.Next(context.Background()); err == nil {
		t.Fatal("Next() after Close() succeeded")
	}

	cancelled, err := NewIterator(
		context.Background(),
		IteratorOptions{Namespace: "crm", Source: source},
	)
	if err != nil {
		t.Fatalf("cancelled NewIterator() error = %v", err)
	}
	defer cancelled.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := cancelled.Next(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled Next() error = %v", err)
	}
}

func TestIteratorDoesNotQuarantineFramingErrors(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "broken.csv", "id\nbad\"tail\ngood\n")
	var calls int
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace:   "crm",
		Source:      singleVertexSource(path),
		RejectLimit: 10,
		OnMalformed: func(context.Context, MalformedRecord) error {
			calls++
			return nil
		},
	})
	if err != nil {
		t.Fatalf("NewIterator() error = %v", err)
	}
	defer iterator.Close()
	if _, err := iterator.Next(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "quote must begin") {
		t.Fatalf("Next() error = %v", err)
	}
	if calls != 0 {
		t.Fatalf("framing error quarantine calls = %d, want 0", calls)
	}
}

func TestIteratorTrailingQuarantineTokensAreIdempotent(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "invalid.csv", "id\n\n")
	var firstToken string
	run := func() string {
		var token string
		iterator, err := NewIterator(context.Background(), IteratorOptions{
			Namespace:   "crm",
			Source:      singleVertexSource(path),
			RejectLimit: 1,
			OnMalformed: func(_ context.Context, malformed MalformedRecord) error {
				token = malformed.Position.Token
				return nil
			},
		})
		if err != nil {
			t.Fatalf("NewIterator() error = %v", err)
		}
		defer iterator.Close()
		if _, err := iterator.Next(context.Background()); !errors.Is(err, io.EOF) {
			t.Fatalf("Next() error = %v, want EOF", err)
		}
		return token
	}
	firstToken = run()
	secondToken := run()
	if firstToken == "" || firstToken != secondToken {
		t.Fatalf("trailing quarantine tokens = %q, %q", firstToken, secondToken)
	}
}

func TestIteratorRejectsExcessiveMappingsAndIndexes(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "headerless.csv", "value\n")
	header := false
	null := ""
	source := config.CSVSource{
		Defaults: config.DelimitedOptions{
			Delimiter: ",",
			Quote:     `"`,
			Escape:    `"`,
			Header:    &header,
			Encoding:  "utf-8",
			NullValue: &null,
		},
		Vertices: []config.CSVVertex{{
			Label:    "Person",
			Path:     path,
			IDColumn: strconv.FormatInt(int64(^uint(0)>>1), 10),
		}},
	}
	iterator, err := NewIterator(context.Background(), IteratorOptions{
		Namespace: "crm",
		Source:    source,
		MaxFields: 8,
	})
	if err != nil {
		t.Fatalf("index NewIterator() error = %v", err)
	}
	defer iterator.Close()
	if _, err := iterator.Next(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "below 8") {
		t.Fatalf("large-index Next() error = %v", err)
	}

	properties := make(map[string]string, 3)
	properties["a"] = "0"
	properties["b"] = "0"
	properties["c"] = "0"
	source.Vertices[0].IDColumn = "0"
	source.Vertices[0].Properties = properties
	if _, err := NewIterator(context.Background(), IteratorOptions{
		Namespace:     "crm",
		Source:        source,
		MaxProperties: 2,
	}); err == nil || !strings.Contains(err.Error(), "maximum is 2") {
		t.Fatalf("property-limited NewIterator() error = %v", err)
	}
}

func nextItem(t *testing.T, iterator *Iterator) sourceItem {
	t.Helper()
	item, err := iterator.Next(context.Background())
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	return sourceItem{Record: item.Record, SizeBytes: item.SizeBytes}
}

type sourceItem struct {
	Record    model.Record
	SizeBytes int64
}

func singleVertexSource(path string) config.CSVSource {
	header := true
	null := "NULL"
	return config.CSVSource{
		Defaults: config.DelimitedOptions{
			Delimiter: ",",
			Quote:     `"`,
			Escape:    `"`,
			Header:    &header,
			Encoding:  "utf-8",
			NullValue: &null,
		},
		Vertices: []config.CSVVertex{{
			Label:    "Person",
			Path:     path,
			IDColumn: "id",
		}},
	}
}

func writeTestFile(t *testing.T, directory, name, contents string) string {
	t.Helper()
	path := filepath.Join(directory, name)
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("write test file: %v", err)
	}
	return path
}
