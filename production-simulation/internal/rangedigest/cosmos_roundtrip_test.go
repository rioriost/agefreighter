package rangedigest

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/source/cosmos"
	"github.com/rioriost/agefreighter/pkg/model"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
)

// A bounded offline JSON transport, not an Azure emulator or GUI qualification.
// json.Unmarshal/Marshal simulates the observed integral-float spelling loss.
type documentClient struct {
	files  map[string]string
	pagers map[string]*documentPager
}
type documentPager struct {
	file *os.File
	scan *bufio.Scanner
	done bool
	page int
}

func (c documentClient) NewQueryPager(container, query string, _ []cosmos.Parameter, o cosmos.QueryOptions) (cosmos.Pager, error) {
	if o.HasContinuationToken {
		p := c.pagers[query]
		if p == nil || strconv.Itoa(p.page) != o.ContinuationToken {
			return nil, errors.New("invalid offline continuation")
		}
		return p, nil
	}
	f, err := os.Open(c.files[query])
	if err != nil {
		return nil, err
	}
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 65536), 1024*1024)
	p := &documentPager{file: f, scan: s}
	c.pagers[query] = p
	return p, nil
}

func (c documentClient) Close() error {
	for _, p := range c.pagers {
		p.file.Close()
	}
	return nil
}
func (p *documentPager) More() bool { return !p.done }
func (p *documentPager) NextPage(ctx context.Context) (cosmos.Page, error) {
	var page cosmos.Page
	for len(page.Items) < 100 {
		if err := ctx.Err(); err != nil {
			p.file.Close()
			return page, err
		}
		if !p.scan.Scan() {
			p.done = true
			p.file.Close()
			if err := p.scan.Err(); err != nil {
				return page, err
			}
			break
		}
		var d any
		if err := json.Unmarshal(p.scan.Bytes(), &d); err != nil {
			p.file.Close()
			return page, err
		}
		b, err := json.Marshal(d)
		if err != nil {
			p.file.Close()
			return page, err
		}
		page.Items = append(page.Items, b)
	}
	// Cosmos pages must not be assumed to follow fixture key order.
	for i, j := 0, len(page.Items)-1; i < j; i, j = i+1, j-1 {
		page.Items[i], page.Items[j] = page.Items[j], page.Items[i]
	}
	p.page++
	page.HasContinuation = !p.done
	page.ContinuationToken = strconv.Itoa(p.page)
	return page, nil
}

func TestCosmosTypedPortableCanonicalParity(t *testing.T) {
	ctx := context.Background()
	manifestPath := os.Getenv("AF_P1_COSMOS_FIXTURE")
	dir := os.Getenv("AF_P1_COSMOS_PORTABLE")
	var converted portable.Manifest
	if manifestPath == "" {
		tmp := t.TempDir()
		original := filepath.Join(tmp, "original")
		dir = filepath.Join(tmp, "portable")
		if _, err := fixture.Generate(ctx, fixture.GenerateConfig{Phase: fixture.PhaseTiny, Output: original, Seed: 20260829, Shards: 4, Workers: 2}); err != nil {
			t.Fatal(err)
		}
		manifestPath = filepath.Join(original, "manifest.json")
		var err error
		converted, err = portable.Export(ctx, manifestPath, dir)
		if err != nil {
			t.Fatal(err)
		}
	} else {
		b, err := os.ReadFile(filepath.Join(dir, "portable-manifest.json"))
		if err != nil {
			t.Fatal(err)
		}
		if err = json.Unmarshal(b, &converted); err != nil {
			t.Fatal(err)
		}
	}
	expected, err := FixtureManifest(ctx, manifestPath, 100000)
	if err != nil {
		t.Fatal(err)
	}
	if converted.FixtureRoot != expected.FixtureRoot {
		t.Fatal("portable fixture identity changed")
	}
	source := config.CosmosSource{Endpoint: "https://example.documents.azure.com:443/", Database: "p1", PageSize: 100}
	client := documentClient{files: map[string]string{}, pagers: map[string]*documentPager{}}
	for _, table := range converted.Tables {
		path := filepath.Join(dir, table.Documents)
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.New()
		_, err = io.Copy(h, f)
		f.Close()
		if err != nil || hex.EncodeToString(h.Sum(nil)) != table.DocumentsHash {
			t.Fatal("document fixture changed")
		}
		client.files[table.Name] = path
		props := map[string]string{}
		types := map[string]string{}
		for _, field := range table.Columns {
			if field != "start_id" && field != "end_id" {
				props[field] = "/" + field
			}
			// Match the fresh GUI fixture exactly: override only the two
			// floating-point properties; all other fields retain inference.
			if field == "score" || field == "distance_km" {
				if table.Types[field] != "float64" {
					t.Fatal("unexpected frozen numeric schema")
				}
				types[field] = "float64"
			}
		}
		if table.Kind == "node" {
			source.Vertices = append(source.Vertices, config.CosmosVertexQuery{Container: "graph", Label: table.Name, Query: table.Name, IDField: "/external_id", Properties: props, PropertyTypes: types})
		} else {
			source.Edges = append(source.Edges, config.CosmosEdgeQuery{Container: "graph", Label: table.Name, Query: table.Name, ExternalIDField: "/relationship_id", Start: config.EndpointMapping{Label: table.StartLabel, Field: "/start_id"}, End: config.EndpointMapping{Label: table.EndLabel, Field: "/end_id"}, Properties: props, PropertyTypes: types})
		}
	}
	i, err := cosmos.NewIterator(ctx, cosmos.IteratorOptions{Namespace: "p1", Source: source, Client: client})
	if err != nil {
		t.Fatal(err)
	}
	defer i.Close()
	b, _ := newRangeBuilder(100000)
	kind, name := "", ""
	var sink canonicalSink
	for {
		item, err := i.Next(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		r := item.Record
		var k, n, id, start, end string
		var props model.Properties
		if r.Vertex != nil {
			k, n, id, props = "v", string(r.Vertex.Label), string(r.Vertex.ExternalID), r.Vertex.Properties
		} else {
			k, n, id, props = "e", string(r.Edge.Label), string(r.Edge.ExternalID), r.Edge.Properties
			start, end = string(r.Edge.Start.ExternalID), string(r.Edge.End.ExternalID)
		}
		if k != kind || n != name {
			if sink != nil {
				if err := finishTargetSink(ctx, sink); err != nil {
					t.Fatal(err)
				}
				b.end()
			}
			kind, name = k, n
			b.begin(k, n)
			sink = targetSink(b, true, 4_000_000)
		}
		key := props["source_key"]
		if key.Kind != model.ValueInteger {
			t.Fatal("integer key changed")
		}
		encoded, err := model.EncodeProperties(props)
		if err != nil {
			t.Fatal(err)
		}
		line := vertexLine(n, key.Integer, id, encoded)
		if k == "e" {
			line = edgeLine(n, key.Integer, id, start, end, encoded)
		}
		if err := sink.add(key.Integer, line); err != nil {
			t.Fatal(err)
		}
	}
	if err := finishTargetSink(ctx, sink); err != nil {
		t.Fatal(err)
	}
	b.end()
	actual := b.result("apache-age", expected.FixtureRoot, "", "")
	if _, err := Compare(expected, actual); err != nil {
		t.Fatal(err)
	}
	t.Logf("OFFLINE ONLY: typed Cosmos conversion and shuffled pages: %d records, %d ranges, root=%s", actual.RecordCount, len(actual.Leaves), actual.RootSHA256)
}
