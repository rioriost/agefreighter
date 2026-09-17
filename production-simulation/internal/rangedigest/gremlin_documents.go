package rangedigest

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/internal/source/cosmos"
	"github.com/rioriost/agefreighter/pkg/model"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
)

// GremlinDocumentsManifest is OFFLINE ONLY: it exercises the production decoder,
// not Azure ingestion, Gremlin queries, or the PostgreSQL target.
func GremlinDocumentsManifest(ctx context.Context, dir string, m portable.Manifest, rangeRows int64) (Manifest, error) {
	if ctx == nil {
		return Manifest{}, errors.New("context is required")
	}
	if m.Version != 1 || m.DocumentFormat != portable.GremlinDocumentFormat || len(m.Tables) == 0 {
		return Manifest{}, errors.New("invalid Gremlin fixture manifest")
	}
	plan, err := fixture.BuildPlan(m.Plan.Phase)
	if err != nil || !reflect.DeepEqual(plan, m.Plan) || (plan.Phase != fixture.PhaseTiny && plan.Phase != fixture.PhaseP1) || len(m.Tables) != len(plan.VertexSpecs)+len(plan.EdgeSpecs) {
		return Manifest{}, errors.New("offline Gremlin verification requires the canonical tiny or P1 plan")
	}
	for index, t := range m.Tables {
		if index < len(plan.VertexSpecs) {
			v := plan.VertexSpecs[index]
			if t.Kind != "node" || t.Name != v.Label || t.Rows != v.Count || t.StartLabel != "" || t.EndLabel != "" {
				return Manifest{}, errors.New("vertex table metadata mismatch")
			}
		} else {
			e := plan.EdgeSpecs[index-len(plan.VertexSpecs)]
			if t.Kind != "edge" || t.Name != e.Type || t.Rows != e.Count || t.StartLabel != e.Start || t.EndLabel != e.End {
				return Manifest{}, errors.New("edge table metadata mismatch")
			}
		}
	}
	check := func() error {
		for _, t := range m.Tables {
			if err := ctx.Err(); err != nil {
				return err
			}
			if filepath.Base(t.Documents) != t.Documents {
				return errors.New("document path must be a basename")
			}
			f, err := os.Open(filepath.Join(dir, t.Documents))
			if err != nil {
				return err
			}
			h := sha256.New()
			_, err = io.Copy(h, f)
			f.Close()
			if err != nil {
				return err
			}
			if hex.EncodeToString(h.Sum(nil)) != t.DocumentsHash {
				return fmt.Errorf("document checksum mismatch: %s", t.Name)
			}
		}
		return ctx.Err()
	}
	if err := check(); err != nil {
		return Manifest{}, err
	}
	client := &gremlinFileClient{files: map[string]string{}, pagers: map[string]*gremlinFilePager{}}
	defer client.Close()
	source := config.CosmosSource{Endpoint: "https://offline.invalid/", Database: "p1", PageSize: 100}
	for _, t := range m.Tables {
		if _, exists := client.files[t.Name]; exists {
			return Manifest{}, errors.New("duplicate table")
		}
		client.files[t.Name] = filepath.Join(dir, t.Documents)
		if t.Kind == "node" {
			source.Vertices = append(source.Vertices, config.CosmosVertexQuery{Container: "graph", Label: t.Name, Query: t.Name, IDField: "/id", DocumentFormat: config.CosmosDocumentGremlin, PartitionKeyProperty: "partitionKey", MaxProperties: 128, PropertyTypes: map[string]string{"score": "float64"}})
		} else if t.Kind == "edge" {
			source.Edges = append(source.Edges, config.CosmosEdgeQuery{Container: "graph", Label: t.Name, Query: t.Name, ExternalIDField: "/id", Start: config.EndpointMapping{Label: t.StartLabel, Field: "/_vertexId"}, End: config.EndpointMapping{Label: t.EndLabel, Field: "/_sink"}, DocumentFormat: config.CosmosDocumentGremlin, PartitionKeyProperty: "partitionKey", MaxProperties: 128, PropertyTypes: map[string]string{"distance_km": "float64"}})
		} else {
			return Manifest{}, errors.New("invalid table kind")
		}
	}
	iterator, err := cosmos.NewIterator(ctx, cosmos.IteratorOptions{Namespace: "p1", Source: source, Client: client})
	if err != nil {
		return Manifest{}, err
	}
	defer iterator.Close()
	builder, err := newRangeBuilder(rangeRows)
	if err != nil {
		return Manifest{}, err
	}
	kind, name := "", ""
	var sink canonicalSink
	finish := func() error {
		if sink == nil {
			return nil
		}
		if err := finishTargetSink(ctx, sink); err != nil {
			return err
		}
		return builder.end()
	}
	for {
		item, err := iterator.Next(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return Manifest{}, err
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
			if err := finish(); err != nil {
				return Manifest{}, err
			}
			kind, name = k, n
			if err := builder.begin(k, n); err != nil {
				return Manifest{}, err
			}
			sink = targetSink(builder, true, 4_000_000)
		}
		key := props["source_key"]
		if key.Kind != model.ValueInteger {
			return Manifest{}, errors.New("source key lost integer type")
		}
		encoded, err := model.EncodeProperties(props)
		if err != nil {
			return Manifest{}, err
		}
		line := vertexLine(n, key.Integer, id, encoded)
		if k == "e" {
			line = edgeLine(n, key.Integer, id, start, end, encoded)
		}
		if err := sink.add(key.Integer, line); err != nil {
			return Manifest{}, err
		}
	}
	if sink == nil {
		return Manifest{}, errors.New("empty Gremlin fixture")
	}
	if err := finish(); err != nil {
		return Manifest{}, err
	}
	if err := check(); err != nil {
		return Manifest{}, err
	}
	result := builder.result("cosmos-gremlin-offline", m.FixtureRoot, "", "")
	result.CanonicalVersion = GremlinCanonicalVersion
	return result, nil
}

type gremlinFileClient struct {
	files  map[string]string
	pagers map[string]*gremlinFilePager
}
type gremlinFilePager struct {
	file *os.File
	scan *bufio.Scanner
	done bool
	page int
}

func (c *gremlinFileClient) NewQueryPager(_ string, query string, _ []cosmos.Parameter, o cosmos.QueryOptions) (cosmos.Pager, error) {
	if o.HasContinuationToken {
		p := c.pagers[query]
		if p == nil || strconv.Itoa(p.page) != o.ContinuationToken {
			return nil, errors.New("invalid offline continuation")
		}
		return p, nil
	}
	path, ok := c.files[query]
	if !ok {
		return nil, errors.New("unknown offline query")
	}
	if c.pagers[query] != nil {
		return nil, errors.New("duplicate offline query")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 65536), 1024*1024)
	p := &gremlinFilePager{file: f, scan: s}
	c.pagers[query] = p
	return p, nil
}
func (c *gremlinFileClient) Close() error {
	for _, p := range c.pagers {
		p.file.Close()
	}
	return nil
}
func (p *gremlinFilePager) More() bool { return !p.done }
func (p *gremlinFilePager) NextPage(ctx context.Context) (cosmos.Page, error) {
	var page cosmos.Page
	for len(page.Items) < 100 {
		if err := ctx.Err(); err != nil {
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
		decoder := json.NewDecoder(bytes.NewReader(p.scan.Bytes()))
		decoder.UseNumber()
		if err := decoder.Decode(&d); err != nil {
			return page, err
		}
		var extra any
		if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
			return page, errors.New("offline document line must contain exactly one JSON value")
		}
		// Simulate integral-float spelling loss without rounding integer identities.
		normalizeGremlinNumbers(d)
		b, err := json.Marshal(d)
		if err != nil {
			return page, err
		}
		page.Items = append(page.Items, b)
	}
	for i, j := 0, len(page.Items)-1; i < j; i, j = i+1, j-1 {
		page.Items[i], page.Items[j] = page.Items[j], page.Items[i]
	}
	p.page++
	page.HasContinuation = !p.done
	page.ContinuationToken = strconv.Itoa(p.page)
	return page, nil
}
func normalizeGremlinNumbers(v any) any {
	switch x := v.(type) {
	case map[string]any:
		for k, v := range x {
			x[k] = normalizeGremlinNumbers(v)
		}
	case []any:
		for i, v := range x {
			x[i] = normalizeGremlinNumbers(v)
		}
	case json.Number:
		s := string(x)
		if whole, fraction, ok := strings.Cut(s, "."); ok && fraction != "" && strings.Trim(fraction, "0") == "" {
			return json.Number(whole)
		}
	}
	return v
}
