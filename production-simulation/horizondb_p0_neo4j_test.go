package simulation_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/neo4j/neo4j-go-driver/v6/neo4j"
	"github.com/rioriost/agefreighter/internal/config"
	neosource "github.com/rioriost/agefreighter/internal/source/neo4j"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
	"github.com/rioriost/agefreighter/production-simulation/internal/rangedigest"
)

const horizonP0Neo4jSeedRows = 512

func TestHorizonDBP0Neo4jSourceIntegration(t *testing.T) {
	if os.Getenv("AGEFREIGHTER_REQUIRE_HORIZON_P0_NEO4J_SOURCE") != "1" {
		t.Skip("explicit P0 source-only Neo4j fixture preflight required")
	}
	output := os.Getenv("AGEFREIGHTER_HORIZON_NEO4J_SOURCE_OUTPUT")
	run := os.Getenv("AGEFREIGHTER_HORIZON_NEO4J_SOURCE_RUN_ID")
	source := os.Getenv("AGEFREIGHTER_HORIZON_NEO4J_SOURCE")
	if !filepath.IsAbs(output) {
		t.Fatal("new absolute source evidence directory required")
	}
	if _, err := horizonP0Neo4jScope(source, run, "base"); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(output, 0700); err != nil {
		t.Fatal(err)
	}
	paths, manifests, directories := horizonP0Fixtures(t, output)
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Minute)
	defer cancel()
	if err := seedHorizonP0Neo4j(t, ctx, paths, manifests, directories, source, run, output); err != nil {
		t.Fatal(err)
	}
	t.Log("P0 Neo4j source hashes, actual counts and production iterator verified; no HorizonDB migration")
}

func validateHorizonP0Neo4jURI(source, uri string) error {
	expected := "bolt://127.0.0.1:7689"
	if source == "neo4j526" {
		expected = "bolt://127.0.0.1:7690"
	} else if source != "neo4j44" {
		return errors.New("pinned Neo4j 4.4 or 5.26 source required")
	}
	if uri != expected {
		return errors.New("dedicated loopback P0 Neo4j endpoint required")
	}
	return nil
}

func horizonP0Neo4jScope(source, run, dataset string) (string, error) {
	if (source != "neo4j44" && source != "neo4j526") ||
		!regexp.MustCompile(`^r[a-z0-9]{6,30}$`).MatchString(run) ||
		(dataset != "base" && dataset != "append" && dataset != "upsert") {
		return "", errors.New("new run-scoped immutable P0 Neo4j dataset required")
	}
	return "hs_p0_" + source + "_" + run + "_" + dataset, nil
}

func horizonP0Neo4jQueries(job *config.LoadJob, manifest portable.Manifest, source, run, dataset string) error {
	scope, err := horizonP0Neo4jScope(source, run, dataset)
	if err != nil {
		return err
	}
	if job.Source.Type != config.SourceNeo4j || job.Source.Neo4j == nil {
		return errors.New("explicit P0 Neo4j source required")
	}
	n := job.Source.Neo4j
	if err := validateHorizonP0Neo4jURI(source, n.URI); err != nil {
		return err
	}
	n.Discovery, n.SourceID, n.FetchRows = nil, "horizon-p0-"+source, 20_000
	n.Vertices, n.Edges = nil, nil
	identifier := regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_]*$`)
	for _, table := range manifest.Tables {
		if !identifier.MatchString(table.Name) || (table.Kind != "node" && table.Kind != "edge") {
			return errors.New("ordinary fixture label and kind required")
		}
		entity, match := "n", "MATCH (n:`"+scope+"_"+table.Name+"`)"
		if table.Kind == "edge" {
			if !identifier.MatchString(table.StartLabel) || !identifier.MatchString(table.EndLabel) {
				return errors.New("ordinary fixture endpoint labels required")
			}
			entity, match = "r", "MATCH (s)-[r:`"+scope+"_"+table.Name+"`]->(e)"
		}
		props, columns := map[string]string{}, []string{}
		for _, name := range table.Columns {
			if !regexp.MustCompile(`^[a-z][a-z0-9_]*$`).MatchString(name) {
				return errors.New("ordinary scalar fixture column required")
			}
			value := entity + "." + name
			switch name {
			case "start_id":
				value = "s.external_id"
			case "end_id":
				value = "e.external_id"
			default:
				props[name] = name
			}
			columns = append(columns, value+" AS "+name)
		}
		projection := " RETURN " + strings.Join(columns, ",") + " ORDER BY source_key LIMIT $pageRows"
		query := match + " WHERE " + entity + ".source_key > $afterKey" + projection
		initial := match + projection
		if table.Kind == "node" {
			n.Vertices = append(n.Vertices, config.VertexQuery{Label: table.Name, KeyField: "source_key",
				Query: query, InitialQuery: initial, IDField: "external_id", Properties: props})
		} else {
			n.Edges = append(n.Edges, config.EdgeQuery{Label: table.Name, KeyField: "source_key",
				Query: query, InitialQuery: initial, ExternalIDField: "relationship_id", Properties: props,
				Start: config.EndpointMapping{Label: table.StartLabel, Field: "start_id"},
				End:   config.EndpointMapping{Label: table.EndLabel, Field: "end_id"}})
		}
	}
	return nil
}

func seedHorizonP0Neo4j(t *testing.T, ctx context.Context, paths rangedigest.Scenario,
	manifests map[string]portable.Manifest, directories map[string]string, source, run, output string,
) error {
	t.Helper()
	scopes, jobs := map[string]string{}, map[string]config.LoadJob{}
	for _, dataset := range []string{"base", "append", "upsert"} {
		manifest, ok := manifests[dataset]
		path := map[string]string{"base": paths.Base, "append": paths.Append, "upsert": paths.Upsert}[dataset]
		f, err := fixture.Verify(path)
		if err != nil {
			return err
		}
		if !ok || f.Plan.Phase != fixture.PhaseP0 || manifest.FixtureRoot != f.RootSHA256 {
			return errors.New("verified P0 portable provenance required before Neo4j source writes")
		}
		job := tinyJob(manifest, directories[dataset], source, 17)
		if err := horizonP0Neo4jQueries(&job, manifest, source, run, dataset); err != nil {
			return err
		}
		scopes[dataset], err = horizonP0Neo4jScope(source, run, dataset)
		if err != nil {
			return err
		}
		jobs[dataset] = job
	}
	uri := jobs["base"].Source.Neo4j.URI
	password := os.Getenv("AGEFREIGHTER_NEO4J_PASSWORD")
	if password == "" {
		return errors.New("private Neo4j fixture password required")
	}
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth("neo4j", password, ""))
	if err != nil {
		return err
	}
	defer driver.Close(context.Background())
	session := driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j", FetchSize: horizonP0Neo4jSeedRows})
	defer session.Close(context.Background())
	result, err := session.Run(ctx, "CALL dbms.components() YIELD versions RETURN versions[0] AS version", nil)
	if err != nil {
		return err
	}
	record, err := result.Single(ctx)
	if err != nil {
		return err
	}
	version, _ := record.Get("version")
	expected := "4.4.48"
	if source == "neo4j526" {
		expected = "5.26.30"
	}
	if version != expected {
		return fmt.Errorf("expected pinned Neo4j %s source, got %v", expected, version)
	}
	execute := func(query string) error {
		result, err := session.Run(ctx, query, nil)
		if err != nil {
			return err
		}
		_, err = result.Consume(ctx)
		return err
	}
	lookup := strings.TrimSuffix(scopes["base"], "base") + "lookup"
	for _, table := range manifests["base"].Tables {
		if table.Kind == "node" {
			label := lookup + "_" + table.Name
			if err := execute("CREATE INDEX `" + label + "_id` FOR (n:`" + label + "`) ON (n.external_id)"); err != nil {
				return err
			}
		}
	}
	datasets := map[string]any{}
	for _, dataset := range []string{"base", "append", "upsert"} {
		scope, manifest := scopes[dataset], manifests[dataset]
		for _, table := range manifest.Tables {
			label := scope + "_" + table.Name
			pattern, entity := "(n:`"+label+"`)", "n"
			if table.Kind == "edge" {
				pattern, entity = "()-[r:`"+label+"`]-()", "r"
			}
			if err := execute("CREATE INDEX `" + label + "_key` FOR " + pattern + " ON (" + entity + ".source_key)"); err != nil {
				return err
			}
		}
		if err := execute("CALL db.awaitIndexes(120)"); err != nil {
			return err
		}
		var count int64
		for _, table := range manifest.Tables {
			label := scope + "_" + table.Name
			labels := "`" + label + "`"
			// Upsert-only copies must not duplicate endpoint lookup identities.
			if dataset != "upsert" {
				labels += ":`" + lookup + "_" + table.Name + "`"
			}
			query := "UNWIND $rows AS row CREATE (n:" + labels + ") SET n=row.props"
			countQuery := "MATCH (n:`" + label + "`) RETURN count(n) AS count"
			if table.Kind == "edge" {
				query = "UNWIND $rows AS row MATCH (s:`" + lookup + "_" + table.StartLabel + "` {external_id:row.start}) " +
					"MATCH (e:`" + lookup + "_" + table.EndLabel + "` {external_id:row.end}) " +
					"CREATE (s)-[r:`" + label + "`]->(e) SET r=row.props"
				countQuery = "MATCH ()-[r:`" + label + "`]->() RETURN count(r) AS count"
			}
			file, err := os.Open(filepath.Join(directories[dataset], table.Documents))
			if err != nil {
				return err
			}
			hash := sha256.New()
			decoder := json.NewDecoder(io.TeeReader(file, hash))
			decoder.UseNumber()
			rows := make([]map[string]any, 0, horizonP0Neo4jSeedRows)
			var read int64
			flush := func() error {
				if len(rows) == 0 {
					return nil
				}
				result, err := session.Run(ctx, query, map[string]any{"rows": rows})
				if err != nil {
					return err
				}
				summary, err := result.Consume(ctx)
				if err != nil {
					return err
				}
				created := summary.Counters().NodesCreated()
				if table.Kind == "edge" {
					created = summary.Counters().RelationshipsCreated()
				}
				if created != len(rows) {
					return errors.New("Neo4j source batch count or physical endpoints differ")
				}
				rows = rows[:0]
				return nil
			}
			seedErr := func() error {
				for {
					var doc map[string]any
					if err := decoder.Decode(&doc); errors.Is(err, io.EOF) {
						return flush()
					} else if err != nil {
						return err
					}
					props := map[string]any{}
					for _, name := range table.Columns {
						if name != "start_id" && name != "end_id" {
							props[name] = tinyNativeValue(t, doc[name])
							if table.Types[name] == "float64" && doc[name] != nil {
								number, ok := doc[name].(json.Number)
								if !ok {
									return errors.New("typed Neo4j floating-point fixture property required")
								}
								value, err := number.Float64()
								if err != nil {
									return err
								}
								props[name] = value
							}
						}
					}
					rows = append(rows, map[string]any{"props": props, "start": doc["start_id"], "end": doc["end_id"]})
					read++
					if len(rows) == horizonP0Neo4jSeedRows {
						if err := flush(); err != nil {
							return err
						}
					}
				}
			}()
			if err := errors.Join(seedErr, file.Close()); err != nil {
				return fmt.Errorf("seed %s/%s: %w", dataset, table.Name, err)
			}
			if read != table.Rows || hex.EncodeToString(hash.Sum(nil)) != table.DocumentsHash {
				return errors.New("immutable Neo4j source document hash or input count differs")
			}
			result, err := session.Run(ctx, countQuery, nil)
			if err != nil {
				return err
			}
			record, err := result.Single(ctx)
			if err != nil {
				return err
			}
			actual, _ := record.Get("count")
			if actual != table.Rows {
				return errors.New("actual Neo4j source dataset table count differs")
			}
			count += read
		}
		if count != manifest.Plan.VertexTotal+manifest.Plan.EdgeTotal {
			return errors.New("seeded P0 Neo4j dataset count differs")
		}
		if err := verifyHorizonP0Neo4jIterator(ctx, jobs[dataset], password, count); err != nil {
			return err
		}
		path := map[string]string{"base": paths.Base, "append": paths.Append, "upsert": paths.Upsert}[dataset]
		if _, err := fixture.Verify(path); err != nil {
			return err
		}
		datasets[dataset] = map[string]any{"scope": scope, "records": count,
			"fixtureRootSha256": manifest.FixtureRoot, "tablesVerified": len(manifest.Tables)}
	}
	body, err := json.MarshalIndent(map[string]any{"source": source, "sourceVersion": version,
		"maximumSeedBatchRows": horizonP0Neo4jSeedRows, "productionIteratorVerified": true, "datasets": datasets}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(output, "neo4j-source-seed.json"), append(body, '\n'), 0600)
}

func verifyHorizonP0Neo4jIterator(ctx context.Context, job config.LoadJob, password string, expected int64) error {
	n := job.Source.Neo4j
	client, err := neosource.NewSDKClient(ctx, n.URI, n.Database, n.Username, password, n.FetchRows)
	if err != nil {
		return err
	}
	it, err := neosource.NewIterator(ctx, neosource.IteratorOptions{
		Namespace: job.Source.Namespace, Source: *n, Client: client, MaxProperties: 64,
	})
	if err != nil {
		return errors.Join(err, client.Close())
	}
	var count int64
	for {
		_, err := it.Next(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return errors.Join(err, it.Close())
		}
		count++
	}
	if count != expected {
		return errors.Join(errors.New("production Neo4j iterator count differs"), it.Close())
	}
	return it.Close()
}

func TestHorizonP0Neo4jConfigScope(t *testing.T) {
	for _, source := range []string{"neo4j44", "neo4j526"} {
		uri, env := "bolt://127.0.0.1:7689", "AGEFREIGHTER_NEO4J44_URI"
		if source == "neo4j526" {
			uri, env = "bolt://127.0.0.1:7690", "AGEFREIGHTER_NEO4J526_URI"
		}
		t.Setenv(env, uri)
		for _, invalid := range []string{"bolt://127.0.0.1:7687", "bolt://127.0.0.1:7688",
			"bolt://database.example:7689", uri + "?address=database.example", uri + "/other"} {
			if validateHorizonP0Neo4jURI(source, invalid) == nil {
				t.Fatal("accepted an unrelated Neo4j fixture endpoint")
			}
		}
		manifest := portable.Manifest{Tables: []portable.Table{
			{Name: "Supplier", Kind: "node", Columns: []string{"source_key", "external_id", "name"}},
			{Name: "Product", Kind: "node", Columns: []string{"source_key", "external_id", "name"}},
			{Name: "SUPPLIES", Kind: "edge", StartLabel: "Supplier", EndLabel: "Product",
				Columns: []string{"source_key", "relationship_id", "start_id", "end_id", "quantity"}},
		}}
		for _, dataset := range []string{"base", "append", "upsert"} {
			job := tinyJob(manifest, t.TempDir(), source, 17)
			if err := horizonP0Neo4jQueries(&job, manifest, source, "r20261118p0neo04", dataset); err != nil {
				t.Fatal(err)
			}
			n := job.Source.Neo4j
			if n.Discovery != nil || n.SourceID != "horizon-p0-"+source || n.Vertices[0].Label != "Supplier" ||
				n.Edges[0].Start.Label != "Supplier" || n.Edges[0].End.Label != "Product" ||
				n.Edges[0].Properties["start_id"] != "" || !strings.Contains(n.Edges[0].Query, "s.external_id AS start_id") ||
				!strings.Contains(n.Vertices[0].Query, source+"_r20261118p0neo04_"+dataset+"_Supplier") ||
				strings.Contains(n.Vertices[0].InitialQuery, "$afterKey") {
				t.Fatal("Neo4j source identity, scalar projection or dataset isolation differs")
			}
			path := filepath.Join(t.TempDir(), "job.json")
			writeTinyJSON(t, path, job)
			if _, err := config.Load(path); err != nil {
				t.Fatal(err)
			}
		}
		for _, run := range []string{"tiny", "r20261118p0neo04` MATCH (n) DETACH DELETE n //"} {
			if _, err := horizonP0Neo4jScope(source, run, "base"); err == nil {
				t.Fatal("accepted an unrelated or injectable Neo4j source scope")
			}
		}
	}
	if _, err := horizonP0Neo4jScope("postgresql", "r20261118p0neo04", "base"); err == nil {
		t.Fatal("accepted an unrelated source kind")
	}
}
