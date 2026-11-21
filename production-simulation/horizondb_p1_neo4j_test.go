//go:build linux || darwin

package simulation_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/neo4j/neo4j-go-driver/v6/neo4j"
	"github.com/rioriost/agefreighter/internal/config"
	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
)

const p1Neo4jPassword = "/dev/shm/agefreighter-p1-neo4j/password"

func TestHorizonDBP1Neo4j44PG17Integration(t *testing.T) {
	runHorizonDBP1(t, 17, "neo4j44")
}

func TestHorizonDBP1Neo4j44PG18Integration(t *testing.T) {
	runHorizonDBP1(t, 18, "neo4j44")
}

func TestHorizonDBP1Neo4j526PG17Integration(t *testing.T) {
	runHorizonDBP1(t, 17, "neo4j526")
}

func TestHorizonDBP1Neo4j526PG18Integration(t *testing.T) {
	runHorizonDBP1(t, 18, "neo4j526")
}

func setP1Neo4jConnection(job *config.LoadJob, source string) {
	job.Source.Neo4j.URI = horizonNeo4jURI(source)
	job.Source.Neo4j.Password = &config.SecretRef{File: p1Neo4jPassword}
}

func configureP1Neo4jJob(job *config.LoadJob, manifest portable.Manifest, a p1Admission, dataset string) error {
	setP1Neo4jConnection(job, a.source())
	return horizonNeo4jQueries(job, manifest, a.source(), a.RunID, dataset, fixture.PhaseP1)
}

func validateP1Neo4jWorker(t *testing.T, job config.LoadJob, a p1Admission, dataset, fixtureRoot string) error {
	t.Helper()
	var manifest portable.Manifest
	p1ReadJSON(t, filepath.Join(a.Output, "csv-"+dataset, portable.ManifestFilename), &manifest)
	if manifest.FixtureRoot == "" || manifest.FixtureRoot != fixtureRoot {
		return errors.New("Neo4j loader fixture root differs from its admitted dataset")
	}
	want := tinyJob(manifest, "", a.source(), a.writerMajor())
	if err := configureP1Neo4jJob(&want, manifest, a, dataset); err != nil {
		return err
	}
	if job.Source.Type != config.SourceNeo4j || !reflect.DeepEqual(job.Source.Neo4j, want.Source.Neo4j) {
		return errors.New("Neo4j loader differs from its admitted connection or run-scoped keyset queries")
	}
	return nil
}

func prepareP1Neo4j(t *testing.T, ctx context.Context, a p1Admission, paths map[string]string, exports map[string]portable.Manifest) func() {
	t.Helper()
	info, err := os.Lstat(p1Neo4jPassword)
	if err != nil || !info.Mode().IsRegular() || info.Mode().Perm() != 0600 {
		t.Fatal("require a private Neo4j fixture credential")
	}
	secret, err := os.ReadFile(p1Neo4jPassword)
	if err != nil || !regexp.MustCompile(`^[a-f0-9]{64}$`).Match(secret) {
		t.Fatal("require the dedicated Neo4j fixture credential")
	}
	password := string(secret)
	directories := make(map[string]string, len(p1FixtureOrder))
	for _, dataset := range p1FixtureOrder {
		directories[dataset] = filepath.Join(a.Output, "csv-"+dataset)
	}
	seedCtx, stopSeed := context.WithTimeout(ctx, 50*time.Minute)
	defer stopSeed()
	if err := seedHorizonNeo4j(t, seedCtx, p1Scenario(paths, "upsert"), exports, directories,
		a.source(), a.RunID, a.Output, password, fixture.PhaseP1); err != nil {
		failP1Neo4j(t, "Neo4j fixture seed", err)
	}
	t.Log("Pinned Neo4j source fixture seeded and counted")
	root := filepath.Dir(a.FixtureDir)
	cmd := exec.CommandContext(seedCtx, "/bin/bash",
		filepath.Join(root, "candidate", "production-simulation", "scripts", "bootstrap-horizondb-p1-neo4j.sh"), "freeze")
	cmd.Dir = root
	var diagnostic bytes.Buffer
	cmd.Stderr = &diagnostic
	if err := cmd.Run(); err != nil {
		if strings.TrimSpace(diagnostic.String()) == "Neo4j source did not shut down cleanly" {
			t.Fatal("Neo4j source did not shut down cleanly before read-only restart")
		}
		failP1Neo4j(t, "Neo4j read-only restart", err)
	}
	before, err := hashP1Neo4j(seedCtx, a, exports, password)
	if err != nil {
		failP1Neo4j(t, "Neo4j initial source digest", err)
	}
	writeTinyJSON(t, filepath.Join(a.Output, "neo4j-source-before.json"), before)
	t.Log("Neo4j source write rejected and indexed keyset scans verified")
	return func() {
		verifyCtx, stop := context.WithTimeout(ctx, 20*time.Minute)
		defer stop()
		after, err := hashP1Neo4j(verifyCtx, a, exports, password)
		if err != nil {
			failP1Neo4j(t, "Neo4j final source digest", err)
		}
		writeTinyJSON(t, filepath.Join(a.Output, "neo4j-source-after.json"), after)
		if !reflect.DeepEqual(before, after) {
			t.Fatal("Neo4j source counts or ordered property and endpoint digests changed")
		}
	}
}

func failP1Neo4j(t *testing.T, step string, err error) {
	t.Helper()
	var neoErr *neo4j.Neo4jError
	if errors.As(err, &neoErr) && regexp.MustCompile(`^Neo\.[A-Za-z.]+$`).MatchString(neoErr.Code) {
		t.Fatalf("%s failed (%s)", step, neoErr.Code)
	}
	t.Fatalf("%s failed (%T)", step, err)
}

type p1Neo4jEvidence struct {
	ReadOnly       bool            `json:"readOnly"`
	IndexedKeysets bool            `json:"indexedKeysets"`
	Tables         []p1SourceTable `json:"tables"`
}

func hashP1Neo4j(ctx context.Context, a p1Admission, exports map[string]portable.Manifest, password string) (p1Neo4jEvidence, error) {
	var evidence p1Neo4jEvidence
	driver, err := neo4j.NewDriverWithContext(horizonNeo4jURI(a.source()), neo4j.BasicAuth("neo4j", password, ""))
	if err != nil {
		return evidence, err
	}
	defer driver.Close(context.Background())
	writer := driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j", AccessMode: neo4j.AccessModeWrite})
	defer writer.Close(context.Background())
	tx, err := writer.BeginTransaction(ctx)
	if err != nil {
		return evidence, err
	}
	result, err := tx.Run(ctx, "CREATE (:P1ReadOnlyProbe)", nil)
	if err == nil {
		_, err = result.Consume(ctx)
	}
	rollbackErr := tx.Close(ctx)
	var writeError *neo4j.Neo4jError
	if !errors.As(err, &writeError) || (writeError.Code != "Neo.ClientError.General.ForbiddenOnReadOnlyDatabase" &&
		writeError.Code != "Neo.ClientError.General.WriteOnReadOnlyAccessDatabase") {
		return evidence, errors.Join(errors.New("Neo4j did not reject a write on the frozen fixture database"), err, rollbackErr)
	}
	evidence.ReadOnly = true
	session := driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j", AccessMode: neo4j.AccessModeRead, FetchSize: 2048})
	defer session.Close(context.Background())
	for _, dataset := range p1FixtureOrder {
		manifest := exports[dataset]
		job := tinyJob(manifest, "", a.source(), a.writerMajor())
		if err := configureP1Neo4jJob(&job, manifest, a, dataset); err != nil {
			return evidence, err
		}
		queries := make(map[string]string, len(manifest.Tables))
		for _, v := range job.Source.Neo4j.Vertices {
			queries[v.Label] = v.Query
		}
		for _, e := range job.Source.Neo4j.Edges {
			queries[e.Label] = e.Query
		}
		for _, table := range manifest.Tables {
			operator := "NodeIndexSeekByRange"
			if table.Kind == "edge" {
				operator = "DirectedRelationshipIndexSeekByRange"
			}
			for _, key := range []any{nil, int64(0)} {
				if err := requireP1Neo4jKeyset(ctx, session, queries[table.Name], operator, key); err != nil {
					return evidence, err
				}
			}
			rows, digest, err := digestP1Neo4jTable(ctx, session, queries[table.Name])
			if err != nil {
				return evidence, err
			}
			if rows != table.Rows {
				return evidence, fmt.Errorf("Neo4j source %s/%s row count differs", dataset, table.Name)
			}
			evidence.Tables = append(evidence.Tables, p1SourceTable{dataset, table.Name, rows, digest})
		}
	}
	evidence.IndexedKeysets = true
	return evidence, nil
}

func requireP1Neo4jKeyset(ctx context.Context, session neo4j.Session, query, seek string, afterKey any) error {
	result, err := session.Run(ctx, "EXPLAIN "+query, map[string]any{"afterKey": afterKey, "pageRows": int64(20_000)})
	if err != nil {
		return err
	}
	summary, err := result.Consume(ctx)
	if err != nil {
		return err
	}
	plan := summary.Plan()
	limited := false
	for plan != nil {
		op, _, _ := strings.Cut(plan.Operator(), "@")
		if op == "Limit" {
			limited = true
		}
		if op == seek && limited && len(plan.Children()) == 0 {
			return nil
		}
		if op != "ProduceResults" && op != "Projection" && op != "Limit" {
			break
		}
		children := plan.Children()
		if len(children) != 1 {
			break
		}
		plan = children[0]
	}
	return errors.New("Neo4j keyset must limit an ordered index range seek without scanning or sorting")
}

func digestP1Neo4jTable(ctx context.Context, session neo4j.Session, query string) (int64, string, error) {
	const page = 20_000
	sum := sha256.New()
	encoder := json.NewEncoder(sum)
	rows, last := int64(0), int64(-1)
	for {
		cursor, err := session.Run(ctx, query, map[string]any{"afterKey": last, "pageRows": int64(page)})
		if err != nil {
			return 0, "", err
		}
		read := 0
		for cursor.Next(ctx) {
			record := cursor.Record()
			key, _ := record.Get("source_key")
			value, ok := key.(int64)
			if !ok || value <= last || read >= page {
				return 0, "", errors.New("Neo4j source scan lost unique, increasing keys or exceeded its page bound")
			}
			if err := encoder.Encode(record.Values); err != nil {
				return 0, "", err
			}
			last = value
			read++
		}
		if err := cursor.Err(); err != nil {
			return 0, "", err
		}
		rows += int64(read)
		if read < page {
			return rows, hex.EncodeToString(sum.Sum(nil)), nil
		}
	}
}

func TestHorizonDBP1Neo4jWorkerScope(t *testing.T) {
	a := p1Admission{RunID: "rp1test01", Output: t.TempDir(),
		Host: "afhs-20261118-pg17.rw.scope.japaneast.horizondb.azure.com"}
	fixtureDir := filepath.Join(a.Output, "fixture")
	if _, err := fixture.Generate(t.Context(), fixture.GenerateConfig{
		Phase: fixture.PhaseTiny, Output: fixtureDir, Seed: 20260829, Shards: 1, Workers: 1,
	}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(fixtureDir, "manifest.json")
	manifest, err := portable.Export(t.Context(), path, filepath.Join(a.Output, "csv-append"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := portable.Export(t.Context(), path, filepath.Join(a.Output, "csv-base")); err != nil {
		t.Fatal(err)
	}
	for _, source := range []string{"neo4j44", "neo4j526"} {
		t.Run(source, func(t *testing.T) {
			a := a
			a.Source = source
			job := tinyJob(manifest, "", source, a.writerMajor())
			if err := configureP1Neo4jJob(&job, manifest, a, "append"); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(a.Output, source+"-job.json")
			writeTinyJSON(t, path, job)
			loaded, err := config.Load(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := validateP1Neo4jWorker(t, loaded, a, "append", manifest.FixtureRoot); err != nil {
				t.Fatal(err)
			}
			if validateP1Neo4jWorker(t, loaded, a, "base", manifest.FixtureRoot) == nil {
				t.Fatal("accepted a different Neo4j dataset")
			}
			if validateP1Neo4jWorker(t, loaded, a, "append", "unadmitted-root") == nil {
				t.Fatal("accepted an unadmitted fixture root")
			}
			other := map[string]string{"neo4j44": "neo4j526", "neo4j526": "neo4j44"}[source]
			wrongRoute := a
			wrongRoute.Source = other
			if validateP1Neo4jWorker(t, loaded, wrongRoute, "append", manifest.FixtureRoot) == nil {
				t.Fatal("accepted a different Neo4j route")
			}
			loaded.Source.Neo4j.URI = horizonNeo4jURI(other)
			if validateP1Neo4jWorker(t, loaded, a, "append", manifest.FixtureRoot) == nil {
				t.Fatal("accepted a different Neo4j endpoint")
			}
		})
	}
}
