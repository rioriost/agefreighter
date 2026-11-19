//go:build linux || darwin

package simulation_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/rangedigest"
)

const p1FixtureReceiptName = "worker-fixture.json"

type p1FixtureDataset struct {
	Vertices        int64  `json:"vertices"`
	Edges           int64  `json:"edges"`
	RawSHA256       string `json:"rawSha256"`
	CanonicalSHA256 string `json:"canonicalSha256"`
}

type p1FixturePreparation struct {
	Commit           string                      `json:"candidateCommit"`
	ExecutableSHA256 string                      `json:"executableSha256"`
	PreparedAt       time.Time                   `json:"preparedAtUtc"`
	Datasets         map[string]p1FixtureDataset `json:"datasets"`
}

var p1FixtureOrder = [...]string{"base", "append", "upsert"}

func p1FixtureSpecification(name string) p1FixtureDataset {
	v, e := int64(p1BaseVertices), int64(p1BaseEdges)
	if name != "base" {
		v, e = v/100, e/100
	}
	return p1FixtureDataset{v, e, p1Expected[name].raw, p1Expected[name].canonical}
}

func (p p1FixturePreparation) validate(a p1Admission, now time.Time) error {
	if p.Commit != a.Commit || p.ExecutableSHA256 != a.ExecutableSHA256 {
		return errors.New("fixture preparation must use the admitted candidate and executable")
	}
	if p.PreparedAt.Before(a.Deadline.Add(-8*time.Hour)) || p.PreparedAt.After(a.WriterReadyAt) || p.PreparedAt.After(now) {
		return errors.New("fixture preparation must finish before writer admission, within the current lease")
	}
	if len(p.Datasets) != len(p1FixtureOrder) {
		return errors.New("fixture preparation does not include exactly the approved P1 datasets")
	}
	for _, name := range p1FixtureOrder {
		if p.Datasets[name] != p1FixtureSpecification(name) {
			return fmt.Errorf("fixture preparation does not match the approved P1 %s counts or roots", name)
		}
	}
	return nil
}

func verifiedP1Fixtures(t *testing.T, ctx context.Context, root string) (map[string]string, map[string]rangedigest.Manifest, map[string]p1FixtureDataset) {
	t.Helper()
	paths := make(map[string]string, len(p1FixtureOrder))
	expected := make(map[string]rangedigest.Manifest, len(p1FixtureOrder))
	datasets := make(map[string]p1FixtureDataset, len(p1FixtureOrder))
	for _, name := range p1FixtureOrder {
		paths[name] = filepath.Join(root, name, "manifest.json")
		input, err := fixture.Verify(paths[name])
		if err != nil {
			t.Fatal(err)
		}
		if input.Plan.Phase != fixture.PhaseP1 || input.Seed != 20260829 || input.Shards != 64 {
			t.Fatalf("%s is not the fixed P1 phase, seed and shard count", name)
		}
		want := p1FixtureSpecification(name)
		if input.RootSHA256 != want.RawSHA256 ||
			input.Plan.VertexTotal != want.Vertices || input.Plan.EdgeTotal != want.Edges {
			t.Fatalf("%s differs from the frozen P1 input", name)
		}
		digest, err := rangedigest.ScenarioManifest(ctx, p1Scenario(paths, name), 100_000, false)
		if err != nil {
			t.Fatal(err)
		}
		datasets[name] = p1FixtureDataset{
			Vertices: input.Plan.VertexTotal, Edges: input.Plan.EdgeTotal,
			RawSHA256: input.RootSHA256, CanonicalSHA256: digest.RootSHA256,
		}
		wantRows := int64(p1BaseRows)
		if name != "base" {
			wantRows += p1DeltaRows
		}
		if datasets[name] != want || digest.RecordCount != wantRows {
			t.Fatalf("%s independent canonical row count or SHA-256 differs", name)
		}
		expected[name] = digest
	}
	return paths, expected, datasets
}

func TestHorizonDBP1PrepareWorkerFixture(t *testing.T) {
	root := os.Getenv("AGEFREIGHTER_HORIZON_P1_PREPARE_DIR")
	if root == "" {
		t.Skip("explicit P1 worker fixture preparation required")
	}
	commit := os.Getenv("AGEFREIGHTER_HORIZON_P1_PREPARE_COMMIT")
	if os.Getenv(p1Gate) != "" || os.Getenv(p1GatePG18) != "" || os.Getenv(p1PGGate) != "" || os.Getenv(p1ReceiptEnv) != "" ||
		!filepath.IsAbs(root) || filepath.Clean(root) != root || !p1CommitPattern.MatchString(commit) {
		t.Fatal("require a new absolute fixture directory, exact candidate commit and no live pilot admission")
	}
	var fs syscall.Statfs_t
	if err := syscall.Statfs(filepath.Dir(root), &fs); err != nil || fs.Bavail > fs.Blocks ||
		(float64(fs.Bavail)-float64(fs.Blocks)/5)*float64(fs.Bsize) < 8<<30 {
		t.Fatal("require 8 GiB for fixture and CSV staging before the worker disk reaches 80%")
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(root, 0700); err != nil {
		t.Fatal(err)
	}
	ctx, stop := context.WithTimeout(t.Context(), 30*time.Minute)
	defer stop()
	base := filepath.Join(root, "base")
	if _, err := fixture.Generate(ctx, fixture.GenerateConfig{
		Phase: fixture.PhaseP1, Output: base, Seed: 20260829, Shards: 64, Workers: 4,
	}); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []fixture.DeltaKind{fixture.DeltaAppend, fixture.DeltaUpsert} {
		if _, err := fixture.GenerateDelta(ctx, filepath.Join(base, "manifest.json"), filepath.Join(root, string(kind)), kind); err != nil {
			t.Fatal(err)
		}
	}
	_, _, datasets := verifiedP1Fixtures(t, ctx, root)
	path := filepath.Join(root, p1FixtureReceiptName)
	writeTinyJSON(t, path, p1FixturePreparation{
		Commit: commit, ExecutableSHA256: p1FileSHA(t, executable), PreparedAt: time.Now().UTC(), Datasets: datasets,
	})
	t.Logf("prepared P1 source-only fixture; receipt SHA-256 %s", p1FileSHA(t, path))
}

func TestHorizonDBP1FixturePreparationBinding(t *testing.T) {
	now := time.Now()
	a := p1Admission{Commit: "candidate", ExecutableSHA256: "binary", WriterReadyAt: now.Add(-time.Minute), Deadline: now.Add(7 * time.Hour)}
	for _, tt := range []struct {
		name   string
		change func(*p1FixturePreparation)
	}{
		{name: "exact fixture"},
		{name: "different executable", change: func(p *p1FixturePreparation) { p.ExecutableSHA256 = "older" }},
		{name: "different candidate", change: func(p *p1FixturePreparation) { p.Commit = "older" }},
		{name: "previous lease", change: func(p *p1FixturePreparation) { p.PreparedAt = now.Add(-2 * time.Hour) }},
		{name: "completed after writer readiness", change: func(p *p1FixturePreparation) { p.PreparedAt = now }},
		{name: "missing delta", change: func(p *p1FixturePreparation) { delete(p.Datasets, "upsert") }},
		{name: "incorrect canonical delta", change: func(p *p1FixturePreparation) {
			d := p.Datasets["upsert"]
			d.CanonicalSHA256 = p.Datasets["append"].CanonicalSHA256
			p.Datasets["upsert"] = d
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p := p1FixturePreparation{a.Commit, a.ExecutableSHA256, now.Add(-2 * time.Minute), map[string]p1FixtureDataset{
				"base": p1FixtureSpecification("base"), "append": p1FixtureSpecification("append"), "upsert": p1FixtureSpecification("upsert"),
			}}
			if tt.change != nil {
				tt.change(&p)
			}
			if err := p.validate(a, now); (err != nil) != (tt.change != nil) {
				t.Fatalf("unexpected preparation binding result: %v", err)
			}
		})
	}
}
