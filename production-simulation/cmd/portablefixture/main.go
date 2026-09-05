package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"syscall"

	"github.com/rioriost/agefreighter/production-simulation/internal/portable"
	"github.com/rioriost/agefreighter/production-simulation/internal/rangedigest"
)

func main() {
	manifest := flag.String("manifest", "", "verified production-simulation fixture manifest")
	output := flag.String("output", "", "new portable fixture directory")
	flag.Parse()
	if flag.NArg() != 0 || *manifest == "" || *output == "" {
		flag.Usage()
		os.Exit(2)
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	result, err := portable.Export(ctx, *manifest, *output)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := verify(ctx, *manifest, *output, result); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func verify(ctx context.Context, input, output string, result portable.Manifest) error {
	expected, err := rangedigest.FixtureManifest(ctx, input, 100_000)
	if err != nil {
		return err
	}
	source := result.CSVSource()
	for i := range source.Vertices {
		source.Vertices[i].Path = filepath.Join(output, source.Vertices[i].Path)
	}
	for i := range source.Edges {
		source.Edges[i].Path = filepath.Join(output, source.Edges[i].Path)
	}
	actual, err := rangedigest.CSVManifest(ctx, source, result.FixtureRoot, 100_000)
	if err != nil {
		return err
	}
	if expected.RootSHA256 != actual.RootSHA256 || expected.RecordCount != actual.RecordCount || !slices.Equal(expected.Leaves, actual.Leaves) {
		return fmt.Errorf("portable CSV canonical mismatch")
	}
	f, err := os.OpenFile(filepath.Join(output, "canonical-verification.json"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	err = json.NewEncoder(f).Encode(map[string]any{"outcome": "pass", "expected": expected, "actual": actual})
	if err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}
