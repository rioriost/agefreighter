package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
)

const defaultSeed uint64 = 20260829

func main() {
	os.Exit(run(context.Background(), os.Args[1:], os.Stdout, os.Stderr))
}

func run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		usage(stderr)
		return 2
	}
	var err error
	switch args[0] {
	case "generate":
		err = runGenerate(ctx, args[1:], stdout, stderr)
	case "verify":
		err = runVerify(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		usage(stdout)
		return 0
	default:
		err = fmt.Errorf("unknown command %q", args[0])
	}
	if err != nil {
		fmt.Fprintf(stderr, "fixturegen: %v\n", err)
		return 1
	}
	return 0
}

func runGenerate(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("generate", flag.ContinueOnError)
	flags.SetOutput(stderr)
	phase := flags.String("phase", "", "tiny, p0, p1, p2, or p3")
	output := flags.String("output", "", "new output directory")
	shards := flags.Int("shards", 1, "number of output shards")
	workers := flags.Int("workers", 0, "parallel writers; zero uses host CPUs")
	seedText := flags.String("seed", strconv.FormatUint(defaultSeed, 10), "unsigned deterministic seed")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("generate accepts flags only")
	}
	seed, err := strconv.ParseUint(*seedText, 10, 64)
	if err != nil {
		return fmt.Errorf("parse seed: %w", err)
	}
	manifest, err := fixture.Generate(ctx, fixture.GenerateConfig{
		Phase: fixture.Phase(*phase), Output: *output, Shards: *shards, Workers: *workers, Seed: seed,
	})
	if err != nil {
		return err
	}
	fmt.Fprintf(stdout, "phase=%s vertices=%d edges=%d files=%d root_sha256=%s\n",
		manifest.Plan.Phase, manifest.Plan.VertexTotal, manifest.Plan.EdgeTotal,
		len(manifest.Files), manifest.RootSHA256)
	return nil
}

func runVerify(args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("verify", flag.ContinueOnError)
	flags.SetOutput(stderr)
	manifestPath := flags.String("manifest", "", "fixture manifest path")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *manifestPath == "" {
		return errors.New("verify requires --manifest and accepts no positional arguments")
	}
	manifest, err := fixture.Verify(*manifestPath)
	if err != nil {
		return err
	}
	fmt.Fprintf(stdout, "verified phase=%s vertices=%d edges=%d files=%d root_sha256=%s\n",
		manifest.Plan.Phase, manifest.Plan.VertexTotal, manifest.Plan.EdgeTotal,
		len(manifest.Files), manifest.RootSHA256)
	return nil
}

func usage(writer io.Writer) {
	fmt.Fprintln(writer, "usage: fixturegen generate [flags]")
	fmt.Fprintln(writer, "       fixturegen verify --manifest PATH")
}
