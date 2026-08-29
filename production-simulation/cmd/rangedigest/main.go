package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/rioriost/agefreighter/production-simulation/internal/rangedigest"
)

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
	case "fixture":
		err = runFixture(ctx, args[1:], stdout, stderr)
	case "target":
		err = runTarget(ctx, args[1:], stdout, stderr)
	case "compare":
		err = runCompare(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		usage(stdout)
		return 0
	default:
		err = fmt.Errorf("unknown command %q", args[0])
	}
	if err != nil {
		fmt.Fprintf(stderr, "rangedigest: %v\n", err)
		return 1
	}
	return 0
}

func runFixture(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("fixture", flag.ContinueOnError)
	flags.SetOutput(stderr)
	manifestPath := flags.String("manifest", "", "fixture manifest path")
	rangeRows := flags.Int64("range-rows", 100_000, "canonical records per range")
	output := flags.String("output", "", "new output manifest")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *manifestPath == "" || *output == "" {
		return errors.New("fixture requires --manifest and --output")
	}
	manifest, err := rangedigest.FixtureManifest(ctx, *manifestPath, *rangeRows)
	if err != nil {
		return err
	}
	if err := writeNewJSON(*output, manifest); err != nil {
		return err
	}
	return printSummary(stdout, manifest)
}

func runTarget(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("target", flag.ContinueOnError)
	flags.SetOutput(stderr)
	manifestPath := flags.String("manifest", "", "fixture manifest path")
	rangeRows := flags.Int64("range-rows", 100_000, "canonical records per range")
	jobID := flags.String("job-id", "", "committed agefreighter job ID")
	dsnEnvironment := flags.String("dsn-env", "", "environment variable containing target DSN")
	output := flags.String("output", "", "new output manifest")
	timeout := flags.Duration("timeout", 24*time.Hour, "target export timeout")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *manifestPath == "" || *jobID == "" ||
		*dsnEnvironment == "" || *output == "" {
		return errors.New("target requires --manifest, --job-id, --dsn-env, and --output")
	}
	dsn := os.Getenv(*dsnEnvironment)
	if dsn == "" {
		return fmt.Errorf("DSN environment variable %s is empty", *dsnEnvironment)
	}
	exportContext, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	manifest, err := rangedigest.TargetManifest(
		exportContext, dsn, *manifestPath, *jobID, *rangeRows,
	)
	if err != nil {
		return err
	}
	if err := writeNewJSON(*output, manifest); err != nil {
		return err
	}
	return printSummary(stdout, manifest)
}

func runCompare(args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("compare", flag.ContinueOnError)
	flags.SetOutput(stderr)
	expectedPath := flags.String("expected", "", "fixture digest manifest")
	actualPath := flags.String("actual", "", "target digest manifest")
	output := flags.String("output", "", "optional new comparison JSON")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *expectedPath == "" || *actualPath == "" {
		return errors.New("compare requires --expected and --actual")
	}
	var expected, actual rangedigest.Manifest
	if err := readJSON(*expectedPath, &expected); err != nil {
		return err
	}
	if err := readJSON(*actualPath, &actual); err != nil {
		return err
	}
	comparison, compareErr := rangedigest.Compare(expected, actual)
	if *output != "" {
		if err := writeNewJSON(*output, comparison); err != nil {
			return err
		}
	}
	encoded, err := json.Marshal(comparison)
	if err != nil {
		return err
	}
	fmt.Fprintln(stdout, string(encoded))
	return compareErr
}

func readJSON(path string, target any) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("decode %s: %w", path, err)
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return fmt.Errorf("decode %s: trailing JSON", path)
	}
	return nil
}

func writeNewJSON(path string, value any) error {
	if path == "" {
		return errors.New("output path is empty")
	}
	if _, err := os.Stat(path); err == nil {
		return fmt.Errorf("output already exists: %s", path)
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	directory := filepath.Dir(path)
	file, err := os.CreateTemp(directory, ".rangedigest-*")
	if err != nil {
		return err
	}
	temporary := file.Name()
	defer os.Remove(temporary)
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Chmod(0o644); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(temporary, path)
}

func printSummary(writer io.Writer, manifest rangedigest.Manifest) error {
	_, err := fmt.Fprintf(
		writer,
		"source=%s rows=%d leaves=%d root_sha256=%s\n",
		manifest.Source,
		manifest.RecordCount,
		len(manifest.Leaves),
		manifest.RootSHA256,
	)
	return err
}

func usage(writer io.Writer) {
	fmt.Fprintln(writer, "usage: rangedigest fixture --manifest PATH --output NEW_PATH [--range-rows N]")
	fmt.Fprintln(writer, "       rangedigest target --manifest PATH --job-id UUID --dsn-env NAME --output NEW_PATH [--range-rows N]")
	fmt.Fprintln(writer, "       rangedigest compare --expected PATH --actual PATH [--output NEW_PATH]")
}
