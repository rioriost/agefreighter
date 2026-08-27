package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/spf13/cobra"
)

const benchmarkDSNEnvironment = "AGEFREIGHTER_AGE_TEST_DSN"

const (
	fixtureVertices   = 4
	fixtureEdges      = 6
	benchmarkVertices = 100_000
	benchmarkEdges    = 500_000
)

func NewGenerateCommand() *cobra.Command {
	command := &cobra.Command{
		Use:   "generate",
		Short: "Generate deterministic source datasets",
		Args:  cobra.NoArgs,
	}
	command.AddCommand(
		newDatasetCommand(
			"fixture",
			"Generate a small integration fixture",
			DatasetSpec{Vertices: fixtureVertices, Edges: fixtureEdges, Seed: 1},
			false,
		),
		newDatasetCommand(
			"benchmark",
			"Generate a configurable bulk-load benchmark dataset",
			DatasetSpec{
				Vertices: benchmarkVertices,
				Edges:    benchmarkEdges,
				Seed:     1,
			},
			true,
		),
	)
	return command
}

func NewBenchmarkCommand() *cobra.Command {
	var (
		workload         string
		strategy         string
		rows             int
		endpointVertices int
		propertyBytes    int
		timeout          time.Duration
	)
	command := &cobra.Command{
		Use:   "benchmark-age-copy",
		Short: "Benchmark Apache AGE bulk-write strategies",
		Args:  cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			dsn := os.Getenv(benchmarkDSNEnvironment)
			if dsn == "" {
				return fmt.Errorf("%s is required", benchmarkDSNEnvironment)
			}
			ctx, cancel := context.WithTimeout(command.Context(), timeout)
			defer cancel()
			result, err := RunBulkBenchmark(ctx, BulkBenchmarkOptions{
				DSN:              dsn,
				Workload:         BenchmarkWorkload(workload),
				Strategy:         BenchmarkStrategy(strategy),
				Rows:             rows,
				EndpointVertices: endpointVertices,
				PropertyBytes:    propertyBytes,
				OperationTimeout: timeout,
			})
			if err != nil {
				return err
			}
			encoder := json.NewEncoder(command.OutOrStdout())
			if err := encoder.Encode(result); err != nil {
				return fmt.Errorf("write benchmark result: %w", err)
			}
			return nil
		},
	}
	command.Flags().StringVar(&workload, "workload", string(BenchmarkEdges), "vertices or edges")
	command.Flags().StringVar(&strategy, "strategy", string(BenchmarkDirect), "COPY strategy")
	command.Flags().IntVar(&rows, "rows", 100_000, "rows to write")
	command.Flags().IntVar(&endpointVertices, "endpoint-vertices", 10_000, "preloaded edge endpoints")
	command.Flags().IntVar(&propertyBytes, "property-bytes", 64, "payload string bytes")
	command.Flags().DurationVar(&timeout, "timeout", 10*time.Minute, "operation timeout")
	return command
}

func NewBenchmarkReportCommand() *cobra.Command {
	var format string
	command := &cobra.Command{
		Use:   "benchmark-report [FILE...]",
		Short: "Normalize AGE benchmark results into a deterministic report",
		Args:  cobra.ArbitraryArgs,
		RunE: func(command *cobra.Command, paths []string) error {
			reportFormat := BenchmarkReportFormat(format)
			if err := validateBenchmarkReportFormat(reportFormat); err != nil {
				return err
			}
			readers, files, err := openBenchmarkInputs(
				paths,
				command.InOrStdin(),
			)
			if err != nil {
				return err
			}
			report, normalizeErr := NormalizeBenchmarkReport(readers)
			closeErr := closeBenchmarkInputs(files)
			if err := errors.Join(normalizeErr, closeErr); err != nil {
				return err
			}
			return WriteBenchmarkReport(
				command.OutOrStdout(),
				report,
				reportFormat,
			)
		},
	}
	command.Flags().StringVar(
		&format,
		"format",
		string(BenchmarkReportJSON),
		"output format: json or markdown",
	)
	return command
}

func NewInspectCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "inspect JOB",
		Short: "Inspect source mappings and target configuration without connecting",
		Args:  cobra.ExactArgs(1),
		RunE: func(command *cobra.Command, args []string) error {
			report, err := InspectConfiguration(args[0])
			if err != nil {
				return err
			}
			encoder := json.NewEncoder(command.OutOrStdout())
			encoder.SetIndent("", "  ")
			if err := encoder.Encode(report); err != nil {
				return fmt.Errorf("write inspection: %w", err)
			}
			return nil
		},
	}
}

func openBenchmarkInputs(
	paths []string,
	stdin io.Reader,
) ([]io.Reader, []*os.File, error) {
	if len(paths) == 0 {
		return []io.Reader{stdin}, nil, nil
	}
	if len(paths) > BenchmarkReportMaxInputFiles {
		return nil, nil, fmt.Errorf(
			"benchmark report accepts at most %d input files",
			BenchmarkReportMaxInputFiles,
		)
	}
	readers := make([]io.Reader, 0, len(paths))
	files := make([]*os.File, 0, len(paths))
	stdinUsed := false
	for _, path := range paths {
		if path == "-" {
			if stdinUsed {
				return nil, nil, errors.Join(
					errors.New("standard input can be specified only once"),
					closeBenchmarkInputs(files),
				)
			}
			stdinUsed = true
			readers = append(readers, stdin)
			continue
		}
		file, err := os.Open(path)
		if err != nil {
			return nil, nil, errors.Join(
				fmt.Errorf("open benchmark input %q: %w", path, err),
				closeBenchmarkInputs(files),
			)
		}
		files = append(files, file)
		readers = append(readers, file)
	}
	return readers, files, nil
}

func validateBenchmarkReportFormat(format BenchmarkReportFormat) error {
	switch format {
	case BenchmarkReportJSON, BenchmarkReportMarkdown:
		return nil
	default:
		return fmt.Errorf("unsupported benchmark report format %q", format)
	}
}

func closeBenchmarkInputs(files []*os.File) error {
	var result error
	for _, file := range files {
		if err := file.Close(); err != nil {
			result = errors.Join(result, fmt.Errorf(
				"close benchmark input %q: %w",
				file.Name(),
				err,
			))
		}
	}
	return result
}

func newDatasetCommand(
	use string,
	summary string,
	defaults DatasetSpec,
	configurable bool,
) *cobra.Command {
	var output string
	spec := defaults
	command := &cobra.Command{
		Use:   use,
		Short: summary,
		Args:  cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			manifest, err := GenerateDataset(output, spec)
			if err != nil {
				return err
			}
			_, err = fmt.Fprintf(
				command.OutOrStdout(),
				"generated %d vertices and %d edges in %s (seed %d)\n",
				manifest.Vertices,
				manifest.Edges,
				output,
				manifest.Seed,
			)
			return err
		},
	}
	command.Flags().StringVar(&output, "output", "", "new output directory")
	_ = command.MarkFlagRequired("output")
	if configurable {
		command.Flags().Uint64Var(&spec.Vertices, "vertices", defaults.Vertices, "number of vertices")
		command.Flags().Uint64Var(&spec.Edges, "edges", defaults.Edges, "number of edges")
		command.Flags().Uint64Var(&spec.Seed, "seed", defaults.Seed, "deterministic generator seed")
	}
	return command
}
