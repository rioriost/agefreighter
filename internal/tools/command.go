package tools

import (
	"context"
	"encoding/json"
	"fmt"
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
