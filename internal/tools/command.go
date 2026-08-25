package tools

import (
	"fmt"

	"github.com/spf13/cobra"
)

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
