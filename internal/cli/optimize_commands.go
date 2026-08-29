package cli

import (
	"fmt"

	"github.com/rioriost/agefreighter/internal/app"
	reportcontract "github.com/rioriost/agefreighter/internal/report"
	"github.com/spf13/cobra"
)

func newOptimizeCommand() *cobra.Command {
	var (
		targetPath  string
		formatValue string
		outputPath  string
		analyze     bool
		queryPaths  []string
	)
	command := &cobra.Command{
		Use:   "optimize",
		Short: "Report evidence-based target optimization recommendations",
		Args:  cobra.NoArgs,
		PreRunE: func(command *cobra.Command, _ []string) error {
			return validateDiagnosticOutputFlags(
				formatValue,
				outputPath,
				command.Flags().Changed("output"),
			)
		},
		RunE: func(command *cobra.Command, _ []string) error {
			document, err := app.OptimizationReport(
				command.Context(),
				targetPath,
				app.OptimizeOptions{
					Analyze: analyze, QueryPaths: queryPaths,
				},
			)
			if err != nil {
				return fmt.Errorf("optimize: %w", err)
			}
			return renderDiagnosticOutput(
				command,
				document,
				reportcontract.Format(formatValue),
				outputPath,
			)
		},
	}
	command.Flags().StringVar(
		&targetPath,
		"target",
		"",
		"load job containing target configuration",
	)
	command.Flags().StringVar(
		&formatValue,
		"format",
		string(reportcontract.FormatJSON),
		"report format: json or markdown",
	)
	command.Flags().StringVar(
		&outputPath,
		"output",
		"",
		"write report to a new permission-restricted file",
	)
	command.Flags().BoolVar(
		&analyze,
		"apply-analyze",
		false,
		"run bounded ANALYZE on allowlisted owned relations",
	)
	command.Flags().StringArrayVar(
		&queryPaths,
		"queries",
		nil,
		"local Cypher query file (repeat for multiple files)",
	)
	_ = command.MarkFlagRequired("target")
	return command
}
