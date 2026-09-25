package cli

import (
	"fmt"

	"github.com/rioriost/agefreighter/internal/app"
	reportcontract "github.com/rioriost/agefreighter/internal/report"
	"github.com/spf13/cobra"
)

func newInventoryCommand() *cobra.Command {
	var formatValue string
	command := &cobra.Command{
		Use:   "inventory JOB",
		Short: "Read exact source totals for guided capacity planning",
		Args:  cobra.ExactArgs(1),
		PreRunE: func(_ *cobra.Command, _ []string) error {
			return validateDiagnosticOutputFlags(formatValue, "", false)
		},
		RunE: func(command *cobra.Command, args []string) error {
			document, err := app.SourceInventory(
				command.Context(),
				args[0],
				app.InventoryOptions{},
			)
			if err != nil {
				return fmt.Errorf("inventory: %w", err)
			}
			return renderDiagnosticOutput(
				command,
				document,
				reportcontract.Format(formatValue),
				"",
			)
		},
	}
	command.Flags().StringVar(
		&formatValue,
		"format",
		string(reportcontract.FormatJSON),
		"report format: json or markdown",
	)
	return command
}
