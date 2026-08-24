package cli

import (
	"encoding/json"
	"fmt"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/spf13/cobra"
)

func newValidateCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "validate JOB",
		Short: "Validate a load job without connecting to its source or target",
		Args:  cobra.ExactArgs(1),
		RunE: func(command *cobra.Command, args []string) error {
			job, err := config.Load(args[0])
			if err != nil {
				return fmt.Errorf("validate job: %w", err)
			}
			_, err = fmt.Fprintf(
				command.OutOrStdout(),
				"valid: %s (%s -> %s, mode=%s)\n",
				job.Metadata.Name,
				job.Source.Type,
				job.Target.Type,
				job.Target.Mode,
			)
			return err
		},
	}
}

func newPlanCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "plan JOB",
		Short: "Print a static load plan without connecting to its source or target",
		Args:  cobra.ExactArgs(1),
		RunE: func(command *cobra.Command, args []string) error {
			job, err := config.Load(args[0])
			if err != nil {
				return fmt.Errorf("plan job: %w", err)
			}
			encoder := json.NewEncoder(command.OutOrStdout())
			encoder.SetIndent("", "  ")
			if err := encoder.Encode(config.BuildStaticPlan(job)); err != nil {
				return fmt.Errorf("write plan: %w", err)
			}
			return nil
		},
	}
}
