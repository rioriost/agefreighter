package cli

import (
	"encoding/json"
	"fmt"

	"github.com/rioriost/agefreighter/internal/app"
	"github.com/spf13/cobra"
)

func newLoadCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "load JOB",
		Short: "Load a validated job into Apache AGE",
		Args:  cobra.ExactArgs(1),
		RunE: func(command *cobra.Command, args []string) error {
			result, err := app.Load(command.Context(), args[0])
			if err != nil {
				return fmt.Errorf("load job %s: %w", result.JobID, err)
			}
			return writeJSON(command, result)
		},
	}
}

func newResumeCommand() *cobra.Command {
	var jobPath string
	command := &cobra.Command{
		Use:   "resume JOB_ID",
		Short: "Resume a failed load job",
		Args:  cobra.ExactArgs(1),
		RunE: func(command *cobra.Command, args []string) error {
			result, err := app.Resume(command.Context(), jobPath, args[0])
			if err != nil {
				return fmt.Errorf("resume: %w", err)
			}
			return writeJSON(command, result)
		},
	}
	command.Flags().StringVar(&jobPath, "job", "", "validated load job configuration")
	_ = command.MarkFlagRequired("job")
	return command
}

func newStatusCommand() *cobra.Command {
	var targetPath string
	command := &cobra.Command{
		Use:   "status JOB_ID",
		Short: "Show durable load job status",
		Args:  cobra.ExactArgs(1),
		RunE: func(command *cobra.Command, args []string) error {
			result, err := app.Status(command.Context(), targetPath, args[0])
			if err != nil {
				return fmt.Errorf("status: %w", err)
			}
			return writeJSON(command, result)
		},
	}
	command.Flags().StringVar(&targetPath, "target", "", "load job containing target configuration")
	_ = command.MarkFlagRequired("target")
	return command
}

func newVerifyCommand() *cobra.Command {
	var targetPath string
	command := &cobra.Command{
		Use:   "verify JOB_ID",
		Short: "Verify that a load job completed",
		Args:  cobra.ExactArgs(1),
		RunE: func(command *cobra.Command, args []string) error {
			result, err := app.Verify(command.Context(), targetPath, args[0])
			if err != nil {
				return fmt.Errorf("verify: %w", err)
			}
			return writeJSON(command, result)
		},
	}
	command.Flags().StringVar(&targetPath, "target", "", "load job containing target configuration")
	_ = command.MarkFlagRequired("target")
	return command
}

func newCleanupCommand() *cobra.Command {
	var targetPath string
	command := &cobra.Command{
		Use:   "cleanup JOB_ID",
		Short: "Remove a retained graph replacement backup",
		Args:  cobra.ExactArgs(1),
		RunE: func(command *cobra.Command, args []string) error {
			result, err := app.Cleanup(command.Context(), targetPath, args[0])
			if err != nil {
				return fmt.Errorf("cleanup: %w", err)
			}
			return writeJSON(command, result)
		},
	}
	command.Flags().StringVar(&targetPath, "target", "", "replace load job containing target configuration")
	_ = command.MarkFlagRequired("target")
	return command
}

func writeJSON(command *cobra.Command, value any) error {
	encoder := json.NewEncoder(command.OutOrStdout())
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return fmt.Errorf("write command output: %w", err)
	}
	return nil
}
