package cli

import (
	"context"
	"fmt"
	"io"

	"github.com/rioriost/agefreighter/internal/version"
	"github.com/spf13/cobra"
)

func NewAgefreighter(stdout, stderr io.Writer) *cobra.Command {
	return newRoot(
		"agefreighter",
		"Validated, resumable graph migration into Apache AGE",
		stdout,
		stderr,
	)
}

func NewTools(stdout, stderr io.Writer) *cobra.Command {
	return newRoot(
		"agefreighter-tools",
		"Diagnostics, fixtures, and benchmarks for agefreighter",
		stdout,
		stderr,
	)
}

func Execute(command *cobra.Command, args []string) error {
	command.SetArgs(args)
	return command.Execute()
}

func ExecuteContext(
	ctx context.Context,
	command *cobra.Command,
	args []string,
) error {
	command.SetContext(ctx)
	command.SetArgs(args)
	return command.Execute()
}

func newRoot(name, summary string, stdout, stderr io.Writer) *cobra.Command {
	command := &cobra.Command{
		Use:           name,
		Short:         summary,
		SilenceErrors: true,
		SilenceUsage:  true,
	}
	command.SetOut(stdout)
	command.SetErr(stderr)
	command.AddCommand(newVersionCommand(name))
	if name == "agefreighter" {
		command.AddCommand(
			newValidateCommand(),
			newPlanCommand(),
			newLoadCommand(),
			newResumeCommand(),
			newStatusCommand(),
			newVerifyCommand(),
			newCleanupCommand(),
		)
	}
	return command
}

func newVersionCommand(program string) *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print build version information",
		Args:  cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			_, err := fmt.Fprintln(command.OutOrStdout(), version.Current().String(program))
			return err
		},
	}
}
