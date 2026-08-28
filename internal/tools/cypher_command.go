package tools

import (
	"errors"
	"fmt"

	"github.com/rioriost/agefreighter/internal/cypher"
	"github.com/spf13/cobra"
)

var ErrCypherStrict = errors.New(
	"Cypher compatibility check failed in strict mode",
)

func NewCheckCypherCommand() *cobra.Command {
	var (
		format    string
		targetAGE string
		strict    bool
	)
	command := &cobra.Command{
		Use:   "check-cypher FILE ...",
		Short: "Statically check local Cypher for Apache AGE compatibility",
		Args:  cobra.MinimumNArgs(1),
		PreRunE: func(_ *cobra.Command, _ []string) error {
			if targetAGE != cypher.TargetAGEVersion {
				return fmt.Errorf(
					"unsupported Apache AGE target %q; only %s is cataloged",
					targetAGE,
					cypher.TargetAGEVersion,
				)
			}
			switch cypher.Format(format) {
			case cypher.FormatJSON, cypher.FormatMarkdown:
				return nil
			default:
				return fmt.Errorf(
					"unsupported output format %q; use json or markdown",
					format,
				)
			}
		},
		RunE: func(command *cobra.Command, paths []string) error {
			result, err := cypher.AnalyzeFiles(
				command.Context(),
				paths,
				cypher.Options{},
			)
			if err != nil {
				return fmt.Errorf("check Cypher: %w", err)
			}
			output, err := cypher.Render(result, cypher.Format(format))
			if err != nil {
				return fmt.Errorf("render Cypher report: %w", err)
			}
			if err := cypher.WriteOutput(
				command.Context(),
				command.OutOrStdout(),
				output,
			); err != nil {
				return errors.New("write Cypher report")
			}
			if strict && cypher.StrictFailure(result) {
				return ErrCypherStrict
			}
			return nil
		},
	}
	command.Flags().StringVar(
		&format,
		"format",
		string(cypher.FormatJSON),
		"output format: json or markdown",
	)
	command.Flags().StringVar(
		&targetAGE,
		"target-age",
		cypher.TargetAGEVersion,
		"Apache AGE compatibility rule catalog",
	)
	command.Flags().BoolVar(
		&strict,
		"strict",
		false,
		"fail when any query is unsupported or unknown",
	)
	return command
}
