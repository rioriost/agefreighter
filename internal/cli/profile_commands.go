package cli

import (
	"fmt"

	"github.com/rioriost/agefreighter/internal/app"
	reportcontract "github.com/rioriost/agefreighter/internal/report"
	"github.com/spf13/cobra"
)

func newProfileCommand() *cobra.Command {
	var (
		modeValue   string
		sampleSize  int
		formatValue string
	)
	command := &cobra.Command{
		Use:   "profile JOB",
		Short: "Profile configured source mappings without opening the target",
		Args:  cobra.ExactArgs(1),
		PreRunE: func(_ *cobra.Command, _ []string) error {
			switch app.ProfileMode(modeValue) {
			case app.ProfileSample, app.ProfileExact:
			default:
				return fmt.Errorf("unsupported profile mode %q; use sample or exact", modeValue)
			}
			if sampleSize < 1 || sampleSize > app.MaxProfileSampleSize {
				return fmt.Errorf(
					"--sample-size must be within 1..%d",
					app.MaxProfileSampleSize,
				)
			}
			return validateDiagnosticOutputFlags(formatValue, "", false)
		},
		RunE: func(command *cobra.Command, args []string) error {
			document, err := app.SourceProfile(command.Context(), args[0], app.ProfileOptions{
				Mode:       app.ProfileMode(modeValue),
				SampleSize: sampleSize,
			})
			if err != nil {
				return fmt.Errorf("profile: %w", err)
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
		&modeValue,
		"mode",
		string(app.ProfileSample),
		"profile mode: sample or exact",
	)
	command.Flags().IntVar(
		&sampleSize,
		"sample-size",
		app.DefaultProfileSampleSize,
		"maximum sampled rows in sample mode",
	)
	command.Flags().StringVar(
		&formatValue,
		"format",
		string(reportcontract.FormatJSON),
		"report format: json or markdown",
	)
	return command
}
