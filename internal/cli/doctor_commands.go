package cli

import (
	"errors"
	"fmt"
	"time"

	"github.com/rioriost/agefreighter/internal/app"
	reportcontract "github.com/rioriost/agefreighter/internal/report"
	"github.com/spf13/cobra"
)

func newDoctorCommand() *cobra.Command {
	var (
		targetPath  string
		formatValue string
		outputPath  string
		persist     bool
	)
	command := &cobra.Command{
		Use:   "doctor",
		Short: "Diagnose an Apache AGE target without changing it",
		Args:  cobra.NoArgs,
		PreRunE: func(command *cobra.Command, _ []string) error {
			return validateDiagnosticOutputFlags(
				formatValue,
				outputPath,
				command.Flags().Changed("output"),
			)
		},
		RunE: func(command *cobra.Command, _ []string) error {
			document, err := app.Doctor(
				command.Context(),
				targetPath,
				app.DoctorOptions{Persist: persist},
			)
			if err != nil {
				return fmt.Errorf("doctor: %w", err)
			}
			return renderDiagnosticOutput(
				command,
				document,
				reportcontract.Format(formatValue),
				outputPath,
			)
		},
	}
	command.PersistentFlags().StringVar(
		&targetPath,
		"target",
		"",
		"load job containing target configuration",
	)
	command.PersistentFlags().StringVar(
		&formatValue,
		"format",
		string(reportcontract.FormatJSON),
		"report format: json or markdown",
	)
	command.PersistentFlags().StringVar(
		&outputPath,
		"output",
		"",
		"write report to a new permission-restricted file",
	)
	command.Flags().BoolVar(
		&persist,
		"persist",
		false,
		"persist the final bounded report in a current metadata schema",
	)
	_ = command.MarkPersistentFlagRequired("target")
	command.AddCommand(newDoctorHistoryCommand(
		&targetPath,
		&formatValue,
		&outputPath,
	))
	return command
}

func newDoctorHistoryCommand(
	targetPath, formatValue, outputPath *string,
) *cobra.Command {
	limit := app.DefaultDoctorHistory
	command := &cobra.Command{
		Use:   "history",
		Short: "Read bounded persisted doctor history",
		Args:  cobra.NoArgs,
		PreRunE: func(command *cobra.Command, _ []string) error {
			if err := validateDiagnosticOutputFlags(
				*formatValue,
				*outputPath,
				command.Flags().Changed("output"),
			); err != nil {
				return err
			}
			if limit <= 0 || limit > app.MaxDoctorHistory {
				return fmt.Errorf(
					"--limit must be within 1..%d",
					app.MaxDoctorHistory,
				)
			}
			return nil
		},
		RunE: func(command *cobra.Command, _ []string) error {
			document, err := app.DoctorHistory(
				command.Context(),
				*targetPath,
				limit,
				time.Time{},
			)
			if err != nil {
				return fmt.Errorf("doctor history: %w", err)
			}
			return renderDiagnosticOutput(
				command,
				document,
				reportcontract.Format(*formatValue),
				*outputPath,
			)
		},
	}
	command.Flags().IntVar(
		&limit,
		"limit",
		app.DefaultDoctorHistory,
		"maximum persisted reports to return (1..100)",
	)
	return command
}

func validateDiagnosticOutputFlags(
	formatValue, outputPath string,
	outputSet bool,
) error {
	switch reportcontract.Format(formatValue) {
	case reportcontract.FormatJSON, reportcontract.FormatMarkdown:
	default:
		return fmt.Errorf(
			"unsupported report format %q; use json or markdown",
			formatValue,
		)
	}
	if outputSet && outputPath == "" {
		return errors.New("--output requires a non-empty file path")
	}
	return nil
}

func renderDiagnosticOutput(
	command *cobra.Command,
	document reportcontract.Document,
	format reportcontract.Format,
	outputPath string,
) error {
	if err := command.Context().Err(); err != nil {
		return err
	}
	output, err := reportcontract.Render(document, format)
	if err != nil {
		return fmt.Errorf("render report: %w", err)
	}
	if err := command.Context().Err(); err != nil {
		return err
	}
	if outputPath == "" {
		if _, err := command.OutOrStdout().Write(output); err != nil {
			return fmt.Errorf("write report output: %w", err)
		}
		return nil
	}
	return writeExclusiveReport(outputPath, output)
}
