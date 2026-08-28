package cli

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/rioriost/agefreighter/internal/app"
	reportcontract "github.com/rioriost/agefreighter/internal/report"
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
	var (
		targetPath  string
		level       string
		counts      bool
		integrity   bool
		limit       int
		formatValue string
		outputPath  string
	)
	command := &cobra.Command{
		Use:   "verify JOB_ID",
		Short: "Verify a completed load job and optionally run deep checks",
		Args:  cobra.ExactArgs(1),
		PreRunE: func(command *cobra.Command, _ []string) error {
			switch level {
			case "catalog":
			case "counts":
				counts = true
			default:
				return fmt.Errorf("unsupported verification level %q; use catalog or counts", level)
			}
			if limit < 1 || limit > app.MaxIntegrityLimit {
				return fmt.Errorf(
					"--limit must be within 1..%d", app.MaxIntegrityLimit,
				)
			}
			switch reportcontract.Format(formatValue) {
			case reportcontract.FormatJSON, reportcontract.FormatMarkdown:
			default:
				return fmt.Errorf(
					"unsupported report format %q; use json or markdown", formatValue,
				)
			}
			if command.Flags().Changed("output") && outputPath == "" {
				return errors.New("--output requires a non-empty file path")
			}
			if !counts && !integrity &&
				(command.Flags().Changed("format") ||
					command.Flags().Changed("output") ||
					command.Flags().Changed("limit")) {
				return errors.New("--format, --output, and --limit require --counts or --integrity")
			}
			return nil
		},
		RunE: func(command *cobra.Command, args []string) error {
			if counts || integrity {
				document, err := app.VerificationReport(
					command.Context(), targetPath, args[0],
					app.VerifyOptions{
						Counts: counts, Integrity: integrity, Limit: limit,
					},
				)
				if err != nil {
					return fmt.Errorf("verify: %w", err)
				}
				output, err := reportcontract.Render(
					document, reportcontract.Format(formatValue),
				)
				if err != nil {
					return fmt.Errorf("render verification report: %w", err)
				}
				if outputPath == "" {
					if _, err := command.OutOrStdout().Write(output); err != nil {
						return fmt.Errorf("write verification report: %w", err)
					}
				} else if err := writeExclusiveReport(outputPath, output); err != nil {
					return err
				}
				if document.Outcome == reportcontract.OutcomeFail {
					return errors.New("deep verification failed")
				}
				return nil
			}
			result, err := app.Verify(command.Context(), targetPath, args[0])
			if err != nil {
				return fmt.Errorf("verify: %w", err)
			}
			return writeJSON(command, result)
		},
	}
	command.Flags().StringVar(&targetPath, "target", "", "load job containing target configuration")
	command.Flags().StringVar(
		&level, "level", "catalog", "verification level: catalog or counts",
	)
	command.Flags().BoolVar(
		&counts, "counts", false,
		"compare persisted per-label counters with exact live identity counts",
	)
	command.Flags().BoolVar(
		&integrity, "integrity", false,
		"run deterministic bounded identity and endpoint consistency checks",
	)
	command.Flags().IntVar(
		&limit, "limit", app.DefaultIntegrityLimit,
		"maximum identity and physical rows checked per label",
	)
	command.Flags().StringVar(
		&formatValue, "format", string(reportcontract.FormatJSON),
		"deep-verification report format: json or markdown",
	)
	command.Flags().StringVar(
		&outputPath, "output", "",
		"write deep-verification report to a new permission-restricted file",
	)
	_ = command.MarkFlagRequired("target")
	return command
}

func newReportCommand() *cobra.Command {
	var (
		targetPath    string
		formatValue   string
		outputPath    string
		includeCounts bool
		limitBatches  int
	)
	command := &cobra.Command{
		Use:   "report JOB_ID",
		Short: "Generate a bounded migration report from durable metadata",
		Args:  cobra.ExactArgs(1),
		PreRunE: func(command *cobra.Command, _ []string) error {
			switch reportcontract.Format(formatValue) {
			case reportcontract.FormatJSON, reportcontract.FormatMarkdown:
			default:
				return fmt.Errorf(
					"unsupported report format %q; use json or markdown",
					formatValue,
				)
			}
			if command.Flags().Changed("limit-batches") &&
				(limitBatches <= 0 || limitBatches > app.MaxReportBatches) {
				return fmt.Errorf(
					"--limit-batches must be within 1..%d",
					app.MaxReportBatches,
				)
			}
			if command.Flags().Changed("output") && outputPath == "" {
				return errors.New("--output requires a non-empty file path")
			}
			return nil
		},
		RunE: func(command *cobra.Command, args []string) error {
			document, err := app.MigrationReport(
				command.Context(),
				targetPath,
				args[0],
				app.ReportOptions{
					IncludeCounts: includeCounts,
					LimitBatches:  limitBatches,
				},
			)
			if err != nil {
				return fmt.Errorf("report: %w", err)
			}
			output, err := reportcontract.Render(
				document,
				reportcontract.Format(formatValue),
			)
			if err != nil {
				return fmt.Errorf("render report: %w", err)
			}
			if outputPath == "" {
				if _, err := command.OutOrStdout().Write(output); err != nil {
					return fmt.Errorf("write report output: %w", err)
				}
				return nil
			}
			if err := writeExclusiveReport(outputPath, output); err != nil {
				return err
			}
			return nil
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
		&includeCounts,
		"include-counts",
		false,
		"run exact per-label identity counts with operation deadlines",
	)
	command.Flags().IntVar(
		&limitBatches,
		"limit-batches",
		0,
		"include at most N recent batch attempts",
	)
	_ = command.MarkFlagRequired("target")
	return command
}

func writeExclusiveReport(path string, data []byte) (resultErr error) {
	if path == "" {
		return errors.New("report output path is required")
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("create report output %q: %w", path, err)
	}
	closed := false
	defer func() {
		if !closed {
			resultErr = errors.Join(resultErr, file.Close())
		}
		if resultErr != nil {
			removeErr := os.Remove(path)
			if removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
				resultErr = errors.Join(
					resultErr,
					fmt.Errorf("remove incomplete report output %q: %w", path, removeErr),
				)
			}
		}
	}()
	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("inspect report output %q: %w", path, err)
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("report output %q is not a regular file", path)
	}
	if err := file.Chmod(0o600); err != nil {
		return fmt.Errorf("restrict report output %q: %w", path, err)
	}
	if _, err := io.Copy(file, bytes.NewReader(data)); err != nil {
		return fmt.Errorf("write report output %q: %w", path, err)
	}
	if err := file.Sync(); err != nil {
		return fmt.Errorf("sync report output %q: %w", path, err)
	}
	if err := file.Close(); err != nil {
		closed = true
		return fmt.Errorf("close report output %q: %w", path, err)
	}
	closed = true
	return nil
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
