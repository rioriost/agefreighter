package cli

import (
	"encoding/json"
	"fmt"

	"github.com/rioriost/agefreighter/internal/config"
	"github.com/spf13/cobra"
)

type validationResult struct {
	SchemaVersion int              `json:"schemaVersion"`
	Valid         bool             `json:"valid"`
	APIVersion    string           `json:"apiVersion"`
	Kind          string           `json:"kind"`
	Job           string           `json:"job"`
	Source        validationSource `json:"source"`
	Target        validationTarget `json:"target"`
}

type validationSource struct {
	Type config.SourceType `json:"type"`
}

type validationTarget struct {
	Type config.TargetType `json:"type"`
	Mode config.LoadMode   `json:"mode"`
}

func newValidateCommand() *cobra.Command {
	var formatValue string
	command := &cobra.Command{
		Use:   "validate JOB",
		Short: "Validate a load job without connecting to its source or target",
		Args:  cobra.ExactArgs(1),
		PreRunE: func(_ *cobra.Command, _ []string) error {
			switch formatValue {
			case "text", "json":
				return nil
			default:
				return fmt.Errorf(
					"unsupported validation format %q; use text or json",
					formatValue,
				)
			}
		},
		RunE: func(command *cobra.Command, args []string) error {
			job, err := config.Load(args[0])
			if err != nil {
				return fmt.Errorf("validate job: %w", err)
			}
			if formatValue == "json" {
				return writeJSON(command, validationResult{
					SchemaVersion: 1,
					Valid:         true,
					APIVersion:    job.APIVersion,
					Kind:          job.Kind,
					Job:           job.Metadata.Name,
					Source: validationSource{
						Type: job.Source.Type,
					},
					Target: validationTarget{
						Type: job.Target.Type,
						Mode: job.Target.Mode,
					},
				})
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
	command.Flags().StringVar(
		&formatValue,
		"format",
		"text",
		"validation output format: text or json",
	)
	return command
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
