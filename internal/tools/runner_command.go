package tools

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/rioriost/agefreighter/internal/runner"
	"github.com/spf13/cobra"
)

func NewRunnerCommand() *cobra.Command {
	root := &cobra.Command{Use: "runner", Short: "Private Linux guest assessment protocol", Args: cobra.NoArgs}
	dispatch := &cobra.Command{Use: "dispatch", Short: "Receive one protected JSON control request on stdin", Args: cobra.NoArgs, RunE: func(command *cobra.Command, _ []string) error {
		request, err := runner.Decode(command.InOrStdin())
		if err != nil {
			return err
		}
		manager, err := runner.LinuxManager()
		if err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(command.Context(), 30*time.Second)
		defer cancel()
		var result any
		switch request.Action {
		case "ready":
			result, err = manager.Ready(ctx)
		case "profile", "inventory":
			result, err = manager.Submit(ctx, request)
		case "status":
			result, err = manager.Status(request.Workflow, request.Operation)
		case "report":
			result, err = manager.Report(request.Workflow, request.Operation, request.Offset)
		case "export-report":
			result, err = manager.ExportReport(ctx, request)
		case "import-csv":
			result, err = manager.SubmitCSV(ctx, request)
		default:
			return errors.New("unsupported guest operation")
		}
		if err != nil {
			return err
		}
		return json.NewEncoder(command.OutOrStdout()).Encode(result)
	}}
	var workflow, operation string
	worker := &cobra.Command{Use: "worker", Short: "Execute one claimed assessment (systemd only)", Hidden: true, Args: cobra.NoArgs, RunE: func(command *cobra.Command, _ []string) error {
		manager, err := runner.LinuxManager()
		if err != nil {
			return err
		}
		return manager.Work(command.Context(), workflow, operation)
	}}
	worker.Flags().StringVar(&workflow, "workflow", "", "retained workflow UUID")
	worker.Flags().StringVar(&operation, "operation", "", "retained operation UUID")
	_ = worker.MarkFlagRequired("workflow")
	_ = worker.MarkFlagRequired("operation")
	root.AddCommand(dispatch, worker)
	return root
}
