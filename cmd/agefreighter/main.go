package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/rioriost/agefreighter/internal/cli"
	"github.com/rioriost/agefreighter/internal/observability"
)

func run(args []string, stdout, stderr io.Writer) int {
	return runContext(context.Background(), args, stdout, stderr)
}

func runContext(
	ctx context.Context,
	args []string,
	stdout, stderr io.Writer,
) int {
	runtime, err := observability.NewFromEnvironment(
		ctx,
		"agefreighter",
		stderr,
	)
	if err != nil {
		fmt.Fprintf(stderr, "initialize observability: %v\n", err)
		return 1
	}
	commandName := ""
	if len(args) > 0 {
		commandName = args[0]
	}
	err = runtime.Run(ctx, commandName, func(commandContext context.Context) error {
		command := cli.NewAgefreighter(stdout, stderr)
		return cli.ExecuteContext(commandContext, command, args)
	})
	shutdownContext, cancel := context.WithTimeout(
		context.WithoutCancel(ctx),
		5*time.Second,
	)
	defer cancel()
	if shutdownErr := runtime.Shutdown(shutdownContext); shutdownErr != nil {
		if !runtime.LogExportError(ctx, shutdownErr) {
			fmt.Fprintln(stderr, "telemetry export failed")
		}
	}
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}

func main() {
	ctx, stop := signal.NotifyContext(
		context.Background(),
		os.Interrupt,
		syscall.SIGTERM,
	)
	defer stop()
	os.Exit(runContext(ctx, os.Args[1:], os.Stdout, os.Stderr))
}
