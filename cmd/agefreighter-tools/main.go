package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/rioriost/agefreighter/internal/cli"
	"github.com/rioriost/agefreighter/internal/observability"
	"github.com/rioriost/agefreighter/internal/tools"
)

func run(args []string, stdout, stderr io.Writer) int {
	return runContext(context.Background(), args, stdout, stderr)
}

func runContext(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	runtime, err := observability.NewFromEnvironment(
		ctx,
		"agefreighter-tools",
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
		command := cli.NewTools(stdout, stderr)
		command.AddCommand(
			tools.NewGenerateCommand(),
			tools.NewBenchmarkCommand(),
			tools.NewBenchmarkReportCommand(),
			tools.NewInspectCommand(),
			tools.NewConvertGremlinCommand(),
		)
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

func signalContext(parent context.Context) (context.Context, context.CancelFunc) {
	return signalContextWithExit(parent, os.Exit)
}

func signalContextWithExit(
	parent context.Context,
	forceExit func(int),
) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(parent)
	signals := []os.Signal{
		os.Interrupt,
		syscall.SIGHUP,
		syscall.SIGTERM,
	}
	notifications := make(chan os.Signal, 2)
	signal.Notify(notifications, signals...)
	done := make(chan struct{})
	var once sync.Once
	stop := func() {
		once.Do(func() {
			signal.Stop(notifications)
			cancel()
			close(done)
		})
	}
	go func() {
		select {
		case <-notifications:
			cancel()
			select {
			case next := <-notifications:
				forceExit(signalExitCode(next))
			case <-done:
			}
		case <-parent.Done():
			stop()
		case <-done:
		}
	}()
	return ctx, stop
}

func signalExitCode(received os.Signal) int {
	switch received {
	case os.Interrupt:
		return 130
	case syscall.SIGHUP:
		return 129
	case syscall.SIGTERM:
		return 143
	default:
		return 1
	}
}

func execute(args []string, stdout, stderr io.Writer) int {
	ctx, stop := signalContext(context.Background())
	defer stop()
	return runContext(ctx, args, stdout, stderr)
}

func main() {
	os.Exit(execute(os.Args[1:], os.Stdout, os.Stderr))
}
