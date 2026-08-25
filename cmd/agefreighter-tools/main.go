package main

import (
	"fmt"
	"io"
	"os"

	"github.com/rioriost/agefreighter/internal/cli"
	"github.com/rioriost/agefreighter/internal/tools"
)

func run(args []string, stdout, stderr io.Writer) int {
	command := cli.NewTools(stdout, stderr)
	command.AddCommand(tools.NewGenerateCommand())
	if err := cli.Execute(command, args); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}
