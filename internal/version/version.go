package version

import "fmt"

var (
	Version   = "dev"
	Commit    = "none"
	BuildDate = "unknown"
)

type Info struct {
	Version   string
	Commit    string
	BuildDate string
}

func Current() Info {
	return Info{
		Version:   Version,
		Commit:    Commit,
		BuildDate: BuildDate,
	}
}

func (info Info) String(program string) string {
	return fmt.Sprintf(
		"%s %s (commit: %s, built: %s)",
		program,
		info.Version,
		info.Commit,
		info.BuildDate,
	)
}
