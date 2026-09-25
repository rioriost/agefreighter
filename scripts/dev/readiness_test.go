package dev_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"go.yaml.in/yaml/v3"
)

func TestWorkflowPostgreSQLReadinessWaitsForTCP(t *testing.T) {
	const fakeDocker = `#!/bin/sh
case "$1" in
pull|rm|run|logs) exit 0 ;;
exec)
	shift 2
	test "$1" = pg_isready || exit 2
	printf '%s\n' "$*" >> "$FAKE_READY_LOG"
	if [ "$2" != -h ] || [ "$3" != 127.0.0.1 ]; then
		exit 0
	fi
	if [ ! -f "$FAKE_READY_STATE" ]; then
		touch "$FAKE_READY_STATE"
		exit 1
	fi
	test "$FAKE_READY_MODE" = ready
	;;
*) exit 2 ;;
esac
`
	matched := 0
	for _, name := range []string{"ci.yml", "release.yml", "azure-integration.yml"} {
		path := filepath.Join(scriptDirectory(t), "..", "..", ".github", "workflows", name)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var workflow struct {
			Jobs map[string]struct {
				Steps []struct {
					Name string
					Run  string
				}
			}
		}
		if err := yaml.Unmarshal(data, &workflow); err != nil {
			t.Fatal(err)
		}
		for job, definition := range workflow.Jobs {
			for _, step := range definition.Steps {
				if step.Name != "Start pinned PostgreSQL 19 beta target" {
					continue
				}
				matched++
				for _, mode := range []string{"ready", "unavailable"} {
					t.Run(name+"/"+job+"/"+mode, func(t *testing.T) {
						dir := t.TempDir()
						for name, body := range map[string]string{"docker": fakeDocker, "sleep": "#!/bin/sh\nexit 0\n"} {
							if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0700); err != nil {
								t.Fatal(err)
							}
						}
						log := filepath.Join(dir, "probes")
						command := exec.CommandContext(t.Context(), "bash", "-c", step.Run)
						command.Env = append(os.Environ(),
							"PATH="+dir+string(os.PathListSeparator)+os.Getenv("PATH"),
							"PGGRAPH_CONTAINER=agefreighter-readiness-fixture",
							"PGGRAPH_IMAGE=readiness-fixture",
							"PGGRAPH_PASSWORD=fixture-only",
							"FAKE_READY_LOG="+log, "FAKE_READY_STATE="+filepath.Join(dir, "state"),
							"FAKE_READY_MODE="+mode,
						)
						output, err := command.CombinedOutput()
						if (err == nil) != (mode == "ready") {
							t.Fatalf("readiness result = %v: %s", err, output)
						}
						probes, err := os.ReadFile(log)
						if err != nil {
							t.Fatal(err)
						}
						want := 2
						if mode == "unavailable" {
							want = 60
						}
						lines := strings.Split(strings.TrimSpace(string(probes)), "\n")
						if len(lines) != want {
							t.Fatalf("readiness used %d probes, want %d; temporary socket must not pass", len(lines), want)
						}
						for _, line := range lines {
							if line != "pg_isready -h 127.0.0.1 -U postgres -d agefreighter" {
								t.Fatalf("unexpected readiness transport: %q", line)
							}
						}
					})
				}
			}
		}
	}
	if matched != 4 {
		t.Fatalf("tested %d PostgreSQL workflow startup gates, want 4", matched)
	}
}

func TestLocalPostgreSQLReadinessUsesTCP(t *testing.T) {
	for name, want := range map[string]int{"dev.sh": 2, "pggraph-apple-container.sh": 1} {
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join(scriptDirectory(t), name))
			if err != nil {
				t.Fatal(err)
			}
			matched := 0
			for _, line := range strings.Split(strings.ReplaceAll(string(data), "\\\n", " "), "\n") {
				fields := strings.Fields(line)
				for index, field := range fields {
					if field != "pg_isready" {
						continue
					}
					matched++
					if len(fields) <= index+2 || fields[index+1] != "-h" || fields[index+2] != "127.0.0.1" {
						t.Fatalf("readiness can accept temporary Unix-socket bootstrap: %s", line)
					}
				}
			}
			if matched != want {
				t.Fatalf("tested %d local PostgreSQL readiness probes, want %d", matched, want)
			}
		})
	}
}
