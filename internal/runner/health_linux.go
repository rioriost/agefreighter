package runner

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

func (m Manager) health(ctx context.Context) (*GuestHealth, error) {
	var disk syscall.Statfs_t
	if syscall.Statfs(m.Root, &disk) != nil || disk.Blocks == 0 {
		return nil, errors.New("guest storage evidence is unavailable")
	}
	mem, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return nil, errors.New("guest swap evidence is unavailable")
	}
	values := map[string]uint64{}
	for _, line := range strings.Split(string(mem), "\n") {
		p := strings.Fields(line)
		if len(p) == 3 && (p[0] == "SwapTotal:" || p[0] == "SwapFree:") && p[2] == "kB" {
			n, e := strconv.ParseUint(p[1], 10, 64)
			if e != nil {
				return nil, e
			}
			values[p[0]] = n
		}
	}
	total, ok := values["SwapTotal:"]
	free, ok2 := values["SwapFree:"]
	if !ok || !ok2 || free > total {
		return nil, errors.New("guest swap evidence is invalid")
	}
	active, err := filepath.Glob(filepath.Join(m.Root, "*", "active"))
	if err != nil {
		return nil, err
	}
	cmd := exec.CommandContext(ctx, "journalctl", "-k", "-b", "--no-pager", "--grep=Out of memory|Killed process", "-o", "cat")
	output := &boundedOutput{limit: 64 << 10}
	stderr := &boundedOutput{limit: 4096}
	cmd.Stdout = output
	cmd.Stderr = stderr
	runErr := cmd.Run()
	// journalctl --grep returns 1 (with empty stdout/stderr) when no records
	// match. Permission, missing-journal and other diagnostics are not zero OOMs.
	noMatches := false
	var exit *exec.ExitError
	if errors.As(runErr, &exit) && exit.ExitCode() == 1 && output.Len() == 0 && stderr.Len() == 0 {
		noMatches = true
	}
	if runErr != nil && !noMatches || output.overflow || stderr.overflow || stderr.Len() != 0 {
		return nil, errors.New("guest OOM evidence is unavailable")
	}
	ooms := 0
	for _, line := range strings.Split(output.String(), "\n") {
		if strings.Contains(line, "Out of memory") || strings.Contains(line, "Killed process") {
			ooms++
		}
	}
	return &GuestHealth{Idle: len(active) == 0, StorageUsedPercent: 100 * (1 - float64(disk.Bavail)/float64(disk.Blocks)), SwapUsedBytes: (total - free) * 1024, OOMEvents: ooms}, nil
}
