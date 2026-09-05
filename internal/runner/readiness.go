package runner

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/rioriost/agefreighter/internal/version"
)

type Readiness struct {
	Version       int    `json:"version"`
	OS            string `json:"os"`
	Architecture  string `json:"architecture"`
	BootID        string `json:"bootId"`
	CLIVersion    string `json:"cliVersion"`
	Commit        string `json:"commit"`
	ArchiveSHA256 string `json:"archiveSha256"`
	Ready         bool   `json:"ready"`
}

func (m Manager) Ready(ctx context.Context) (Readiness, error) {
	base := filepath.Dir(m.Root)
	marker, err := os.Stat(filepath.Join(base, "bootstrap.complete"))
	if err != nil || !marker.Mode().IsRegular() {
		return Readiness{}, errors.New("guest bootstrap is not complete")
	}
	digest, err := os.ReadFile(filepath.Join(base, "evidence", "archive.sha256"))
	if err != nil {
		return Readiness{}, errors.New("guest artifact checksum is unavailable")
	}
	sha := strings.TrimSpace(string(digest))
	if len(sha) != 64 || strings.Trim(sha, "0123456789abcdef") != "" {
		return Readiness{}, errors.New("guest artifact checksum is invalid")
	}
	boot, err := m.BootID()
	if err != nil {
		return Readiness{}, err
	}
	deadline, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(deadline, m.CLI, "version")
	output := &boundedOutput{limit: 4096}
	cmd.Stdout = output
	if err := cmd.Run(); err != nil || output.overflow || strings.TrimSpace(output.String()) != version.Current().String("agefreighter") {
		return Readiness{}, errors.New("installed CLI and tools versions do not match")
	}
	return Readiness{Version: 1, OS: runtime.GOOS, Architecture: runtime.GOARCH, BootID: boot, CLIVersion: version.Current().Version, Commit: version.Current().Commit, ArchiveSHA256: sha, Ready: true}, nil
}
