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
	Version       int          `json:"version"`
	OS            string       `json:"os"`
	Architecture  string       `json:"architecture"`
	BootID        string       `json:"bootId"`
	CLIVersion    string       `json:"cliVersion"`
	Commit        string       `json:"commit"`
	ArchiveSHA256 string       `json:"archiveSha256"`
	Ready         bool         `json:"ready"`
	Capabilities  []string     `json:"capabilities,omitempty"`
	Health        *GuestHealth `json:"health,omitempty"`
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
	installed, err := m.installedVersion(deadline)
	if err != nil || strings.TrimSpace(installed) != version.Current().String("agefreighter") {
		return Readiness{}, errors.New("installed CLI and tools versions do not match")
	}
	probe := m.health
	if m.healthProbe != nil {
		probe = m.healthProbe
	}
	health, err := probe(ctx)
	if err != nil {
		return Readiness{}, err
	}
	return Readiness{Version: 1, OS: runtime.GOOS, Architecture: runtime.GOARCH, BootID: boot, CLIVersion: version.Current().Version, Commit: version.Current().Commit, ArchiveSHA256: sha, Ready: true, Capabilities: []string{
		"csv-inventory-v1", "csv-migration-v1",
		"neo4j-inventory-v1", "neo4j-migration-v1",
		"postgresql-inventory-v1", "postgresql-migration-v1",
		"cosmos-nosql-inventory-v1", "cosmos-nosql-migration-v1",
	}, Health: health}, nil
}

func (m Manager) installedVersion(ctx context.Context) (string, error) {
	if m.versionProbe != nil {
		return m.versionProbe(ctx)
	}
	cmd := exec.CommandContext(ctx, m.CLI, "version")
	output := &boundedOutput{limit: 4096}
	cmd.Stdout = output
	if err := cmd.Run(); err != nil {
		return "", err
	}
	if output.overflow {
		return "", errors.New("installed CLI version output exceeds its bound")
	}
	return output.String(), nil
}

type GuestHealth struct {
	Idle               bool    `json:"idle"`
	StorageUsedPercent float64 `json:"storageUsedPercent"`
	SwapUsedBytes      uint64  `json:"swapUsedBytes"`
	OOMEvents          int     `json:"oomEvents"`
}
