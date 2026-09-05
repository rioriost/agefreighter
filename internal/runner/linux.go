package runner

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

func LinuxManager() (Manager, error) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" || os.Geteuid() != 0 {
		return Manager{}, errors.New("runner control requires the privileged Linux x64 guest")
	}
	root := "/var/lib/agefreighter/workflows"
	if err := privateDirectory("/var/lib/agefreighter"); err != nil {
		return Manager{}, err
	}
	if err := os.MkdirAll(root, 0700); err != nil {
		return Manager{}, err
	}
	if err := privateDirectory(root); err != nil {
		return Manager{}, err
	}
	return Manager{Root: root, UnitDirectory: "/etc/systemd/system", CLI: "/usr/local/bin/agefreighter", Tools: "/usr/local/bin/agefreighter-tools",
		BootID: func() (string, error) {
			data, err := os.ReadFile("/proc/sys/kernel/random/boot_id")
			if err != nil {
				return "", err
			}
			id := strings.TrimSpace(string(data))
			if !uuid.MatchString(id) {
				return "", errors.New("invalid boot ID")
			}
			return id, nil
		},
		Start: func(ctx context.Context, name string) error {
			if err := exec.CommandContext(ctx, "/bin/systemctl", "daemon-reload").Run(); err != nil {
				return err
			}
			return exec.CommandContext(ctx, "/bin/systemctl", "start", "--no-block", name).Run()
		},
	}, nil
}
