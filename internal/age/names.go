package age

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"regexp"
)

const (
	MinGraphNameBytes = 3
	MaxNameBytes      = 63
)

var (
	graphNamePattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.-]*[A-Za-z0-9_]$`)
	labelNamePattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	ErrInvalidName   = errors.New("invalid Apache AGE name")
)

func ValidateGraphName(name string) error {
	if len(name) < MinGraphNameBytes ||
		len(name) > MaxNameBytes ||
		!graphNamePattern.MatchString(name) {
		return fmt.Errorf(
			"%w: graph name must be %d..%d bytes and match %s",
			ErrInvalidName,
			MinGraphNameBytes,
			MaxNameBytes,
			graphNamePattern,
		)
	}
	return nil
}

func ValidateLabelName(name string) error {
	if len(name) == 0 ||
		len(name) > MaxNameBytes ||
		!labelNamePattern.MatchString(name) {
		return fmt.Errorf(
			"%w: label name must be 1..%d bytes and match %s",
			ErrInvalidName,
			MaxNameBytes,
			labelNamePattern,
		)
	}
	return nil
}

type DerivedNameKind string

const (
	ShadowName DerivedNameKind = "shadow"
	BackupName DerivedNameKind = "backup"
)

func DeriveGraphName(base string, kind DerivedNameKind, token string) (string, error) {
	if err := ValidateGraphName(base); err != nil {
		return "", err
	}
	if kind != ShadowName && kind != BackupName {
		return "", fmt.Errorf("%w: unsupported derived name kind %q", ErrInvalidName, kind)
	}
	sum := sha256.Sum256([]byte(base + "\x00" + string(kind) + "\x00" + token))
	suffix := "_" + string(kind) + "_" + hex.EncodeToString(sum[:6])
	prefixBytes := MaxNameBytes - len(suffix)
	if len(base) > prefixBytes {
		base = base[:prefixBytes]
	}
	derived := base + suffix
	if err := ValidateGraphName(derived); err != nil {
		return "", err
	}
	return derived, nil
}
