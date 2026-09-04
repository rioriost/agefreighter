package pggraph

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"unicode"

	"github.com/jackc/pgx/v5"
)

const maxIdentifierBytes = 63

// QuoteIdentifier returns a PostgreSQL identifier that is safe to interpolate
// into generated DDL. Values must never be interpolated as SQL literals.
func QuoteIdentifier(identifier string) string {
	return pgx.Identifier{identifier}.Sanitize()
}

// PhysicalName produces a stable, collision-resistant ASCII identifier for a
// logical graph label. A hash is always present because slug normalization is
// intentionally lossy.
func PhysicalName(prefix, label string) string {
	digest := sha256.Sum256([]byte(label))
	suffix := hex.EncodeToString(digest[:6])
	slug := asciiSlug(label)
	if slug == "" {
		slug = "label"
	}
	available := maxIdentifierBytes - len(prefix) - len(suffix) - 1
	if available < 1 {
		available = 1
	}
	if len(slug) > available {
		slug = strings.TrimRight(slug[:available], "_")
		if slug == "" {
			slug = "x"
		}
	}
	return prefix + slug + "_" + suffix
}

func asciiSlug(value string) string {
	var result strings.Builder
	previousUnderscore := false
	for _, character := range strings.ToLower(value) {
		if character <= unicode.MaxASCII &&
			((character >= 'a' && character <= 'z') ||
				(character >= '0' && character <= '9')) {
			result.WriteRune(character)
			previousUnderscore = false
			continue
		}
		if result.Len() > 0 && !previousUnderscore {
			result.WriteByte('_')
			previousUnderscore = true
		}
	}
	return strings.Trim(result.String(), "_")
}
