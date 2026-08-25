package csv

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
)

func bindManifest(
	ctx context.Context,
	mappings []fileMapping,
) (string, error) {
	digest := sha256.New()
	_, _ = fmt.Fprintf(digest, "csv-manifest-v1:%d:", len(mappings))
	for index := range mappings {
		mapping := &mappings[index]
		value, err := fingerprint(
			ctx,
			mapping.path,
			mapping.fingerprintInput,
		)
		if err != nil {
			return "", err
		}
		mapping.fingerprint = value
		_, _ = fmt.Fprintf(digest, "%d:%s:", index, value)
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func fingerprint(ctx context.Context, path string, semantic []byte) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open CSV fingerprint source %q: %w", path, err)
	}
	defer file.Close()
	return fingerprintFile(ctx, file, path, semantic)
}

func fingerprintFile(
	ctx context.Context,
	file *os.File,
	path string,
	semantic []byte,
) (string, error) {
	before, err := file.Stat()
	if err != nil {
		return "", fmt.Errorf("stat CSV fingerprint source %q: %w", path, err)
	}
	if !before.Mode().IsRegular() {
		return "", fmt.Errorf("CSV source %q is not a regular file", path)
	}
	digest := sha256.New()
	_, _ = fmt.Fprintf(digest, "%d:%d:", before.Size(), before.ModTime().UnixNano())
	buffer := make([]byte, 64<<10)
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		count, readErr := file.Read(buffer)
		if count > 0 {
			_, _ = digest.Write(buffer[:count])
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			return "", fmt.Errorf("hash CSV source %q: %w", path, readErr)
		}
	}
	if _, err := digest.Write(semantic); err != nil {
		return "", fmt.Errorf("hash CSV mapping %q: %w", path, err)
	}
	after, err := file.Stat()
	if err != nil {
		return "", fmt.Errorf("restat CSV fingerprint source %q: %w", path, err)
	}
	if before.Size() != after.Size() ||
		before.ModTime().UnixNano() != after.ModTime().UnixNano() {
		return "", fmt.Errorf("CSV source %q changed while fingerprinting", path)
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}
