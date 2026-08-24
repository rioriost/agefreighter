package config

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"
)

type ByteSize int64

const (
	kilobyte ByteSize = 1_000
	megabyte          = 1_000 * kilobyte
	gigabyte          = 1_000 * megabyte
	terabyte          = 1_000 * gigabyte
	kibibyte          = 1_024
	mebibyte          = 1_024 * kibibyte
	gibibyte          = 1_024 * mebibyte
	tebibyte          = 1_024 * gibibyte
)

var byteUnits = []struct {
	suffix string
	size   ByteSize
}{
	{"TiB", tebibyte},
	{"GiB", gibibyte},
	{"MiB", mebibyte},
	{"KiB", kibibyte},
	{"TB", terabyte},
	{"GB", gigabyte},
	{"MB", megabyte},
	{"KB", kilobyte},
	{"B", 1},
}

func ParseByteSize(value string) (ByteSize, error) {
	for _, unit := range byteUnits {
		if !strings.HasSuffix(value, unit.suffix) {
			continue
		}
		number := strings.TrimSuffix(value, unit.suffix)
		if number == "" {
			break
		}
		parsed, err := strconv.ParseInt(number, 10, 64)
		if err != nil || parsed <= 0 {
			break
		}
		if parsed > int64(ByteSize(^uint64(0)>>1))/int64(unit.size) {
			break
		}
		return ByteSize(parsed) * unit.size, nil
	}
	return 0, fmt.Errorf("invalid byte size: use a positive integer with B, KB, MB, GB, TB, KiB, MiB, GiB, or TiB")
}

func (size *ByteSize) UnmarshalText(text []byte) error {
	parsed, err := ParseByteSize(string(text))
	if err != nil {
		return err
	}
	*size = parsed
	return nil
}

func (size ByteSize) MarshalText() ([]byte, error) {
	return []byte(size.String()), nil
}

func (size ByteSize) MarshalJSON() ([]byte, error) {
	return json.Marshal(size.String())
}

func (size ByteSize) String() string {
	for _, unit := range byteUnits {
		if size >= unit.size && size%unit.size == 0 {
			return fmt.Sprintf("%d%s", size/unit.size, unit.suffix)
		}
	}
	return fmt.Sprintf("%dB", size)
}

type Duration time.Duration

func ParseDuration(value string) (Duration, error) {
	parsed, err := time.ParseDuration(value)
	if err != nil || parsed <= 0 {
		return 0, fmt.Errorf("invalid duration: use a positive Go duration such as 30s or 5m")
	}
	return Duration(parsed), nil
}

func (duration *Duration) UnmarshalText(text []byte) error {
	parsed, err := ParseDuration(string(text))
	if err != nil {
		return err
	}
	*duration = parsed
	return nil
}

func (duration Duration) MarshalText() ([]byte, error) {
	return []byte(duration.String()), nil
}

func (duration Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(duration.String())
}

func (duration Duration) String() string {
	return time.Duration(duration).String()
}
