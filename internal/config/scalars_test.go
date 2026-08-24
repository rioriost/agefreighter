package config

import (
	"encoding/json"
	"testing"
	"time"
)

func TestParseByteSize(t *testing.T) {
	tests := []struct {
		input string
		want  ByteSize
	}{
		{input: "1B", want: 1},
		{input: "2KB", want: 2_000},
		{input: "3MiB", want: 3 * mebibyte},
		{input: "4GiB", want: 4 * gibibyte},
	}
	for _, test := range tests {
		t.Run(test.input, func(t *testing.T) {
			got, err := ParseByteSize(test.input)
			if err != nil {
				t.Fatalf("ParseByteSize() error = %v", err)
			}
			if got != test.want {
				t.Fatalf("ParseByteSize() = %d, want %d", got, test.want)
			}
		})
	}
}

func TestParseByteSizeRejectsInvalidValues(t *testing.T) {
	for _, input := range []string{"", "0B", "-1MiB", "1.5GiB", "10", "1XB", "999999999999999999999TiB"} {
		t.Run(input, func(t *testing.T) {
			if _, err := ParseByteSize(input); err == nil {
				t.Fatalf("ParseByteSize(%q) error = nil, want error", input)
			}
		})
	}
}

func TestByteSizeTextAndJSON(t *testing.T) {
	var size ByteSize
	if err := size.UnmarshalText([]byte("16MiB")); err != nil {
		t.Fatalf("UnmarshalText() error = %v", err)
	}
	if got := size.String(); got != "16MiB" {
		t.Fatalf("String() = %q, want 16MiB", got)
	}
	text, err := size.MarshalText()
	if err != nil {
		t.Fatalf("MarshalText() error = %v", err)
	}
	if string(text) != "16MiB" {
		t.Fatalf("MarshalText() = %q, want 16MiB", text)
	}
	encoded, err := json.Marshal(size)
	if err != nil {
		t.Fatalf("MarshalJSON() error = %v", err)
	}
	if string(encoded) != `"16MiB"` {
		t.Fatalf("MarshalJSON() = %s, want quoted size", encoded)
	}
	if err := size.UnmarshalText([]byte("invalid")); err == nil {
		t.Fatal("UnmarshalText() error = nil, want error")
	}
}

func TestByteSizeStringUsesBytesWhenNotDivisible(t *testing.T) {
	if got := ByteSize(1025).String(); got != "1025B" {
		t.Fatalf("String() = %q, want 1025B", got)
	}
}

func TestDurationTextAndJSON(t *testing.T) {
	parsed, err := ParseDuration("1m30s")
	if err != nil {
		t.Fatalf("ParseDuration() error = %v", err)
	}
	if parsed != Duration(90*time.Second) {
		t.Fatalf("ParseDuration() = %s, want 1m30s", parsed)
	}

	var duration Duration
	if err := duration.UnmarshalText([]byte("250ms")); err != nil {
		t.Fatalf("UnmarshalText() error = %v", err)
	}
	text, err := duration.MarshalText()
	if err != nil {
		t.Fatalf("MarshalText() error = %v", err)
	}
	if string(text) != "250ms" {
		t.Fatalf("MarshalText() = %q, want 250ms", text)
	}
	encoded, err := json.Marshal(duration)
	if err != nil {
		t.Fatalf("MarshalJSON() error = %v", err)
	}
	if string(encoded) != `"250ms"` {
		t.Fatalf("MarshalJSON() = %s, want quoted duration", encoded)
	}
}

func TestParseDurationRejectsInvalidValues(t *testing.T) {
	for _, input := range []string{"", "0s", "-1s", "tomorrow"} {
		if _, err := ParseDuration(input); err == nil {
			t.Fatalf("ParseDuration(%q) error = nil, want error", input)
		}
	}
	var duration Duration
	if err := duration.UnmarshalText([]byte("invalid")); err == nil {
		t.Fatal("UnmarshalText() error = nil, want error")
	}
}
