package version

import "testing"

func TestCurrent(t *testing.T) {
	originalVersion, originalCommit, originalBuildDate := Version, Commit, BuildDate
	t.Cleanup(func() {
		Version, Commit, BuildDate = originalVersion, originalCommit, originalBuildDate
	})
	Version, Commit, BuildDate = "2.0.0", "abc123", "2026-08-24T12:00:00Z"

	got := Current()

	want := Info{
		Version:   "2.0.0",
		Commit:    "abc123",
		BuildDate: "2026-08-24T12:00:00Z",
	}
	if got != want {
		t.Fatalf("Current() = %#v, want %#v", got, want)
	}
}

func TestInfoString(t *testing.T) {
	info := Info{
		Version:   "2.0.0",
		Commit:    "abc123",
		BuildDate: "2026-08-24T12:00:00Z",
	}

	got := info.String("agefreighter")
	want := "agefreighter 2.0.0 (commit: abc123, built: 2026-08-24T12:00:00Z)"
	if got != want {
		t.Fatalf("Info.String() = %q, want %q", got, want)
	}
}
