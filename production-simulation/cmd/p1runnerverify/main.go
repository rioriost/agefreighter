// p1runnerverify is an isolated qualification harness, not an AGEFreighter
// installation or a production migration command. It reads the committed graph
// and regenerates the frozen P1 fixture independently of the CSV importer.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"time"

	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
	"github.com/rioriost/agefreighter/production-simulation/internal/rangedigest"
)

const fixtureRoot = "f74220f6c58f0c1a62f80a567520ffcde43a2499ba48100667ee7b78ff4e2e2f"
const canonicalRoot = "bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70"

type qualificationFailure struct{ stage, code string }

func (f *qualificationFailure) Error() string { return "P1 qualification failed" }

// Only fixed stage/code identifiers leave the verifier. Never serialize driver
// errors, source properties, connection strings, command arguments or stdin.
func retainFailure(path string, cause error) error {
	stage, code := "setup", "invalid-input-or-output"
	var failure *qualificationFailure
	if errors.As(cause, &failure) {
		stage, code = failure.stage, failure.code
	}

	data, err := json.Marshal(struct {
		Version int    `json:"version"`
		Outcome string `json:"outcome"`
		Stage   string `json:"stage"`
		Code    string `json:"code"`
	}{1, "fail", stage, code})
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	_, err = f.Write(append(data, '\n'))
	return errors.Join(err, f.Close())
}

func readonlyDSN(raw, host string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil || !regexp.MustCompile(`^[a-z][a-z0-9-]+\.postgres\.database\.azure\.com$`).MatchString(host) || u.Scheme != "postgresql" || u.Hostname() != host || u.Port() != "5432" || u.User == nil || u.User.Username() != "afadmin" || u.Path != "/agefreighter" || u.Fragment != "" {
		return "", errors.New("expected reviewed private target")
	}
	p, ok := u.User.Password()
	q, err := url.ParseQuery(u.RawQuery)
	if !ok || len(p) < 24 || err != nil || len(q) != 1 || len(q["sslmode"]) != 1 || q.Get("sslmode") != "verify-full" {
		return "", errors.New("expected verified TLS credentials")
	}
	q.Set("default_transaction_read_only", "on")
	q.Set("statement_timeout", "1200000")
	u.RawQuery = q.Encode()
	return u.String(), nil
}

func run(ctx context.Context, args []string, input io.Reader) error {
	if len(args) != 2 || !regexp.MustCompile(`^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$`).MatchString(args[0]) {
		return errors.New("expected committed job ID and target hostname")
	}
	secret, err := io.ReadAll(io.LimitReader(input, 8193))
	if err != nil || len(secret) > 8192 {
		return errors.New("invalid private input")
	}
	dsn, err := readonlyDSN(string(secret), args[1])
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, 20*time.Minute)
	defer cancel()
	// Same seed/shards and byte root as the original Mac P1 fixture. Refuse
	// any generator drift before querying the target. No source data is edited.
	f, err := fixture.Generate(ctx, fixture.GenerateConfig{Phase: fixture.PhaseP1, Output: "fixture", Shards: 64, Workers: 2, Seed: 20260829})
	if err != nil || f.RootSHA256 != fixtureRoot {
		return &qualificationFailure{"fixture-generation", "fixture-generation-or-root"}
	}
	expected, err := rangedigest.FixtureManifest(ctx, filepath.Join("fixture", "manifest.json"), 100000)
	if err != nil || expected.RootSHA256 != canonicalRoot || expected.RecordCount != 5600000 || len(expected.Leaves) != 64 {
		return &qualificationFailure{"fixture-digest", "fixture-digest-or-coverage"}
	}
	actual, err := rangedigest.TargetManifest(ctx, dsn, filepath.Join("fixture", "manifest.json"), args[0], 100000)
	if err != nil {
		code := "target-read-or-canonicalization"
		if errors.Is(err, rangedigest.ErrSourceKeyOrder) {
			code = "source-key-order"
		}
		return &qualificationFailure{"target-digest", code}
	}
	comparison, compareErr := rangedigest.Compare(expected, actual)
	result := struct {
		Version     int                    `json:"version"`
		JobID       string                 `json:"jobId"`
		GeneratedAt string                 `json:"generatedAt"`
		ReadOnly    bool                   `json:"readOnly"`
		Expected    rangedigest.Manifest   `json:"expected"`
		Actual      rangedigest.Manifest   `json:"actual"`
		Comparison  rangedigest.Comparison `json:"comparison"`
	}{1, args[0], time.Now().UTC().Format(time.RFC3339Nano), true, expected, actual, comparison}
	file, err := os.OpenFile("result.json", os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	err = json.NewEncoder(file).Encode(result)
	closeErr := file.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	if compareErr != nil {
		return &qualificationFailure{"comparison", "canonical-mismatch"}
	}
	return nil
}

func main() {
	if err := run(context.Background(), os.Args[1:], os.Stdin); err != nil {
		// Keep prior evidence immutable. Receipt failure still exits nonzero;
		// failure.json is diagnostic evidence, never a passing result.json.
		_ = retainFailure("failure.json", err)
		// Never print driver diagnostics or credentials into ARM output.
		os.Stderr.WriteString("Independent P1 qualification failed; retained results must be reviewed.\n")
		os.Exit(1)
	}
}
