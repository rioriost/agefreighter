//go:build linux || darwin

package simulation_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"golang.org/x/sys/unix"
)

const (
	p1WorkerName       = "afhs-20261118-p1-csv"
	p1MetricsProvider  = "Microsoft.HorizonDB/clusters"
	p1StorageBudget    = 128 << 30
	p1MaximumMetricAge = 5 * time.Minute
)

var p1WriterMetrics = []string{
	"CpuUtilization", "StorageUsed", "PageReadIops", "PageReadThroughput",
	"WALWriteIops", "WALWriteThroughput", "CommitLatency", "WriteLatency",
}

type p1Metric struct {
	At      time.Time `json:"bucketUtc"`
	Maximum float64   `json:"maximum"`
}

func p1ReadWriterMetrics(r io.Reader, resource string, notBefore, now time.Time) (map[string]p1Metric, error) {
	var response struct {
		Interval string
		Value    []struct {
			ID         string
			Name       struct{ Value string }
			ErrorCode  string
			Timeseries []struct {
				Data []struct {
					TimeStamp time.Time
					Maximum   *float64
				}
			}
		}
	}
	d := json.NewDecoder(r)
	if err := d.Decode(&response); err != nil {
		return nil, errors.New("invalid HorizonDB metrics response")
	}
	if err := d.Decode(new(any)); !errors.Is(err, io.EOF) ||
		response.Interval != "PT1M" || len(response.Value) != len(p1WriterMetrics) {
		return nil, errors.New("require the complete one-minute HorizonDB metrics response")
	}
	current := make(map[string]p1Metric, len(p1WriterMetrics))
	names := make(map[string]bool, len(p1WriterMetrics))
	for _, name := range p1WriterMetrics {
		names[name] = true
	}
	for _, series := range response.Value {
		name := series.Name.Value
		if !names[name] || !strings.EqualFold(series.ID, resource+"/providers/Microsoft.Insights/metrics/"+name) ||
			(series.ErrorCode != "" && series.ErrorCode != "Success") || len(series.Timeseries) > 1 {
			return nil, errors.New("unexpected HorizonDB metric identity, dimensions or provider error")
		}
		delete(names, name)
		if len(series.Timeseries) == 0 {
			continue
		}
		for _, point := range series.Timeseries[0].Data {
			at := point.TimeStamp
			if point.Maximum == nil || at.Before(notBefore) || now.Sub(at) > p1MaximumMetricAge ||
				at.Add(time.Minute).After(now) {
				continue
			}
			if !at.Equal(at.Truncate(time.Minute)) || *point.Maximum < 0 ||
				(name == "CpuUtilization" && *point.Maximum > 100) {
				return nil, fmt.Errorf("invalid %s metric point", name)
			}
			previous := current[name]
			if previous.At.IsZero() || (name == "StorageUsed" && *point.Maximum > previous.Maximum) ||
				(at.After(previous.At) && (name != "StorageUsed" || *point.Maximum == previous.Maximum)) {
				current[name] = p1Metric{at, *point.Maximum}
			}
		}
	}
	for _, name := range []string{"CpuUtilization", "StorageUsed"} {
		if _, ok := current[name]; !ok {
			return nil, fmt.Errorf("no fresh %s sample for this writer", name)
		}
	}
	return current, nil
}

func (a p1Admission) writerResource() string {
	subscription := strings.SplitN(a.SnapshotID, "/resourceGroups/", 2)[0]
	return subscription + "/resourceGroups/rg-agefreighter-horizon-scale-20261118/providers/" + p1MetricsProvider + "/afhs-20261118-pg17"
}

func readP1Metrics(ctx context.Context, client *http.Client, identity *azidentity.ManagedIdentityCredential, a p1Admission) (map[string]p1Metric, error) {
	token, err := identity.GetToken(ctx, policy.TokenRequestOptions{Scopes: []string{"https://management.azure.com/.default"}})
	if err != nil {
		return nil, errors.New("worker system identity could not authenticate for HorizonDB metrics")
	}
	now := time.Now().UTC()
	end := now.Truncate(time.Minute)
	resource := a.writerResource()
	params := url.Values{
		"api-version": {"2023-10-01"}, "metricnamespace": {p1MetricsProvider},
		"metricnames": {strings.Join(p1WriterMetrics, ",")}, "aggregation": {"Maximum"},
		"interval": {"PT1M"}, "timespan": {end.Add(-p1MaximumMetricAge).Format(time.RFC3339) + "/" + end.Format(time.RFC3339)},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		"https://management.azure.com"+resource+"/providers/Microsoft.Insights/metrics?"+params.Encode(), nil)
	if err != nil {
		return nil, errors.New("invalid HorizonDB metric request")
	}
	req.Header.Set("Authorization", "Bearer "+token.Token)
	result, err := client.Do(req)
	if err != nil {
		return nil, errors.New("HorizonDB metrics request did not complete")
	}
	defer result.Body.Close()
	if result.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HorizonDB metrics unavailable (HTTP %d)", result.StatusCode)
	}
	return p1ReadWriterMetrics(io.LimitReader(result.Body, 1<<20), resource, a.WriterReadyAt.Truncate(time.Minute).Add(time.Minute), now)
}

type p1HostSample struct {
	FixtureDiskUsedPercent float64  `json:"fixtureDiskUsedPercent"`
	OutputDiskUsedPercent  float64  `json:"outputDiskUsedPercent"`
	CPUJiffies             []uint64 `json:"cpuJiffies"`
	OutputDiskCounters     []uint64 `json:"outputDiskCounters"`
}

func p1DiskUsage(path string, initial os.FileInfo) (float64, string, error) {
	current, err := os.Stat(path)
	if err != nil || !os.SameFile(initial, current) {
		return 0, "", errors.New("P1 worker lost its fixture or evidence directory")
	}
	var fs syscall.Statfs_t
	if err := syscall.Statfs(path, &fs); err != nil || fs.Blocks == 0 || fs.Bavail > fs.Blocks {
		return 0, "", errors.New("P1 worker volume utilization is unavailable")
	}
	stat, ok := current.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, "", errors.New("P1 worker volume identity is unavailable")
	}
	device := uint64(stat.Dev)
	return 100 * float64(fs.Blocks-fs.Bavail) / float64(fs.Blocks),
		fmt.Sprintf("/sys/dev/block/%d:%d/stat", unix.Major(device), unix.Minor(device)), nil
}

func p1Counters(fields []string, count int) ([]uint64, error) {
	if len(fields) < count {
		return nil, errors.New("missing Linux resource counters")
	}
	counters := make([]uint64, count)
	for i := range counters {
		v, err := strconv.ParseUint(fields[i], 10, 64)
		if err != nil {
			return nil, errors.New("malformed Linux resource counter")
		}
		counters[i] = v
	}
	return counters, nil
}

func readP1Host(a p1Admission, fixture, output os.FileInfo) (p1HostSample, error) {
	var sample p1HostSample
	swaps, err := os.ReadFile("/proc/swaps")
	if err != nil || !strings.HasPrefix(string(swaps), "Filename") || strings.Contains(strings.TrimSpace(string(swaps)), "\n") {
		return sample, errors.New("P1 worker has active swap or cannot verify swap status")
	}
	vm, err := os.ReadFile("/proc/vmstat")
	if err != nil || !strings.Contains("\n"+string(vm)+"\n", "\noom_kill 0\n") {
		return sample, errors.New("P1 worker reports OOM or cannot verify its OOM counter")
	}
	stat, err := os.ReadFile("/proc/stat")
	line, _, _ := strings.Cut(string(stat), "\n")
	fields := strings.Fields(line)
	if err != nil || len(fields) == 0 || fields[0] != "cpu" {
		return sample, errors.New("P1 worker CPU counters unavailable")
	}
	sample.CPUJiffies, err = p1Counters(fields[1:], 8)
	if err != nil {
		return sample, err
	}
	var fixtureDevice, outputDevice string
	sample.FixtureDiskUsedPercent, fixtureDevice, err = p1DiskUsage(a.FixtureDir, fixture)
	if err != nil {
		return sample, err
	}
	sample.OutputDiskUsedPercent, outputDevice, err = p1DiskUsage(a.Output, output)
	if err != nil {
		return sample, err
	}
	if fixtureDevice != outputDevice {
		return sample, errors.New("P1 fixtures and evidence must use the planned durable worker disk")
	}
	raw, err := os.ReadFile(outputDevice)
	if err != nil {
		return sample, errors.New("durable P1 worker disk I/O counters unavailable")
	}
	sample.OutputDiskCounters, err = p1Counters(strings.Fields(string(raw)), 11)
	return sample, err
}

func checkP1Storage(host p1HostSample, metrics map[string]p1Metric) error {
	if host.FixtureDiskUsedPercent >= 80 || host.OutputDiskUsedPercent >= 80 {
		return errors.New("P1 worker storage reached 80%")
	}
	storage, ok := metrics["StorageUsed"]
	if !ok {
		return errors.New("HorizonDB storage usage is unavailable")
	}
	if storage.Maximum >= p1StorageBudget*4.0/5 {
		return errors.New("HorizonDB storage reached 80% of the P1 planning allowance")
	}
	return nil
}

func recordP1TelemetryStop(log *json.Encoder, err error) error {
	return errors.Join(err, log.Encode(map[string]any{"stoppedAtUtc": time.Now().UTC(), "reason": err.Error()}))
}

func watchP1Telemetry(ctx context.Context, cancelPilot context.CancelFunc, ticks <-chan time.Time, collect func() error, log *json.Encoder) error {
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticks:
			if err := collect(); err != nil {
				if ctx.Err() != nil {
					return nil
				}
				cancelPilot()
				return recordP1TelemetryStop(log, err)
			}
		}
	}
}

func startP1Telemetry(t *testing.T, ctx context.Context, cancelPilot context.CancelFunc, a p1Admission) func(time.Time) {
	t.Helper()
	host, err := os.Hostname()
	if err != nil || host != p1WorkerName {
		t.Fatal("require the dedicated isolated P1 worker VM")
	}
	fixture, fixtureErr := os.Stat(a.FixtureDir)
	output, outputErr := os.Stat(a.Output)
	if fixtureErr != nil || outputErr != nil || !fixture.IsDir() || !output.IsDir() {
		t.Fatal("P1 telemetry requires the durable fixture and evidence directories")
	}
	identity, err := azidentity.NewManagedIdentityCredential(nil)
	if err != nil {
		t.Fatal("P1 telemetry could not initialize the worker system identity")
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	client := &http.Client{
		Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	log, err := os.OpenFile(filepath.Join(a.Output, "telemetry.jsonl"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	monitor, stop := context.WithCancel(ctx)
	encoder := json.NewEncoder(log)
	latest := make(map[string]time.Time)
	collect := func() error {
		host, err := readP1Host(a, fixture, output)
		if err != nil {
			return err
		}
		probe, cancel := context.WithTimeout(monitor, 20*time.Second)
		defer cancel()
		metrics, err := readP1Metrics(probe, client, identity, a)
		if err != nil {
			return err
		}
		if err := encoder.Encode(map[string]any{"observedAtUtc": time.Now().UTC(), "worker": host, "writer": metrics}); err != nil {
			return err
		}
		for name, point := range metrics {
			if point.At.After(latest[name]) {
				latest[name] = point.At
			}
		}
		return checkP1Storage(host, metrics)
	}
	if err := collect(); err != nil {
		stop()
		client.CloseIdleConnections()
		t.Fatal(errors.Join(recordP1TelemetryStop(encoder, err), log.Close()))
	}
	done := make(chan error, 1)
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		done <- errors.Join(watchP1Telemetry(monitor, cancelPilot, ticker.C, collect, encoder), log.Close())
	}()
	return func(loadedSince time.Time) {
		stop()
		defer client.CloseIdleConnections()
		if err := <-done; err != nil {
			t.Error(err)
		}
		if !loadedSince.IsZero() {
			for _, name := range p1WriterMetrics[2:] {
				if latest[name].Before(loadedSince) {
					t.Errorf("no %s metric from the P1 load interval", name)
				}
			}
		}
	}
}

func TestHorizonDBP1WriterMetricScopeAndFreshness(t *testing.T) {
	now := time.Date(2026, 11, 20, 7, 35, 40, 0, time.UTC)
	notBefore := now.Truncate(time.Minute).Add(-3 * time.Minute)
	resource := p1Admission{SnapshotID: "/subscriptions/00000000-0000-0000-0000-000000000001/resourceGroups/retained"}.writerResource()
	for _, tt := range []struct {
		name    string
		change  func(values []map[string]any)
		wantErr string
	}{
		{name: "null idle WAL is not zero"},
		{name: "old lease", change: func(v []map[string]any) {
			v[0]["timeseries"] = []any{map[string]any{"data": []any{map[string]any{"timeStamp": notBefore.Add(-time.Minute), "maximum": 0}}}}
		}, wantErr: "no fresh CpuUtilization"},
		{name: "expired storage", change: func(v []map[string]any) {
			v[1]["timeseries"] = []any{map[string]any{"data": []any{map[string]any{"timeStamp": now.Truncate(time.Minute).Add(-5 * time.Minute), "maximum": 0}}}}
		}, wantErr: "no fresh StorageUsed"},
		{name: "different cluster", change: func(v []map[string]any) {
			v[1]["id"] = strings.ReplaceAll(v[1]["id"].(string), "-pg17", "-pg18")
		}, wantErr: "metric identity"},
		{name: "provider partial error", change: func(v []map[string]any) {
			v[4]["errorCode"] = "Forbidden"
		}, wantErr: "provider error"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			points := []any{map[string]any{"timeStamp": notBefore, "maximum": 0}}
			values := make([]map[string]any, 0, len(p1WriterMetrics))
			for _, name := range p1WriterMetrics {
				data := points
				switch name {
				case "WALWriteIops":
					data = []any{map[string]any{"timeStamp": notBefore}}
				case "StorageUsed":
					data = []any{
						map[string]any{"timeStamp": notBefore, "maximum": p1StorageBudget},
						map[string]any{"timeStamp": notBefore.Add(time.Minute), "maximum": 1},
					}
				}
				values = append(values, map[string]any{
					"id":   resource + "/providers/Microsoft.Insights/metrics/" + name,
					"name": map[string]string{"value": name}, "timeseries": []any{map[string]any{"data": data}},
				})
			}
			if tt.change != nil {
				tt.change(values)
			}
			var body bytes.Buffer
			if err := json.NewEncoder(&body).Encode(map[string]any{"interval": "PT1M", "value": values}); err != nil {
				t.Fatal(err)
			}
			metrics, err := p1ReadWriterMetrics(&body, resource, notBefore, now)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("want %q, got %v", tt.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := metrics["WALWriteIops"]; ok {
				t.Fatal("an idle null bucket was mistaken for a measured zero")
			}
			if err := checkP1Storage(p1HostSample{}, metrics); err == nil {
				t.Fatal("a fresh storage breach was hidden by a later lower reading")
			}
		})
	}
}

func TestHorizonDBP1StorageStop(t *testing.T) {
	belowLimit := map[string]p1Metric{"StorageUsed": {Maximum: p1StorageBudget*4.0/5 - 1}}
	if err := checkP1Storage(p1HostSample{FixtureDiskUsedPercent: 79.9, OutputDiskUsedPercent: 79.9}, belowLimit); err != nil {
		t.Fatal(err)
	}
	if err := checkP1Storage(p1HostSample{OutputDiskUsedPercent: 80}, belowLimit); err == nil {
		t.Fatal("worker at 80% accepted")
	}
	if err := checkP1Storage(p1HostSample{}, nil); err == nil {
		t.Fatal("missing HorizonDB storage metric accepted")
	}
}

func TestHorizonDBP1TelemetryCancelsActiveCases(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	caseCtx, stopCase := context.WithCancel(ctx)
	defer stopCase()
	ticks := make(chan time.Time, 2)
	ticks <- time.Now()
	ticks <- time.Now()
	var log bytes.Buffer
	samples := 0
	if err := watchP1Telemetry(ctx, cancel, ticks, func() error {
		samples++
		if samples == 1 {
			return nil
		}
		return checkP1Storage(p1HostSample{OutputDiskUsedPercent: 80}, nil)
	}, json.NewEncoder(&log)); err == nil || !errors.Is(caseCtx.Err(), context.Canceled) {
		t.Fatalf("storage breach did not stop the active case: %v / %v", err, caseCtx.Err())
	}
	var result struct {
		StoppedAtUTC time.Time
		Reason       string
	}
	if err := json.Unmarshal(log.Bytes(), &result); err != nil || result.StoppedAtUTC.IsZero() ||
		result.Reason != "P1 worker storage reached 80%" {
		t.Fatalf("missing terminal telemetry evidence: %q, %v", &log, err)
	}
}
