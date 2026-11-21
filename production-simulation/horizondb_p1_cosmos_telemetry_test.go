//go:build linux || darwin

package simulation_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
)

var (
	errP1CosmosMetricTimeout   = errors.New("Cosmos metrics transport timed out")
	errP1CosmosMetricForbidden = errors.New("Cosmos metrics permission not yet visible")
)

func readP1CosmosMetric(ctx context.Context, client *http.Client, token, resource, metric, aggregation, filter string) (map[time.Time]float64, error) {
	end := time.Now().UTC().Truncate(time.Minute)
	query := url.Values{
		"api-version": {"2023-10-01"}, "metricnamespace": {"Microsoft.DocumentDB/databaseAccounts"},
		"metricnames": {metric}, "aggregation": {aggregation}, "interval": {"PT1M"},
		"timespan": {end.Add(-12*time.Minute).Format(time.RFC3339) + "/" + end.Format(time.RFC3339)},
	}
	if filter != "" {
		query.Set("$filter", filter)
	}
	ctx, stop := context.WithTimeout(ctx, p1MetricAttempt)
	defer stop()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		"https://management.azure.com"+resource+"/providers/Microsoft.Insights/metrics?"+query.Encode(), nil)
	if err != nil {
		return nil, errors.New("invalid Cosmos metrics request")
	}
	req.Header.Set("Authorization", "Bearer "+token)
	res, err := client.Do(req)
	if err != nil {
		var timeout net.Error
		if errors.As(err, &timeout) && timeout.Timeout() {
			return nil, errP1CosmosMetricTimeout
		}
		return nil, fmt.Errorf("Cosmos %s metric request did not complete", metric)
	}
	defer res.Body.Close()
	if res.StatusCode == http.StatusForbidden {
		return nil, errP1CosmosMetricForbidden
	}
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Cosmos %s metric unavailable (HTTP %d)", metric, res.StatusCode)
	}
	var response struct {
		Interval string
		Value    []struct {
			ID        string
			ErrorCode string
			Series    []struct {
				Data []struct {
					TimeStamp time.Time
					Count     *float64
					Total     *float64
				}
			} `json:"timeseries"`
		}
	}
	d := json.NewDecoder(io.LimitReader(res.Body, 1<<20))
	if err := d.Decode(&response); err != nil || !errors.Is(d.Decode(new(any)), io.EOF) ||
		response.Interval != "PT1M" || len(response.Value) != 1 {
		return nil, fmt.Errorf("invalid one-minute Cosmos %s metric response", metric)
	}
	series := response.Value[0]
	if !strings.EqualFold(series.ID, resource+"/providers/Microsoft.Insights/metrics/"+metric) ||
		(series.ErrorCode != "" && series.ErrorCode != "Success") || len(series.Series) > 1 {
		return nil, fmt.Errorf("unexpected Cosmos %s metric identity, dimensions or provider error", metric)
	}
	points := make(map[time.Time]float64)
	for _, sub := range series.Series {
		for _, point := range sub.Data {
			value := point.Total
			if aggregation == "Count" {
				value = point.Count
			}
			at := point.TimeStamp.UTC()
			if value == nil || !at.Before(end) {
				continue
			}
			if *value < 0 || !at.Equal(at.Truncate(time.Minute)) {
				return nil, errors.New("invalid Cosmos metric sample")
			}
			if _, duplicate := points[at]; duplicate {
				return nil, errors.New("duplicate Cosmos metric sample")
			}
			points[at] = *value
		}
	}
	return points, nil
}

func p1SustainedCosmosThrottles(points map[time.Time]float64) bool {
	for minute := range points {
		sustained := true
		for i := range 5 {
			if points[minute.Add(time.Duration(i)*time.Minute)] <= 0 {
				sustained = false
				break
			}
		}
		if sustained {
			return true
		}
	}
	return false
}

func startP1CosmosTelemetry(t *testing.T, ctx context.Context, cancelPilot context.CancelFunc, resource, output string) func(bool) {
	t.Helper()
	identity, err := azidentity.NewManagedIdentityCredential(nil)
	if err != nil {
		t.Fatal("Cosmos telemetry requires the worker system identity")
	}
	log, err := os.OpenFile(filepath.Join(output, "cosmos-telemetry.jsonl"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	client := &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}}
	monitor, stop := context.WithCancel(ctx)
	encoder := json.NewEncoder(log)
	start := time.Now().UTC().Truncate(time.Minute)
	observedRU := false
	collect := func() error {
		probe, cancel := context.WithTimeout(monitor, 45*time.Second)
		defer cancel()
		token, err := identity.GetToken(probe, policy.TokenRequestOptions{Scopes: []string{"https://management.azure.com/.default"}})
		if err != nil {
			return errors.New("worker system identity could not authenticate for Cosmos metrics")
		}
		read := func(metric, aggregation, filter string) (map[time.Time]float64, error) {
			values, err := readP1CosmosMetric(probe, client, token.Token, resource, metric, aggregation, filter)
			if errors.Is(err, errP1CosmosMetricTimeout) && probe.Err() == nil {
				return readP1CosmosMetric(probe, client, token.Token, resource, metric, aggregation, filter)
			}
			return values, err
		}
		charge, err := read("TotalRequestUnits", "Total", "")
		if err != nil {
			return err
		}
		throttles, err := read("TotalRequests", "Count", "StatusCode eq '429'")
		if err != nil {
			return err
		}
		for at, value := range charge {
			observedRU = observedRU || !at.Before(start) && value > 0
		}
		if err := encoder.Encode(map[string]any{
			"observedAtUtc": time.Now().UTC(), "requestUnits": charge, "requestsThrottled": throttles,
		}); err != nil {
			return err
		}
		if p1SustainedCosmosThrottles(throttles) {
			return errors.New("Cosmos source reported five consecutive minutes of throttling")
		}
		return nil
	}
	err = collect()
	for until := time.Now().Add(2 * time.Minute); errors.Is(err, errP1CosmosMetricForbidden) && time.Now().Before(until) && monitor.Err() == nil; {
		select {
		case <-monitor.Done():
			err = monitor.Err()
		case <-time.After(15 * time.Second):
			err = collect()
		}
	}
	if err != nil {
		stop()
		client.CloseIdleConnections()
		t.Fatal(errors.Join(recordP1TelemetryStop(encoder, err), log.Close()))
	}
	done := make(chan error, 1)
	go func() {
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()
		done <- errors.Join(watchP1Telemetry(monitor, cancelPilot, ticker.C, collect, encoder), log.Close())
	}()
	return func(required bool) {
		stop()
		defer client.CloseIdleConnections()
		if err := <-done; err != nil {
			t.Error(err)
		}
		if required && !observedRU {
			t.Error("no Cosmos request-unit metrics from this worker phase")
		}
	}
}

func TestHorizonDBP1CosmosSustainedThrottles(t *testing.T) {
	minute := time.Now().UTC().Truncate(time.Minute)
	points := make(map[time.Time]float64)
	for i := range 5 {
		points[minute.Add(time.Duration(i)*time.Minute)] = 1
	}
	if !p1SustainedCosmosThrottles(points) {
		t.Fatal("five consecutive throttled minutes must stop the lane")
	}
	delete(points, minute.Add(2*time.Minute))
	points[minute.Add(5*time.Minute)] = 2
	if p1SustainedCosmosThrottles(points) {
		t.Fatal("separated throttle bursts must not count as sustained")
	}
}
