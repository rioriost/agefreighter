package observability

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestJSONLoggingContractOmitsArgumentsAndErrors(t *testing.T) {
	var output bytes.Buffer
	runtime, err := New(t.Context(), Options{
		ServiceName: "agefreighter",
		LogWriter:   &output,
		LogFormat:   "json",
		LogLevel:    slog.LevelInfo,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	secret := "postgres://user:password@example.invalid/database"
	runErr := runtime.Run(t.Context(), "load", func(context.Context) error {
		return errors.New(secret)
	})
	if runErr == nil || runErr.Error() != secret {
		t.Fatalf("Run() error = %v", runErr)
	}
	if strings.Contains(output.String(), secret) ||
		strings.Contains(output.String(), "password") {
		t.Fatalf("JSON log exposed command error: %s", output.String())
	}

	decoder := json.NewDecoder(&output)
	var events []map[string]any
	for decoder.More() {
		var event map[string]any
		if err := decoder.Decode(&event); err != nil {
			t.Fatalf("decode JSON log: %v", err)
		}
		events = append(events, event)
	}
	if len(events) != 2 ||
		events[0]["event"] != "command_started" ||
		events[1]["event"] != "command_failed" ||
		events[1]["failed"] != true ||
		events[1]["command"] != "load" ||
		events[1]["service"] != "agefreighter" ||
		events[1]["schema_version"] != float64(1) {
		t.Fatalf("JSON events = %#v", events)
	}
	for _, event := range events {
		if _, ok := event["time"]; !ok {
			t.Fatalf("JSON event missing time: %#v", event)
		}
		if _, ok := event["level"]; !ok {
			t.Fatalf("JSON event missing level: %#v", event)
		}
		if _, ok := event["msg"]; !ok {
			t.Fatalf("JSON event missing msg: %#v", event)
		}
	}
}

func TestTraceExportUsesSafeCommandAttributes(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	runtime, err := New(t.Context(), Options{
		ServiceName:   "agefreighter-tools",
		LogWriter:     &bytes.Buffer{},
		TraceExporter: exporter,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := runtime.Run(t.Context(), "inspect", func(context.Context) error {
		return nil
	}); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if err := runtime.ForceFlush(t.Context()); err != nil {
		t.Fatalf("ForceFlush() error = %v", err)
	}
	spans := exporter.GetSpans()
	if len(spans) != 1 || spans[0].Name != "cli.inspect" {
		t.Fatalf("exported spans = %#v", spans)
	}
	attributes := make(map[string]string)
	for _, item := range spans[0].Attributes {
		attributes[string(item.Key)] = item.Value.AsString()
	}
	if attributes["cli.program"] != "agefreighter-tools" ||
		attributes["cli.command"] != "inspect" {
		t.Fatalf("span attributes = %#v", attributes)
	}
	if err := runtime.Shutdown(t.Context()); err != nil {
		t.Fatalf("Shutdown() error = %v", err)
	}
}

func TestRuntimeValidationAndSafeCommandNames(t *testing.T) {
	if _, err := New(t.Context(), Options{
		LogWriter: &bytes.Buffer{},
	}); err == nil {
		t.Fatal("New() accepted empty service name")
	}
	if _, err := New(t.Context(), Options{
		ServiceName: "test",
	}); err == nil {
		t.Fatal("New() accepted nil log writer")
	}
	if _, _, err := newLogger(&bytes.Buffer{}, "xml", slog.LevelInfo); err == nil {
		t.Fatal("newLogger() accepted invalid format")
	}
	if err := (*Runtime)(nil).Run(t.Context(), "version", func(context.Context) error {
		return nil
	}); err == nil {
		t.Fatal("nil Runtime.Run() succeeded")
	}
	runtime, err := New(t.Context(), Options{
		ServiceName: "test",
		LogWriter:   &bytes.Buffer{},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := runtime.Run(t.Context(), "version", nil); err == nil {
		t.Fatal("Runtime.Run() accepted nil command")
	}
	if err := runtime.Shutdown(t.Context()); err != nil {
		t.Fatalf("disabled Shutdown() error = %v", err)
	}
	if err := runtime.ForceFlush(t.Context()); err != nil {
		t.Fatalf("disabled ForceFlush() error = %v", err)
	}
	if runtime.LogExportError(t.Context(), nil) {
		t.Fatal("nil export error was logged")
	}
	if (*Runtime)(nil).LogExportError(t.Context(), errors.New("ignored")) {
		t.Fatal("nil runtime reported export log")
	}

	tests := map[string]string{
		"":             "root",
		"--help":       "root",
		"load":         "load",
		"benchmark-1":  "other",
		"secret=value": "other",
	}
	for input, want := range tests {
		if got := safeCommandName(input); got != want {
			t.Fatalf("safeCommandName(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestEnvironmentOptions(t *testing.T) {
	for _, name := range []string{
		LogFormatEnvironment,
		LogLevelEnvironment,
		OTLPEndpointEnvironment,
		OTLPTraceEndpointEnvironment,
		TracesExporterEnvironment,
		SDKDisabledEnvironment,
	} {
		t.Setenv(name, "")
	}
	options, err := optionsFromEnvironment("test", &bytes.Buffer{})
	if err != nil {
		t.Fatalf("optionsFromEnvironment() error = %v", err)
	}
	if options.TraceEnabled || options.LogFormat != "" ||
		options.LogLevel != slog.LevelInfo {
		t.Fatalf("default options = %#v", options)
	}

	t.Setenv(LogFormatEnvironment, "JSON")
	t.Setenv(LogLevelEnvironment, "DEBUG")
	t.Setenv(OTLPEndpointEnvironment, "http://collector:4317")
	options, err = optionsFromEnvironment("test", &bytes.Buffer{})
	if err != nil {
		t.Fatalf("enabled options error = %v", err)
	}
	if !options.TraceEnabled || options.LogFormat != "json" ||
		options.LogLevel != slog.LevelDebug {
		t.Fatalf("enabled options = %#v", options)
	}

	t.Setenv(OTLPEndpointEnvironment, "")
	t.Setenv(OTLPTraceEndpointEnvironment, "http://collector:4318")
	options, err = optionsFromEnvironment("test", &bytes.Buffer{})
	if err != nil || !options.TraceEnabled {
		t.Fatalf("trace-specific endpoint options = %#v, %v", options, err)
	}

	t.Setenv(SDKDisabledEnvironment, "true")
	options, err = optionsFromEnvironment("test", &bytes.Buffer{})
	if err != nil || options.TraceEnabled {
		t.Fatalf("disabled SDK options = %#v, %v", options, err)
	}
}

func TestEnvironmentOptionErrors(t *testing.T) {
	tests := []struct {
		name  string
		value string
	}{
		{name: LogFormatEnvironment, value: "yaml"},
		{name: LogLevelEnvironment, value: "verbose"},
		{name: SDKDisabledEnvironment, value: "sometimes"},
		{name: TracesExporterEnvironment, value: "zipkin"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, name := range []string{
				LogFormatEnvironment,
				LogLevelEnvironment,
				OTLPEndpointEnvironment,
				OTLPTraceEndpointEnvironment,
				TracesExporterEnvironment,
				SDKDisabledEnvironment,
			} {
				t.Setenv(name, "")
			}
			t.Setenv(test.name, test.value)
			if _, err := optionsFromEnvironment("test", &bytes.Buffer{}); err == nil {
				t.Fatal("optionsFromEnvironment() succeeded")
			}
		})
	}

	for _, name := range []string{
		LogFormatEnvironment,
		LogLevelEnvironment,
		OTLPEndpointEnvironment,
		OTLPTraceEndpointEnvironment,
		TracesExporterEnvironment,
		SDKDisabledEnvironment,
	} {
		t.Setenv(name, "")
	}
	t.Setenv(TracesExporterEnvironment, "otlp")
	if _, err := optionsFromEnvironment("test", &bytes.Buffer{}); err == nil {
		t.Fatal("explicit OTLP exporter accepted missing endpoint")
	}
}

func TestTextLoggingAndFilteredLevel(t *testing.T) {
	var output bytes.Buffer
	runtime, err := New(t.Context(), Options{
		ServiceName: "tools",
		LogWriter:   &output,
		LogFormat:   "text",
		LogLevel:    slog.LevelWarn,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := runtime.Run(t.Context(), "version", func(context.Context) error {
		return nil
	}); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if output.Len() != 0 {
		t.Fatalf("filtered log output = %q", output.String())
	}
	if !runtime.LogExportError(t.Context(), errors.New("secret exporter message")) {
		t.Fatal("enabled logger did not report export error")
	}
	if !strings.Contains(output.String(), "telemetry_export_failed") ||
		strings.Contains(output.String(), "secret exporter message") {
		t.Fatalf("export error log = %q", output.String())
	}
}
