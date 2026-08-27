package observability

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.43.0"
	"go.opentelemetry.io/otel/trace"

	"github.com/rioriost/agefreighter/internal/version"
)

const (
	LogFormatEnvironment         = "AGEFREIGHTER_LOG_FORMAT"
	LogLevelEnvironment          = "AGEFREIGHTER_LOG_LEVEL"
	OTLPEndpointEnvironment      = "OTEL_EXPORTER_OTLP_ENDPOINT"
	OTLPTraceEndpointEnvironment = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
	TracesExporterEnvironment    = "OTEL_TRACES_EXPORTER"
	SDKDisabledEnvironment       = "OTEL_SDK_DISABLED"

	instrumentationName = "github.com/rioriost/agefreighter"
)

type Options struct {
	ServiceName   string
	LogWriter     io.Writer
	LogFormat     string
	LogLevel      slog.Level
	TraceEnabled  bool
	TraceExporter sdktrace.SpanExporter
}

type Runtime struct {
	serviceName string
	logger      *slog.Logger
	logEnabled  bool
	tracer      trace.Tracer
	provider    *sdktrace.TracerProvider
}

func NewFromEnvironment(
	ctx context.Context,
	serviceName string,
	logWriter io.Writer,
) (*Runtime, error) {
	options, err := optionsFromEnvironment(serviceName, logWriter)
	if err != nil {
		return nil, err
	}
	return New(ctx, options)
}

func New(ctx context.Context, options Options) (*Runtime, error) {
	if strings.TrimSpace(options.ServiceName) == "" {
		return nil, errors.New("observability service name is required")
	}
	if options.LogWriter == nil {
		return nil, errors.New("observability log writer is required")
	}
	logger, logEnabled, err := newLogger(
		options.LogWriter,
		options.LogFormat,
		options.LogLevel,
	)
	if err != nil {
		return nil, err
	}
	runtime := &Runtime{
		serviceName: options.ServiceName,
		logger:      logger,
		logEnabled:  logEnabled,
		tracer:      trace.NewNoopTracerProvider().Tracer(instrumentationName),
	}
	if !options.TraceEnabled && options.TraceExporter == nil {
		return runtime, nil
	}

	exporter := options.TraceExporter
	if exporter == nil {
		exporter, err = otlptracegrpc.New(ctx)
		if err != nil {
			return nil, fmt.Errorf("create OTLP trace exporter: %w", err)
		}
	}
	build := version.Current()
	serviceResource, err := resource.Merge(
		resource.Default(),
		resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceName(options.ServiceName),
			semconv.ServiceVersion(build.Version),
			attribute.String("vcs.ref.head.revision", build.Commit),
		),
	)
	if err != nil {
		resourceErr := fmt.Errorf("create OpenTelemetry resource: %w", err)
		if shutdownErr := exporter.Shutdown(ctx); shutdownErr != nil {
			return nil, errors.Join(
				resourceErr,
				fmt.Errorf("shut down trace exporter: %w", shutdownErr),
			)
		}
		return nil, resourceErr
	}
	runtime.provider = sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(serviceResource),
	)
	runtime.tracer = runtime.provider.Tracer(instrumentationName)
	otel.SetErrorHandler(otel.ErrorHandlerFunc(func(err error) {
		runtime.LogExportError(context.Background(), err)
	}))
	return runtime, nil
}

func (runtime *Runtime) Run(
	ctx context.Context,
	command string,
	execute func(context.Context) error,
) error {
	if runtime == nil {
		return errors.New("observability runtime is nil")
	}
	if execute == nil {
		return errors.New("observability command function is required")
	}
	command = safeCommandName(command)
	ctx, span := runtime.tracer.Start(
		ctx,
		"cli."+command,
		trace.WithAttributes(
			attribute.String("cli.program", runtime.serviceName),
			attribute.String("cli.command", command),
		),
	)
	started := time.Now()
	runtime.log(ctx, slog.LevelInfo, "command_started", command, 0, false)
	err := execute(ctx)
	elapsed := time.Since(started)
	if err != nil {
		span.SetStatus(codes.Error, "command failed")
		runtime.log(ctx, slog.LevelError, "command_failed", command, elapsed, true)
	} else {
		span.SetStatus(codes.Ok, "")
		runtime.log(ctx, slog.LevelInfo, "command_completed", command, elapsed, false)
	}
	span.End()
	return err
}

func (runtime *Runtime) Shutdown(ctx context.Context) error {
	if runtime == nil || runtime.provider == nil {
		return nil
	}
	if err := runtime.provider.Shutdown(ctx); err != nil {
		return fmt.Errorf("shut down OpenTelemetry: %w", err)
	}
	return nil
}

func (runtime *Runtime) ForceFlush(ctx context.Context) error {
	if runtime == nil || runtime.provider == nil {
		return nil
	}
	if err := runtime.provider.ForceFlush(ctx); err != nil {
		return fmt.Errorf("flush OpenTelemetry: %w", err)
	}
	return nil
}

func (runtime *Runtime) LogExportError(ctx context.Context, err error) bool {
	if runtime == nil || err == nil {
		return false
	}
	runtime.log(ctx, slog.LevelWarn, "telemetry_export_failed", "", 0, true)
	return runtime.logEnabled
}

func (runtime *Runtime) log(
	ctx context.Context,
	level slog.Level,
	event string,
	command string,
	elapsed time.Duration,
	failed bool,
) {
	if !runtime.logEnabled {
		return
	}
	attributes := []any{
		slog.Int("schema_version", 1),
		slog.String("event", event),
		slog.String("service", runtime.serviceName),
	}
	if spanContext := trace.SpanContextFromContext(ctx); spanContext.IsValid() {
		attributes = append(
			attributes,
			slog.String("trace_id", spanContext.TraceID().String()),
			slog.String("span_id", spanContext.SpanID().String()),
		)
	}
	if command != "" {
		attributes = append(attributes, slog.String("command", command))
	}
	if elapsed > 0 {
		attributes = append(attributes, slog.Int64("duration_ms", elapsed.Milliseconds()))
	}
	if failed {
		attributes = append(attributes, slog.Bool("failed", true))
	}
	runtime.logger.Log(ctx, level, event, attributes...)
}

func optionsFromEnvironment(serviceName string, writer io.Writer) (Options, error) {
	format := strings.ToLower(strings.TrimSpace(os.Getenv(LogFormatEnvironment)))
	switch format {
	case "", "off", "text", "json":
	default:
		return Options{}, fmt.Errorf(
			"%s must be off, text, or json",
			LogFormatEnvironment,
		)
	}
	level, err := parseLogLevel(os.Getenv(LogLevelEnvironment))
	if err != nil {
		return Options{}, err
	}
	traceEnabled, err := traceEnabledFromEnvironment()
	if err != nil {
		return Options{}, err
	}
	return Options{
		ServiceName:  serviceName,
		LogWriter:    writer,
		LogFormat:    format,
		LogLevel:     level,
		TraceEnabled: traceEnabled,
	}, nil
}

func newLogger(
	writer io.Writer,
	format string,
	level slog.Level,
) (*slog.Logger, bool, error) {
	options := &slog.HandlerOptions{Level: level}
	switch format {
	case "", "off":
		return slog.New(slog.NewTextHandler(io.Discard, options)), false, nil
	case "text":
		return slog.New(slog.NewTextHandler(writer, options)), true, nil
	case "json":
		return slog.New(slog.NewJSONHandler(writer, options)), true, nil
	default:
		return nil, false, fmt.Errorf("unsupported log format %q", format)
	}
}

func parseLogLevel(raw string) (slog.Level, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "info":
		return slog.LevelInfo, nil
	case "debug":
		return slog.LevelDebug, nil
	case "warn":
		return slog.LevelWarn, nil
	case "error":
		return slog.LevelError, nil
	default:
		return 0, fmt.Errorf(
			"%s must be debug, info, warn, or error",
			LogLevelEnvironment,
		)
	}
}

func traceEnabledFromEnvironment() (bool, error) {
	if raw := strings.TrimSpace(os.Getenv(SDKDisabledEnvironment)); raw != "" {
		disabled, err := strconv.ParseBool(raw)
		if err != nil {
			return false, fmt.Errorf("%s must be a boolean", SDKDisabledEnvironment)
		}
		if disabled {
			return false, nil
		}
	}
	exporter := strings.ToLower(strings.TrimSpace(os.Getenv(TracesExporterEnvironment)))
	switch exporter {
	case "none":
		return false, nil
	case "", "otlp":
	default:
		return false, fmt.Errorf(
			"%s must be otlp or none",
			TracesExporterEnvironment,
		)
	}
	if strings.TrimSpace(os.Getenv(OTLPEndpointEnvironment)) == "" &&
		strings.TrimSpace(os.Getenv(OTLPTraceEndpointEnvironment)) == "" {
		if exporter == "otlp" {
			return false, fmt.Errorf(
				"%s or %s is required when %s=otlp",
				OTLPEndpointEnvironment,
				OTLPTraceEndpointEnvironment,
				TracesExporterEnvironment,
			)
		}
		return false, nil
	}
	return true, nil
}

func safeCommandName(command string) string {
	command = strings.TrimSpace(command)
	if command == "" || strings.HasPrefix(command, "-") {
		return "root"
	}
	switch command {
	case "version",
		"validate",
		"plan",
		"load",
		"resume",
		"status",
		"verify",
		"cleanup",
		"generate",
		"benchmark-age-copy",
		"benchmark-report",
		"inspect",
		"convert-gremlin":
		return command
	default:
		return "other"
	}
}
