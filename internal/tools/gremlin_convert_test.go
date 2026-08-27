package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestOpenAIGremlinConverter(t *testing.T) {
	const (
		apiKey  = "test-api-key"
		model   = "test-model"
		gremlin = "g.V().hasLabel('Person')"
	)
	server := httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, request *http.Request) {
			if request.Method != http.MethodPost {
				t.Errorf("method = %s, want POST", request.Method)
			}
			if got := request.Header.Get("Authorization"); got != "Bearer "+apiKey {
				t.Errorf("Authorization = %q", got)
			}
			var body map[string]any
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Fatalf("decode request: %v", err)
			}
			if body["model"] != model ||
				body["input"] != gremlin ||
				body["store"] != false {
				t.Errorf("request = %#v", body)
			}
			text, ok := body["text"].(map[string]any)
			if !ok {
				t.Fatalf("text = %#v", body["text"])
			}
			format, ok := text["format"].(map[string]any)
			if !ok || format["type"] != "json_schema" || format["strict"] != true {
				t.Fatalf("text.format = %#v", text["format"])
			}
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{
				"status":"completed",
				"model":"test-model-2026-01-01",
				"output":[{
					"type":"message",
					"content":[{
						"type":"output_text",
						"text":"{\"status\":\"converted\",\"cypher\":\"MATCH (n:Person) RETURN n\",\"confidence\":\"high\",\"warnings\":[]}"
					}]
				}]
			}`)
		},
	))
	defer server.Close()

	converter := &openAIGremlinConverter{
		client:   server.Client(),
		endpoint: server.URL,
	}
	result, err := converter.Convert(t.Context(), apiKey, model, gremlin)
	if err != nil {
		t.Fatalf("Convert() error = %v", err)
	}
	if result.FormatVersion != conversionFormatVersion ||
		result.Status != "converted" ||
		result.Cypher != "MATCH (n:Person) RETURN n" ||
		result.Confidence != "high" ||
		result.Model != "test-model-2026-01-01" ||
		result.Warnings == nil {
		t.Fatalf("result = %#v", result)
	}
}

func TestOpenAIGremlinConverterRefusal(t *testing.T) {
	server := newOpenAIResponseServer(t, http.StatusOK, `{
		"status":"completed",
		"model":"test-model",
		"output":[{
			"type":"message",
			"content":[{"type":"refusal","refusal":"sensitive echoed input"}]
		}]
	}`)
	defer server.Close()
	converter := &openAIGremlinConverter{
		client:   server.Client(),
		endpoint: server.URL,
	}

	result, err := converter.Convert(
		t.Context(),
		"test-key",
		"test-model",
		"g.V()",
	)
	if err != nil {
		t.Fatalf("Convert() error = %v", err)
	}
	if result.Status != "refused" ||
		result.Cypher != "" ||
		strings.Contains(strings.Join(result.Warnings, " "), "sensitive") {
		t.Fatalf("result = %#v", result)
	}
}

func TestOpenAIGremlinConverterRejectsUnsafeResponses(t *testing.T) {
	tests := map[string]struct {
		status int
		body   string
		want   string
	}{
		"http error redacts body": {
			status: http.StatusUnauthorized,
			body: `{"error":{"message":"secret-query-and-token",` +
				`"type":"invalid_request_error","code":"invalid_api_key"}}`,
			want: "HTTP status 401",
		},
		"incomplete response": {
			status: http.StatusOK,
			body:   `{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}`,
			want:   `status "incomplete"`,
		},
		"invalid API JSON": {
			status: http.StatusOK,
			body:   `{`,
			want:   "invalid response",
		},
		"unknown structured field": {
			status: http.StatusOK,
			body: completedOpenAIResponse(
				`{"status":"converted","cypher":"MATCH (n) RETURN n",` +
					`"confidence":"high","warnings":[],"extra":true}`,
			),
			want: "invalid structured conversion output",
		},
		"non cypher output": {
			status: http.StatusOK,
			body: completedOpenAIResponse(
				`{"status":"converted","cypher":"Here is your query",` +
					`"confidence":"high","warnings":[]}`,
			),
			want: "does not start with an openCypher clause",
		},
		"unsupported with query": {
			status: http.StatusOK,
			body: completedOpenAIResponse(
				`{"status":"unsupported","cypher":"MATCH (n) RETURN n",` +
					`"confidence":"low","warnings":[]}`,
			),
			want: "query for an unsupported conversion",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			server := newOpenAIResponseServer(t, test.status, test.body)
			defer server.Close()
			converter := &openAIGremlinConverter{
				client:   server.Client(),
				endpoint: server.URL,
			}
			_, err := converter.Convert(
				t.Context(),
				"test-key",
				"test-model",
				"g.V()",
			)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Convert() error = %v, want %q", err, test.want)
			}
			if strings.Contains(err.Error(), "secret-query-and-token") {
				t.Fatalf("Convert() leaked response body: %v", err)
			}
		})
	}
}

func TestConvertGremlinCommand(t *testing.T) {
	t.Setenv(OpenAIAPIKeyEnvironment, "test-key")
	t.Setenv(OpenAIModelEnvironment, "env-model")
	converter := &recordingGremlinConverter{
		result: GremlinConversionResult{
			FormatVersion: conversionFormatVersion,
			Status:        "converted",
			Cypher:        "RETURN 1",
			Confidence:    "high",
			Warnings:      []string{},
			Model:         "env-model",
		},
	}
	var output bytes.Buffer
	command := newConvertGremlinCommand(converter)
	command.SetOut(&output)
	command.SetArgs([]string{"--query", "g.V().count()"})

	if err := command.Execute(); err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if converter.apiKey != "test-key" ||
		converter.model != "env-model" ||
		converter.gremlin != "g.V().count()" {
		t.Fatalf("converter call = %#v", converter)
	}
	var result GremlinConversionResult
	if err := json.Unmarshal(output.Bytes(), &result); err != nil {
		t.Fatalf("decode output: %v", err)
	}
	if result.Status != "converted" || result.Cypher != "RETURN 1" {
		t.Fatalf("result = %#v", result)
	}
}

func TestConvertGremlinCommandInputValidation(t *testing.T) {
	tests := map[string]struct {
		args  []string
		key   string
		stdin string
		want  string
	}{
		"missing source": {
			key:  "key",
			want: "exactly one",
		},
		"multiple sources": {
			args: []string{"--query", "g.V()", "--input", "-"},
			key:  "key",
			want: "exactly one",
		},
		"missing key": {
			args: []string{"--query", "g.V()"},
			want: OpenAIAPIKeyEnvironment + " is required",
		},
		"invalid model": {
			args: []string{"--query", "g.V()", "--model", "bad model"},
			key:  "key",
			want: "unsupported character",
		},
		"empty stdin": {
			args:  []string{"--input", "-"},
			key:   "key",
			stdin: "  ",
			want:  "must not be empty",
		},
		"invalid timeout": {
			args: []string{"--query", "g.V()", "--timeout", "0s"},
			key:  "key",
			want: "timeout must be positive",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			t.Setenv(OpenAIAPIKeyEnvironment, test.key)
			command := newConvertGremlinCommand(&recordingGremlinConverter{})
			command.SetIn(strings.NewReader(test.stdin))
			command.SetArgs(test.args)
			err := command.Execute()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Execute() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestConvertGremlinCommandUsesTimeoutAndReportsWriteFailure(t *testing.T) {
	t.Setenv(OpenAIAPIKeyEnvironment, "test-key")
	converter := &recordingGremlinConverter{
		result: GremlinConversionResult{
			FormatVersion: conversionFormatVersion,
			Status:        "converted",
			Cypher:        "RETURN 1",
			Confidence:    "high",
			Warnings:      []string{},
			Model:         "test-model",
		},
	}
	command := newConvertGremlinCommand(converter)
	command.SetOut(failingGremlinWriter{})
	command.SetArgs([]string{
		"--query", "g.V()",
		"--timeout", "20ms",
	})

	err := command.Execute()
	if err == nil || !strings.Contains(err.Error(), "write conversion result") {
		t.Fatalf("Execute() error = %v", err)
	}
	if converter.deadlineRemaining <= 0 ||
		converter.deadlineRemaining > 20*time.Millisecond {
		t.Fatalf("deadline remaining = %v", converter.deadlineRemaining)
	}
}

func TestGremlinInputBounds(t *testing.T) {
	if _, err := validateGremlinInput(
		strings.Repeat("x", maxGremlinInputBytes+1),
	); err == nil {
		t.Fatal("validateGremlinInput() accepted oversized input")
	}
	if _, err := validateGremlinInput("g.V()\x00"); err == nil {
		t.Fatal("validateGremlinInput() accepted NUL")
	}
	oversized := strings.NewReader(strings.Repeat("x", maxGremlinInputBytes+1))
	if _, err := readGremlinInput("", "-", oversized); err == nil {
		t.Fatal("readGremlinInput() accepted oversized stdin")
	}
}

func TestValidateOpenAIConversion(t *testing.T) {
	if !hasOpenCypherPrefix(" optional MATCH (n) RETURN n") {
		t.Fatal("hasOpenCypherPrefix() rejected OPTIONAL MATCH")
	}
	if hasOpenCypherPrefix("This is not Cypher") {
		t.Fatal("hasOpenCypherPrefix() accepted prose")
	}
	if got := safeResponseStatus("server-injected-status"); got != "unknown" {
		t.Fatalf("safeResponseStatus() = %q", got)
	}
	if got := safeAPIIdentifier("unsafe value"); got != "unknown" {
		t.Fatalf("safeAPIIdentifier() = %q", got)
	}
	if got := safeAPIIdentifier("rate_limit_exceeded"); got != "rate_limit_exceeded" {
		t.Fatalf("safeAPIIdentifier() = %q", got)
	}
	if err := validateOpenAIModel(strings.Repeat("a", 129)); err == nil {
		t.Fatal("validateOpenAIModel() accepted oversized name")
	}
}

func newOpenAIResponseServer(
	t *testing.T,
	status int,
	body string,
) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, _ *http.Request) {
			writer.WriteHeader(status)
			_, _ = io.WriteString(writer, body)
		},
	))
}

func completedOpenAIResponse(output string) string {
	encoded, _ := json.Marshal(output)
	return `{"status":"completed","model":"test-model","output":[{` +
		`"type":"message","content":[{"type":"output_text","text":` +
		string(encoded) + `}]}]}`
}

type recordingGremlinConverter struct {
	result            GremlinConversionResult
	err               error
	apiKey            string
	model             string
	gremlin           string
	deadlineRemaining time.Duration
}

func (converter *recordingGremlinConverter) Convert(
	ctx context.Context,
	apiKey string,
	model string,
	gremlin string,
) (GremlinConversionResult, error) {
	converter.apiKey = apiKey
	converter.model = model
	converter.gremlin = gremlin
	if deadline, ok := ctx.Deadline(); ok {
		converter.deadlineRemaining = time.Until(deadline)
	}
	return converter.result, converter.err
}

type failingGremlinWriter struct{}

func (failingGremlinWriter) Write([]byte) (int, error) {
	return 0, errors.New("write failed")
}
