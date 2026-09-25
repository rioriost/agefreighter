package runner

import (
	"bytes"
	"encoding/json"
	"encoding/pem"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rioriost/agefreighter/internal/config"
)

type failedReader struct{}

func (failedReader) Read([]byte) (int, error) { return 0, errors.New("private read failure") }

func TestProtocolSeparatesReadOnlyRequestsAndExecutionCapabilities(t *testing.T) {
	base := Request{Version: 1, Workflow: workflowID, Operation: operationID, Action: "status"}
	for _, action := range []string{"ready", "status", "report"} {
		t.Run(action, func(t *testing.T) {
			r := base
			r.Action = action
			data, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := Decode(bytes.NewReader(data))
			if err != nil || decoded.Action != action {
				t.Fatalf("read-only request: %+v %v", decoded, err)
			}
		})
	}
	for _, tc := range []struct {
		name   string
		change func(*Request)
	}{
		{"version", func(r *Request) { r.Version++ }},
		{"workflow", func(r *Request) { r.Workflow = "../escape" }},
		{"operation", func(r *Request) { r.Operation = "" }},
		{"configuration", func(r *Request) { r.Configuration = json.RawMessage(`{}`) }},
		{"secrets", func(r *Request) { r.Secrets = map[string]string{"AGEFREIGHTER_SOURCE_PASSWORD": "private"} }},
		{"boot", func(r *Request) { r.ExpectedBootID = bootID }},
		{"offset", func(r *Request) { r.Offset = 1 }},
		{"negative-report-offset", func(r *Request) { r.Action, r.Offset = "report", -1 }},
		{"import", func(r *Request) { r.Import = &CSVImport{} }},
		{"export", func(r *Request) { r.Export = &ReportExport{} }},
		{"resume", func(r *Request) { r.Resume = &ResumeBinding{} }},
		{"assessment-without-config", func(r *Request) { r.Action, r.ExpectedBootID = "profile", bootID }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := base
			tc.change(&r)
			data, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := Decode(bytes.NewReader(data)); err == nil {
				t.Fatal("mixed or invalid control accepted")
			}
		})
	}
	encoded, err := json.Marshal(base)
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range []io.Reader{failedReader{}, bytes.NewReader(append(encoded, []byte(` {}`)...)), strings.NewReader(`null`)} {
		if _, err := Decode(input); err == nil || strings.Contains(err.Error(), "private") {
			t.Fatalf("unsafe decode result: %v", err)
		}
	}
}

func TestSourceCABundleBoundariesAndFailedStagingLeaveNoCertificate(t *testing.T) {
	ca := testSourceCAPEM(t)
	block, _ := pem.Decode([]byte(ca))
	withHeader := *block
	withHeader.Headers = map[string]string{"Unreviewed": "value"}
	for _, tc := range []struct{ name, value string }{
		{"empty", ""}, {"whitespace", " \n\t"},
		{"oversized", strings.Repeat("x", (64<<10)+1)},
		{"invalid-der", string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: []byte("not a certificate")}))},
		{"pem-headers", string(pem.EncodeToMemory(&withHeader))},
		{"too-many", strings.Repeat(ca, 17)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := validateSourceCA([]byte(tc.value)); err == nil {
				t.Fatal("unreviewed CA bundle accepted")
			}
		})
	}
	if err := validateSourceCA([]byte(strings.Repeat(ca, 16))); err != nil {
		t.Fatalf("maximum certificate count: %v", err)
	}
	for _, dsn := range []string{"postgresql://%", "postgresql://source/db?sslmode=require", "postgresql://source/db?sslmode=verify-full&sslrootcert=other"} {
		dir := t.TempDir()
		input := map[string]string{"AGEFREIGHTER_SOURCE_CA_PEM": ca, "AGEFREIGHTER_SOURCE_DSN": dsn}
		if _, _, err := stageSourceCA(dir, input); err == nil {
			t.Fatal("unreviewed TLS source accepted")
		}
		if _, err := os.Stat(filepath.Join(dir, "source-ca.pem")); !os.IsNotExist(err) {
			t.Fatal("failed source binding retained staged CA")
		}
		if input["AGEFREIGHTER_SOURCE_CA_PEM"] != ca || input["AGEFREIGHTER_SOURCE_DSN"] != dsn {
			t.Fatal("failed staging mutated protected inputs")
		}
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "source-ca.pem")
	if err := writeNew(path, []byte("retained")); err != nil {
		t.Fatal(err)
	}
	if _, _, err := stageSourceCA(dir, map[string]string{"AGEFREIGHTER_SOURCE_CA_PEM": ca}); err == nil {
		t.Fatal("staging overwrote retained certificate")
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "retained" {
		t.Fatal("certificate evidence changed")
	}
	if _, _, err := stageSourceCA(dir, map[string]string{"AGEFREIGHTER_SOURCE_CA_PEM": "invalid"}); err == nil {
		t.Fatal("invalid staged certificate accepted")
	}
}

func TestRetainedUploadsRequireRegularSealedVertexAndEdgeFiles(t *testing.T) {
	m, r, _ := testManager(t)
	root := filepath.Join(m.Root, r.Workflow)
	job, err := config.Parse(r.Configuration)
	if err != nil {
		t.Fatal(err)
	}
	vertex := job.Source.CSV.Vertices[0]
	job.Source.CSV.Edges = []config.CSVEdge{{Path: vertex.Path}}
	configuration, err := json.Marshal(job)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateUploadPaths(configuration, root); err != nil {
		t.Fatalf("sealed edge upload: %v", err)
	}
	seal, err := os.ReadFile(vertex.Path + ".seal.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(vertex.Path+".seal.json", []byte(`{}`), 0600); err != nil {
		t.Fatal(err)
	}
	if err := validateUploadPaths(configuration, root); err == nil {
		t.Fatal("upload path admitted without matching full-content seal")
	}
	if err := os.WriteFile(vertex.Path+".seal.json", seal, 0600); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"relative.csv", filepath.Join(root, "uploads"), filepath.Join(root, "elsewhere.csv")} {
		if insideUploads(root, path) {
			t.Fatalf("uncontained path accepted: %s", path)
		}
	}
	if err := validateUploadPaths([]byte(`{`), root); err == nil {
		t.Fatal("invalid retained JSON accepted")
	}
	job.Source.CSV.Vertices[0].Path = filepath.Join(root, "uploads", "directory")
	if err := os.Mkdir(job.Source.CSV.Vertices[0].Path, 0700); err != nil {
		t.Fatal(err)
	}
	configuration, err = json.Marshal(job)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateUploadPaths(configuration, root); err == nil {
		t.Fatal("directory admitted as CSV")
	}
	if err := os.Remove(vertex.Path); err != nil {
		t.Fatal(err)
	}
	if err := verifyCSVSeal(vertex.Path); err == nil {
		t.Fatal("seal without file admitted")
	}
}

func TestResumeBindingRejectsNoncanonicalAndOutOfRangeIdentities(t *testing.T) {
	base := ResumeBinding{PreviousOperation: operationID, JobID: operationID, ConfigSHA256: strings.Repeat("a", 64), Fingerprint: strings.Repeat("b", 64), GenerationID: "9223372036854775807", CommittedRows: "0"}
	if !validResumeBinding(&base) {
		t.Fatal("lossless 64-bit boundary rejected")
	}
	for _, value := range []string{"0", "-1", "01", "+1", "1.0", "9223372036854775808", ""} {
		b := base
		b.GenerationID = value
		if validResumeBinding(&b) {
			t.Fatalf("noncanonical generation accepted: %q", value)
		}
	}
	for _, value := range []string{"-1", "00", "+0", "1e2", "9223372036854775808"} {
		b := base
		b.CommittedRows = value
		if validResumeBinding(&b) {
			t.Fatalf("noncanonical count accepted: %q", value)
		}
	}
	base.Fingerprint = strings.Repeat("G", 64)
	if validResumeBinding(&base) || validResumeBinding(nil) {
		t.Fatal("invalid fingerprint/binding accepted")
	}
}

func TestConfigurationRejectsUnreviewedSourceAndCredentialCombinations(t *testing.T) {
	ca := testSourceCAPEM(t)
	for _, tc := range []struct {
		name, fixture, action, want string
		change                      func(*config.LoadJob, *Request)
	}{
		{"trial", "trial.yaml", "profile", "trial writes", nil},
		{"network-as-csv", "neo4j.yaml", "migrate-csv", "reviewed CSV source", nil},
		{"csv-as-network", "", "migrate-source", "network migration requires", nil},
		{"target-tls", "", "migrate-csv", "verified TLS", func(_ *config.LoadJob, r *Request) {
			r.Secrets["AGEFREIGHTER_TARGET_DSN"] = strings.Replace(testMigrationDSN(), "verify-full", "require", 1)
		}},
		{"postgres-credential", "postgresql.yaml", "migrate-source", "protected source connection", func(_ *config.LoadJob, r *Request) {
			delete(r.Secrets, "AGEFREIGHTER_SOURCE_DSN")
		}},
		{"csv-migration-ca", "", "migrate-csv", "custom source CA", func(_ *config.LoadJob, r *Request) {
			r.Secrets["AGEFREIGHTER_SOURCE_CA_PEM"] = ca
		}},
		{"file-secret", "", "profile", "approved environment handles", func(j *config.LoadJob, _ *Request) {
			j.Target.Connection = config.SecretRef{File: "/unreviewed/credential"}
		}},
		{"invalid-ca", "", "profile", "certificate-only", func(_ *config.LoadJob, r *Request) {
			r.Secrets["AGEFREIGHTER_SOURCE_CA_PEM"] = "not a certificate"
		}},
		{"csv-assessment-ca", "", "profile", "custom source CA", func(_ *config.LoadJob, r *Request) {
			r.Secrets["AGEFREIGHTER_SOURCE_CA_PEM"] = ca
		}},
		{"edge-escape", "csv.yaml", "profile", "verified upload directory", func(j *config.LoadJob, _ *Request) {
			j.Source.CSV.Edges[0].Path = "/unreviewed/edge.csv"
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, r, starts := testManager(t)
			job, err := config.Parse(r.Configuration)
			if err != nil {
				t.Fatal(err)
			}
			vertexPath := job.Source.CSV.Vertices[0].Path
			if tc.fixture != "" {
				data, err := os.ReadFile(filepath.Join("../config/testdata/valid", tc.fixture))
				if err != nil {
					t.Fatal(err)
				}
				job, err = config.Parse(data)
				if err != nil {
					t.Fatal(err)
				}
			}
			job.Target.Connection = config.SecretRef{Env: "AGEFREIGHTER_TARGET_DSN"}
			job.Target.Mode = config.LoadCreate
			r.Action, r.Secrets = tc.action, map[string]string{"AGEFREIGHTER_TARGET_DSN": testMigrationDSN()}
			if job.Source.PostgreSQL != nil {
				job.Source.PostgreSQL.Connection = config.SecretRef{Env: "AGEFREIGHTER_SOURCE_DSN"}
				r.Secrets["AGEFREIGHTER_SOURCE_DSN"] = catalogConnection
			}
			if job.Source.Neo4j != nil {
				job.Source.Neo4j.Password = &config.SecretRef{Env: "AGEFREIGHTER_SOURCE_PASSWORD"}
				r.Secrets["AGEFREIGHTER_SOURCE_PASSWORD"] = "fixture-source-password"
			}
			if job.Source.CSV != nil {
				job.Source.CSV.Vertices[0].Path = vertexPath
			}
			if tc.change != nil {
				tc.change(&job, &r)
			}
			r.Configuration, err = json.Marshal(job)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := m.Submit(t.Context(), r); err == nil || !strings.Contains(err.Error(), tc.want) || *starts != 0 {
				t.Fatalf("configuration admitted or wrong gate: %v starts=%d", err, *starts)
			}
			if _, err := os.Stat(filepath.Join(m.Root, r.Workflow, "active")); !os.IsNotExist(err) {
				t.Fatal("invalid source configuration acquired a lease")
			}
		})
	}
}

func TestCatalogValidationRejectsMalformedScopeAndCertificate(t *testing.T) {
	for _, mode := range []string{"configuration", "certificate"} {
		t.Run(mode, func(t *testing.T) {
			m, r, starts := catalogManager(t)
			if mode == "configuration" {
				r.Configuration = json.RawMessage(`{}`)
			} else {
				r.Secrets["AGEFREIGHTER_SOURCE_CA_PEM"] = "invalid certificate"
			}
			if _, err := m.Submit(t.Context(), r); err == nil || *starts != 0 {
				t.Fatal("unreviewed catalog request started")
			}
		})
	}
}

func TestCapabilityControlRejectsMalformedImportAndRecoveryCredentials(t *testing.T) {
	r := csvRequest("id\n1\n")
	r.Offset = 1
	wire, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Decode(bytes.NewReader(wire)); err == nil {
		t.Fatal("CSV import accepted unrelated offset")
	}
	_, r, _ = resumeFixture(t)
	r.Secrets["AGEFREIGHTER_TARGET_DSN"] = "invalid"
	wire, err = json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Decode(bytes.NewReader(wire)); err == nil {
		t.Fatal("recovery accepted unreviewed target credentials")
	}
	export := ReportExport{URL: exportCapability("c", time.Now()) + "&sig=bad;query", SHA256: strings.Repeat("a", 64), Bytes: 1}
	if err := validateReportExport(workflowID, operationID, export, time.Now()); err == nil || strings.Contains(err.Error(), "bad;query") {
		t.Fatalf("unsafe export query handling: %v", err)
	}
}
