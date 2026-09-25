package runner

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/binary"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgproto3"
	"github.com/jackc/pgx/v5/pgtype"
	"golang.org/x/net/dns/dnsmessage"
)

// This wire double exercises the real pgx/TLS and read-only transaction paths,
// not database qualification. DNS is intercepted in-process and can resolve
// only this fixture to loopback; no cloud or operating-system DNS changes occur.
type targetFixture struct {
	t         *testing.T
	mode      string
	graph     string
	digest    string
	committed bool
	queries   chan string
}

type targetResult struct {
	key    string
	values []any
}

func newTargetFixture(t *testing.T, mode, graph, digest string, committed bool) *targetFixture {
	t.Helper()
	// migrationConnection requires port 5432 and its callers connect directly
	// through pgx. DNS can redirect the hostname, not the port: an ephemeral
	// listener would require a new production dial seam or a weaker DSN policy.
	// Never connect to or alter another service to make the fixture fit.
	listener, err := net.Listen("tcp4", "127.0.0.1:5432")
	if err != nil {
		t.Fatalf("target TLS fixture requires exclusive loopback port 5432; no external database will be used: %v", err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	const hostname = "afpg-test.postgres.database.azure.com"
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: hostname},
		DNSNames: []string{hostname}, NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour),
		IsCA: true, BasicConstraintsValid: true, KeyUsage: x509.KeyUsageCertSign | x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	ca := filepath.Join(t.TempDir(), "target-ca.pem")
	if err := os.WriteFile(ca, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PGSSLROOTCERT", ca)
	t.Setenv("PGSSLCERT", "")
	t.Setenv("PGSSLKEY", "")
	t.Setenv("PGSERVICE", "")
	oldResolver := net.DefaultResolver
	net.DefaultResolver = &net.Resolver{
		PreferGo: true,
		Dial: func(ctx context.Context, _, _ string) (net.Conn, error) {
			client, server := net.Pipe()
			go func() {
				defer server.Close()
				_ = server.SetDeadline(time.Now().Add(5 * time.Second))
				var length [2]byte
				if _, err := io.ReadFull(server, length[:]); err != nil {
					return
				}
				data := make([]byte, binary.BigEndian.Uint16(length[:]))
				if _, err := io.ReadFull(server, data); err != nil {
					return
				}
				var request dnsmessage.Message
				if request.Unpack(data) != nil || len(request.Questions) != 1 {
					return
				}
				q := request.Questions[0]
				response := dnsmessage.Message{Header: dnsmessage.Header{ID: request.ID, Response: true, RecursionAvailable: true}, Questions: request.Questions}
				if q.Name.String() != hostname+"." {
					response.RCode = dnsmessage.RCodeNameError
				} else if q.Type == dnsmessage.TypeA {
					response.Answers = []dnsmessage.Resource{{
						Header: dnsmessage.ResourceHeader{Name: q.Name, Type: dnsmessage.TypeA, Class: dnsmessage.ClassINET, TTL: 1},
						Body:   &dnsmessage.AResource{A: [4]byte{127, 0, 0, 1}},
					}}
				}
				data, err := response.Pack()
				if err != nil {
					return
				}
				binary.BigEndian.PutUint16(length[:], uint16(len(data)))
				_, _ = server.Write(append(length[:], data...))
			}()
			return client, nil
		},
	}
	t.Cleanup(func() { net.DefaultResolver = oldResolver })
	f := &targetFixture{t: t, mode: mode, graph: graph, digest: digest, committed: committed, queries: make(chan string, 32)}
	done := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err == nil {
			defer conn.Close()
			_ = conn.SetDeadline(time.Now().Add(10 * time.Second))
			err = f.serve(conn, &tls.Config{Certificates: []tls.Certificate{{Certificate: [][]byte{der}, PrivateKey: key}}, MinVersion: tls.VersionTLS12})
		}
		done <- err
	}()
	t.Cleanup(func() {
		_ = listener.Close()
		if err := <-done; err != nil && !errors.Is(err, net.ErrClosed) {
			t.Errorf("target wire fixture: %v", err)
		}
	})
	return f
}

func (f *targetFixture) result(query string) targetResult {
	now := time.Now().UTC().Add(-time.Minute)
	switch {
	case strings.HasPrefix(query, "begin"):
		return targetResult{key: "begin"}
	case query == "rollback":
		return targetResult{key: "rollback"}
	case strings.Contains(query, "server_version_num"):
		version, encrypted := int64(180001), true
		if f.mode == "old-version" {
			version = 170000
		}
		if f.mode == "unencrypted" {
			encrypted = false
		}
		return targetResult{"version", []any{version, encrypted}}
	case query == "CREATE EXTENSION IF NOT EXISTS age":
		return targetResult{key: "extension"}
	case strings.Contains(query, "shared_preload_libraries"):
		return targetResult{"preload", []any{"age"}}
	case strings.HasPrefix(query, "SET search_path"):
		return targetResult{key: "search-path"}
	case strings.Contains(query, "SELECT EXISTS"):
		return targetResult{"graph-exists", []any{f.mode == "existing-graph"}}
	case strings.Contains(query, "FROM agefreighter_meta.load_job AS job"):
		status, fingerprint := "failed", strings.Repeat("a", 64)
		if f.committed {
			status = "committed"
		}
		if f.mode == "identity-mismatch" {
			fingerprint = strings.Repeat("f", 64)
		}
		return targetResult{"job", []any{
			operationID, "wire fixture", "csv", "create", "apache-age", "", f.graph,
			"", fingerprint, status, int64(17), int64(1), "", int64(1), int64(12),
			int64(0), int64(0), "", now, now, now, nil, nil,
		}}
	case strings.Contains(query, "FROM agefreighter_meta.graph_generation"):
		state := "loading"
		if f.committed {
			state = "active"
		}
		return targetResult{"generation", []any{int64(17), operationID, f.graph, int64(100), int64(101), int64(0), int64(1), state, now, now}}
	case strings.Contains(query, "FROM agefreighter_meta.job_verification"):
		digest := f.digest
		if f.mode == "mapping-mismatch" {
			digest = strings.Repeat("f", 64)
		}
		return targetResult{"verification", []any{operationID, digest, strings.Repeat("b", 64), "{}"}}
	default:
		f.t.Errorf("unreviewed target query: %q", query)
		return targetResult{key: "unexpected"}
	}
}

func fixtureFields(values []any, formats []int16) []pgproto3.FieldDescription {
	fields := make([]pgproto3.FieldDescription, len(values))
	for i, value := range values {
		oid := uint32(pgtype.TextOID)
		switch value.(type) {
		case int64:
			oid = pgtype.Int8OID
		case bool:
			oid = pgtype.BoolOID
		case time.Time, nil:
			oid = pgtype.TimestamptzOID
		}
		var format int16
		if len(formats) == 1 {
			format = formats[0]
		} else if len(formats) > i {
			format = formats[i]
		}
		fields[i] = pgproto3.FieldDescription{Name: []byte(fmt.Sprintf("column%d", i)), DataTypeOID: oid, DataTypeSize: -1, TypeModifier: -1, Format: format}
	}
	return fields
}

func (f *targetFixture) serve(conn net.Conn, config *tls.Config) error {
	backend := pgproto3.NewBackend(conn, conn)
	message, err := backend.ReceiveStartupMessage()
	if err != nil {
		return err
	}
	if _, ok := message.(*pgproto3.SSLRequest); !ok {
		return fmt.Errorf("target did not negotiate verified TLS: %T", message)
	}
	if _, err := conn.Write([]byte("S")); err != nil {
		return err
	}
	secure := tls.Server(conn, config)
	if err := secure.Handshake(); err != nil {
		return err
	}
	backend = pgproto3.NewBackend(secure, secure)
	message, err = backend.ReceiveStartupMessage()
	if err != nil {
		return err
	}
	startup, ok := message.(*pgproto3.StartupMessage)
	if !ok || startup.Parameters["user"] != "afadmin" || startup.Parameters["database"] != "agefreighter" {
		return fmt.Errorf("unexpected startup: %T", message)
	}
	if f.mode == "authentication" {
		backend.Send(&pgproto3.ErrorResponse{Severity: "FATAL", Code: "28P01", Message: "PRIVATE-TARGET-DETAIL"})
		return backend.Flush()
	}
	backend.Send(&pgproto3.AuthenticationOk{})
	backend.Send(&pgproto3.ParameterStatus{Name: "server_version", Value: "18.1"})
	backend.Send(&pgproto3.ReadyForQuery{TxStatus: 'I'})
	if err := backend.Flush(); err != nil {
		return err
	}
	statements := map[string]string{}
	var boundQuery string
	var formats []int16
	types := pgtype.NewMap()
	execute := func(query string, simple bool) error {
		f.queries <- query
		result := f.result(query)
		if f.mode == result.key {
			backend.Send(&pgproto3.ErrorResponse{Severity: "ERROR", Code: "42501", Message: "PRIVATE-TARGET-DETAIL"})
			return nil
		}
		if len(result.values) != 0 {
			fields := fixtureFields(result.values, formats)
			if simple {
				backend.Send(&pgproto3.RowDescription{Fields: fields})
			}
			row := make([][]byte, len(fields))
			for i, field := range fields {
				value, err := types.Encode(field.DataTypeOID, field.Format, result.values[i], nil)
				if err != nil {
					return err
				}
				if value == nil && result.values[i] != nil {
					value = []byte{}
				}
				row[i] = value
			}
			backend.Send(&pgproto3.DataRow{Values: row})
		}
		backend.Send(&pgproto3.CommandComplete{CommandTag: []byte("SELECT 1")})
		return nil
	}
	for {
		message, err := backend.Receive()
		if err != nil {
			return err
		}
		switch msg := message.(type) {
		case *pgproto3.Parse:
			statements[msg.Name] = msg.Query
			backend.Send(&pgproto3.ParseComplete{})
		case *pgproto3.Describe:
			query := statements[msg.Name]
			var parameters []uint32
			if strings.Contains(query, "$1") {
				parameters = []uint32{pgtype.TextOID}
			}
			backend.Send(&pgproto3.ParameterDescription{ParameterOIDs: parameters})
			result := f.result(query)
			if len(result.values) == 0 {
				backend.Send(&pgproto3.NoData{})
			} else {
				backend.Send(&pgproto3.RowDescription{Fields: fixtureFields(result.values, nil)})
			}
		case *pgproto3.Bind:
			boundQuery, formats = statements[msg.PreparedStatement], msg.ResultFormatCodes
			if strings.Contains(boundQuery, "agefreighter_meta.") && (len(msg.Parameters) != 1 || string(msg.Parameters[0]) != operationID) {
				return errors.New("metadata query was not bound to the original job")
			}
			if strings.Contains(boundQuery, "SELECT EXISTS") && (len(msg.Parameters) != 1 || string(msg.Parameters[0]) != f.graph) {
				return errors.New("graph existence query was not bound to the reviewed graph")
			}
			backend.Send(&pgproto3.BindComplete{})
		case *pgproto3.Close:
			delete(statements, msg.Name)
			backend.Send(&pgproto3.CloseComplete{})
		case *pgproto3.Execute:
			if err := execute(boundQuery, false); err != nil {
				return err
			}
		case *pgproto3.Query:
			formats = nil
			if err := execute(msg.String, true); err != nil {
				return err
			}
			backend.Send(&pgproto3.ReadyForQuery{TxStatus: 'I'})
			if f.mode == "begin" {
				// pgx discards a connection after a failed BEGIN. There is no
				// transaction to inspect or roll back on this connection.
				return backend.Flush()
			}
		case *pgproto3.Sync:
			backend.Send(&pgproto3.ReadyForQuery{TxStatus: 'I'})
		case *pgproto3.Terminate:
			return nil
		default:
			return fmt.Errorf("unexpected frontend message: %T", msg)
		}
		if err := backend.Flush(); err != nil {
			return err
		}
	}
}

func (f *targetFixture) assertReadOnly(t *testing.T) {
	t.Helper()
	var queries []string
	for len(f.queries) != 0 {
		queries = append(queries, <-f.queries)
	}
	if len(queries) < 2 || queries[0] != "begin isolation level repeatable read read only" || queries[len(queries)-1] != "rollback" {
		t.Fatalf("inspection transaction not bounded and read-only: %q", queries)
	}
	for _, query := range queries[1 : len(queries)-1] {
		if !strings.HasPrefix(strings.TrimSpace(query), "SELECT") {
			t.Fatalf("inspection issued a write: %q", query)
		}
	}
}
