package postgres

import (
	"bytes"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

const (
	resumeTokenVersion  = 1
	resumeTokenPrefix   = "postgresql:v1:"
	maxResumeTokenBytes = 64 << 10
	maxResumeKeyBytes   = 16 << 10
)

type resumeKey struct {
	Type  string `json:"t"`
	Value string `json:"x"`
}

type resumeTokenPayload struct {
	Version      int        `json:"v"`
	Fingerprint  string     `json:"fp"`
	MappingIndex int        `json:"mi"`
	MappingKind  string     `json:"mk"`
	Consumed     int64      `json:"n"`
	Rejected     int        `json:"r"`
	Key          *resumeKey `json:"k,omitempty"`
}

type resumeState struct {
	fingerprint  string
	mappingIndex int
	mappingKind  mappingKind
	consumed     int64
	rejected     int
	key          *keyValue
}

func formatResumeToken(state resumeState) (string, error) {
	var key *resumeKey
	if state.key != nil {
		if len(state.key.text) > maxResumeKeyBytes {
			return "", errors.New("PostgreSQL resume key is too large")
		}
		key = &resumeKey{Type: state.key.kind, Value: state.key.text}
	}
	encoded, err := json.Marshal(resumeTokenPayload{
		Version: resumeTokenVersion, Fingerprint: state.fingerprint,
		MappingIndex: state.mappingIndex, MappingKind: state.mappingKind.String(),
		Consumed: state.consumed, Rejected: state.rejected, Key: key,
	})
	if err != nil {
		return "", errors.New("encode PostgreSQL resume token")
	}
	token := resumeTokenPrefix + base64.RawURLEncoding.EncodeToString(encoded)
	if len(token) > maxResumeTokenBytes {
		return "", errors.New("PostgreSQL resume token is too large")
	}
	return token, nil
}

func parseResumeToken(token string) (resumeState, error) {
	if len(token) > maxResumeTokenBytes {
		return resumeState{}, errors.New("PostgreSQL resume token is too large")
	}
	if !strings.HasPrefix(token, resumeTokenPrefix) {
		return resumeState{}, errors.New("PostgreSQL resume token has an unrecognized format")
	}
	raw, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(token, resumeTokenPrefix))
	if err != nil {
		return resumeState{}, errors.New("PostgreSQL resume token is not valid base64url")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var payload resumeTokenPayload
	if err := decoder.Decode(&payload); err != nil {
		return resumeState{}, errors.New("PostgreSQL resume token payload is not valid")
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return resumeState{}, errors.New("PostgreSQL resume token has trailing content")
	}
	if payload.Version != resumeTokenVersion {
		return resumeState{}, fmt.Errorf(
			"PostgreSQL resume token version %d is not supported", payload.Version,
		)
	}
	fingerprintBytes, err := hex.DecodeString(payload.Fingerprint)
	if err != nil || len(fingerprintBytes) != 32 {
		return resumeState{}, errors.New("PostgreSQL resume token fingerprint is invalid")
	}
	if payload.MappingIndex < 0 {
		return resumeState{}, errors.New("PostgreSQL resume token mapping index is out of range")
	}
	var kind mappingKind
	switch payload.MappingKind {
	case "vertex":
		kind = vertexMapping
	case "edge":
		kind = edgeMapping
	default:
		return resumeState{}, errors.New("PostgreSQL resume token mapping kind is invalid")
	}
	if payload.Consumed < 0 {
		return resumeState{}, errors.New("PostgreSQL resume token consumed count is invalid")
	}
	if payload.Rejected < 0 {
		return resumeState{}, errors.New("PostgreSQL resume token rejected count is invalid")
	}
	var key *keyValue
	if payload.Key != nil {
		if len(payload.Key.Value) > maxResumeKeyBytes {
			return resumeState{}, errors.New("PostgreSQL resume key is too large")
		}
		parsed, err := parseStoredKey(payload.Key.Type, payload.Key.Value)
		if err != nil {
			return resumeState{}, err
		}
		key = &parsed
	}
	return resumeState{
		fingerprint: payload.Fingerprint, mappingIndex: payload.MappingIndex,
		mappingKind: kind, consumed: payload.Consumed,
		rejected: payload.Rejected, key: key,
	}, nil
}

func parseStoredKey(kind, value string) (keyValue, error) {
	if kind != keyNumber {
		return keyValue{}, errors.New("PostgreSQL resume key type is invalid")
	}
	return parseNumberKey(value)
}
