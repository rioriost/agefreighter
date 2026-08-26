package neo4j

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
	resumeTokenPrefix   = "neo4j:v1:"
	maxResumeTokenBytes = 64 << 10
)

type resumeTokenPayload struct {
	Version      int    `json:"v"`
	Fingerprint  string `json:"fp"`
	MappingIndex int    `json:"mi"`
	MappingKind  string `json:"mk"`
	Consumed     int64  `json:"n"`
	Rejected     int    `json:"r"`
	LastKey      *int64 `json:"k,omitempty"`
}

type resumeState struct {
	fingerprint  string
	mappingIndex int
	mappingKind  mappingKind
	consumed     int64
	rejected     int
	lastKey      *int64
}

func formatResumeToken(state resumeState) (string, error) {
	encoded, err := json.Marshal(resumeTokenPayload{
		Version: resumeTokenVersion, Fingerprint: state.fingerprint,
		MappingIndex: state.mappingIndex, MappingKind: state.mappingKind.String(),
		Consumed: state.consumed, Rejected: state.rejected, LastKey: state.lastKey,
	})
	if err != nil {
		return "", errors.New("encode Neo4j resume token")
	}
	token := resumeTokenPrefix + base64.RawURLEncoding.EncodeToString(encoded)
	if len(token) > maxResumeTokenBytes {
		return "", errors.New("Neo4j resume token is too large")
	}
	return token, nil
}

func parseResumeToken(token string) (resumeState, error) {
	if len(token) > maxResumeTokenBytes {
		return resumeState{}, errors.New("Neo4j resume token is too large")
	}
	if !strings.HasPrefix(token, resumeTokenPrefix) {
		return resumeState{}, errors.New("Neo4j resume token has an unrecognized format")
	}
	raw, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(token, resumeTokenPrefix))
	if err != nil {
		return resumeState{}, errors.New("Neo4j resume token is not valid base64url")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var payload resumeTokenPayload
	if err := decoder.Decode(&payload); err != nil {
		return resumeState{}, errors.New("Neo4j resume token payload is not valid")
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return resumeState{}, errors.New("Neo4j resume token has trailing content")
	}
	if payload.Version != resumeTokenVersion {
		return resumeState{}, fmt.Errorf("Neo4j resume token version %d is not supported", payload.Version)
	}
	fingerprint, err := hex.DecodeString(payload.Fingerprint)
	if err != nil || len(fingerprint) != sha256Size {
		return resumeState{}, errors.New("Neo4j resume token fingerprint is invalid")
	}
	if payload.MappingIndex < 0 {
		return resumeState{}, errors.New("Neo4j resume token mapping index is out of range")
	}
	var kind mappingKind
	switch payload.MappingKind {
	case "vertex":
		kind = vertexMapping
	case "edge":
		kind = edgeMapping
	default:
		return resumeState{}, errors.New("Neo4j resume token mapping kind is invalid")
	}
	if payload.Consumed < 1 {
		return resumeState{}, errors.New("Neo4j resume token consumed count is invalid")
	}
	if payload.Rejected < 0 {
		return resumeState{}, errors.New("Neo4j resume token rejected count is invalid")
	}
	if payload.LastKey == nil {
		return resumeState{}, errors.New("Neo4j resume token key is inconsistent")
	}
	return resumeState{
		fingerprint: payload.Fingerprint, mappingIndex: payload.MappingIndex,
		mappingKind: kind, consumed: payload.Consumed, rejected: payload.Rejected,
		lastKey: payload.LastKey,
	}, nil
}

const sha256Size = 32
