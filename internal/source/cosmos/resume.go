package cosmos

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// resumeTokenVersion is bumped whenever the resume token's payload shape
// changes in a way that is not backward compatible.
const resumeTokenVersion = 1

// resumeTokenPrefix makes the token self-describing (versioned) while
// remaining opaque to callers; it is not intended to be parsed by anything
// other than parseResumeToken.
const resumeTokenPrefix = "cosmos-nosql:v1:"

const maxResumeTokenBytes = 256 << 10

// resumeTokenPayload is the JSON shape embedded (base64url-encoded) in a
// resume token. It never contains document content; Continuation is the
// Cosmos continuation token needed to reopen the exact page currently being
// processed (empty/HasContinuation=false for a mapping's first page).
type resumeTokenPayload struct {
	Version         int    `json:"v"`
	Fingerprint     string `json:"fp"`
	MappingIndex    int    `json:"mi"`
	MappingKind     string `json:"mk"`
	HasContinuation bool   `json:"hc,omitempty"`
	Continuation    string `json:"c,omitempty"`
	Consumed        int    `json:"n"`
	Rejected        int    `json:"r"`
}

// resumeState is the decoded, validated form of a resumeTokenPayload used
// internally by Iterator.
type resumeState struct {
	fingerprint     string
	mappingIndex    int
	mappingKind     mappingKind
	hasContinuation bool
	continuation    string
	consumed        int
	rejected        int
}

// formatResumeToken renders state as a versioned, opaque resume token. The
// caller must never log or print the returned value; it should only be
// surfaced through model.SourcePosition.Token for checkpointing.
func formatResumeToken(state resumeState) string {
	payload := resumeTokenPayload{
		Version:         resumeTokenVersion,
		Fingerprint:     state.fingerprint,
		MappingIndex:    state.mappingIndex,
		MappingKind:     state.mappingKind.String(),
		HasContinuation: state.hasContinuation,
		Continuation:    state.continuation,
		Consumed:        state.consumed,
		Rejected:        state.rejected,
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		// payload only contains builtin types, so this can never fail.
		panic("cosmos: resume token payload encoding failed: " + err.Error())
	}
	return resumeTokenPrefix + base64.RawURLEncoding.EncodeToString(encoded)
}

// parseResumeToken decodes and validates a resume token previously produced
// by formatResumeToken. Error messages never include the raw token value.
func parseResumeToken(token string) (resumeState, error) {
	if len(token) > maxResumeTokenBytes {
		return resumeState{}, errors.New("Cosmos resume token is too large")
	}
	if !strings.HasPrefix(token, resumeTokenPrefix) {
		return resumeState{}, errors.New("Cosmos resume token has an unrecognized format")
	}
	raw, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(token, resumeTokenPrefix))
	if err != nil {
		return resumeState{}, errors.New("Cosmos resume token is not valid base64url")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var payload resumeTokenPayload
	if err := decoder.Decode(&payload); err != nil {
		return resumeState{}, errors.New("Cosmos resume token payload is not valid")
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return resumeState{}, errors.New("Cosmos resume token has trailing content")
	}
	if payload.Version != resumeTokenVersion {
		return resumeState{}, fmt.Errorf(
			"Cosmos resume token version %d is not supported", payload.Version,
		)
	}
	if payload.MappingIndex < 0 {
		return resumeState{}, errors.New("Cosmos resume token mapping index is out of range")
	}
	var kind mappingKind
	switch payload.MappingKind {
	case "vertex":
		kind = vertexMapping
	case "edge":
		kind = edgeMapping
	default:
		return resumeState{}, errors.New("Cosmos resume token mapping kind is invalid")
	}
	if payload.Consumed < 0 {
		return resumeState{}, errors.New("Cosmos resume token consumed count is invalid")
	}
	if payload.Rejected < 0 {
		return resumeState{}, errors.New("Cosmos resume token rejected count is invalid")
	}
	return resumeState{
		fingerprint:     payload.Fingerprint,
		mappingIndex:    payload.MappingIndex,
		mappingKind:     kind,
		hasContinuation: payload.HasContinuation,
		continuation:    payload.Continuation,
		consumed:        payload.Consumed,
		rejected:        payload.Rejected,
	}, nil
}
