package cosmos

import "testing"

func TestResumeTokenRoundTrip(t *testing.T) {
	state := resumeState{
		fingerprint:     "abc123",
		mappingIndex:    2,
		mappingKind:     edgeMapping,
		hasContinuation: true,
		continuation:    "cont-token-value",
		consumed:        7,
		rejected:        3,
	}
	token := formatResumeToken(state)
	if token == "" {
		t.Fatal("formatResumeToken: expected non-empty token")
	}
	parsed, err := parseResumeToken(token)
	if err != nil {
		t.Fatalf("parseResumeToken: %v", err)
	}
	if parsed != state {
		t.Errorf("parseResumeToken round-trip = %+v, want %+v", parsed, state)
	}
}

func TestResumeTokenOpaqueAndVersioned(t *testing.T) {
	token := formatResumeToken(resumeState{mappingKind: vertexMapping})
	if len(token) < len(resumeTokenPrefix) || token[:len(resumeTokenPrefix)] != resumeTokenPrefix {
		t.Fatalf("formatResumeToken: token does not carry the expected version prefix: %q", token)
	}
}

func TestParseResumeTokenRejectsMalformedInput(t *testing.T) {
	cases := map[string]string{
		"wrong prefix":     "not-a-cosmos-token",
		"bad base64":       resumeTokenPrefix + "***not-base64***",
		"trailing content": resumeTokenPrefix + "eyJ2IjoxfQ" + "eyJ2IjoxfQ",
	}
	for name, token := range cases {
		if _, err := parseResumeToken(token); err == nil {
			t.Errorf("%s: expected parseResumeToken to reject %q", name, token)
		}
	}
}

func TestParseResumeTokenRejectsInvalidPayloadFields(t *testing.T) {
	base := resumeState{fingerprint: "fp", mappingIndex: 0, mappingKind: vertexMapping}

	negativeMapping := base
	negativeMapping.mappingIndex = -1
	if _, err := parseResumeToken(formatResumeToken(negativeMapping)); err == nil {
		t.Error("expected negative mapping index to be rejected")
	}

	negativeConsumed := base
	negativeConsumed.consumed = -1
	if _, err := parseResumeToken(formatResumeToken(negativeConsumed)); err == nil {
		t.Error("expected negative consumed count to be rejected")
	}

	negativeRejected := base
	negativeRejected.rejected = -1
	if _, err := parseResumeToken(formatResumeToken(negativeRejected)); err == nil {
		t.Error("expected negative rejected count to be rejected")
	}
}

func TestParseResumeTokenRejectsInvalidMappingKind(t *testing.T) {
	// mappingKind(99) has no String() case, so it renders as "unknown",
	// which parseResumeToken must reject.
	token := formatResumeToken(resumeState{mappingKind: mappingKind(99)})
	if _, err := parseResumeToken(token); err == nil {
		t.Error("expected invalid mapping kind to be rejected")
	}
}

func TestParseResumeTokenRejectsUnsupportedVersion(t *testing.T) {
	token := resumeTokenPrefix + "eyJmcCI6ImZwIiwibWkiOjAsIm1rIjoidmVydGV4IiwidiI6Mn0"
	if _, err := parseResumeToken(token); err == nil {
		t.Error("expected unsupported version to be rejected")
	}
}
