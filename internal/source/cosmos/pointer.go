package cosmos

import (
	"fmt"
	"strconv"
	"strings"
)

// pointer is a parsed RFC 6901 JSON Pointer used to project a field out of a
// decoded Cosmos DB document.
type pointer struct {
	raw    string
	tokens []string
}

// parsePointer parses and validates an RFC 6901 JSON Pointer. The pointer
// must be non-empty and start with "/"; every "~" escape must be followed
// by "0" or "1". config.validateJSONPointer performs the same syntax check
// at configuration time, so this should only fail for pointers that bypass
// configuration validation (for example when an Iterator is built directly
// in tests).
func parsePointer(raw string) (pointer, error) {
	if raw == "" {
		return pointer{}, fmt.Errorf("JSON pointer must not be empty")
	}
	if raw[0] != '/' {
		return pointer{}, fmt.Errorf("JSON pointer %q must start with /", raw)
	}
	parts := strings.Split(raw[1:], "/")
	tokens := make([]string, len(parts))
	for index, part := range parts {
		for cursor := 0; cursor < len(part); cursor++ {
			if part[cursor] != '~' {
				continue
			}
			if cursor+1 >= len(part) || (part[cursor+1] != '0' && part[cursor+1] != '1') {
				return pointer{}, fmt.Errorf("JSON pointer %q has an invalid ~ escape", raw)
			}
		}
		part = strings.ReplaceAll(part, "~1", "/")
		part = strings.ReplaceAll(part, "~0", "~")
		tokens[index] = part
	}
	return pointer{raw: raw, tokens: tokens}, nil
}

// resolve walks document (the generic tree produced by decodeDocument)
// following the pointer's tokens, returning the value at that location and
// true, or false if any segment of the path does not exist.
func (p pointer) resolve(document any) (any, bool) {
	current := document
	for _, token := range p.tokens {
		switch node := current.(type) {
		case map[string]any:
			value, ok := node[token]
			if !ok {
				return nil, false
			}
			current = value
		case []any:
			index, err := strconv.Atoi(token)
			if err != nil || index < 0 || index >= len(node) {
				return nil, false
			}
			current = node[index]
		default:
			return nil, false
		}
	}
	return current, true
}
