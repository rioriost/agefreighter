package age

import (
	"encoding/json"
	"fmt"

	"github.com/rioriost/agefreighter/pkg/model"
)

// maxPropertyDepth and ErrInvalidProperty are kept as package-level names for
// compatibility with existing callers and tests; the canonical encoding rules
// now live in pkg/model so other packages (for example internal/source/cosmos)
// can reuse them without depending on internal/age.
const maxPropertyDepth = model.MaxPropertyDepth

var ErrInvalidProperty = model.ErrInvalidValue

// EncodeProperties canonically encodes properties as a JSON object with
// lexicographically sorted keys. It delegates to pkg/model so the same
// canonical encoding rules are shared across the codebase.
func EncodeProperties(properties model.Properties) ([]byte, error) {
	return model.EncodeProperties(properties)
}

func loadProperties(properties model.Properties, encoded []byte) ([]byte, error) {
	if encoded == nil {
		return EncodeProperties(properties)
	}
	if !json.Valid(encoded) || len(encoded) < 2 ||
		encoded[0] != '{' || encoded[len(encoded)-1] != '}' {
		return nil, fmt.Errorf(
			"%w: encoded properties must be a JSON object",
			ErrInvalidProperty,
		)
	}
	return encoded, nil
}
