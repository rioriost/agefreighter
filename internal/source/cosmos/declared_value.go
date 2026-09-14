package cosmos

import (
	"encoding/json"
	"errors"
	"math"
	"math/big"
	"strconv"
	"strings"

	"github.com/rioriost/agefreighter/pkg/model"
)

// Undeclared fields keep legacy JSON inference. Declared types preserve schema
// intent when Cosmos serializes an integral float such as 1.0 as the number 1.
// No string-to-number/boolean coercion, truncation or recursive array flattening.
func convertDeclaredValue(raw any, declared string) (model.Value, error) {
	if declared == "" {
		return convertValue(raw, 0)
	}
	if raw == nil {
		return model.Value{Kind: model.ValueNull}, nil
	}
	if strings.HasSuffix(declared, "[]") {
		items, ok := raw.([]any)
		if !ok {
			return model.Value{}, errors.New("declared array requires a JSON array")
		}
		values := make([]model.Value, len(items))
		for i, item := range items {
			v, err := convertDeclaredValue(item, strings.TrimSuffix(declared, "[]"))
			if err != nil {
				return model.Value{}, err
			}
			values[i] = v
		}
		return model.Value{Kind: model.ValueList, List: values}, nil
	}
	switch declared {
	case "string":
		if _, ok := raw.(string); ok {
			return convertValue(raw, 0)
		}
	case "boolean":
		if _, ok := raw.(bool); ok {
			return convertValue(raw, 0)
		}
	case "int64", "float64":
		n, ok := raw.(json.Number)
		if !ok {
			break
		}
		text := n.String()
		// Bound exact arithmetic even for malicious exponent/mantissa shapes.
		if len(text) > 128 {
			break
		}
		if at := strings.IndexAny(text, "eE"); at >= 0 {
			exponent, err := strconv.Atoi(text[at+1:])
			if err != nil || exponent < -400 || exponent > 400 {
				break
			}
		}
		rational, ok := new(big.Rat).SetString(text)
		if !ok {
			break
		}
		if declared == "int64" {
			if !rational.IsInt() || !rational.Num().IsInt64() {
				break
			}
			return model.Value{Kind: model.ValueInteger, Integer: rational.Num().Int64()}, nil
		}
		f, err := strconv.ParseFloat(text, 64)
		if err != nil || math.IsNaN(f) || math.IsInf(f, 0) || (f == 0 && rational.Sign() != 0) {
			break
		}
		// Integer-valued source numbers must not silently lose integer precision.
		if rational.IsInt() && new(big.Rat).SetFloat64(f).Cmp(rational) != 0 {
			break
		}
		return model.Value{Kind: model.ValueFloat, Float: f}, nil
	}
	return model.Value{}, errors.New("JSON value does not fit the declared Cosmos property type")
}
