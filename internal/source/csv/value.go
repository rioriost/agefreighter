package csv

import (
	"encoding/json"
	"errors"
	"math"
	"strconv"
	"strings"

	"github.com/rioriost/agefreighter/pkg/model"
)

// decodeCSVValue deliberately does not infer types or include cell contents in
// errors: both coercion and credential/property leaks would be surprising.
func decodeCSVValue(text, valueType string) (model.Value, error) {
	if strings.HasSuffix(valueType, "[]") {
		var cells []json.RawMessage
		if !strings.HasPrefix(strings.TrimSpace(text), "[") || json.Unmarshal([]byte(text), &cells) != nil {
			return model.Value{}, errors.New("expected a JSON array")
		}
		items := make([]model.Value, len(cells))
		base := strings.TrimSuffix(valueType, "[]")
		for index, cell := range cells {
			value := string(cell)
			if base == "string" {
				if value == "null" || json.Unmarshal(cell, &value) != nil {
					return model.Value{}, errors.New("expected a JSON string array item")
				}
			}
			item, err := decodeCSVValue(value, base)
			if err != nil {
				return model.Value{}, err
			}
			items[index] = item
		}
		return model.Value{Kind: model.ValueList, List: items}, nil
	}
	switch valueType {
	case "", "string":
		return model.Value{Kind: model.ValueString, String: text}, nil
	case "int64":
		value, err := strconv.ParseInt(text, 10, 64)
		if err != nil {
			return model.Value{}, errors.New("expected a signed 64-bit integer")
		}
		return model.Value{Kind: model.ValueInteger, Integer: value}, nil
	case "float64":
		value, err := strconv.ParseFloat(text, 64)
		if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
			return model.Value{}, errors.New("expected a finite 64-bit float")
		}
		return model.Value{Kind: model.ValueFloat, Float: value}, nil
	case "boolean":
		if text != "true" && text != "false" {
			return model.Value{}, errors.New("expected true or false")
		}
		return model.Value{Kind: model.ValueBoolean, Boolean: text == "true"}, nil
	default:
		return model.Value{}, errors.New("unsupported CSV property type")
	}
}
