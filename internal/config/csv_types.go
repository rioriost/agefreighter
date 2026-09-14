package config

import (
	"fmt"
	"slices"
)

// ValidateCSVPropertyTypes also protects callers constructing iterators directly.
// Types are keyed by destination property, not by CSV column.
func ValidateCSVPropertyTypes(properties, types map[string]string) error {
	return validateMappedPropertyTypes(properties, types, "CSV")
}

// ValidateCosmosPropertyTypes requires explicit types to refer to mapped fields.
func ValidateCosmosPropertyTypes(properties, types map[string]string) error {
	return validateMappedPropertyTypes(properties, types, "Cosmos")
}

func validateMappedPropertyTypes(properties, types map[string]string, source string) error {
	keys := make([]string, 0, len(types))
	for key := range types {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		if _, ok := properties[key]; !ok {
			return fmt.Errorf("propertyTypes references unmapped property %q", key)
		}
		switch types[key] {
		case "string", "int64", "float64", "boolean", "string[]", "int64[]", "float64[]", "boolean[]":
		default:
			return fmt.Errorf("unsupported %s property type %q for %q", source, types[key], key)
		}
	}
	return nil
}
