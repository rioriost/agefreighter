package config

import (
	"fmt"
	"slices"
	"strings"
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

// Gremlin properties are discovered, not explicitly projected. Declarations
// apply only when a user property exists; they never synthesize missing values.
func ValidateCosmosGremlinPropertyTypes(partition string, maximum int, types map[string]string) error {
	if len(types) > maximum || len(types) > 1024 {
		return fmt.Errorf("too many Gremlin property type declarations")
	}
	keys := make([]string, 0, len(types))
	for key := range types {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		if !validDiscoveryIdentifier(key) || key == "id" || key == "label" || key == partition || strings.HasPrefix(key, "_") {
			return fmt.Errorf("Gremlin propertyTypes must name non-structural user properties")
		}
	}
	return validateMappedPropertyTypes(types, types, "Cosmos Gremlin")
}

func validateCosmosDocumentPropertyTypes(format CosmosDocumentFormat, partition string, maximum int, properties, types map[string]string) error {
	if format == CosmosDocumentGremlin {
		return ValidateCosmosGremlinPropertyTypes(partition, maximum, types)
	}
	return ValidateCosmosPropertyTypes(properties, types)
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
