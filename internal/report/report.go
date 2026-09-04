package report

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/rioriost/agefreighter/internal/version"
)

const (
	SchemaVersion       = 1
	MaxChecks           = 256
	MaxFindings         = 128
	MaxSections         = 64
	MaxFieldsPerSection = 256
	MaxTextBytes        = 8192
	MaxOutputBytes      = 4 << 20
)

type Outcome string

const (
	OutcomePass       Outcome = "pass"
	OutcomeFail       Outcome = "fail"
	OutcomeIncomplete Outcome = "incomplete"
)

type CheckStatus string

const (
	CheckPass        CheckStatus = "pass"
	CheckFail        CheckStatus = "fail"
	CheckWarning     CheckStatus = "warning"
	CheckUnknown     CheckStatus = "unknown"
	CheckUnavailable CheckStatus = "unavailable"
)

type Document struct {
	SchemaVersion       int       `json:"schemaVersion"`
	Command             string    `json:"command"`
	AgefreighterVersion string    `json:"agefreighterVersion"`
	GeneratedAt         time.Time `json:"generatedAt"`
	Outcome             Outcome   `json:"outcome"`
	Job                 *Job      `json:"job,omitempty"`
	Target              *Target   `json:"target,omitempty"`
	Checks              []Check   `json:"checks"`
	Warnings            []Finding `json:"warnings"`
	Errors              []Finding `json:"errors"`
	IncompleteChecks    []string  `json:"incompleteChecks"`
	Sections            []Section `json:"sections"`
}

type Job struct {
	ID                string `json:"id"`
	ConfigFingerprint string `json:"configFingerprint"`
}

type Target struct {
	PostgreSQL VersionValue `json:"postgresql"`
	AGE        VersionValue `json:"age"`
}

type VersionValue struct {
	Value  string      `json:"value,omitempty"`
	Status CheckStatus `json:"status"`
}

type Check struct {
	ID      string      `json:"id"`
	Status  CheckStatus `json:"status"`
	Summary string      `json:"summary"`
	Detail  string      `json:"detail,omitempty"`
}

type Finding struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type Section struct {
	Title  string  `json:"title"`
	Fields []Field `json:"fields"`
}

type Field struct {
	Name   string      `json:"name"`
	Value  string      `json:"value,omitempty"`
	Status CheckStatus `json:"status"`
}

func New(command string, generatedAt time.Time) Document {
	return Document{
		SchemaVersion:       SchemaVersion,
		Command:             command,
		AgefreighterVersion: version.Current().Version,
		GeneratedAt:         generatedAt,
		Outcome:             OutcomeIncomplete,
		Checks:              []Check{},
		Warnings:            []Finding{},
		Errors:              []Finding{},
		IncompleteChecks:    []string{},
		Sections:            []Section{},
	}
}

// Decode validates one bounded JSON report document and rejects unknown fields.
func Decode(data []byte) (Document, error) {
	if len(data) > MaxOutputBytes {
		return Document{}, fmt.Errorf("report input exceeds %d bytes", MaxOutputBytes)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var document Document
	if err := decoder.Decode(&document); err != nil {
		return Document{}, fmt.Errorf("decode report: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return Document{}, errors.New("report input must contain one JSON object")
	}
	if err := validate(document); err != nil {
		return Document{}, err
	}
	return document, nil
}

func canonical(document Document) (Document, error) {
	if err := validate(document); err != nil {
		return Document{}, err
	}
	result := document
	result.GeneratedAt = document.GeneratedAt.UTC()
	result.Checks = slices.Clone(document.Checks)
	result.Warnings = slices.Clone(document.Warnings)
	result.Errors = slices.Clone(document.Errors)
	result.IncompleteChecks = slices.Clone(document.IncompleteChecks)
	result.Sections = slices.Clone(document.Sections)
	for index := range result.Sections {
		result.Sections[index].Fields = slices.Clone(document.Sections[index].Fields)
		slices.SortFunc(result.Sections[index].Fields, func(left, right Field) int {
			return strings.Compare(left.Name, right.Name)
		})
	}
	slices.SortFunc(result.Checks, func(left, right Check) int {
		return strings.Compare(left.ID, right.ID)
	})
	sortFindings(result.Warnings)
	sortFindings(result.Errors)
	slices.Sort(result.IncompleteChecks)
	slices.SortFunc(result.Sections, func(left, right Section) int {
		return strings.Compare(left.Title, right.Title)
	})
	if result.Checks == nil {
		result.Checks = []Check{}
	}
	if result.Warnings == nil {
		result.Warnings = []Finding{}
	}
	if result.Errors == nil {
		result.Errors = []Finding{}
	}
	if result.IncompleteChecks == nil {
		result.IncompleteChecks = []string{}
	}
	if result.Sections == nil {
		result.Sections = []Section{}
	}
	return result, nil
}

func sortFindings(findings []Finding) {
	slices.SortFunc(findings, func(left, right Finding) int {
		if compared := strings.Compare(left.Code, right.Code); compared != 0 {
			return compared
		}
		return strings.Compare(left.Message, right.Message)
	})
}

func validate(document Document) error {
	if document.SchemaVersion != SchemaVersion {
		return fmt.Errorf("unsupported report schema version %d", document.SchemaVersion)
	}
	switch document.Command {
	case "report", "doctor", "verify", "profile", "inventory", "optimize", "check-cypher":
	default:
		return fmt.Errorf("unsupported report command %q", document.Command)
	}
	if err := validateText("agefreighterVersion", document.AgefreighterVersion, true); err != nil {
		return err
	}
	if document.GeneratedAt.IsZero() {
		return errors.New("generatedAt is required")
	}
	if len(document.Checks) > MaxChecks {
		return fmt.Errorf("report has more than %d checks", MaxChecks)
	}
	if len(document.Warnings) > MaxFindings || len(document.Errors) > MaxFindings {
		return fmt.Errorf("report has more than %d warnings or errors", MaxFindings)
	}
	if len(document.IncompleteChecks) > MaxChecks {
		return fmt.Errorf("report has more than %d incomplete checks", MaxChecks)
	}
	if len(document.Sections) > MaxSections {
		return fmt.Errorf("report has more than %d sections", MaxSections)
	}
	if len(document.Checks) == 0 && len(document.Sections) == 0 {
		return errors.New("report must contain at least one check or section")
	}
	switch document.Command {
	case "report":
		if document.Job == nil {
			return errors.New("migration report requires a job")
		}
		if document.Target == nil {
			return errors.New("migration report requires a target")
		}
	case "verify":
		if document.Job == nil {
			return errors.New("verification report requires a job")
		}
	case "doctor", "optimize":
		if document.Target == nil {
			return fmt.Errorf("%s report requires a target", document.Command)
		}
	case "profile", "inventory", "check-cypher":
	}
	if document.Job != nil {
		if !validUUID(document.Job.ID) {
			return errors.New("job ID must be a canonical UUID")
		}
		if !validFingerprint(document.Job.ConfigFingerprint) {
			return errors.New("config fingerprint must be 64 lowercase hexadecimal characters")
		}
	}
	hasFailed := false
	hasIncomplete := len(document.IncompleteChecks) > 0
	seenChecks := make(map[string]struct{}, len(document.Checks))
	for index, check := range document.Checks {
		if err := validateText("check ID", check.ID, true); err != nil {
			return fmt.Errorf("check %d: %w", index+1, err)
		}
		if _, exists := seenChecks[check.ID]; exists {
			return fmt.Errorf("duplicate check ID %q", check.ID)
		}
		seenChecks[check.ID] = struct{}{}
		if err := validateStatus(check.Status); err != nil {
			return fmt.Errorf("check %q: %w", check.ID, err)
		}
		if err := validateText("check summary", check.Summary, true); err != nil {
			return fmt.Errorf("check %q: %w", check.ID, err)
		}
		if err := validateText("check detail", check.Detail, false); err != nil {
			return fmt.Errorf("check %q: %w", check.ID, err)
		}
		hasFailed = hasFailed || check.Status == CheckFail
		hasIncomplete = hasIncomplete ||
			check.Status == CheckUnknown ||
			check.Status == CheckUnavailable
	}
	for _, finding := range append(slices.Clone(document.Warnings), document.Errors...) {
		if err := validateText("finding code", finding.Code, true); err != nil {
			return err
		}
		if err := validateText("finding message", finding.Message, true); err != nil {
			return err
		}
	}
	for _, name := range document.IncompleteChecks {
		if err := validateText("incomplete check", name, true); err != nil {
			return err
		}
	}
	if document.Target != nil {
		if err := validateVersionValue("PostgreSQL", document.Target.PostgreSQL); err != nil {
			return err
		}
		if err := validateVersionValue("Apache AGE", document.Target.AGE); err != nil {
			return err
		}
		hasFailed = hasFailed ||
			document.Target.PostgreSQL.Status == CheckFail ||
			document.Target.AGE.Status == CheckFail
		hasIncomplete = hasIncomplete ||
			document.Target.PostgreSQL.Status == CheckUnknown ||
			document.Target.PostgreSQL.Status == CheckUnavailable ||
			document.Target.AGE.Status == CheckUnknown ||
			document.Target.AGE.Status == CheckUnavailable
	}
	seenSections := make(map[string]struct{}, len(document.Sections))
	for _, section := range document.Sections {
		if err := validateText("section title", section.Title, true); err != nil {
			return err
		}
		if _, exists := seenSections[section.Title]; exists {
			return fmt.Errorf("duplicate section %q", section.Title)
		}
		seenSections[section.Title] = struct{}{}
		if len(section.Fields) > MaxFieldsPerSection {
			return fmt.Errorf(
				"section %q has more than %d fields",
				section.Title,
				MaxFieldsPerSection,
			)
		}
		seenFields := make(map[string]struct{}, len(section.Fields))
		for _, field := range section.Fields {
			if err := validateText("field name", field.Name, true); err != nil {
				return fmt.Errorf("section %q: %w", section.Title, err)
			}
			if _, exists := seenFields[field.Name]; exists {
				return fmt.Errorf("section %q has duplicate field %q", section.Title, field.Name)
			}
			seenFields[field.Name] = struct{}{}
			if err := validateText("field value", field.Value, false); err != nil {
				return fmt.Errorf("section %q field %q: %w", section.Title, field.Name, err)
			}
			if err := validateStatus(field.Status); err != nil {
				return fmt.Errorf("section %q field %q: %w", section.Title, field.Name, err)
			}
			hasFailed = hasFailed || field.Status == CheckFail
			hasIncomplete = hasIncomplete ||
				field.Status == CheckUnknown ||
				field.Status == CheckUnavailable
		}
	}
	hasFailed = hasFailed || len(document.Errors) > 0
	switch document.Outcome {
	case OutcomePass:
		if hasFailed || hasIncomplete {
			return errors.New("passing report contains failed, unknown, or unavailable results")
		}
	case OutcomeFail:
		if !hasFailed {
			return errors.New("failed report contains no failed check or error")
		}
	case OutcomeIncomplete:
		if hasFailed {
			return errors.New("incomplete report contains a failed check or error")
		}
		if !hasIncomplete {
			return errors.New("incomplete report contains no unknown or unavailable result")
		}
	default:
		return fmt.Errorf("unsupported report outcome %q", document.Outcome)
	}
	return nil
}

func validateVersionValue(name string, value VersionValue) error {
	if err := validateStatus(value.Status); err != nil {
		return fmt.Errorf("%s version: %w", name, err)
	}
	if err := validateText(name+" version", value.Value, false); err != nil {
		return err
	}
	if (value.Status == CheckPass || value.Status == CheckFail) &&
		strings.TrimSpace(value.Value) == "" {
		return fmt.Errorf("%s version value is required for status %q", name, value.Status)
	}
	return nil
}

func validateStatus(status CheckStatus) error {
	switch status {
	case CheckPass, CheckFail, CheckWarning, CheckUnknown, CheckUnavailable:
		return nil
	default:
		return fmt.Errorf("unsupported check status %q", status)
	}
}

func validateText(name, value string, required bool) error {
	if required && strings.TrimSpace(value) == "" {
		return fmt.Errorf("%s is required", name)
	}
	if len(value) > MaxTextBytes {
		return fmt.Errorf("%s exceeds %d bytes", name, MaxTextBytes)
	}
	if !utf8.ValidString(value) {
		return fmt.Errorf("%s is not valid UTF-8", name)
	}
	return nil
}

func validUUID(value string) bool {
	if len(value) != 36 ||
		value != strings.ToLower(value) ||
		value[8] != '-' || value[13] != '-' ||
		value[18] != '-' || value[23] != '-' {
		return false
	}
	_, err := hex.DecodeString(strings.ReplaceAll(value, "-", ""))
	return err == nil
}

func validFingerprint(value string) bool {
	if len(value) != 64 || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}
