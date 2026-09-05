package cli

import (
	reportcontract "github.com/rioriost/agefreighter/internal/report"
	"testing"
)

func TestVerificationRequireComplete(t *testing.T) {
	for _, require := range []bool{false, true} {
		for _, outcome := range []reportcontract.Outcome{reportcontract.OutcomePass, reportcontract.OutcomeFail, reportcontract.OutcomeIncomplete} {
			wantError := outcome == reportcontract.OutcomeFail || require && outcome != reportcontract.OutcomePass
			if got := verificationOutcomeError(outcome, require); (got != nil) != wantError {
				t.Errorf("%s require=%v: %v", outcome, require, got)
			}
		}
	}
	cmd := newVerifyCommand()
	if err := cmd.Flags().Set("require-complete", "true"); err != nil {
		t.Fatal(err)
	}
	if err := cmd.PreRunE(cmd, []string{"job"}); err == nil {
		t.Fatal("catalog-only strict mode accepted")
	}
	if err := cmd.Flags().Set("counts", "true"); err != nil {
		t.Fatal(err)
	}
	if err := cmd.PreRunE(cmd, []string{"job"}); err != nil {
		t.Fatal(err)
	}
}
