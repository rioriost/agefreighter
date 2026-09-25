//go:build !linux

package runner

import "context"

// Non-Linux execution is unsupported. Tests may exercise the protocol here,
// but the absence of Linux health evidence cannot authorize a migration.
func (m Manager) health(context.Context) (*GuestHealth, error) { return nil, nil }
