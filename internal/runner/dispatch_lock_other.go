//go:build !linux && !darwin

package runner

import "errors"

func (m Manager) dispatchLock() (func(), error) {
	return nil, errors.New("runner admission requires Linux")
}
