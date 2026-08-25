package age

import "github.com/rioriost/agefreighter/internal/meta"

func (adapter *Adapter) Metadata() (*meta.Store, error) {
	return meta.New(adapter.pool)
}

func (transaction *Transaction) Metadata() (*meta.Store, error) {
	return meta.New(transaction.tx)
}
