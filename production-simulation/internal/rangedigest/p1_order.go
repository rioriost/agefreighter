package rangedigest

import (
	"context"
	"errors"
	"sort"
)

type canonicalSink interface{ add(int64, []byte) error }
type canonicalRecord struct {
	key  int64
	data []byte
}

// One mapping at a time, with independent row and payload ceilings. No source
// properties are discarded/coerced and duplicate keys still fail in rangeBuilder.
type p1OrderBuffer struct {
	builder *rangeBuilder
	rows    []canonicalRecord
	maxRows int64
	bytes   int64
}

const p1MappingByteLimit = 512 * 1024 * 1024

func targetSink(builder *rangeBuilder, unordered bool, count int64) canonicalSink {
	if !unordered {
		return builder
	}
	return &p1OrderBuffer{builder: builder, maxRows: count}
}
func (b *p1OrderBuffer) add(key int64, data []byte) error {
	if b.maxRows < 1 || b.maxRows > 4_000_000 || int64(len(b.rows)) >= b.maxRows || int64(len(data)) > p1MappingByteLimit-b.bytes {
		return errors.New("P1 canonical ordering buffer limit exceeded")
	}
	b.rows = append(b.rows, canonicalRecord{key: key, data: append([]byte(nil), data...)})
	b.bytes += int64(len(data))
	return nil
}
func finishTargetSink(ctx context.Context, sink canonicalSink) error {
	b, ok := sink.(*p1OrderBuffer)
	if !ok {
		return nil
	}
	sort.Slice(b.rows, func(i, j int) bool { return b.rows[i].key < b.rows[j].key })
	for _, row := range b.rows {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := b.builder.add(row.key, row.data); err != nil {
			return err
		}
	}
	b.rows = nil
	return nil
}
