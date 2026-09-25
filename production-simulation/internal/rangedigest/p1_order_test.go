package rangedigest

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func TestP1OrderIndependentAndStrict(t *testing.T) {
	ordered, _ := newRangeBuilder(2)
	unordered, _ := newRangeBuilder(2)
	ordered.begin("v", "Product")
	unordered.begin("v", "Product")
	buffer := targetSink(unordered, true, 3)
	for _, key := range []int64{3, 1, 2} {
		if err := buffer.add(key, []byte{byte(key)}); err != nil {
			t.Fatal(err)
		}
	}
	for _, key := range []int64{1, 2, 3} {
		ordered.add(key, []byte{byte(key)})
	}
	if err := finishTargetSink(context.Background(), buffer); err != nil {
		t.Fatal(err)
	}
	ordered.end()
	unordered.end()
	if !reflect.DeepEqual(ordered.result("a", "b", "c", "d"), unordered.result("a", "b", "c", "d")) {
		t.Fatal("ingestion order changed canonical manifest")
	}
	for _, keys := range [][]int64{{1, 2, 2}, {1, 2, 1}} {
		b, _ := newRangeBuilder(2)
		b.begin("v", "Product")
		for _, k := range keys[:2] {
			b.add(k, nil)
		}
		if !errors.Is(b.add(keys[2], nil), ErrSourceKeyOrder) {
			t.Fatal("ordering escaped at range boundary")
		}
	}
	b, _ := newRangeBuilder(1)
	b.begin("e", "SUPPLIES")
	sink := targetSink(b, true, 2)
	sink.add(1, []byte("a"))
	sink.add(1, []byte("b"))
	if !errors.Is(finishTargetSink(context.Background(), sink), ErrSourceKeyOrder) {
		t.Fatal("duplicate key accepted")
	}
}
func TestP1OrderingBoundsAndCancellation(t *testing.T) {
	b, _ := newRangeBuilder(2)
	b.begin("v", "Product")
	s := targetSink(b, true, 1)
	s.add(1, nil)
	if s.add(2, nil) == nil {
		t.Fatal("row ceiling ignored")
	}
	p := &p1OrderBuffer{builder: b, maxRows: 2, bytes: p1MappingByteLimit}
	if p.add(1, []byte("x")) == nil {
		t.Fatal("byte ceiling ignored")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if !errors.Is(finishTargetSink(ctx, s), context.Canceled) {
		t.Fatal("cancellation ignored")
	}
	if targetSink(b, false, 1) != b {
		t.Fatal("P3 streaming path changed")
	}
}
