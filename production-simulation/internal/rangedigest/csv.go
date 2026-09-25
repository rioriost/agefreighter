package rangedigest

import (
	"context"
	"errors"
	"io"

	"github.com/rioriost/agefreighter/internal/config"
	csvsource "github.com/rioriost/agefreighter/internal/source/csv"
	"github.com/rioriost/agefreighter/pkg/model"
)

// CSVManifest reads the exported data through the actual AGEFreighter connector,
// independently of the fixture's expected digest implementation.
func CSVManifest(ctx context.Context, source config.CSVSource, fixtureRoot string, rangeRows int64) (Manifest, error) {
	iterator, err := csvsource.NewIterator(ctx, csvsource.IteratorOptions{Namespace: "p1", Source: source})
	if err != nil {
		return Manifest{}, err
	}
	defer iterator.Close()
	builder, err := newRangeBuilder(rangeRows)
	if err != nil {
		return Manifest{}, err
	}
	kind, name := "", ""
	for {
		item, err := iterator.Next(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return Manifest{}, err
		}
		record := item.Record
		var nextKind, nextName string
		var props model.Properties
		var id, start, end string
		if record.Vertex != nil {
			nextKind, nextName = "v", string(record.Vertex.Label)
			props = record.Vertex.Properties
			id = string(record.Vertex.ExternalID)
		} else {
			nextKind, nextName = "e", string(record.Edge.Label)
			props = record.Edge.Properties
			id = string(record.Edge.ExternalID)
			start = string(record.Edge.Start.ExternalID)
			end = string(record.Edge.End.ExternalID)
		}
		if kind != nextKind || name != nextName {
			if kind != "" {
				if err := builder.end(); err != nil {
					return Manifest{}, err
				}
			}
			kind, name = nextKind, nextName
			if err := builder.begin(kind, name); err != nil {
				return Manifest{}, err
			}
		}
		key := props["source_key"]
		if key.Kind != model.ValueInteger {
			return Manifest{}, errors.New("source key lost its integer type")
		}
		encoded, err := model.EncodeProperties(props)
		if err != nil {
			return Manifest{}, err
		}
		line := vertexLine(name, key.Integer, id, encoded)
		if kind == "e" {
			line = edgeLine(name, key.Integer, id, start, end, encoded)
		}
		if err := builder.add(key.Integer, line); err != nil {
			return Manifest{}, err
		}
	}
	if kind == "" {
		return Manifest{}, errors.New("CSV fixture is empty")
	}
	if err := builder.end(); err != nil {
		return Manifest{}, err
	}
	return builder.result("csv", fixtureRoot, "", ""), nil
}
