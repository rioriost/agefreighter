package rangedigest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/rioriost/agefreighter/production-simulation/internal/fixture"
)

const GremlinCanonicalVersion = "agefreighter-production-simulation-gremlin-partition64-v1"

func gremlinFixtureID(spec fixture.VertexSpec, key int64, id string) ([]byte, error) {
	if key < spec.FirstKey || key >= spec.FirstKey+spec.Count {
		return nil, fmt.Errorf("Gremlin endpoint outside frozen label range")
	}
	partition := spec.Label + "-" + strconv.FormatInt((key-spec.FirstKey)%64, 10)
	return json.Marshal([2]string{partition, id})
}

func gremlinFixtureVertex(spec fixture.VertexSpec, row []string) (int64, []byte, error) {
	key, line, err := fixtureVertex(spec.Label, row)
	if err != nil {
		return 0, nil, err
	}
	fields := bytes.SplitN(line, []byte{0}, 5)
	fields[3], err = gremlinFixtureID(spec, key, row[1])
	if err != nil {
		return 0, nil, err
	}
	return key, bytes.Join(fields, []byte{0}), nil
}

func gremlinFixtureEdge(spec fixture.EdgeSpec, vertices map[string]fixture.VertexSpec, row []string) (int64, []byte, error) {
	key, line, err := fixtureEdge(spec, vertices, row)
	if err != nil {
		return 0, nil, err
	}
	fields := bytes.SplitN(line, []byte{0}, 7)
	startKey, _ := strconv.ParseInt(row[2], 10, 64)
	endKey, _ := strconv.ParseInt(row[3], 10, 64) // validated by fixtureEdge
	fields[3], err = gremlinFixtureID(vertices[spec.Start], startKey, row[1])
	if err != nil {
		return 0, nil, err
	}
	fields[4], err = gremlinFixtureID(vertices[spec.Start], startKey, string(fields[4]))
	if err != nil {
		return 0, nil, err
	}
	fields[5], err = gremlinFixtureID(vertices[spec.End], endKey, string(fields[5]))
	if err != nil {
		return 0, nil, err
	}
	return key, bytes.Join(fields, []byte{0}), nil
}

// CompareOfflineGremlin cannot be used to claim target qualification. The input
// role and canonical version are checked before reusing the leaf comparator.
func CompareOfflineGremlin(expected, actual Manifest) (Comparison, error) {
	if expected.Source != "fixture" || actual.Source != "cosmos-gremlin-offline" || expected.CanonicalVersion != GremlinCanonicalVersion || actual.CanonicalVersion != GremlinCanonicalVersion {
		return Comparison{Status: "fail"}, fmt.Errorf("not an offline Gremlin comparison")
	}
	return compareManifests(expected, actual, GremlinCanonicalVersion, "cosmos-gremlin-offline")
}
