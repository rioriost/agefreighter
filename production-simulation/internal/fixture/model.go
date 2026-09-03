package fixture

import (
	"fmt"
	"sort"
)

const ManifestVersion = 1

type Phase string

const (
	PhaseTiny Phase = "tiny"
	PhaseP0   Phase = "p0"
	PhaseP1   Phase = "p1"
	PhaseP2   Phase = "p2"
	PhaseP3   Phase = "p3"
)

type VertexSpec struct {
	Label    string `json:"label"`
	Count    int64  `json:"count"`
	FirstKey int64  `json:"firstKey"`
}

type EdgeSpec struct {
	Type     string `json:"type"`
	Start    string `json:"start"`
	End      string `json:"end"`
	Count    int64  `json:"count"`
	FirstKey int64  `json:"firstKey"`
}

type Plan struct {
	Phase       Phase        `json:"phase"`
	VertexTotal int64        `json:"vertexTotal"`
	EdgeTotal   int64        `json:"edgeTotal"`
	VertexSpecs []VertexSpec `json:"vertices"`
	EdgeSpecs   []EdgeSpec   `json:"edges"`
}

type weightedVertex struct {
	label string
	full  int64
}

type weightedEdge struct {
	typeName string
	start    string
	end      string
	full     int64
}

var vertexModel = []weightedVertex{
	{label: "Supplier", full: 4_000_000},
	{label: "Facility", full: 2_000_000},
	{label: "Product", full: 20_000_000},
	{label: "PurchaseOrder", full: 45_000_000},
	{label: "Shipment", full: 35_000_000},
	{label: "Lot", full: 50_000_000},
	{label: "Location", full: 1_000_000},
	{label: "Carrier", full: 100_000},
	{label: "Customer", full: 2_900_000},
}

var edgeModel = []weightedEdge{
	{typeName: "SUPPLIES", start: "Supplier", end: "Product", full: 40_000_000},
	{typeName: "PRODUCED_AT", start: "Product", end: "Facility", full: 30_000_000},
	{typeName: "PLACED_WITH", start: "PurchaseOrder", end: "Supplier", full: 45_000_000},
	{typeName: "CONTAINS", start: "PurchaseOrder", end: "Product", full: 100_000_000},
	{typeName: "FULFILLS", start: "Shipment", end: "PurchaseOrder", full: 45_000_000},
	{typeName: "ORIGINATES_AT", start: "Shipment", end: "Facility", full: 35_000_000},
	{typeName: "DESTINED_FOR", start: "Shipment", end: "Location", full: 35_000_000},
	{typeName: "CARRIED_BY", start: "Shipment", end: "Carrier", full: 35_000_000},
	{typeName: "INCLUDED_IN", start: "Lot", end: "Shipment", full: 35_000_000},
}

func BuildPlan(phase Phase) (Plan, error) {
	vertexTotal, edgeTotal, ok := phaseTotals(phase)
	if !ok {
		return Plan{}, fmt.Errorf("unknown phase %q", phase)
	}

	vertexWeights := make([]int64, len(vertexModel))
	for index, item := range vertexModel {
		vertexWeights[index] = item.full
	}
	vertexCounts := allocate(vertexWeights, vertexTotal)

	edgeWeights := make([]int64, len(edgeModel))
	for index, item := range edgeModel {
		edgeWeights[index] = item.full
	}
	edgeCounts := allocate(edgeWeights, edgeTotal)

	plan := Plan{Phase: phase, VertexTotal: vertexTotal, EdgeTotal: edgeTotal}
	nextVertexKey := int64(1)
	for index, item := range vertexModel {
		plan.VertexSpecs = append(plan.VertexSpecs, VertexSpec{
			Label: item.label, Count: vertexCounts[index], FirstKey: nextVertexKey,
		})
		nextVertexKey += vertexCounts[index]
	}
	nextEdgeKey := int64(1)
	for index, item := range edgeModel {
		plan.EdgeSpecs = append(plan.EdgeSpecs, EdgeSpec{
			Type: item.typeName, Start: item.start, End: item.end,
			Count: edgeCounts[index], FirstKey: nextEdgeKey,
		})
		nextEdgeKey += edgeCounts[index]
	}
	return plan, nil
}

func phaseTotals(phase Phase) (int64, int64, bool) {
	switch phase {
	case PhaseTiny:
		return 160, 400, true
	case PhaseP0:
		return 100_000, 358_000, true
	case PhaseP1:
		return 1_600_000, 4_000_000, true
	case PhaseP2:
		return 16_000_000, 40_000_000, true
	case PhaseP3:
		return 160_000_000, 400_000_000, true
	default:
		return 0, 0, false
	}
}

type remainder struct {
	index int
	value int64
}

func allocate(weights []int64, target int64) []int64 {
	if target < int64(len(weights)) {
		panic("allocation target is smaller than the model")
	}
	totalWeight := int64(0)
	for _, weight := range weights {
		totalWeight += weight
	}
	counts := make([]int64, len(weights))
	remainders := make([]remainder, len(weights))
	allocated := int64(0)
	for index, weight := range weights {
		product := weight * target
		counts[index] = product / totalWeight
		remainders[index] = remainder{index: index, value: product % totalWeight}
		allocated += counts[index]
	}
	sort.SliceStable(remainders, func(left, right int) bool {
		return remainders[left].value > remainders[right].value
	})
	for index := int64(0); index < target-allocated; index++ {
		counts[remainders[index%int64(len(remainders))].index]++
	}
	for index, count := range counts {
		if count != 0 {
			continue
		}
		donor := 0
		for candidate := range counts {
			if counts[candidate] > counts[donor] {
				donor = candidate
			}
		}
		if counts[donor] <= 1 {
			panic("allocation cannot preserve non-empty model entries")
		}
		counts[donor]--
		counts[index] = 1
	}
	return counts
}
