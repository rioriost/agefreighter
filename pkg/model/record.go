package model

type Label string
type Namespace string
type ExternalID string

type RecordKind uint8

const (
	RecordInvalid RecordKind = iota
	RecordVertex
	RecordEdge
)

type ValueKind uint8

const (
	ValueNull ValueKind = iota
	ValueBoolean
	ValueInteger
	ValueFloat
	ValueString
	ValueList
	ValueObject
)

type Value struct {
	Kind    ValueKind
	Boolean bool
	Integer int64
	Float   float64
	String  string
	List    []Value
	Object  map[string]Value
}

type Properties map[string]Value

type SourcePosition struct {
	Connector string
	Resource  string
	Offset    int64
	Line      int64
	Token     string
}

type Vertex struct {
	Label             Label
	Namespace         Namespace
	ExternalID        ExternalID
	Properties        Properties
	EncodedProperties []byte `json:"-"`
	Position          SourcePosition
}

type Endpoint struct {
	Label      Label
	Namespace  Namespace
	ExternalID ExternalID
}

type Edge struct {
	Label             Label
	Namespace         Namespace
	ExternalID        ExternalID
	Start             Endpoint
	End               Endpoint
	Properties        Properties
	EncodedProperties []byte `json:"-"`
	Position          SourcePosition
}

type Record struct {
	Vertex *Vertex
	Edge   *Edge
}

func VertexRecord(vertex Vertex) Record {
	return Record{Vertex: &vertex}
}

func EdgeRecord(edge Edge) Record {
	return Record{Edge: &edge}
}

func (record Record) Kind() RecordKind {
	switch {
	case record.Vertex != nil && record.Edge == nil:
		return RecordVertex
	case record.Edge != nil && record.Vertex == nil:
		return RecordEdge
	default:
		return RecordInvalid
	}
}

func (record Record) SourcePosition() (SourcePosition, bool) {
	switch record.Kind() {
	case RecordVertex:
		return record.Vertex.Position, true
	case RecordEdge:
		return record.Edge.Position, true
	default:
		return SourcePosition{}, false
	}
}
