package model

type Label string
type Namespace string
type ExternalID string

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
	Label      Label
	Namespace  Namespace
	ExternalID ExternalID
	Properties Properties
	Position   SourcePosition
}

type Endpoint struct {
	Label      Label
	Namespace  Namespace
	ExternalID ExternalID
}

type Edge struct {
	Label      Label
	Namespace  Namespace
	ExternalID ExternalID
	Start      Endpoint
	End        Endpoint
	Properties Properties
	Position   SourcePosition
}
