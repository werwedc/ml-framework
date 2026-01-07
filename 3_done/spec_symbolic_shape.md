# Spec: Symbolic Shape Representation

## Overview
Implement a shape representation that uses symbolic dimensions instead of concrete integers.

## Requirements

### Class: SymbolicShape
- Properties:
  - `Dimensions`: ReadOnlyCollection<SymbolicDimension>
  - `Rank`: int - Number of dimensions

- Methods:
  - `GetDimension(int index)`: SymbolicDimension
  - `SetDimension(int index, SymbolicDimension dim)`: SymbolicShape - Return new shape
  - `IsFullyKnown()`: bool - All dimensions have concrete values
  - `IsPartiallyKnown()`: bool - At least one dimension is known
  - `ToConcrete()`: int[] - Convert to concrete array (throws if not fully known)
  - `Equals(SymbolicShape other)`: bool
  - `Clone()`: SymbolicShape
  - `ToString()`: string - Format like "[batch_size, seq_len, 512]"

### Class: SymbolicShapeFactory
- Factory methods:
  - `Create(params SymbolicDimension[] dims)`
  - `FromConcrete(params int[] dims)` - Convert concrete to symbolic
  - `Scalar()`: SymbolicShape - Rank 0
  - `Vector(int length)`: SymbolicShape - Rank 1 with known length
  - `Matrix(int rows, int cols)`: SymbolicShape - Rank 2

### Class: ShapeComparer
- Static methods:
  - `AreCompatible(SymbolicShape a, SymbolicShape b)`: bool - Check if shapes can be broadcast
  - `GetBroadcastShape(SymbolicShape a, SymbolicShape b)`: SymbolicShape
  - `CanReshape(SymbolicShape from, SymbolicShape to)`: bool

### Unit Tests
- Test shape creation with various combinations
- Test dimension access and modification
- IsFullyKnown/IsPartiallyKnown edge cases
- Test concrete conversion
- Test broadcasting compatibility
- Test reshape validation
- Test equality and cloning

## Implementation Notes
- Immutable design - all modifications return new instances
- Cache IsFullyKnown/IsPartiallyKnown results
- Support negative indexing for GetDimension
- Use structural equality for comparison

## Dependencies
- spec_symbolic_dimension.md

## Success Criteria
- Can represent shapes with any mix of known/unknown dimensions
- Broadcasting checks work correctly
- Reshape validation prevents impossible operations
- Conversion to/from concrete shapes is seamless
