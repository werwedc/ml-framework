# Spec: Symbolic Dimension Core

## Overview
Implement the fundamental building block for dynamic shapes - the symbolic dimension class that represents unknown tensor dimensions.

## Requirements

### Class: SymbolicDimension
- Properties:
  - `Name`: string (e.g., "batch_size", "seq_len")
  - `Value`: int? (null for truly unknown, or concrete value when known)
  - `MinValue`: int (lower bound, default 0)
  - `MaxValue`: int? (upper bound, null means unbounded)

- Methods:
  - `IsKnown()`: bool - Returns true if Value is set
  - `IsBounded()`: bool - Returns true if MaxValue is set
  - `Equals(other)`: bool - Compare symbolic dimensions
  - `Clone()`: SymbolicDimension - Create a copy
  - `WithConstraints(int? min, int? max)`: SymbolicDimension - Return new instance with bounds

### Class: SymbolicDimensionFactory
- Factory methods:
  - `Create(string name, int? value = null)`
  - `CreateBounded(string name, int min, int max)`
  - `CreateRange(string name, int min)` - unbounded above

### Unit Tests
- Test creation with various parameter combinations
- Test bounds checking (IsBounded, IsKnown)
- Test equality and inequality
- Test cloning preserves values
- Test constraint application (WithConstraints)

## Implementation Notes
- Use immutable design - operations return new instances
- Override GetHashCode() and Equals() for dictionary keys
- Support comparison operators for constraint evaluation

## Dependencies
- None (core type)

## Success Criteria
- Can represent known, unknown, and partially known dimensions
- Enforces immutable semantics
- Thread-safe (immutable)
