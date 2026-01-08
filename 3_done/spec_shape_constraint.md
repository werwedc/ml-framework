# Spec: Shape Constraint System

## Overview
Implement a constraint system for symbolic dimensions to express relationships and bounds.

## Requirements

### Interface: IShapeConstraint
- Methods:
  - `Validate(SymbolicDimension dim)`: bool - Check if constraint is satisfied
  - `ToString()`: string - Human-readable description

### Class: RangeConstraint : IShapeConstraint
- Properties:
  - `MinValue`: int (inclusive)
  - `MaxValue`: int (inclusive)

- Methods:
  - `Validate(SymbolicDimension dim)`: bool
  - `Clamp(int value)`: int - Force value into range

### Class: EqualityConstraint : IShapeConstraint
- Properties:
  - `TargetValue`: int

- Methods:
  - `Validate(SymbolicDimension dim)`: bool

### Class: ModuloConstraint : IShapeConstraint
- Properties:
  - `Divisor`: int

- Methods:
  - `Validate(SymbolicDimension dim)`: bool - Check if value % Divisor == 0

### Class: ShapeConstraintBuilder
- Fluent API for building constraint lists:
  - `Min(int value)`: this
  - `Max(int value)`: this
  - `Equal(int value)`: this
  - `Modulo(int divisor)`: this
  - `Build()`: List<IShapeConstraint>

### Class: ConstraintValidator
- Methods:
  - `ValidateAll(IEnumerable<SymbolicDimension> dims, Dictionary<string, List<IShapeConstraint>> constraints)`: bool
  - `GetViolations(...)`: List<string> - Return descriptive error messages

### Unit Tests
- Test each constraint type individually
- Test constraint builder API
- Test validator with multiple constraints
- Test violation messages are descriptive

## Implementation Notes
- Constraints are immutable
- Support combination of multiple constraints via AND logic
- Cache constraint evaluation results for performance

## Dependencies
- spec_symbolic_dimension.md

## Success Criteria
- Can express common shape constraints (ranges, divisibility)
- Validation is fast and provides clear error messages
- Fluent API is intuitive
