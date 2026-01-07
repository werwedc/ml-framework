# Spec: Symbolic Tensor API

## Overview
Implement user-facing API for creating and working with symbolic tensors.

## Requirements

### Static Class: Tensor
- Factory Methods:
  - `Symbolic(params SymbolicDimension[] dims)`: Tensor
  - `Symbolic(SymbolicShape shape)`: Tensor
  - `Zeros(params SymbolicDimension[] dims)`: Tensor
  - `Ones(params SymbolicDimension[] dims)`: Tensor
  - `Random(params SymbolicDimension[] dims)`: Tensor - throws if dims not fully known

- Shape Methods:
  - `ShapeHint(string dimName, SymbolicConstraint constraint)`: Tensor - Attach constraint to dimension
  - `WithBounds(string dimName, int min, int? max)`: Tensor
  - `GetShape()`: SymbolicShape

### Extension Methods for Tensor:
- `ResizeTo(params int[] concreteDims)`: Tensor - Concrete resize at runtime
- `ResizeTo(SymbolicShape symbolicDims)`: Tensor - Symbolic resize
- `GetDimension(string name)`: SymbolicDimension

### Class: SymbolicTensor : Tensor
- Properties:
  - `SymbolicShape`: SymbolicShape
  - `Constraints`: Dictionary<string, List<IShapeConstraint>> - Dim name -> constraints

- Methods:
  - `ValidateShape()`: bool - Check all constraints
  - `GetConcreteShape(params int[] values)`: int[] - Substitute symbolic dimensions with values
  - `CanInstantiateWith(params int[] dims)`: bool

### Class: TensorShapeRegistry
- Methods:
  - `RegisterBinding(string dimName, int value)`: void - Bind symbolic dimension to concrete value
  - `GetBinding(string dimName)`: int?
  - `ClearBindings()`: void
  - `Clone()`: TensorShapeRegistry

### Unit Tests
- Test symbolic tensor creation
- Test shape hints and constraints
- Test constraint validation
- Test shape substitution with concrete values
- Test extension methods on Tensor
- Test registry binding management

## Implementation Notes
- SymbolicTensor is a placeholder - not usable for actual computation until shapes bound
- Provide clear error messages when trying to instantiate unbound symbolic tensor
- Support chained method calls for constraint specification
- Thread-safe constraint checking

## Dependencies
- spec_symbolic_dimension.md
- spec_shape_constraint.md
- spec_symbolic_shape.md

## Success Criteria
- API is intuitive and follows framework conventions
- Chainable constraint specification
- Clear errors when shapes cannot be instantiated
- Seamless transition from symbolic to concrete
