# Spec: Broadcasting with Symbolic Dimensions

## Overview
Implement broadcasting rules that work with symbolic dimensions to support dynamic tensor operations.

## Requirements

### Class: BroadcastingRule
- Properties:
  - `DimensionIndex`: int
  - `IsBroadcastable`: bool
  - `OutputSize`: SymbolicDimension

- Methods:
  - `Apply(SymbolicDimension dim1, SymbolicDimension dim2)`: SymbolicDimension

### Class: SymbolicBroadcastingEngine
- Methods:
  - `CanBroadcast(SymbolicShape shape1, SymbolicShape shape2)`: bool
  - `GetBroadcastShape(SymbolicShape shape1, SymbolicShape shape2)`: SymbolicShape
  - `GetBroadcastPlan(SymbolicShape shape1, SymbolicShape shape2)`: List<BroadcastingRule>
  - `InferBroadcastConstraints(SymbolicShape shape1, SymbolicShape shape2)`: List<SymbolicConstraint>

### Broadcasting Logic:
1. Align shapes from right to left
2. For each dimension pair:
   - If equal: output = that dimension
   - If one is 1: output = the other dimension
   - If one is unknown (symbolic): output = symbolic dimension (add constraint that >= 1)
   - If both symbolic and not equal: incompatible (unless they can be proven equal via constraints)
   - Else: incompatible

### Class: BroadcastedTensor
- Properties:
  - `OriginalTensor`: Tensor
  - `BroadcastShape`: SymbolicShape
  - `BroadcastPlan`: List<BroadcastingRule>

- Methods:
  - `Materialize()`: Tensor - Apply broadcasting to produce actual tensor
  - `GetStrides()`: int[] - Calculate broadcasted strides

### Unit Tests
- Test broadcasting with concrete shapes
- Test broadcasting with symbolic shapes
- Test broadcasting with mixed concrete/symbolic
- Test incompatible shape detection
- Test constraint inference
- Test plan generation
- Test broadcasted tensor materialization
- Test edge cases (scalar, 1-dim, mismatched ranks)

## Implementation Notes
- Follow NumPy broadcasting semantics
- Symbolic dimensions can be constrained during broadcasting (e.g., must be >= 1)
- Provide clear error messages for incompatible broadcasts
- Support inverse broadcasting (identify broadcast source)
- Cache broadcast plans for performance

## Dependencies
- spec_symbolic_dimension.md
- spec_symbolic_shape.md
- spec_shape_constraint.md

## Success Criteria
- Correctly implements NumPy broadcasting rules
- Handles symbolic dimensions correctly
- Detects incompatible broadcasts early
- Constraints are inferred properly
- Plan generation is efficient
