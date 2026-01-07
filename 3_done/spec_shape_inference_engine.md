# Spec: Shape Inference Engine

## Overview
Implement a base shape inference engine that can determine output shapes for operations given input shapes.

## Requirements

### Interface: IShapeInferenceRule
- Methods:
  - `CanInfer(Operation op, List<SymbolicShape> inputs)`: bool
  - `Infer(Operation op, List<SymbolicShape> inputs)`: List<SymbolicShape>

### Class: ShapeInferenceEngine
- Properties:
  - `Rules`: Dictionary<string, IShapeInferenceRule> - Operation name -> rule

- Methods:
  - `RegisterRule(string opName, IShapeInferenceRule rule)`: void
  - `UnregisterRule(string opName)`: void
  - `Infer(Operation op, List<SymbolicShape> inputs)`: List<SymbolicShape>
  - `CanInfer(Operation op, List<SymbolicShape> inputs)`: bool
  - `Validate(Operation op, List<SymbolicShape> inputs)`: bool - Check if shapes are valid for operation

### Base Class: ShapeInferenceRuleBase : IShapeInferenceRule
- Abstract methods (implement in subclasses)
- Default implementations for validation logic

### Built-in Rules (for common operations):

1. **MatMulRule**: Matrix multiplication
   - Input: [M, K], [K, N] -> Output: [M, N]
   - Support broadcasting on batch dimensions

2. **Conv2DRule**: 2D convolution
   - Input: [N, C_in, H, W], weight shape, padding, stride, dilation
   - Output: [N, C_out, H_out, W_out]

3. **TransposeRule**: Tensor transpose
   - Input shape + permutation -> Output shape

4. **ReshapeRule**: Tensor reshape
   - Validate total elements match
   - Handle -1 dimension (inferred)

5. **Add/Sub/Mul/DivRule**: Element-wise operations
   - Broadcasting support

### Class: InferenceContext
- Properties:
  - `TensorShapes`: Dictionary<string, SymbolicShape> - Tensor name -> shape
  - `OperationResults`: Dictionary<string, List<SymbolicShape>> - Op ID -> output shapes

- Methods:
  - `GetShape(string tensorName)`: SymbolicShape?
  - `SetShape(string tensorName, SymbolicShape shape)`: void
  - `RecordInference(string opId, List<SymbolicShape> outputs)`: void

### Unit Tests
- Test each built-in rule with various inputs
- Test rule registration/unregistration
- Test inference engine coordination
- Test context management
- Test broadcasting in element-wise operations
- Test -1 dimension handling in reshape

## Implementation Notes
- Rules are stateless and thread-safe
- Cache inference results in context for same operation
- Support symbolic dimension propagation (e.g., if input is [N, 512], output may be [N, 256])
- Provide detailed error messages for invalid shapes

## Dependencies
- spec_symbolic_shape.md

## Success Criteria
- Can infer shapes for all core operations (matmul, conv, element-wise, reshape)
- Extensible registration system for custom operations
- Clear error messages for shape mismatches
- Handles symbolic dimension propagation correctly
