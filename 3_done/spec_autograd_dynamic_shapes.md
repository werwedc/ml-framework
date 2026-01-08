# Spec: Autograd Engine Updates for Dynamic Shapes

## Overview
Update the autograd engine to correctly handle dynamic intermediate tensors and shape-dependent operations.

## Requirements

### Interface: IDynamicGradientFunction
- Inherits from IGradientFunction
- Additional methods:
  - `GetOutputShape(List<SymbolicShape> inputShapes)`: SymbolicShape
  - `ValidateGradientShape(SymbolicShape gradientShape)`: void

### Class: DynamicGradientTensor
- Properties:
  - `Tensor`: Tensor
  - `Shape`: SymbolicShape
  - `GradientRequired`: bool

- Methods:
  - `AccumulateGradient(Tensor grad)`: void
  - `GetGradient()`: Tensor?
  - `ClearGradient()`: void

### Updated Operations for Autograd:

1. **DynamicMatMulBackward**
   - Handle batch dimension broadcasting
   - Support symbolic batch size
   - Gradient shape: same as input shape

2. **DynamicConv2DBackward**
   - Compute gradient with dynamic spatial dimensions
   - Handle padding/stride with unknown size
   - Validate gradient shapes match input

3. **DynamicReshapeBackward**
   - Reshape gradient to original shape
   - Preserve symbolic dimensions

4. **DynamicBroadcastBackward**
   - Sum gradient over broadcasted dimensions
   - Handle reduction with symbolic sizes
   - Reduce to 1 where broadcast originated

### Class: DynamicAutogradContext
- Properties:
  - `InputShapes`: List<SymbolicShape>
  - `OutputShapes`: List<SymbolicShape>
  - `SavedTensors`: List<Tensor>
  - `GradientShapes`: List<SymbolicShape>

- Methods:
  - `SaveForBackward(Tensor tensor)`: void
  - `GetSavedTensor(int index)`: Tensor
  - `ValidateGradient(int index, SymbolicShape gradShape)`: void

### Class: GradientAccumulatorDynamic
- Methods:
  - `Accumulate(Tensor grad, SymbolicShape shape)`: void
  - `AccumulateBatched(List<Tensor> grads, List<SymbolicShape> shapes)`: void
  - `GetAccumulated(SymbolicShape shape)`: Tensor?
  - `Reset()`: void

### Shape-dependent Operation Support:

- **DynamicMask**: Apply mask with variable sequence length
- **DynamicPad**: Pad to variable sizes
- **DynamicSlice**: Slice with dynamic indices
- **DynamicGather**: Gather with dynamic indices

### Unit Tests
- Test gradient computation with symbolic shapes
- Test gradient shape validation
- Test gradient accumulation with variable batch sizes
- Test backward operations (matmul, conv, reshape)
- Test broadcasted gradients
- Test shape-dependent operations
- Test gradient correctness for dynamic inputs

## Implementation Notes
- Gradient shapes must match input shapes exactly
- Use symbolic shape tracking to validate gradients
- Support batched gradient accumulation with variable sizes
- Handle partial gradients (e.g., from truncated backprop)
- Preserve symbolic dimensions through backward pass

## Dependencies
- spec_symbolic_shape.md
- spec_broadcasting_symbolic.md

## Success Criteria
- Gradients computed correctly for dynamic shapes
- Gradient shapes validated against input shapes
- Accumulation works with variable batch sizes
- Shape-dependent operations produce correct gradients
- Backward pass preserves symbolic information
