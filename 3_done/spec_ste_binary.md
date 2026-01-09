# Spec: STEBinary Custom Function

## Overview
Implement the `STEBinary` custom function that provides a Straight-Through Estimator (STE) for binary activation. This function uses a binary step function in the forward pass but passes gradients through unchanged in the backward pass, enabling gradient flow through binary decisions.

## Requirements

### 1. Class Definition
Create a `STEBinary` class in `src/Autograd/Functions/STEBinary.cs` that inherits from `CustomFunction`.

### 2. Forward Pass

#### Implementation Details
- Input: Single tensor `x`
- Output: Binary tensor where each element is:
  - `1` if x > 0
  - `-1` if x < 0
  - `0` if x == 0 (or configurable behavior)
- Save input tensor `x` for backward pass (though not strictly needed for STE)

#### Mathematical Definition
```
Forward(x):
  if x > 0: return 1
  if x < 0: return -1
  if x == 0: return 0 (or preserve original sign)
```

### 3. Backward Pass

#### Implementation Details
- Input: Single gradient tensor `grad_y`
- Output: Returns the gradient unchanged
- No computation needed - just return `grad_y`
- This is the key "straight-through" property

#### Mathematical Definition
```
Backward(grad_y):
  return grad_y  // Pass through unchanged
```

### 4. Constructor
No parameters needed for basic STE. Optional: Add a parameter to control behavior at zero.

### 5. Edge Cases to Handle
- Zero values in input (decide on behavior: return 0, 1, or -1)
- NaN values in input (should propagate NaN)
- Inf values in input (should treat accordingly)

### 6. Validation
- Validate input tensor is not null
- Validate output tensor is not null
- Validate gradient tensor is not null

## Implementation Notes

### Basic Implementation
```csharp
public class STEBinary : CustomFunction
{
    public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
    {
        var x = inputs[0];
        ctx.SaveForBackward(x);

        // Create binary tensor: 1 where x > 0, -1 where x < 0, 0 where x == 0
        var result = Tensor.Sign(x);  // Assuming framework has Sign operation
        return new[] { result };
    }

    public override Tensor[] Backward(Tensor[] grad_outputs, FunctionContext ctx)
    {
        // Straight-through: pass gradient through unchanged
        return grad_outputs;
    }
}
```

### Alternative Implementation (without Tensor.Sign)
```csharp
public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
{
    var x = inputs[0];
    ctx.SaveForBackward(x);

    // Manual binary computation
    var positive = (x > 0).CastToFloat() * 1.0;
    var negative = (x < 0).CastToFloat() * -1.0;
    var result = positive + negative;

    return new[] { result };
}
```

### Optional: Configurable Zero Behavior
```csharp
private readonly double _zeroValue;

public STEBinary(double zeroValue = 0.0)
{
    _zeroValue = zeroValue;
}

public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
{
    var x = inputs[0];
    ctx.SaveForBackward(x);

    var positive = (x > 0).CastToFloat() * 1.0;
    var negative = (x < 0).CastToFloat() * -1.0;
    var zero = (x == 0).CastToFloat() * _zeroValue;
    var result = positive + negative + zero;

    return new[] { result };
}
```

## Testing Requirements
Create unit tests in `tests/Autograd/Functions/STEBinaryTests.cs`:

1. **Forward Pass Tests**
   - Test with positive values (should return 1)
   - Test with negative values (should return -1)
   - Test with zero values (should return 0 or configured value)
   - Test with mixed values
   - Test with multi-dimensional tensors

2. **Backward Pass Tests**
   - Test that gradient is passed through unchanged
   - Test with various gradient values
   - Verify gradient shape matches input shape
   - Test with gradient = 1.0 (identity)
   - Test with gradient = 0.0

3. **Integration Tests**
   - Create a simple network using STEBinary
   - Run forward pass and verify binary output
   - Run backward pass and verify gradients flow
   - Test in a training loop context

4. **Edge Case Tests**
   - Test with NaN in input
   - Test with Inf in input
   - Test with empty tensor

5. **Gradient Checking**
   - Use numerical gradient checking to verify STE behavior
   - Compare numerical gradient with analytical gradient
   - They should match (since gradient passes through)

## Success Criteria
- [ ] STEBinary class is implemented in `src/Autograd/Functions/STEBinary.cs`
- [ ] Forward pass produces correct binary output (1, -1, 0)
- [ ] Backward pass passes gradient through unchanged
- [ ] Integration with autograd graph works correctly
- [ ] Unit tests cover all scenarios with >90% code coverage
- [ ] Optional: Zero behavior is configurable
