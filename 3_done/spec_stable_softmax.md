# Spec: StableSoftmax Custom Function

## Overview
Implement the `StableSoftmax` custom function that computes softmax with numerical stability by subtracting the maximum value before exponentiation. This prevents overflow/underflow in the exponential computation.

## Requirements

### 1. Class Definition
Create a `StableSoftmax` class in `src/Autograd/Functions/StableSoftmax.cs` that inherits from `CustomFunction`.

### 2. Forward Pass

#### Implementation Details
- Input: Single tensor `x` (typically logits)
- Output: Softmax probabilities with shape matching input
- Computes: `softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))`
- Save output tensor `y` for backward pass (needed for gradient computation)

#### Mathematical Definition
```
max_x = max(x)
exp_x = exp(x - max_x)
sum_exp = sum(exp_x)
y = exp_x / sum_exp
```

#### Dimension Handling
- Support softmax along a specific dimension (default: last dimension)
- Allow `keepDim` parameter for sum/max operations
- Provide constructor to specify dimension and keepDim

### 3. Backward Pass

#### Implementation Details
- Input: Single gradient tensor `grad_y`
- Output: Gradient with respect to input `grad_x`
- Uses the saved softmax output `y`
- Computes gradient: `grad_x = y * (grad_y - sum(grad_y * y, axis=dim, keepdim=True))`
- This is the standard softmax derivative: `dy_i/dx_j = y_i * (δ_ij - y_j)`

#### Mathematical Definition
```
Given y = softmax(x):
grad_x = y * (grad_y - sum(grad_y * y, dim=dim, keepdim=True))
```

### 4. Constructor
Parameters:
- `int dim` - Dimension along which to compute softmax (default: -1 for last dimension)
- `bool keepDim` - Whether to keep reduced dimensions (default: true)

### 5. Edge Cases to Handle
- Very large values (should be handled by max subtraction)
- Very small values (should produce small but non-zero probabilities)
- All equal values (should produce uniform distribution)
- Single element tensors (softmax should be 1.0)
- NaN/Inf in input (should propagate appropriately)

### 6. Validation
- Validate input tensor is not null
- Validate output tensor is not null
- Validate gradient tensor is not null
- Validate dimension is within valid range for tensor shape

## Implementation Notes

### Basic Implementation
```csharp
public class StableSoftmax : CustomFunction
{
    private readonly int _dim;
    private readonly bool _keepDim;

    public StableSoftmax(int dim = -1, bool keepDim = true)
    {
        _dim = dim;
        _keepDim = keepDim;
    }

    public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
    {
        var x = inputs[0];

        // Subtract max for numerical stability
        var max = x.Max(dim: _dim, keepDim: _keepDim);
        var exp = (x - max).Exp();
        var sum = exp.Sum(dim: _dim, keepDim: _keepDim);
        var y = exp / sum;

        ctx.SaveForBackward(y);
        return new[] { y };
    }

    public override Tensor[] Backward(Tensor[] grad_outputs, FunctionContext ctx)
    {
        var grad_y = grad_outputs[0];
        var y = ctx.GetSavedTensor(0);

        // Gradient: y * (grad_y - sum(grad_y * y, dim=dim, keepdim=True))
        var sum = (grad_y * y).Sum(dim: _dim, keepDim: _keepDim);
        var grad_x = y * (grad_y - sum);

        return new[] { grad_x };
    }
}
```

### Tensor Operations Assumed
The implementation assumes the following tensor operations exist:
- `Max(dim, keepDim)` - Element-wise maximum along dimension
- `Exp()` - Element-wise exponential
- `Sum(dim, keepDim)` - Sum along dimension
- Element-wise arithmetic operators (+, -, *, /)

### Broadcasting Considerations
- Ensure operations work correctly with broadcasting
- The `keepDim` parameter ensures shapes align for subtraction

## Testing Requirements
Create unit tests in `tests/Autograd/Functions/StableSoftmaxTests.cs`:

1. **Forward Pass Tests**
   - Test with small values (verify probabilities sum to 1)
   - Test with large values (verify numerical stability)
   - Test with mixed positive and negative values
   - Test with all equal values (should get uniform distribution)
   - Test with single element (should be 1.0)
   - Test with multi-dimensional tensors
   - Test with different dimensions

2. **Numerical Stability Tests**
   - Test with very large values (e.g., 1000, -1000)
   - Verify no overflow or NaN results
   - Compare with standard softmax implementation

3. **Backward Pass Tests**
   - Verify gradient shape matches input shape
   - Test gradient with uniform gradient input (grad_y = 1.0)
   - Test gradient with specific gradient patterns
   - Verify sum of gradients is zero (softmax derivative property)

4. **Integration Tests**
   - Create a simple network using StableSoftmax
   - Run forward pass and verify output properties (sum=1, all positive)
   - Run backward pass and verify gradients flow correctly
   - Test in a classification context

5. **Gradient Checking**
   - Use numerical gradient checking to verify correctness
   - Compare numerical gradient with analytical gradient
   - Test with various input values

6. **Edge Case Tests**
   - Test with NaN in input
   - Test with Inf in input
   - Test with very small values (near underflow)
   - Test with very large values (near overflow)

## Success Criteria
- [ ] StableSoftmax class is implemented in `src/Autograd/Functions/StableSoftmax.cs`
- [ ] Forward pass produces correct softmax probabilities
- [ ] Backward pass computes correct gradient
- [ ] Numerical stability is achieved (no overflow/underflow)
- [ ] Supports configurable dimension and keepDim
- [ ] Integration with autograd graph works correctly
- [ ] Unit tests cover all scenarios with >90% code coverage
- [ ] Gradient checking passes
