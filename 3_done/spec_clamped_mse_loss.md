# Spec: ClampedMSELoss Custom Function

## Overview
Implement the `ClampedMSELoss` custom function that computes Mean Squared Error loss with clamped differences. This prevents extreme gradient values when predictions are far from targets, providing robustness against outliers.

## Requirements

### 1. Class Definition
Create a `ClampedMSELoss` class in `src/Autograd/Functions/ClampedMSELoss.cs` that inherits from `CustomFunction`.

### 2. Forward Pass

#### Implementation Details
- Input: Two tensors - `predictions` and `targets`
- Output: Single scalar tensor representing the clamped MSE loss
- Computes: `mean((clamp(preds - targets, min, max))^2)`
- Save predictions, targets, and difference for backward pass

#### Mathematical Definition
```
diff = predictions - targets
clamped_diff = clamp(diff, min, max)
loss = mean(clamped_diff^2)
```

#### Reduction Handling
- Support reduction modes: 'mean' (default), 'sum', 'none'
- For 'none': return per-element losses
- For 'sum': sum all clamped squared differences
- For 'mean': average all clamped squared differences

### 3. Backward Pass

#### Implementation Details
- Input: Single gradient tensor `grad_loss`
- Output: Two gradients - `grad_preds` and `grad_targets`
- Uses saved tensors: predictions, targets, difference
- Computes gradients only for unclamped differences (where mask is true)
- Gradient formula: `2 * clamped_diff / num_elements * mask`

#### Mathematical Definition
```
mask = (diff >= min) & (diff <= max)
grad_unclamped = 2 * clamped_diff / num_elements
grad_preds = grad_loss * grad_unclamped * mask
grad_targets = grad_loss * (-grad_unclamped * mask)
```

#### Explanation
- When difference is clamped, gradient is zero (no learning from outliers)
- When difference is unclamped, gradient is standard MSE gradient
- Target gradient is negative of prediction gradient (standard MSE property)

### 4. Constructor
Parameters:
- `double clampMin` - Minimum clamp value (e.g., -1.0)
- `double clampMax` - Maximum clamp value (e.g., 1.0)
- `string reduction` - Reduction mode: 'mean', 'sum', or 'none' (default: 'mean')

### 5. Edge Cases to Handle
- Empty tensors
- Zero-dimension tensors
- All differences clamped (gradient should be zero)
- No differences clamped (should behave like standard MSE)
- Single element tensors
- NaN/Inf in input

### 6. Validation
- Validate predictions and targets are not null
- Validate predictions and targets have same shape
- Validate clampMin < clampMax
- Validate reduction parameter is valid
- Validate output is not null

## Implementation Notes

### Basic Implementation
```csharp
public class ClampedMSELoss : CustomFunction
{
    private readonly double _clampMin;
    private readonly double _clampMax;
    private readonly string _reduction;

    public ClampedMSELoss(double clampMin, double clampMax, string reduction = "mean")
    {
        _clampMin = clampMin;
        _clampMax = clampMax;
        _reduction = reduction;
    }

    public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
    {
        var predictions = inputs[0];
        var targets = inputs[1];

        var diff = predictions - targets;
        var clampedDiff = Tensor.Clamp(diff, _clampMin, _clampMax);

        Tensor loss;
        switch (_reduction.ToLower())
        {
            case "none":
                loss = clampedDiff * clampedDiff;
                break;
            case "sum":
                loss = (clampedDiff * clampedDiff).Sum();
                break;
            case "mean":
            default:
                loss = (clampedDiff * clampedDiff).Mean();
                break;
        }

        ctx.SaveForBackward(predictions, targets, diff);
        return new[] { loss };
    }

    public override Tensor[] Backward(Tensor[] grad_outputs, FunctionContext ctx)
    {
        var grad_loss = grad_outputs[0];
        var predictions = ctx.GetSavedTensor(0);
        var targets = ctx.GetSavedTensor(1);
        var diff = ctx.GetSavedTensor(2);

        var clampedDiff = Tensor.Clamp(diff, _clampMin, _clampMax);
        var mask = (diff >= _clampMin) & (diff <= _clampMax);

        Tensor grad_unclamped;
        switch (_reduction.ToLower())
        {
            case "none":
                grad_unclamped = 2 * clampedDiff;
                break;
            case "sum":
                grad_unclamped = 2 * clampedDiff;
                break;
            case "mean":
            default:
                grad_unclamped = 2 * clampedDiff / predictions.NumberOfElements;
                break;
        }

        var grad_preds = grad_loss * grad_unclamped * mask;
        var grad_targets = grad_loss * (-grad_unclamped * mask);

        return new[] { grad_preds, grad_targets };
    }
}
```

### Tensor Operations Assumed
- Element-wise arithmetic (+, -, *, /)
- Comparison operators (>=, <=)
- Logical AND (&) for mask creation
- Tensor.Clamp(value, min, max)
- .Mean() and .Sum() reduction operations
- .NumberOfElements property

### Broadcasting Considerations
- Predictions and targets should have same shape (no broadcasting)
- Gradient tensors should have same shape as inputs

## Testing Requirements
Create unit tests in `tests/Autograd/Functions/ClampedMSELossTests.cs`:

1. **Forward Pass Tests**
   - Test with no clamping (difference within range) - should match standard MSE
   - Test with partial clamping (some differences exceed range)
   - Test with all clamping (all differences exceed range)
   - Test with different clamp values
   - Test with multi-dimensional tensors
   - Test each reduction mode (mean, sum, none)

2. **Backward Pass Tests**
   - Verify gradient shapes match input shapes
   - Test gradient when no clamping (should match standard MSE gradient)
   - Test gradient when all clamped (should be zero)
   - Test gradient when partial clamping
   - Verify pred gradient is negative of target gradient

3. **Integration Tests**
   - Create a simple network using ClampedMSELoss
   - Run forward pass and verify loss computation
   - Run backward pass and verify gradients
   - Compare with standard MSE when clamps are wide enough

4. **Gradient Checking**
   - Use numerical gradient checking to verify correctness
   - Compare numerical gradient with analytical gradient
   - Test with various clamp values

5. **Edge Case Tests**
   - Test with perfect predictions (loss should be 0)
   - Test with identical predictions and targets
   - Test with empty tensors (if supported)
   - Test with NaN/Inf in input
   - Test with clampMin == clampMax (edge case)

6. **Robustness Tests**
   - Test with extreme outliers (should clamp and have zero gradient)
   - Verify gradient is zero for clamped elements
   - Verify gradient is non-zero for unclamped elements

## Success Criteria
- [ ] ClampedMSELoss class is implemented in `src/Autograd/Functions/ClampedMSELoss.cs`
- [ ] Forward pass computes correct clamped MSE loss
- [ ] Backward pass computes correct gradients
- [ ] All three reduction modes work correctly (mean, sum, none)
- [ ] Gradients are zero for clamped differences
- [ ] Integration with autograd graph works correctly
- [ ] Unit tests cover all scenarios with >90% code coverage
- [ ] Gradient checking passes
