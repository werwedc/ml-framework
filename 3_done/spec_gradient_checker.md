# Spec: Gradient Checking Utilities

## Overview
Implement gradient checking utilities that numerically verify custom function gradients using finite differences. This provides a robust way to validate that custom gradient implementations are correct.

## Requirements

### 1. Class Definition
Create a `GradientChecker` class in `src/Autograd/GradientChecker.cs` with static methods for numerical gradient checking.

### 2. Core Method

#### CheckGradients(CustomFunction function, Tensor[] inputs, double epsilon = 1e-6, double tolerance = 1e-4, bool verbose = false)
- Verifies that the backward pass gradients match numerical gradients
- Returns a `GradientCheckResult` object with detailed information
- Uses finite difference method to compute numerical gradients
- Supports functions with multiple inputs and outputs

#### Algorithm (Finite Difference Method)
```
For each input tensor i:
  For each element j:
    1. Save original value: orig = input[i][j]
    2. Perturb positively: input[i][j] = orig + epsilon
    3. Compute f_plus = forward(inputs)
    4. Perturb negatively: input[i][j] = orig - epsilon
    5. Compute f_minus = forward(inputs)
    6. Restore: input[i][j] = orig
    7. numerical_grad[i][j] = (f_plus - f_minus) / (2 * epsilon)

Compare numerical_grad with analytical_grad from backward()
```

### 3. Helper Methods

#### ComputeNumericalGradient(CustomFunction function, Tensor[] inputs, int outputIndex = 0, double epsilon = 1e-6)
- Computes numerical gradient for a specific output
- Returns array of gradient tensors (one per input)
- Uses central difference method for better accuracy

#### CompareGradients(Tensor[] numerical, Tensor[] analytical, double tolerance = 1e-4)
- Compares numerical and analytical gradients element-wise
- Returns true if all elements are within tolerance
- Returns false if any element exceeds tolerance
- Provides maximum absolute difference

#### ComputeRelativeError(Tensor numerical, Tensor analytical)
- Computes relative error between two tensors
- Formula: `|numerical - analytical| / max(|numerical|, |analytical|, epsilon)`
- Returns a new tensor with relative errors
- Handles zero values gracefully

### 4. Result Object

Create `GradientCheckResult` class in `src/Autograd/GradientCheckResult.cs`:
```csharp
public class GradientCheckResult
{
    public bool Passed { get; set; }
    public double MaxAbsoluteDifference { get; set; }
    public double MaxRelativeError { get; set; }
    public List<TensorDifference> Differences { get; set; } = new List<TensorDifference>();
    public string FailureReason { get; set; }
}

public class TensorDifference
{
    public int InputIndex { get; set; }
    public int[] ElementIndex { get; set; }
    public double NumericalValue { get; set; }
    public double AnalyticalValue { get; set; }
    public double AbsoluteDifference { get; set; }
    public double RelativeError { get; set; }
}
```

### 5. Optimization Features

#### Vectorized Checking (Advanced)
- Instead of checking each element individually, check entire tensors
- Compute numerical gradients by perturbing the entire tensor
- Much faster for large tensors

#### Parallel Checking (Advanced)
- Check multiple inputs in parallel
- Check multiple elements within a tensor in parallel
- Requires thread-safe tensor operations

#### Sparse Checking (Optional)
- Check only a random subset of elements for large tensors
- Faster for very large tensors
- Provides probabilistic validation

### 6. Multiple Outputs Support

For functions with multiple outputs:
- Check gradients for each output separately
- Use `outputIndex` parameter to specify which output to check
- Chain rule: `dLoss/dInput = sum(dLoss/dOutput * dOutput/dInput)`

### 7. Edge Cases to Handle
- Inputs that don't require grad (skip)
- Discontinuous functions (may have large numerical errors)
- Functions with very small gradients (use relative error)
- Very large tensors (use sparse checking or vectorized approach)
- NaN/Inf in gradients

## Implementation Notes

### Basic Implementation
```csharp
public static class GradientChecker
{
    public static GradientCheckResult CheckGradients(
        CustomFunction function,
        Tensor[] inputs,
        double epsilon = 1e-6,
        double tolerance = 1e-4,
        bool verbose = false)
    {
        var result = new GradientCheckResult();

        // Compute analytical gradients via backward pass
        var outputs = function.ApplyMany(inputs);
        var loss = outputs.Sum();  // Scalar loss for gradient checking
        var gradLoss = Tensor.OnesLike(loss);
        var analyticalGrads = function.Backward(new[] { gradLoss }, null);

        // Compute numerical gradients
        var numericalGrads = new Tensor[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            if (inputs[i].RequiresGrad)
            {
                numericalGrads[i] = ComputeNumericalGradient(function, inputs, i, epsilon);
            }
        }

        // Compare gradients
        result.Passed = true;
        result.MaxAbsoluteDifference = 0.0;
        result.MaxRelativeError = 0.0;

        for (int i = 0; i < analyticalGrads.Length; i++)
        {
            if (analyticalGrads[i] != null && numericalGrads[i] != null)
            {
                var comparison = CompareTensors(analyticalGrads[i], numericalGrads[i], tolerance);
                if (!comparison.WithinTolerance)
                {
                    result.Passed = false;
                    result.Differences.AddRange(comparison.Differences);
                }
                result.MaxAbsoluteDifference = Math.Max(result.MaxAbsoluteDifference, comparison.MaxAbsDiff);
                result.MaxRelativeError = Math.Max(result.MaxRelativeError, comparison.MaxRelError);
            }
        }

        return result;
    }

    private static Tensor ComputeNumericalGradient(
        CustomFunction function,
        Tensor[] inputs,
        int inputIndex,
        double epsilon)
    {
        var input = inputs[inputIndex];
        var grad = Tensor.ZerosLike(input);

        for (int i = 0; i < input.NumberOfElements; i++)
        {
            var orig = input[i];

            // Positive perturbation
            input[i] = orig + epsilon;
            var fPlus = function.ApplyMany(inputs).Sum();

            // Negative perturbation
            input[i] = orig - epsilon;
            var fMinus = function.ApplyMany(inputs).Sum();

            // Restore
            input[i] = orig;

            // Central difference
            grad[i] = (fPlus - fMinus) / (2 * epsilon);
        }

        return grad;
    }
}
```

### Comparison Implementation
```csharp
private static TensorComparison CompareTensors(
    Tensor analytical,
    Tensor numerical,
    double tolerance)
{
    var comparison = new TensorComparison();

    for (int i = 0; i < analytical.NumberOfElements; i++)
    {
        var ana = analytical[i];
        var num = numerical[i];
        var absDiff = Math.Abs(ana - num);
        var relError = Math.Abs(ana - num) / Math.Max(Math.Abs(ana), Math.Abs(num), epsilon);

        if (absDiff > tolerance && relError > tolerance)
        {
            comparison.WithinTolerance = false;
            comparison.Differences.Add(new TensorDifference
            {
                ElementIndex = new[] { i },
                NumericalValue = num,
                AnalyticalValue = ana,
                AbsoluteDifference = absDiff,
                RelativeError = relError
            });
        }

        comparison.MaxAbsDiff = Math.Max(comparison.MaxAbsDiff, absDiff);
        comparison.MaxRelError = Math.Max(comparison.MaxRelError, relError);
    }

    return comparison;
}
```

## Testing Requirements
Create unit tests in `tests/Autograd/GradientCheckerTests.cs`:

1. **Gradient Checker Tests on Known Functions**
   - Test on simple linear function (should pass easily)
   - Test on squared function (should pass)
   - Test on STEBinary (should pass with known gradient = input)
   - Test on StableSoftmax (should pass with correct gradient)
   - Test on ClampedMSELoss (should pass with correct gradient)

2. **Accuracy Tests**
   - Verify numerical gradient is close to analytical gradient
   - Test with different epsilon values (1e-7, 1e-6, 1e-5)
   - Verify tolerance affects pass/fail correctly

3. **Multiple Input Tests**
   - Test function with multiple inputs
   - Verify all gradients are checked

4. **Edge Case Tests**
   - Test with zero gradients
   - Test with very small gradients
   - Test with NaN gradients
   - Test with functions that don't require grad on all inputs

5. **Integration Tests**
   - Use gradient checker to validate all custom functions
   - Ensure all example functions pass gradient check

## Success Criteria
- [ ] GradientChecker class is implemented in `src/Autograd/GradientChecker.cs`
- [ ] GradientCheckResult class is implemented with all properties
- [ ] Numerical gradient computation works correctly
- [ ] Gradient comparison works correctly
- [ ] All example custom functions pass gradient checking
- [ ] Unit tests cover all scenarios with >90% code coverage
- [ ] Tool is usable for validating new custom functions
