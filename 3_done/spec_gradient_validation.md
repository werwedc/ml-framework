# Spec: Gradient Validation System

## Overview
Implement gradient validation utilities to ensure custom functions return gradients with correct shapes, types, and values. This system provides clear error messages when custom gradient implementations are incorrect.

## Requirements

### 1. Class Definition
Create a `GradientValidator` class in `src/Autograd/GradientValidator.cs` with static validation methods.

### 2. Core Validation Methods

#### ValidateGradientShape(Tensor gradient, Tensor input, string parameterName = "gradient")
- Validates that gradient tensor shape matches input tensor shape
- Throws `GradientShapeException` if shapes don't match
- Provides detailed error message: "Gradient shape {gradShape} does not match input shape {inputShape} for parameter '{parameterName}'"
- Handles broadcastable gradients (if framework supports broadcasting)

#### ValidateGradientType(Tensor gradient, Tensor input, string parameterName = "gradient")
- Validates that gradient tensor dtype matches input tensor dtype
- Throws `GradientTypeException` if dtypes don't match
- Provides detailed error message: "Gradient dtype {gradDtype} does not match input dtype {inputDtype} for parameter '{parameterName}'"

#### ValidateGradients(Tensor[] gradients, Tensor[] inputs)
- Validates that gradients array has same length as inputs array
- Calls ValidateGradientShape() and ValidateGradientType() for each pair
- Allows null gradients (when input doesn't require grad)
- Provides aggregate error message listing all validation failures

#### ValidateGradientHasNoNaN(Tensor gradient, string parameterName = "gradient")
- Checks if gradient contains NaN values
- Throws `GradientNaNException` if NaN detected
- Provides location information: "Gradient for parameter '{parameterName}' contains NaN values"

#### ValidateGradientHasNoInf(Tensor gradient, string parameterName = "gradient")
- Checks if gradient contains infinite values
- Throws `GradientInfException` if Inf detected
- Provides location information: "Gradient for parameter '{parameterName}' contains Inf values"

### 3. Exception Classes

Create custom exception classes in `src/Autograd/Exceptions/`:

#### GradientShapeException
- Inherits from InvalidOperationException
- Stores expected shape and actual shape
- Properties: ExpectedShape, ActualShape, ParameterName

#### GradientTypeException
- Inherits from InvalidOperationException
- Stores expected dtype and actual dtype
- Properties: ExpectedDtype, ActualDtype, ParameterName

#### GradientNaNException
- Inherits from InvalidOperationException
- Stores parameter name and tensor index of first NaN
- Properties: ParameterName, NaNIndex

#### GradientInfException
- Inherits from InvalidOperationException
- Stores parameter name and tensor index of first Inf
- Properties: ParameterName, InfIndex, IsPositiveInfinity

### 4. Gradient Comparison Utilities

#### AreGradientsEqual(Tensor grad1, Tensor grad2, double tolerance = 1e-6)
- Compares two gradient tensors for approximate equality
- Returns true if all elements are within tolerance
- Returns false if shapes don't match
- Useful for gradient checking tests

#### GetGradientDifference(Tensor grad1, Tensor grad2)
- Computes absolute difference between two gradients
- Returns new tensor with element-wise absolute differences
- Useful for debugging gradient implementations

### 5. Shape Comparison Helper

#### AreShapesCompatible(Tensor shape1, Tensor shape2)
- Checks if two tensor shapes are compatible (same or broadcastable)
- Returns true if shapes match or can be broadcast together
- Used in ValidateGradientShape to support broadcasting

## Implementation Notes

### GradientValidationResult Class
```csharp
public class GradientValidationResult
{
    public bool IsValid { get; set; }
    public List<string> Errors { get; set; } = new List<string>();
    public List<string> Warnings { get; set; } = new List<string>();
}
```

This allows collecting all validation errors at once rather than throwing on first error.

### Aggregate Validation
```csharp
public static GradientValidationResult ValidateGradientsAggregate(
    Tensor[] gradients,
    Tensor[] inputs,
    bool checkNaN = true,
    bool checkInf = true)
{
    var result = new GradientValidationResult();

    // Validate count
    if (gradients.Length != inputs.Length)
    {
        result.Errors.Add($"Gradient count ({gradients.Length}) does not match input count ({inputs.Length})");
        result.IsValid = false;
        return result;
    }

    // Validate each gradient
    for (int i = 0; i < gradients.Length; i++)
    {
        var grad = gradients[i];
        var input = inputs[i];

        if (grad != null)
        {
            // Shape check
            if (!AreShapesCompatible(grad, input))
                result.Errors.Add($"Gradient {i} shape {grad.Shape} incompatible with input shape {input.Shape}");

            // Type check
            if (grad.DataType != input.DataType)
                result.Errors.Add($"Gradient {i} dtype {grad.DataType} does not match input dtype {input.DataType}");

            // NaN/Inf checks
            if (checkNaN && ContainsNaN(grad))
                result.Errors.Add($"Gradient {i} contains NaN values");

            if (checkInf && ContainsInf(grad))
                result.Errors.Add($"Gradient {i} contains Inf values");
        }
    }

    result.IsValid = result.Errors.Count == 0;
    return result;
}
```

## Testing Requirements
Create unit tests in `tests/Autograd/GradientValidatorTests.cs`:

1. **Shape Validation Tests**
   - Test with matching shapes (should pass)
   - Test with mismatching shapes (should throw)
   - Test with broadcastable shapes (should pass if supported)

2. **Type Validation Tests**
   - Test with matching dtypes (should pass)
   - Test with mismatching dtypes (should throw)

3. **Array Validation Tests**
   - Test with correct number of gradients
   - Test with too few gradients (should throw)
   - Test with too many gradients (should throw)
   - Test with null gradients (should pass - indicates no gradient needed)

4. **NaN/Inf Detection Tests**
   - Test gradients without NaN/Inf (should pass)
   - Test gradients with NaN (should throw)
   - Test gradients with Inf (should throw)
   - Test gradients with both positive and negative infinity

5. **Aggregate Validation Tests**
   - Test with multiple validation failures
   - Verify all errors are collected
   - Test with valid gradients (no errors)

## Success Criteria
- [ ] GradientValidator class is implemented in `src/Autograd/GradientValidator.cs`
- [ ] All four custom exception classes are implemented
- [ ] Shape, type, NaN, and Inf validation work correctly
- [ ] Aggregate validation collects all errors
- [ ] Helper utilities (AreGradientsEqual, GetGradientDifference) work correctly
- [ ] All error messages are clear and informative
- [ ] Unit tests cover all scenarios with >90% code coverage
