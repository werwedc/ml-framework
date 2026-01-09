# Spec: Shape Validation Logic

## Overview
Implement comprehensive shape validation logic that integrates with existing operations to catch shape mismatches early.

## Requirements

### Class: ShapeValidator
- Location: `src/Core/ShapeValidator.cs`

```csharp
public class ShapeValidator
{
    // Validate before matrix multiplication
    public static ValidationResult ValidateMatrixMultiply(long[] shape1, long[] shape2)

    // Validate before convolution
    public static ValidationResult ValidateConv2D(
        long[] inputShape,
        long[] kernelShape,
        int stride,
        int padding)

    // Validate before concatenation
    public static ValidationResult ValidateConcat(
        List<long[]> inputShapes,
        int axis)

    // Validate before broadcast operation
    public static ValidationResult ValidateBroadcast(
        long[] shape1,
        long[] shape2)

    // Generic validation using registry
    public static ValidationResult Validate(
        OperationType operationType,
        params long[][] inputShapes)
}
```

### Validation Rules

**MatrixMultiply:**
- Shape1: [M, K]
- Shape2: [K, N]
- Error if K dimensions don't match

**Conv2D:**
- Input: [N, C, H, W]
- Kernel: [F, C, kH, kW]
- Error if input channels C don't match kernel channels C

**Concat:**
- All shapes must match except on concatenation axis
- Error if dimensions differ on non-concat axes

**Broadcast:**
- Two shapes are compatible if:
  - They are equal, or
  - One dimension is 1, or
  - One dimension doesn't exist

### Integration Helper Methods
```csharp
public class ShapeValidationHelper
{
    // Helper to create ShapeMismatchException from ValidationResult
    public static ShapeMismatchException CreateException(
        ValidationResult result,
        OperationType operationType,
        string layerName,
        params long[][] inputShapes)

    // Helper to extract problem description
    public static string BuildProblemDescription(
        OperationType operationType,
        params long[][] inputShapes)
}
```

## Tests
- Create `tests/Core/ShapeValidatorTests.cs`
- Test MatrixMultiply validation (valid and invalid cases)
- Test Conv2D validation (valid and invalid cases)
- Test Concat validation (valid and invalid cases)
- Test Broadcast validation (various scenarios)
- Test generic validation through registry
- Test exception creation helper

## Success Criteria
- [ ] ShapeValidator class with all validation methods
- [ ] Correct validation rules for each operation type
- [ ] ShapeValidationHelper for exception creation
- [ ] Comprehensive unit tests covering edge cases
- [ ] Clear error messages in ValidationResult
