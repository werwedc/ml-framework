# Technical Spec: Operation Metadata Registry

## Overview
Create a central registry that stores shape requirements and validation rules for all supported operations. This registry will be used by the diagnostics system to understand what shapes are expected for each operation type.

## Requirements

### Registry Interface
```csharp
public interface IOperationMetadataRegistry
{
    // Register shape requirements for an operation
    void RegisterOperation(
        OperationType operationType,
        OperationShapeRequirements requirements);

    // Get shape requirements for an operation
    OperationShapeRequirements GetRequirements(OperationType operationType);

    // Check if operation is registered
    bool IsRegistered(OperationType operationType);

    // Validate shapes against operation requirements
    ValidationResult ValidateShapes(
        OperationType operationType,
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters = null);
}
```

### OperationShapeRequirements Class
```csharp
public class OperationShapeRequirements
{
    // Number of input tensors required
    public int InputCount { get; set; }

    // Expected dimension count for each input
    // e.g., new[] { 2, 2 } for matrix multiply (2D x 2D)
    public int[] ExpectedDimensions { get; set; }

    // Dimension constraints: key=input_index, value=(dimension_index, constraint)
    // e.g., { [0, 1]: (1, "must_match", [1, 0]) }
    // Means: input[0].dim[1] must match input[1].dim[0]
    public Dictionary<int, Dictionary<int, DimensionConstraint>> DimensionConstraints { get; set; }

    // Human-readable description of shape requirements
    public string Description { get; set; }

    // Format string for error messages
    public string ErrorMessageFormat { get; set; }

    // Optional: Custom validation logic
    public Func<IEnumerable<long[]>, IDictionary<string, object>, ValidationResult> CustomValidator { get; set; }
}
```

### DimensionConstraint Class
```csharp
public class DimensionConstraint
{
    public enum ConstraintType
    {
        MustMatch,           // Must match another dimension
        MustEqual,           // Must equal a specific value
        MustBePositive,      // Must be > 0
        MustBeMultipleOf,    // Must be multiple of a value
        MustDivide,          // Must divide another dimension evenly
        Any                  // Any value is acceptable
    }

    public ConstraintType Type { get; set; }
    public int? TargetInputIndex { get; set; }
    public int? TargetDimensionIndex { get; set; }
    public long? FixedValue { get; set; }
    public long? MultipleOf { get; set; }
}
```

### ValidationResult Class
```csharp
public class ValidationResult
{
    public bool IsValid { get; set; }
    public List<string> Errors { get; set; }
    public List<string> Warnings { get; set; }

    public static ValidationResult Success()
    {
        return new ValidationResult { IsValid = true };
    }

    public static ValidationResult Failure(params string[] errors)
    {
        return new ValidationResult
        {
            IsValid = false,
            Errors = new List<string>(errors)
        };
    }
}
```

### Default Implementation
Create `DefaultOperationMetadataRegistry` implementing `IOperationMetadataRegistry`:
- Pre-register common operations (MatrixMultiply, Conv2D, Concat, Stack, etc.)
- Use concurrent dictionary for thread-safe operations
- Provide method to register custom operations

### Pre-Registered Operations

#### Matrix Multiply
```csharp
new OperationShapeRequirements
{
    InputCount = 2,
    ExpectedDimensions = new[] { 2, 2 },
    DimensionConstraints = new Dictionary<int, Dictionary<int, DimensionConstraint>>
    {
        {
            0, new Dictionary<int, DimensionConstraint>
            {
                { 1, new DimensionConstraint { Type = DimensionConstraint.ConstraintType.MustMatch,
                                                TargetInputIndex = 1,
                                                TargetDimensionIndex = 0 } }
            }
        }
    },
    Description = "Matrix multiplication: [batch, m] × [m, n] → [batch, n]",
    ErrorMessageFormat = "Dimension 1 of input ({0}) does not match dimension 0 of weight ({1})"
}
```

#### Conv2D
```csharp
new OperationShapeRequirements
{
    InputCount = 2,
    ExpectedDimensions = new[] { 4, 4 }, // NCHW format
    Description = "2D Convolution: [N, C_in, H, W] × [C_out, C_in, kH, kW] → [N, C_out, H_out, W_out]",
    CustomValidator = ValidateConv2DShapes
}
```

## Deliverables
- File: `src/Diagnostics/IOperationMetadataRegistry.cs`
- File: `src/Diagnostics/OperationShapeRequirements.cs`
- File: `src/Diagnostics/DimensionConstraint.cs`
- File: `src/Diagnostics/ValidationResult.cs`
- File: `src/Diagnostics/DefaultOperationMetadataRegistry.cs`

## Testing Requirements
Create unit tests in `tests/Diagnostics/OperationMetadataRegistryTests.cs`:
- Test registration and retrieval of operations
- Test shape validation for MatrixMultiply
- Test shape validation for Conv2D
- Test dimension constraint matching
- Test custom validators
- Test error messages generation
- Test pre-registered operations

## Notes
- Registry should be thread-safe (use ConcurrentDictionary)
- Provide clear error messages for invalid registrations
- Consider making registry a singleton for easy access
- Support both NCHW and NHWC formats for convolution operations
- Custom validators can handle complex logic like output shape calculations
