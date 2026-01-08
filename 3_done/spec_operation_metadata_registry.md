# Spec: Operation Metadata Registry

## Overview
Create a centralized registry of shape requirements and validation rules for all ML operations.

## Requirements

### Enum: OperationType
- Location: `src/Core/OperationType.cs`

```csharp
public enum OperationType
{
    MatrixMultiply,
    Linear,
    Conv2D,
    Concat,
    Stack,
    ReduceSum,
    ReduceMean,
    Transpose,
    Reshape,
    Broadcast
}
```

### Interface: IOperationMetadata
- Location: `src/Core/IOperationMetadata.cs`

```csharp
public interface IOperationMetadata
{
    OperationType Type { get; }
    string Name { get; }
    int RequiredInputTensors { get; }
    bool ValidateInputShapes(params long[][] inputShapes);
    long[] InferOutputShape(params long[][] inputShapes);
}
```

### Class: OperationMetadataRegistry
- Location: `src/Core/OperationMetadataRegistry.cs`

```csharp
public class OperationMetadataRegistry
{
    // Singleton instance
    public static OperationMetadataRegistry Instance { get; }

    // Register operation metadata
    public void Register(OperationType type, IOperationMetadata metadata)

    // Get metadata for operation type
    public IOperationMetadata GetMetadata(OperationType type)

    // Validate operation inputs
    public ValidationResult Validate(OperationType type, params long[][] inputShapes)

    // Infer output shape
    public long[] InferOutputShape(OperationType type, params long[][] inputShapes)
}
```

### Struct: ValidationResult
- Location: `src/Core/ValidationResult.cs`

```csharp
public struct ValidationResult
{
    public bool IsValid { get; }
    public string ErrorMessage { get; }
    public List<string> SuggestedFixes { get; }
}
```

### Initial Operation Implementations
Create metadata for basic operations in `src/Core/Operations/`:
- `MatrixMultiplyMetadata.cs` - Validate inner dimensions match
- `Conv2DMetadata.cs` - Validate channel dimensions
- `ConcatMetadata.cs` - Validate all dimensions except concatenation axis match

## Tests
- Create `tests/Core/OperationMetadataRegistryTests.cs`
- Test singleton pattern
- Test registration and retrieval
- Test validation for MatrixMultiply
- Test validation for Conv2D
- Test validation for Concat
- Test shape inference

## Success Criteria
- [ ] OperationType enum defined
- [ ] IOperationMetadata interface defined
- [ ] OperationMetadataRegistry implemented with singleton
- [ ] Metadata implementations for basic operations
- [ ] Unit tests pass
- [ ] Extensible design for adding new operations
