# Spec: Shape Mismatch Exception

## Overview
Create a custom exception class with rich diagnostic information for tensor shape mismatches.

## Requirements

### Class: ShapeMismatchException
- Inherit from `System.Exception`
- Location: `src/Exceptions/ShapeMismatchException.cs`

### Properties
```csharp
public string LayerName { get; set; }
public OperationType OperationType { get; set; }
public List<long[]> InputShapes { get; set; }
public List<long[]> ExpectedShapes { get; set; }
public string ProblemDescription { get; set; }
public List<string> SuggestedFixes { get; set; }
```

### Methods
```csharp
// Generate formatted diagnostic report
public string GetDiagnosticReport()
```

### Output Format
```
MLFramework.ShapeMismatchException: Matrix multiplication failed in layer 'encoder.fc2'

Input shape:    [32, 256]
Expected:       [*, 128]

Problem: Dimension 1 of input (256) does not match dimension 0 of weight (128)

Suggested fixes:
1. Check layer configuration
2. Verify input features match expected dimension
```

### Dependencies
- `OperationType` enum (to be defined in spec_operation_metadata_registry.md)

## Tests
- Create `tests/Exceptions/ShapeMismatchExceptionTests.cs`
- Test property initialization
- Test GetDiagnosticReport() output format
- Test exception serialization/deserialization

## Success Criteria
- [ ] ShapeMismatchException class created with all properties
- [ ] GetDiagnosticReport() generates formatted output
- [ ] Unit tests pass
- [ ] Code follows project coding standards
