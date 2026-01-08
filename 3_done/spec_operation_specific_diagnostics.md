# Spec: Operation-Specific Diagnostics

## Overview
Implement diagnostic formatters that generate human-readable messages for different operation types.

## Requirements

### Interface: IDiagnosticFormatter
- Location: `src/Diagnostics/IDiagnosticFormatter.cs`

```csharp
public interface IDiagnosticFormatter
{
    OperationType SupportedOperation { get; }
    string FormatError(ValidationResult result, params long[][] inputShapes);
    List<string> GenerateSuggestions(ValidationResult result);
}
```

### Class: DiagnosticFormatterRegistry
- Location: `src/Diagnostics/DiagnosticFormatterRegistry.cs`

```csharp
public class DiagnosticFormatterRegistry
{
    public static DiagnosticFormatterRegistry Instance { get; }
    public void Register(IDiagnosticFormatter formatter);
    public string FormatError(OperationType type, ValidationResult result, params long[][] inputShapes);
    public List<string> GetSuggestions(OperationType type, ValidationResult result);
}
```

### Formatter Implementations

**MatrixMultiplyDiagnosticFormatter**
- Location: `src/Diagnostics/MatrixMultiplyDiagnosticFormatter.cs`

```csharp
// Output example:
// "Matrix multiplication: Shape [32, 256] × [128, 10] invalid"
// "Problem: Inner dimensions 256 and 128 must match"
```

**Conv2DDiagnosticFormatter**
- Location: `src/Diagnostics/Conv2DDiagnosticFormatter.cs`

```csharp
// Output example:
// "Conv2D: Input [32, 3, 224, 224] with kernel [64, 3, 3, 3]"
// "Problem: Input channels (3) match kernel channels (3)"
// "Output shape: [32, 64, 222, 222]"
```

**ConcatDiagnosticFormatter**
- Location: `src/Diagnostics/ConcatDiagnosticFormatter.cs`

```csharp
// Output example:
// "Concat on axis 1: [32, 128] + [32, 64]"
// "Valid: All dimensions match except on axis 1"
// "Output shape: [32, 192]"
```

**BroadcastDiagnosticFormatter**
- Location: `src/Diagnostics/BroadcastDiagnosticFormatter.cs`

```csharp
// Output example:
// "Broadcast: [32, 1] → [32, 10]"
// "Problem: Cannot broadcast dimension 1 (size 1) to size 10 without compatible shape"
```

### Suggestion Generation
Each formatter should provide relevant suggestions:
- MatrixMultiply: Check layer configurations, adjust input/output features
- Conv2D: Check channel configurations, verify kernel parameters
- Concat: Check tensor shapes, consider reshape operations
- Broadcast: Check broadcasting rules, consider explicit expansion

## Tests
- Create `tests/Diagnostics/DiagnosticFormatterTests.cs`
- Test each formatter's error messages
- Test suggestion generation for each operation
- Test registry registration and retrieval
- Test with various shape mismatch scenarios

## Success Criteria
- [ ] IDiagnosticFormatter interface defined
- [ ] DiagnosticFormatterRegistry implemented
- [ ] Formatters for MatrixMultiply, Conv2D, Concat, Broadcast
- [ ] Clear, actionable error messages
- [ ] Relevant suggestions for each operation type
- [ ] Unit tests pass
