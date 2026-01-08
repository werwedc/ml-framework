# Technical Spec: Shape Mismatch Exception

## Overview
Create a custom exception class `ShapeMismatchException` that provides rich diagnostic information for tensor shape mismatches in the ML framework.

## Requirements

### Core Exception Properties
The exception must expose the following properties:
- `LayerName` (string) - Name of the layer/module where error occurred
- `OperationType` (enum) - Type of operation (MatrixMultiply, Conv2D, Concat, etc.)
- `InputShapes` (IEnumerable<long[]>) - Shapes of input tensors
- `ExpectedShapes` (IEnumerable<long[]>) - Expected shapes for the operation
- `ProblemDescription` (string) - Human-readable description of the problem
- `SuggestedFixes` (IReadOnlyList<string>) - List of suggested fixes
- `BatchSize` (long?) - Batch size if applicable
- `PreviousLayerContext` (string) - Context about previous layer

### Exception Interface
```csharp
public class ShapeMismatchException : Exception
{
    // Properties (listed above)

    public ShapeMismatchException(
        string layerName,
        OperationType operationType,
        IEnumerable<long[]> inputShapes,
        IEnumerable<long[]> expectedShapes,
        string problemDescription,
        IEnumerable<string> suggestedFixes = null,
        long? batchSize = null,
        string previousLayerContext = null)
        : base(GenerateMessage(layerName, operationType, inputShapes, expectedShapes))
    {
        // Initialize all properties
    }

    // Get formatted diagnostic report
    public string GetDiagnosticReport();

    // Optional: Visualize shape flow (can be null/throw NotImplementedException for now)
    public void VisualizeShapeFlow(string outputPath)
    {
        // Can be implemented in future spec
        throw new NotImplementedException("Shape visualization not yet implemented");
    }

    private static string GenerateMessage(
        string layerName,
        OperationType operationType,
        IEnumerable<long[]> inputShapes,
        IEnumerable<long[]> expectedShapes)
    {
        // Generate concise exception message
        return $"Shape mismatch in layer '{layerName}' during {operationType} operation";
    }
}
```

### OperationType Enum
```csharp
public enum OperationType
{
    MatrixMultiply,
    Conv2D,
    Conv1D,
    MaxPool2D,
    AveragePool2D,
    Concat,
    Stack,
    Reshape,
    Transpose,
    Flatten,
    Broadcast,
    Unknown
}
```

### GetDiagnosticReport Implementation
The method should return a formatted string similar to the example:
```
ShapeMismatchException: Matrix multiplication failed in layer 'encoder.fc2'

Input shape:    [32, 256]
Weight shape:   [128, 10]
Expected:       [batch_size, input_features] × [input_features, output_features]
                → Requires input_features to match

Problem: Dimension 1 of input (256) does not match dimension 0 of weight (128)

Context:
- Layer: encoder.fc2 (Linear)
- Batch size: 32
- Previous layer output: encoder.fc1 with shape [32, 256]

Suggested fixes:
1. Check encoder.fc1 output features (currently 256) matches fc2 input features (expected 128)
2. Consider adjusting fc1 to output 128 features, or fc2 to accept 256 inputs
3. Verify model configuration matches expected architecture
```

## Deliverables
- File: `src/Exceptions/ShapeMismatchException.cs`
- File: `src/Core/OperationType.cs` (enum definition)

## Testing Requirements
Create unit tests in `tests/Exceptions/ShapeMismatchExceptionTests.cs`:
- Test exception construction with all parameters
- Test exception construction with minimal parameters
- Test GetDiagnosticReport() output formatting
- Test that properties are correctly set
- Test that exception message is generated correctly

## Notes
- Keep GetDiagnosticReport() formatting consistent
- Use StringBuilder for efficient string building
- Consider null-safe handling of optional collections
- This is foundational - other specs will build upon this exception
