# Spec: Formatted Error Reports

## Overview
Create a comprehensive formatted error report generator that combines all diagnostic information into a human-readable report.

## Requirements

### Class: ErrorReportGenerator
- Location: `src/Diagnostics/ErrorReportGenerator.cs`

```csharp
public class ErrorReportGenerator
{
    public static string GenerateReport(
        ShapeMismatchException exception,
        ErrorReportFormat format = ErrorReportFormat.Text)
}

public enum ErrorReportFormat
{
    Text,
    Markdown,
    Html
}
```

### Report Format: Text
```
=================================================================
ML Framework Shape Mismatch Error
=================================================================

Operation: Matrix multiplication
Layer: encoder.fc2

INPUT SHAPES:
  Tensor 1: [32, 256]
  Tensor 2: [128, 10]

EXPECTED SHAPES:
  Tensor 1: [batch_size, input_features]
  Tensor 2: [input_features, output_features]

PROBLEM:
  Dimension 1 of input (256) does not match dimension 0 of weight (128)

CONTEXT:
  - Batch size: 32
  - Previous layer output: encoder.fc1 with shape [32, 256]

SUGGESTED FIXES:
  1. Check encoder.fc1 output features (currently 256) matches fc2 input features (expected 128)
  2. Consider adjusting fc1 to output 128 features, or fc2 to accept 256 inputs
  3. Verify model configuration matches expected architecture

=================================================================
```

### Report Format: Markdown
```markdown
# ML Framework Shape Mismatch Error

## Operation
Matrix multiplication

## Layer
`encoder.fc2`

### Input Shapes
- Tensor 1: `[32, 256]`
- Tensor 2: `[128, 10]`

### Expected Shapes
- Tensor 1: `[batch_size, input_features]`
- Tensor 2: `[input_features, output_features]`

## Problem
Dimension 1 of input (256) does not match dimension 0 of weight (128)

### Context
- Batch size: 32
- Previous layer output: `encoder.fc1` with shape `[32, 256]`

### Suggested Fixes
1. Check encoder.fc1 output features (currently 256) matches fc2 input features (expected 128)
2. Consider adjusting fc1 to output 128 features, or fc2 to accept 256 inputs
3. Verify model configuration matches expected architecture
```

### Report Format: HTML
```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .error-report { font-family: monospace; padding: 20px; background: #ffeeee; }
        .header { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
        .section { margin: 10px 0; }
        .shape { background: #ffffee; padding: 5px; }
        .problem { color: red; font-weight: bold; }
        .suggestion { margin-left: 20px; }
    </style>
</head>
<body>
    <div class="error-report">
        <div class="header">ML Framework Shape Mismatch Error</div>
        <div class="section">
            <strong>Operation:</strong> Matrix multiplication<br>
            <strong>Layer:</strong> encoder.fc2
        </div>
        <!-- ... rest of report -->
    </div>
</body>
</html>
```

### Helper Methods
```csharp
private static string FormatTensorShapes(List<long[]> shapes)
private static string FormatContext(string layerName, string previousLayer, long[] previousShape)
private static string FormatSuggestions(List<string> suggestions)
private static string GenerateHtmlReport(string textContent)
```

## Tests
- Create `tests/Diagnostics/ErrorReportGeneratorTests.cs`
- Test text format generation
- Test markdown format generation
- Test HTML format generation
- Test with complete exception data
- Test with partial exception data
- Test output readability

## Success Criteria
- [ ] ErrorReportGenerator class with all formats
- [ ] Clean, readable text format
- [ ] Markdown format for documentation
- [ ] HTML format with basic styling
- [ ] Unit tests pass
- [ ] Reports are comprehensive and actionable
