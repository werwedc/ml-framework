# Technical Spec: Diagnostics API

## Overview
Create a public API for enabling, configuring, and using the shape diagnostics system. This provides a clean, user-friendly interface for developers to enable enhanced error reporting and perform proactive shape checking.

## Requirements

### Diagnostics API Class
```csharp
public static class MLFrameworkDiagnostics
{
    private static bool _isEnabled = false;
    private static bool _isVerbose = false;
    private static IOperationMetadataRegistry _registry;
    private static IShapeInferenceEngine _inferenceEngine;

    // Initialize diagnostics system
    public static void Initialize(
        IOperationMetadataRegistry registry = null,
        IShapeInferenceEngine inferenceEngine = null)
    {
        _registry = registry ?? new DefaultOperationMetadataRegistry();
        _inferenceEngine = inferenceEngine ?? new DefaultShapeInferenceEngine(_registry);
    }

    // Enable enhanced error reporting
    public static void EnableDiagnostics(bool verbose = false)
    {
        if (_registry == null || _inferenceEngine == null)
        {
            Initialize();
        }

        _isEnabled = true;
        _isVerbose = verbose;
    }

    // Disable enhanced error reporting
    public static void DisableDiagnostics()
    {
        _isEnabled = false;
        _isVerbose = false;
    }

    // Check if diagnostics is enabled
    public static bool IsEnabled => _isEnabled;

    // Check if verbose mode is enabled
    public static bool IsVerbose => _isVerbose;

    // Programmatically check shapes for an operation
    public static bool CheckShapes(
        OperationType operationType,
        IEnumerable<Tensor> inputTensors,
        IDictionary<string, object> operationParameters = null)
    {
        if (!_isEnabled)
        {
            // If diagnostics disabled, do basic validation only
            return ValidateBasicShapes(operationType, inputTensors);
        }

        var inputShapes = inputTensors.Select(t => t.Shape.ToArray()).ToArray();

        var validationResult = _registry.ValidateShapes(
            operationType,
            inputShapes,
            operationParameters);

        return validationResult.IsValid;
    }

    // Get detailed shape diagnostics for an operation
    public static ShapeDiagnosticsInfo GetShapeDiagnostics(
        OperationType operationType,
        IEnumerable<Tensor> inputTensors,
        string layerName = null,
        IDictionary<string, object> operationParameters = null)
    {
        var inputShapes = inputTensors.Select(t => t.Shape.ToArray()).ToArray();

        var validationResult = _registry.ValidateShapes(
            operationType,
            inputShapes,
            operationParameters);

        var requirements = _registry.GetRequirements(operationType);
        var inferredOutputShape = _inferenceEngine.InferOutputShape(
            operationType,
            inputShapes,
            operationParameters);

        return new ShapeDiagnosticsInfo
        {
            OperationType = operationType,
            LayerName = layerName ?? "unknown",
            InputShapes = inputShapes,
            ExpectedShapes = CalculateExpectedShapes(requirements, inputShapes),
            ActualOutputShape = inferredOutputShape,
            IsValid = validationResult.IsValid,
            Errors = validationResult.Errors,
            Warnings = validationResult.Warnings,
            RequirementsDescription = requirements.Description
        };
    }

    // Get shape diagnostics with context (previous layer, etc.)
    public static ShapeDiagnosticsInfo GetContextualShapeDiagnostics(
        OperationType operationType,
        IEnumerable<Tensor> inputTensors,
        string layerName,
        string previousLayerName = null,
        long[] previousLayerShape = null,
        IDictionary<string, object> operationParameters = null)
    {
        var diagnostics = GetShapeDiagnostics(
            operationType,
            inputTensors,
            layerName,
            operationParameters);

        diagnostics.PreviousLayerName = previousLayerName;
        diagnostics.PreviousLayerShape = previousLayerShape;

        return diagnostics;
    }

    // Generate suggested fixes based on diagnostics
    public static List<string> GenerateSuggestedFixes(ShapeDiagnosticsInfo diagnostics)
    {
        var fixes = new List<string>();

        if (diagnostics.IsValid)
        {
            return fixes;
        }

        // Common fix patterns
        foreach (var error in diagnostics.Errors)
        {
            if (error.Contains("dimension mismatch"))
            {
                fixes.Add("Check input tensor dimensions match the operation requirements");
            }

            if (error.Contains("batch size"))
            {
                fixes.Add("Ensure batch dimensions are consistent across operations");
            }

            if (error.Contains("channel"))
            {
                fixes.Add("Verify channel order (NCHW vs NHWC) is consistent");
            }
        }

        // Operation-specific fixes
        switch (diagnostics.OperationType)
        {
            case OperationType.MatrixMultiply:
                fixes.AddRange(GenerateMatrixMultiplyFixes(diagnostics));
                break;
            case OperationType.Conv2D:
                fixes.AddRange(GenerateConv2DFixes(diagnostics));
                break;
            case OperationType.Concat:
            case OperationType.Stack:
                fixes.AddRange(GenerateConcatStackFixes(diagnostics));
                break;
        }

        return fixes;
    }

    private static List<string> GenerateMatrixMultiplyFixes(ShapeDiagnosticsInfo diagnostics)
    {
        var fixes = new List<string>();

        if (diagnostics.InputShapes.Length >= 2)
        {
            var inputShape = diagnostics.InputShapes[0];
            var weightShape = diagnostics.InputShapes[1];

            if (inputShape.Length == 2 && weightShape.Length == 2)
            {
                fixes.Add($"Adjust weight matrix to [{weightShape[0]}, {inputShape[1]}] to match input shape");
            }
            else if (inputShape.Length == 3 && weightShape.Length == 2)
            {
                fixes.Add($"Adjust weight matrix to [{weightShape[0]}, {inputShape[2]}] to match input shape");
            }
        }

        return fixes;
    }

    private static List<string> GenerateConv2DFixes(ShapeDiagnosticsInfo diagnostics)
    {
        var fixes = new List<string>();

        // Add Conv2D-specific fixes
        fixes.Add("Check kernel size is compatible with input dimensions");
        fixes.Add("Verify padding settings match input size");
        fixes.Add("Consider adjusting stride to prevent output dimension issues");

        return fixes;
    }

    private static List<string> GenerateConcatStackFixes(ShapeDiagnosticsInfo diagnostics)
    {
        var fixes = new List<string>();

        fixes.Add("Ensure all input tensors have matching dimensions except at the concatenation axis");
        fixes.Add("Check that the concatenation axis is within valid range for all tensors");

        return fixes;
    }

    private static long[][] CalculateExpectedShapes(
        OperationShapeRequirements requirements,
        long[][] inputShapes)
    {
        // Calculate expected shapes based on requirements
        // This is simplified - actual implementation would use inference engine
        return new long[][] { inputShapes[0] };
    }

    private static bool ValidateBasicShapes(
        OperationType operationType,
        IEnumerable<Tensor> inputTensors)
    {
        // Basic validation when diagnostics are disabled
        // Just check that tensors are not null and have valid shapes
        foreach (var tensor in inputTensors)
        {
            if (tensor == null || tensor.Shape == null || tensor.Shape.Length == 0)
            {
                return false;
            }
        }
        return true;
    }
}
```

### ShapeDiagnosticsInfo Class
```csharp
public class ShapeDiagnosticsInfo
{
    public OperationType OperationType { get; set; }
    public string LayerName { get; set; }
    public long[][] InputShapes { get; set; }
    public long[][] ExpectedShapes { get; set; }
    public long[] ActualOutputShape { get; set; }
    public bool IsValid { get; set; }
    public List<string> Errors { get; set; }
    public List<string> Warnings { get; set; }
    public string RequirementsDescription { get; set; }
    public string PreviousLayerName { get; set; }
    public long[] PreviousLayerShape { get; set; }

    public string GetFormattedReport()
    {
        var sb = new StringBuilder();

        sb.AppendLine($"Shape Diagnostics for layer '{LayerName}'");
        sb.AppendLine($"Operation: {OperationType}");
        sb.AppendLine($"Valid: {IsValid}");
        sb.AppendLine();

        if (InputShapes != null && InputShapes.Length > 0)
        {
            sb.AppendLine("Input Shapes:");
            for (int i = 0; i < InputShapes.Length; i++)
            {
                sb.AppendLine($"  Input {i}: [{string.Join(", ", InputShapes[i])}]");
            }
            sb.AppendLine();
        }

        if (ExpectedShapes != null && ExpectedShapes.Length > 0)
        {
            sb.AppendLine("Expected Shapes:");
            for (int i = 0; i < ExpectedShapes.Length; i++)
            {
                sb.AppendLine($"  Expected {i}: [{string.Join(", ", ExpectedShapes[i])}]");
            }
            sb.AppendLine();
        }

        if (ActualOutputShape != null)
        {
            sb.AppendLine($"Output Shape: [{string.Join(", ", ActualOutputShape)}]");
            sb.AppendLine();
        }

        if (RequirementsDescription != null)
        {
            sb.AppendLine($"Requirements: {RequirementsDescription}");
            sb.AppendLine();
        }

        if (!IsValid && Errors != null && Errors.Count > 0)
        {
            sb.AppendLine("Errors:");
            foreach (var error in Errors)
            {
                sb.AppendLine($"  - {error}");
            }
            sb.AppendLine();
        }

        if (Warnings != null && Warnings.Count > 0)
        {
            sb.AppendLine("Warnings:");
            foreach (var warning in Warnings)
            {
                sb.AppendLine($"  - {warning}");
            }
            sb.AppendLine();
        }

        if (PreviousLayerName != null)
        {
            sb.AppendLine("Context:");
            sb.AppendLine($"  Previous layer: {PreviousLayerName}");
            if (PreviousLayerShape != null)
            {
                sb.AppendLine($"  Previous output: [{string.Join(", ", PreviousLayerShape)}]");
            }
        }

        return sb.ToString();
    }
}
```

### Tensor Extension for Diagnostics
```csharp
public static class TensorDiagnosticsExtensions
{
    public static string GetShapeString(this Tensor tensor)
    {
        return $"[{string.Join(", ", tensor.Shape)}]";
    }

    public static long GetElementCount(this Tensor tensor)
    {
        long count = 1;
        foreach (var dim in tensor.Shape)
        {
            count *= dim;
        }
        return count;
    }
}
```

## Deliverables
- File: `src/Diagnostics/MLFrameworkDiagnostics.cs`
- File: `src/Diagnostics/ShapeDiagnosticsInfo.cs`
- File: `src/Diagnostics/TensorDiagnosticsExtensions.cs`

## Testing Requirements
Create unit tests in `tests/Diagnostics/MLFrameworkDiagnosticsTests.cs`:
- Test EnableDiagnostics and DisableDiagnostics
- Test IsEnabled and IsVerbose properties
- Test CheckShapes for valid operations
- Test CheckShapes for invalid operations
- Test GetShapeDiagnostics for various operations
- Test GetContextualShapeDiagnostics with previous layer context
- Test GenerateSuggestedFixes for different error scenarios
- Test ShapeDiagnosticsInfo.GetFormattedReport() formatting
- Test that diagnostics are disabled by default
- Test initialization with custom registry and inference engine

## Notes
- Use lazy initialization for registry and inference engine
- Ensure thread-safety for Enable/Disable operations
- Consider making diagnostics opt-in for production use
- Provide clear documentation on when to use vs not use diagnostics
- Keep API simple and intuitive for developers
