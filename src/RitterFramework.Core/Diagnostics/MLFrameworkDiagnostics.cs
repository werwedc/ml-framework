namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Public API for enabling, configuring, and using the shape diagnostics system.
/// Provides a clean, user-friendly interface for developers to enable enhanced error reporting
/// and perform proactive shape checking.
/// </summary>
public static class MLFrameworkDiagnostics
{
    private static bool _isEnabled = false;
    private static bool _isVerbose = false;
    private static IOperationMetadataRegistry _registry;
    private static IShapeInferenceEngine _inferenceEngine;
    private static readonly object _lock = new object();

    /// <summary>
    /// Initialize diagnostics system with optional custom registry and inference engine.
    /// </summary>
    public static void Initialize(
        IOperationMetadataRegistry registry = null,
        IShapeInferenceEngine inferenceEngine = null)
    {
        lock (_lock)
        {
            _registry = registry ?? new DefaultOperationMetadataRegistry();
            _inferenceEngine = inferenceEngine ?? new DefaultShapeInferenceEngine(_registry);
        }
    }

    /// <summary>
    /// Enable enhanced error reporting.
    /// </summary>
    public static void EnableDiagnostics(bool verbose = false)
    {
        lock (_lock)
        {
            if (_registry == null || _inferenceEngine == null)
            {
                Initialize();
            }

            _isEnabled = true;
            _isVerbose = verbose;
        }
    }

    /// <summary>
    /// Disable enhanced error reporting.
    /// </summary>
    public static void DisableDiagnostics()
    {
        lock (_lock)
        {
            _isEnabled = false;
            _isVerbose = false;
        }
    }

    /// <summary>
    /// Check if diagnostics is enabled.
    /// </summary>
    public static bool IsEnabled => _isEnabled;

    /// <summary>
    /// Check if verbose mode is enabled.
    /// </summary>
    public static bool IsVerbose => _isVerbose;

    /// <summary>
    /// Programmatically check shapes for an operation.
    /// </summary>
    public static bool CheckShapes(
        OperationType operationType,
        IEnumerable<global::RitterFramework.Core.Tensor.Tensor> inputTensors,
        IDictionary<string, object> operationParameters = null)
    {
        if (!_isEnabled)
        {
            // If diagnostics disabled, do basic validation only
            return ValidateBasicShapes(operationType, inputTensors);
        }

        var inputShapes = inputTensors.Select(t => t.Shape.Select(d => (long)d).ToArray()).ToArray();

        var validationResult = _registry.ValidateShapes(
            operationType,
            inputShapes,
            operationParameters);

        return validationResult.IsValid;
    }

    /// <summary>
    /// Get detailed shape diagnostics for an operation.
    /// </summary>
    public static ShapeDiagnosticsInfo GetShapeDiagnostics(
        OperationType operationType,
        IEnumerable<global::RitterFramework.Core.Tensor.Tensor> inputTensors,
        string layerName = null,
        IDictionary<string, object> operationParameters = null)
    {
        var inputShapes = inputTensors.Select(t => t.Shape.Select(d => (long)d).ToArray()).ToArray();

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

    /// <summary>
    /// Get shape diagnostics with context (previous layer, etc.).
    /// </summary>
    public static ShapeDiagnosticsInfo GetContextualShapeDiagnostics(
        OperationType operationType,
        IEnumerable<global::RitterFramework.Core.Tensor.Tensor> inputTensors,
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

    /// <summary>
    /// Generate suggested fixes based on diagnostics.
    /// </summary>
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
        IEnumerable<global::RitterFramework.Core.Tensor.Tensor> inputTensors)
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
