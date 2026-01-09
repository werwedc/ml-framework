# Technical Spec: Operation-Specific Diagnostics

## Overview
Implement specialized diagnostic logic and error messages for different operation types. This includes pattern detection, operation-specific validation, and targeted fix suggestions for common operations.

## Requirements

### Operation Diagnostics Registry
Create a registry for operation-specific diagnostic handlers:

```csharp
public interface IOperationDiagnosticsRegistry
{
    void RegisterDiagnosticsHandler(
        OperationType operationType,
        IOperationDiagnosticsHandler handler);

    IOperationDiagnosticsHandler GetHandler(OperationType operationType);

    bool HasHandler(OperationType operationType);
}
```

### Operation Diagnostics Handler Interface
```csharp
public interface IOperationDiagnosticsHandler
{
    // Validate shapes with operation-specific logic
    ValidationResult Validate(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters);

    // Generate operation-specific error messages
    string GenerateErrorMessage(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters,
        string layerName);

    // Generate operation-specific suggestions
    List<string> GenerateSuggestions(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters);

    // Detect common patterns/issues
    List<string> DetectIssues(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters);
}
```

### Base Handler Implementation
```csharp
public abstract class BaseOperationDiagnosticsHandler : IOperationDiagnosticsHandler
{
    public abstract ValidationResult Validate(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters);

    public abstract string GenerateErrorMessage(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters,
        string layerName);

    public virtual List<string> GenerateSuggestions(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        return new List<string>();
    }

    public virtual List<string> DetectIssues(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        return new List<string>();
    }

    protected bool CheckBroadcastCompatibility(long[] shape1, long[] shape2)
    {
        // Check if shapes are broadcast-compatible
        int dim1 = shape1.Length;
        int dim2 = shape2.Length;
        int maxDim = Math.Max(dim1, dim2);

        for (int i = 1; i <= maxDim; i++)
        {
            int idx1 = dim1 - i;
            int idx2 = dim2 - i;

            long dim1Val = idx1 >= 0 ? shape1[idx1] : 1;
            long dim2Val = idx2 >= 0 ? shape2[idx2] : 1;

            if (dim1Val != dim2Val && dim1Val != 1 && dim2Val != 1)
            {
                return false;
            }
        }

        return true;
    }
}
```

### Matrix Multiply Diagnostics Handler
```csharp
public class MatrixMultiplyDiagnosticsHandler : BaseOperationDiagnosticsHandler
{
    public override ValidationResult Validate(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var shapes = inputShapes.ToArray();

        if (shapes.Length != 2)
        {
            return ValidationResult.Failure("Matrix multiplication requires exactly 2 input tensors");
        }

        var shapeA = shapes[0];
        var shapeB = shapes[1];

        // Handle different dimensionality
        if (shapeA.Length < 2 || shapeB.Length < 2)
        {
            return ValidationResult.Failure("Both tensors must have at least 2 dimensions");
        }

        // Get inner dimensions
        int innerDimA = shapeA.Length - 1;
        int innerDimB = shapeB.Length - 2;

        if (shapeA[innerDimA] != shapeB[innerDimB])
        {
            return ValidationResult.Failure(
                $"Inner dimensions mismatch: {shapeA[innerDimA]} != {shapeB[innerDimB]}");
        }

        // Check for potential batch dimension mismatch
        if (shapeA.Length == 3 && shapeB.Length == 2)
        {
            // shapeA: [batch, m, k], shapeB: [k, n] - this is fine
        }
        else if (shapeA.Length == 2 && shapeB.Length == 3)
        {
            // shapeA: [m, k], shapeB: [batch, k, n] - check if k matches
            if (shapeA[1] != shapeB[1])
            {
                return ValidationResult.Failure(
                    $"Batch dimension mismatch: {shapeA[1]} != {shapeB[1]}");
            }
        }
        else if (shapeA.Length == shapeB.Length && shapeA.Length > 2)
        {
            // Check batch dimensions match
            for (int i = 0; i < shapeA.Length - 2; i++)
            {
                if (shapeA[i] != shapeB[i])
                {
                    return ValidationResult.Failure(
                        $"Batch dimension {i} mismatch: {shapeA[i]} != {shapeB[i]}");
                }
            }
        }

        return ValidationResult.Success();
    }

    public override string GenerateErrorMessage(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters,
        string layerName)
    {
        var shapes = inputShapes.ToArray();
        var shapeA = shapes[0];
        var shapeB = shapes[1];

        var sb = new StringBuilder();
        sb.AppendLine($"Matrix multiplication failed in layer '{layerName}'");
        sb.AppendLine();

        if (shapeA.Length == 2 && shapeB.Length == 2)
        {
            sb.AppendLine($"Input shape:    [{shapeA[0]}, {shapeA[1]}]");
            sb.AppendLine($"Weight shape:   [{shapeB[0]}, {shapeB[1]}]");
            sb.AppendLine();
            sb.AppendLine($"Expected:       [m, k] × [k, n] → [m, n]");
            sb.AppendLine($"                → Requires k to match");
            sb.AppendLine();
            sb.AppendLine($"Problem: Dimension 1 of input ({shapeA[1]}) does not match dimension 0 of weight ({shapeB[0]})");
        }
        else if (shapeA.Length == 3)
        {
            sb.AppendLine($"Input shape:    [{shapeA[0]}, {shapeA[1]}, {shapeA[2]}]");
            sb.AppendLine($"Weight shape:   [{shapeB[0]}, {shapeB[1]}]");
            sb.AppendLine();
            sb.AppendLine($"Expected:       [batch, m, k] × [k, n] → [batch, m, n]");
            sb.AppendLine($"                → Requires k to match");
            sb.AppendLine();
            sb.AppendLine($"Problem: Dimension 2 of input ({shapeA[2]}) does not match dimension 0 of weight ({shapeB[0]})");
        }

        return sb.ToString().Trim();
    }

    public override List<string> GenerateSuggestions(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var suggestions = new List<string>();
        var shapes = inputShapes.ToArray();
        var shapeA = shapes[0];
        var shapeB = shapes[1];

        // Common Matrix Multiply issues and fixes
        if (shapeA.Length == 2 && shapeB.Length == 2)
        {
            suggestions.Add($"Transpose weight matrix if needed: currently [{shapeB[0]}, {shapeB[1]}] → [{shapeB[1]}, {shapeB[0]}]");
            suggestions.Add($"Check input tensor is in correct format (not transposed)");
        }
        else if (shapeA.Length == 3)
        {
            suggestions.Add($"Ensure weight matrix shape matches last dimension of input");
            suggestions.Add($"Check if weight matrix should be transposed");
        }

        suggestions.Add("Verify model configuration matches expected architecture");
        suggestions.Add("Consider using torch.nn.Linear with correct in_features parameter");

        return suggestions;
    }

    public override List<string> DetectIssues(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var issues = new List<string>();
        var shapes = inputShapes.ToArray();
        var shapeA = shapes[0];
        var shapeB = shapes[1];

        // Check for transpose issue (common mistake)
        if (shapeA.Length == 2 && shapeB.Length == 2)
        {
            if (shapeA[1] == shapeB[1])
            {
                issues.Add("Possible transpose issue: input and weight have matching last dimension");
            }
        }

        // Check for scalar multiplication
        if (shapeA.Length == 1 || shapeB.Length == 1)
        {
            issues.Add("One of the tensors is 1D - use squeeze/unsqueeze or proper broadcasting");
        }

        return issues;
    }
}
```

### Conv2D Diagnostics Handler
```csharp
public class Conv2DDiagnosticsHandler : BaseOperationDiagnosticsHandler
{
    public override ValidationResult Validate(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var shapes = inputShapes.ToArray();

        if (shapes.Length != 2)
        {
            return ValidationResult.Failure("Conv2D requires exactly 2 input tensors (input and kernel)");
        }

        var inputShape = shapes[0];
        var kernelShape = shapes[1];

        // Input should be 4D (NCHW or NHWC)
        if (inputShape.Length != 4)
        {
            return ValidationResult.Failure($"Input must be 4D, got {inputShape.Length}D");
        }

        // Kernel should be 4D
        if (kernelShape.Length != 4)
        {
            return ValidationResult.Failure($"Kernel must be 4D, got {kernelShape.Length}D");
        }

        // Check channel count matches
        // Assuming NCHW: input[1] == kernel[1]
        int inputChannels = (int)inputShape[1];
        int kernelChannels = (int)kernelShape[1];

        if (inputChannels != kernelChannels)
        {
            return ValidationResult.Failure(
                $"Channel count mismatch: input has {inputChannels} channels, kernel expects {kernelChannels}");
        }

        // Validate output dimensions
        int kernelHeight = (int)kernelShape[2];
        int kernelWidth = (int)kernelShape[3];
        int inputHeight = (int)inputShape[2];
        int inputWidth = (int)inputShape[3];

        int strideH = operationParameters?.TryGetValue("stride", out var s) == true ? ((int[])s)[0] : 1;
        int strideW = operationParameters?.TryGetValue("stride", out var s2) == true ? ((int[])s2)[1] : 1;
        int paddingH = operationParameters?.TryGetValue("padding", out var p) == true ? ((int[])p)[0] : 0;
        int paddingW = operationParameters?.TryGetValue("padding", out var p2) == true ? ((int[])p2)[1] : 0;

        int outputHeight = (inputHeight + 2 * paddingH - kernelHeight) / strideH + 1;
        int outputWidth = (inputWidth + 2 * paddingW - kernelWidth) / strideW + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            return ValidationResult.Failure(
                $"Invalid output dimensions: {outputHeight}x{outputWidth}. " +
                $"Check kernel size ({kernelHeight}x{kernelWidth}), padding ({paddingH}x{paddingW}), " +
                $"and input size ({inputHeight}x{inputWidth})");
        }

        return ValidationResult.Success();
    }

    public override string GenerateErrorMessage(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters,
        string layerName)
    {
        var shapes = inputShapes.ToArray();
        var inputShape = shapes[0];
        var kernelShape = shapes[1];

        var sb = new StringBuilder();
        sb.AppendLine($"Conv2D failed in layer '{layerName}'");
        sb.AppendLine();

        sb.AppendLine($"Input shape:    [{inputShape[0]}, {inputShape[1]}, {inputShape[2]}, {inputShape[3]}]");
        sb.AppendLine($"Kernel shape:   [{kernelShape[0]}, {kernelShape[1]}, {kernelShape[2]}, {kernelShape[3]}]");
        sb.AppendLine();

        if (inputShape.Length == 4 && kernelShape.Length == 4)
        {
            if (inputShape[1] != kernelShape[1])
            {
                sb.AppendLine($"Problem: Input channels ({inputShape[1]}) do not match kernel input channels ({kernelShape[1]})");
            }
        }

        // Add calculation details
        int kernelHeight = (int)kernelShape[2];
        int kernelWidth = (int)kernelShape[3];
        int strideH = operationParameters?.TryGetValue("stride", out var s) == true ? ((int[])s)[0] : 1;
        int strideW = operationParameters?.TryGetValue("stride", out var s2) == true ? ((int[])s2)[1] : 1;
        int paddingH = operationParameters?.TryGetValue("padding", out var p) == true ? ((int[])p)[0] : 0;
        int paddingW = operationParameters?.TryGetValue("padding", out var p2) == true ? ((int[])p2)[1] : 0;

        int inputHeight = (int)inputShape[2];
        int inputWidth = (int)inputShape[3];
        int outputHeight = (inputHeight + 2 * paddingH - kernelHeight) / strideH + 1;
        int outputWidth = (inputWidth + 2 * paddingW - kernelWidth) / strideW + 1;

        sb.AppendLine();
        sb.AppendLine($"Calculation:");
        sb.AppendLine($"  Output height = ({inputHeight} + 2*{paddingH} - {kernelHeight}) / {strideH} + 1 = {outputHeight}");
        sb.AppendLine($"  Output width  = ({inputWidth} + 2*{paddingW} - {kernelWidth}) / {strideW} + 1 = {outputWidth}");

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            sb.AppendLine($"  → Invalid: output dimensions are non-positive!");
        }

        return sb.ToString().Trim();
    }

    public override List<string> GenerateSuggestions(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var suggestions = new List<string>();

        suggestions.Add("Verify input tensor is in NCHW format (batch, channels, height, width)");
        suggestions.Add("Check kernel shape is correct (out_channels, in_channels, kernel_height, kernel_width)");
        suggestions.Add("Ensure input channels match kernel input channels");
        suggestions.Add("Adjust padding if kernel is larger than input");
        suggestions.Add("Consider using 'same' padding to maintain spatial dimensions");

        // Calculate suggested padding
        var shapes = inputShapes.ToArray();
        if (shapes.Length >= 2)
        {
            var inputShape = shapes[0];
            var kernelShape = shapes[1];
            int kernelHeight = (int)kernelShape[2];
            int inputHeight = (int)inputShape[2];

            if (kernelHeight > inputHeight)
            {
                int suggestedPadding = (kernelHeight - inputHeight) / 2 + 1;
                suggestions.Add($"Try padding={suggestedPadding} to handle kernel larger than input");
            }
        }

        return suggestions;
    }

    public override List<string> DetectIssues(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var issues = new List<string>();
        var shapes = inputShapes.ToArray();
        var inputShape = shapes[0];

        // Check for NHWC vs NCHW confusion
        if (inputShape[1] > inputShape[2] && inputShape[1] > inputShape[3])
        {
            issues.Add("Possible NCHW vs NHWC confusion: input channels seem larger than spatial dimensions");
        }

        // Check for kernel larger than input
        var kernelShape = shapes[1];
        int kernelHeight = (int)kernelShape[2];
        int inputHeight = (int)inputShape[2];

        if (kernelHeight > inputHeight)
        {
            issues.Add("Kernel is larger than input spatial dimension - increase padding or use 'same' padding");
        }

        return issues;
    }
}
```

### Concat Diagnostics Handler
```csharp
public class ConcatDiagnosticsHandler : BaseOperationDiagnosticsHandler
{
    public override ValidationResult Validate(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var shapes = inputShapes.ToArray();

        if (shapes.Length < 2)
        {
            return ValidationResult.Failure("Concat requires at least 2 input tensors");
        }

        // Get concatenation axis (default: 0)
        int axis = operationParameters?.TryGetValue("axis", out var a) == true ? (int)a : 0;

        // All tensors should have same number of dimensions
        int dimCount = shapes[0].Length;
        for (int i = 1; i < shapes.Length; i++)
        {
            if (shapes[i].Length != dimCount)
            {
                return ValidationResult.Failure(
                    $"Tensor {i} has {shapes[i].Length} dimensions, but tensor 0 has {dimCount}");
            }
        }

        // All dimensions except axis should match
        for (int i = 0; i < shapes.Length; i++)
        {
            for (int d = 0; d < dimCount; d++)
            {
                if (d == axis) continue; // Skip concatenation axis

                if (shapes[i][d] != shapes[0][d])
                {
                    return ValidationResult.Failure(
                        $"Tensor {i} dimension {d} is {shapes[i][d]}, but tensor 0 has {shapes[0][d]}");
                }
            }
        }

        // Validate axis is within range
        if (axis < 0 || axis >= dimCount)
        {
            return ValidationResult.Failure(
                $"Axis {axis} is out of range for {dimCount}D tensors (valid range: 0 to {dimCount - 1})");
        }

        return ValidationResult.Success();
    }

    public override string GenerateErrorMessage(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters,
        string layerName)
    {
        var shapes = inputShapes.ToArray();
        int axis = operationParameters?.TryGetValue("axis", out var a) == true ? (int)a : 0;

        var sb = new StringBuilder();
        sb.AppendLine($"Concat failed in layer '{layerName}' (axis={axis})");
        sb.AppendLine();

        sb.AppendLine("Input Shapes:");
        for (int i = 0; i < shapes.Length; i++)
        {
            sb.AppendLine($"  Tensor {i}: [{string.Join(", ", shapes[i])}]");
        }

        sb.AppendLine();
        sb.AppendLine("Expected: All tensors must have matching dimensions except at concatenation axis");

        // Find mismatching dimensions
        int dimCount = shapes[0].Length;
        for (int d = 0; d < dimCount; d++)
        {
            if (d == axis) continue;

            bool allMatch = true;
            long expectedDim = shapes[0][d];

            for (int i = 1; i < shapes.Length; i++)
            {
                if (shapes[i][d] != expectedDim)
                {
                    allMatch = false;
                    break;
                }
            }

            if (!allMatch)
            {
                sb.AppendLine();
                sb.AppendLine($"Problem: Dimension {d} does not match across all tensors");
                for (int i = 0; i < shapes.Length; i++)
                {
                    sb.AppendLine($"  Tensor {i}: {shapes[i][d]}");
                }
            }
        }

        return sb.ToString().Trim();
    }

    public override List<string> GenerateSuggestions(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var suggestions = new List<string>();

        suggestions.Add("Ensure all tensors have the same number of dimensions");
        suggestions.Add("Check that all non-concatenation dimensions match exactly");
        suggestions.Add("Consider using unsqueeze/squeeze to add/remove dimensions before concatenation");
        suggestions.Add("Verify the concatenation axis is correct");

        // Detect axis issues
        int axis = operationParameters?.TryGetValue("axis", out var a) == true ? (int)a : 0;
        var shapes = inputShapes.ToArray();

        int dimCount = shapes[0].Length;
        if (axis < 0 || axis >= dimCount)
        {
            suggestions.Add($"Axis {axis} is invalid - try axis 0, 1, 2, or 3");
        }

        // Check for transpose issues
        bool possibleTranspose = true;
        for (int d = 0; d < dimCount; d++)
        {
            long firstDim = shapes[0][d];
            for (int i = 1; i < shapes.Length; i++)
            {
                if (shapes[i][d] != firstDim)
                {
                    possibleTranspose = false;
                    break;
                }
            }
        }

        if (possibleTranspose && shapes.Length == 2)
        {
            suggestions.Add("All dimensions match - did you mean to use stack instead of concat?");
        }

        return suggestions;
    }
}
```

### Registry Implementation
```csharp
public class OperationDiagnosticsRegistry : IOperationDiagnosticsRegistry
{
    private readonly Dictionary<OperationType, IOperationDiagnosticsHandler> _handlers;

    public OperationDiagnosticsRegistry()
    {
        _handlers = new Dictionary<OperationType, IOperationDiagnosticsHandler>();

        // Register default handlers
        RegisterDiagnosticsHandler(OperationType.MatrixMultiply, new MatrixMultiplyDiagnosticsHandler());
        RegisterDiagnosticsHandler(OperationType.Conv2D, new Conv2DDiagnosticsHandler());
        RegisterDiagnosticsHandler(OperationType.Concat, new ConcatDiagnosticsHandler());
        RegisterDiagnosticsHandler(OperationType.Conv1D, new Conv1DDiagnosticsHandler());
        RegisterDiagnosticsHandler(OperationType.MaxPool2D, new PoolingDiagnosticsHandler(OperationType.MaxPool2D));
        RegisterDiagnosticsHandler(OperationType.AveragePool2D, new PoolingDiagnosticsHandler(OperationType.AveragePool2D));
    }

    public void RegisterDiagnosticsHandler(OperationType operationType, IOperationDiagnosticsHandler handler)
    {
        _handlers[operationType] = handler;
    }

    public IOperationDiagnosticsHandler GetHandler(OperationType operationType)
    {
        return _handlers.TryGetValue(operationType, out var handler) ? handler : null;
    }

    public bool HasHandler(OperationType operationType)
    {
        return _handlers.ContainsKey(operationType);
    }
}
```

## Deliverables
- File: `src/Diagnostics/IOperationDiagnosticsHandler.cs`
- File: `src/Diagnostics/BaseOperationDiagnosticsHandler.cs`
- File: `src/Diagnostics/MatrixMultiplyDiagnosticsHandler.cs`
- File: `src/Diagnostics/Conv2DDiagnosticsHandler.cs`
- File: `src/Diagnostics/ConcatDiagnosticsHandler.cs`
- File: `src/Diagnostics/IOperationDiagnosticsRegistry.cs`
- File: `src/Diagnostics/OperationDiagnosticsRegistry.cs`
- File: `src/Diagnostics/Conv1DDiagnosticsHandler.cs` (simpler version of Conv2D)
- File: `src/Diagnostics/PoolingDiagnosticsHandler.cs` (shared for MaxPool2D and AveragePool2D)

## Testing Requirements
Create unit tests in `tests/Diagnostics/OperationDiagnosticsHandlerTests.cs`:
- Test MatrixMultiply handler validation for various shapes
- Test MatrixMultiply handler error message generation
- Test MatrixMultiply handler suggestion generation
- Test MatrixMultiply handler issue detection
- Test Conv2D handler validation with channel mismatches
- Test Conv2D handler validation with invalid output dimensions
- Test Conv2D handler error message includes calculations
- Test Conv2D handler detects NHWC vs NCHW issues
- Test Concat handler validates matching dimensions
- Test Concat handler detects axis issues
- Test Concat handler suggests stack when appropriate

Create integration tests in `tests/Integration/OperationDiagnosticsIntegrationTests.cs`:
- Test registry correctly routes to appropriate handlers
- Test custom handler registration
- Test handler generates appropriate ShapeMismatchException

## Notes
- Each handler should be focused and testable
- Use the existing OperationMetadataRegistry as a source of truth
- Handlers can use the ShapeInferenceEngine for validation
- Consider adding handlers for more operations in future
- Make handler registration extensible for custom operations
