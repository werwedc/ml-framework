using System.Text;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Diagnostic handler for pooling operations (MaxPool2D and AveragePool2D).
/// Provides specialized validation and error messages for pooling.
/// </summary>
public class PoolingDiagnosticsHandler : BaseOperationDiagnosticsHandler
{
    private readonly OperationType _operationType;

    /// <summary>
    /// Create a new pooling diagnostics handler.
    /// </summary>
    /// <param name="operationType">The pooling operation type (MaxPool2D or AveragePool2D).</param>
    public PoolingDiagnosticsHandler(OperationType operationType)
    {
        _operationType = operationType;
    }

    /// <inheritdoc/>
    public override ValidationResult Validate(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var shapes = inputShapes.ToArray();

        if (shapes.Length != 1)
        {
            return ValidationResult.Failure($"{_operationType} requires exactly 1 input tensor");
        }

        var inputShape = shapes[0];

        // Input should be 4D (NCHW)
        if (inputShape.Length != 4)
        {
            return ValidationResult.Failure($"Input must be 4D, got {inputShape.Length}D");
        }

        // Get pooling parameters
        int kernelSize = operationParameters?.TryGetValue("kernel_size", out var ks) == true ? (int)ks : 2;
        int stride = operationParameters?.TryGetValue("stride", out var s) == true ? (int)s : kernelSize;
        int padding = operationParameters?.TryGetValue("padding", out var p) == true ? (int)p : 0;

        // Validate output dimensions
        int inputHeight = (int)inputShape[2];
        int inputWidth = (int)inputShape[3];

        int outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            return ValidationResult.Failure(
                $"Invalid output dimensions: {outputHeight}x{outputWidth}. " +
                $"Check kernel size ({kernelSize}), padding ({padding}), " +
                $"and input size ({inputHeight}x{inputWidth})");
        }

        return ValidationResult.Success();
    }

    /// <inheritdoc/>
    public override string GenerateErrorMessage(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters,
        string layerName)
    {
        var shapes = inputShapes.ToArray();
        var inputShape = shapes[0];

        var sb = new StringBuilder();
        sb.AppendLine($"{_operationType} failed in layer '{layerName}'");
        sb.AppendLine();

        sb.AppendLine($"Input shape:    [{inputShape[0]}, {inputShape[1]}, {inputShape[2]}, {inputShape[3]}]");

        // Add pooling parameters
        int kernelSize = operationParameters?.TryGetValue("kernel_size", out var ks) == true ? (int)ks : 2;
        int stride = operationParameters?.TryGetValue("stride", out var s) == true ? (int)s : kernelSize;
        int padding = operationParameters?.TryGetValue("padding", out var p) == true ? (int)p : 0;

        sb.AppendLine();
        sb.AppendLine($"Parameters:");
        sb.AppendLine($"  Kernel size:  {kernelSize}");
        sb.AppendLine($"  Stride:       {stride}");
        sb.AppendLine($"  Padding:      {padding}");

        // Add calculation details
        int inputHeight = (int)inputShape[2];
        int inputWidth = (int)inputShape[3];
        int outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;

        sb.AppendLine();
        sb.AppendLine($"Calculation:");
        sb.AppendLine($"  Output height = ({inputHeight} + 2*{padding} - {kernelSize}) / {stride} + 1 = {outputHeight}");
        sb.AppendLine($"  Output width  = ({inputWidth} + 2*{padding} - {kernelSize}) / {stride} + 1 = {outputWidth}");

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            sb.AppendLine($"  → Invalid: output dimensions are non-positive!");
        }

        return sb.ToString().Trim();
    }

    /// <inheritdoc/>
    public override List<string> GenerateSuggestions(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var suggestions = new List<string>();

        suggestions.Add("Verify input tensor is in NCHW format (batch, channels, height, width)");
        suggestions.Add("Adjust kernel size if it's too large for input dimensions");
        suggestions.Add("Increase padding if kernel is larger than input");
        suggestions.Add("Consider using 'same' padding to maintain spatial dimensions");

        // Calculate suggested padding
        var shapes = inputShapes.ToArray();
        var inputShape = shapes[0];

        int kernelSize = operationParameters?.TryGetValue("kernel_size", out var ks) == true ? (int)ks : 2;
        int inputHeight = (int)inputShape[2];

        if (kernelSize > inputHeight)
        {
            int suggestedPadding = (kernelSize - inputHeight) / 2 + 1;
            suggestions.Add($"Try padding={suggestedPadding} to handle kernel larger than input");
        }

        return suggestions;
    }

    /// <inheritdoc/>
    public override List<string> DetectIssues(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var issues = new List<string>();
        var shapes = inputShapes.ToArray();
        var inputShape = shapes[0];

        // Check for kernel larger than input
        int kernelSize = operationParameters?.TryGetValue("kernel_size", out var ks) == true ? (int)ks : 2;
        int inputHeight = (int)inputShape[2];

        if (kernelSize > inputHeight)
        {
            issues.Add("Kernel is larger than input spatial dimension - increase padding or use 'same' padding");
        }

        return issues;
    }
}
