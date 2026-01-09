using System.Text;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Diagnostic handler for 1D convolution operations.
/// Provides specialized validation and error messages for Conv1D.
/// </summary>
public class Conv1DDiagnosticsHandler : BaseOperationDiagnosticsHandler
{
    /// <inheritdoc/>
    public override ValidationResult Validate(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var shapes = inputShapes.ToArray();

        if (shapes.Length != 2)
        {
            return ValidationResult.Failure("Conv1D requires exactly 2 input tensors (input and kernel)");
        }

        var inputShape = shapes[0];
        var kernelShape = shapes[1];

        // Input should be 3D (NCL or NLC)
        if (inputShape.Length != 3)
        {
            return ValidationResult.Failure($"Input must be 3D, got {inputShape.Length}D");
        }

        // Kernel should be 3D
        if (kernelShape.Length != 3)
        {
            return ValidationResult.Failure($"Kernel must be 3D, got {kernelShape.Length}D");
        }

        // Check channel count matches
        // Assuming NCL: input[1] == kernel[1]
        int inputChannels = (int)inputShape[1];
        int kernelChannels = (int)kernelShape[1];

        if (inputChannels != kernelChannels)
        {
            return ValidationResult.Failure(
                $"Channel count mismatch: input has {inputChannels} channels, kernel expects {kernelChannels}");
        }

        // Validate output dimensions
        int kernelSize = (int)kernelShape[2];
        int inputLength = (int)inputShape[2];

        int stride = operationParameters?.TryGetValue("stride", out var s) == true ? (int)s : 1;
        int padding = operationParameters?.TryGetValue("padding", out var p) == true ? (int)p : 0;

        int outputLength = (inputLength + 2 * padding - kernelSize) / stride + 1;

        if (outputLength <= 0)
        {
            return ValidationResult.Failure(
                $"Invalid output length: {outputLength}. " +
                $"Check kernel size ({kernelSize}), padding ({padding}), " +
                $"and input length ({inputLength})");
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
        var kernelShape = shapes[1];

        var sb = new StringBuilder();
        sb.AppendLine($"Conv1D failed in layer '{layerName}'");
        sb.AppendLine();

        sb.AppendLine($"Input shape:    [{inputShape[0]}, {inputShape[1]}, {inputShape[2]}]");
        sb.AppendLine($"Kernel shape:   [{kernelShape[0]}, {kernelShape[1]}, {kernelShape[2]}]");
        sb.AppendLine();

        if (inputShape.Length == 3 && kernelShape.Length == 3)
        {
            if (inputShape[1] != kernelShape[1])
            {
                sb.AppendLine($"Problem: Input channels ({inputShape[1]}) do not match kernel input channels ({kernelShape[1]})");
            }
        }

        // Add calculation details
        int kernelSize = (int)kernelShape[2];
        int stride = operationParameters?.TryGetValue("stride", out var s) == true ? (int)s : 1;
        int padding = operationParameters?.TryGetValue("padding", out var p) == true ? (int)p : 0;

        int inputLength = (int)inputShape[2];
        int outputLength = (inputLength + 2 * padding - kernelSize) / stride + 1;

        sb.AppendLine();
        sb.AppendLine($"Calculation:");
        sb.AppendLine($"  Output length = ({inputLength} + 2*{padding} - {kernelSize}) / {stride} + 1 = {outputLength}");

        if (outputLength <= 0)
        {
            sb.AppendLine($"  → Invalid: output length is non-positive!");
        }

        return sb.ToString().Trim();
    }

    /// <inheritdoc/>
    public override List<string> GenerateSuggestions(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var suggestions = new List<string>();

        suggestions.Add("Verify input tensor is in NCL format (batch, channels, length)");
        suggestions.Add("Check kernel shape is correct (out_channels, in_channels, kernel_size)");
        suggestions.Add("Ensure input channels match kernel input channels");
        suggestions.Add("Adjust padding if kernel is larger than input");
        suggestions.Add("Consider using 'same' padding to maintain sequence length");

        // Calculate suggested padding
        var shapes = inputShapes.ToArray();
        if (shapes.Length >= 2)
        {
            var inputShape = shapes[0];
            var kernelShape = shapes[1];
            int kernelSize = (int)kernelShape[2];
            int inputLength = (int)inputShape[2];

            if (kernelSize > inputLength)
            {
                int suggestedPadding = (kernelSize - inputLength) / 2 + 1;
                suggestions.Add($"Try padding={suggestedPadding} to handle kernel larger than input");
            }
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

        // Check for NLC vs NCL confusion
        if (inputShape[1] > inputShape[2])
        {
            issues.Add("Possible NCL vs NLC confusion: input channels seem larger than sequence length");
        }

        // Check for kernel larger than input
        var kernelShape = shapes[1];
        int kernelSize = (int)kernelShape[2];
        int inputLength = (int)inputShape[2];

        if (kernelSize > inputLength)
        {
            issues.Add("Kernel is larger than input sequence length - increase padding or use 'same' padding");
        }

        return issues;
    }
}
