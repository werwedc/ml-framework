using System.Text;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Diagnostic handler for 2D convolution operations.
/// Provides specialized validation and error messages for Conv2D.
/// </summary>
public class Conv2DDiagnosticsHandler : BaseOperationDiagnosticsHandler
{
    /// <inheritdoc/>
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

    /// <inheritdoc/>
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

    /// <inheritdoc/>
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
