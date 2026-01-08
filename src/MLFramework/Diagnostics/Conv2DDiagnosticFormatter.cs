using RitterFramework.Core;

namespace MLFramework.Diagnostics;

/// <summary>
/// Diagnostic formatter for 2D convolution operations.
/// Provides specific error messages and suggestions for convolution-related issues.
/// </summary>
public class Conv2DDiagnosticFormatter : IDiagnosticFormatter
{
    /// <summary>
    /// Gets the operation type supported by this formatter.
    /// </summary>
    public OperationType SupportedOperation => OperationType.Conv2D;

    /// <summary>
    /// Formats a Conv2D error message.
    /// </summary>
    public string FormatError(ValidationResult result, params long[][] inputShapes)
    {
        if (inputShapes.Length < 2)
        {
            return $"Conv2D: Insufficient input shapes provided. {result.ErrorMessage}";
        }

        var inputShape = $"[{string.Join(", ", inputShapes[0])}]";
        var kernelShape = $"[{string.Join(", ", inputShapes[1])}]";

        var message = $"Conv2D: Input {inputShape} with kernel {kernelShape}\n";

        // Check for channel matching
        if (inputShapes[0].Length == 4 && inputShapes[1].Length == 4)
        {
            var inputChannels = inputShapes[0][1]; // NCHW format
            var kernelChannels = inputShapes[1][1];

            if (inputChannels == kernelChannels)
            {
                message += $"Status: Input channels ({inputChannels}) match kernel channels ({kernelChannels})\n";

                // Calculate output shape (simplified without padding/stride)
                var outputHeight = inputShapes[0][2] - inputShapes[1][2] + 1;
                var outputWidth = inputShapes[0][3] - inputShapes[1][3] + 1;
                var outputChannels = inputShapes[1][0];

                if (outputHeight > 0 && outputWidth > 0)
                {
                    message += $"Output shape: [{inputShapes[0][0]}, {outputChannels}, {outputHeight}, {outputWidth}]";
                }
            }
            else
            {
                message += $"Problem: Input channels ({inputChannels}) do not match kernel channels ({kernelChannels})";
            }
        }
        else
        {
            message += $"Problem: {result.ErrorMessage}";
        }

        return message;
    }

    /// <summary>
    /// Generates suggestions for fixing Conv2D issues.
    /// </summary>
    public List<string> GenerateSuggestions(ValidationResult result)
    {
        var suggestions = new List<string>
        {
            "Check channel configurations - input channels must match kernel in-channels",
            "Verify kernel parameters (height, width, in-channels, out-channels)",
            "Ensure input tensor is in correct format (NCHW: [batch, channels, height, width])",
            "Check padding and stride settings if output shape is incorrect"
        };

        // Add specific suggestions based on the error message
        if (result.ErrorMessage != null)
        {
            var errorMsg = result.ErrorMessage.ToLower();

            if (errorMsg.Contains("channel"))
            {
                suggestions.Add("Adjust either input tensor channels or kernel in-channels to match");
            }

            if (errorMsg.Contains("shape"))
            {
                suggestions.Add("Verify input tensor dimensions are compatible with kernel size");
            }
        }

        return suggestions;
    }
}
