using System;

namespace RitterFramework.Core.Operations;

/// <summary>
/// Metadata for 2D convolution operations.
/// Validates that input channels match kernel channels.
/// Input shape: [N, C, H, W]
/// Kernel shape: [F, C, kH, kW]
/// </summary>
public class Conv2DMetadata : IOperationMetadata
{
    private readonly int _stride;
    private readonly int _padding;

    /// <summary>
    /// Creates a new Conv2DMetadata with default stride and padding.
    /// </summary>
    public Conv2DMetadata() : this(stride: 1, padding: 0)
    {
    }

    /// <summary>
    /// Creates a new Conv2DMetadata with specified stride and padding.
    /// </summary>
    /// <param name="stride">The stride of the convolution.</param>
    /// <param name="padding">The padding size.</param>
    public Conv2DMetadata(int stride, int padding)
    {
        if (stride <= 0)
        {
            throw new ArgumentException("Stride must be positive", nameof(stride));
        }

        if (padding < 0)
        {
            throw new ArgumentException("Padding cannot be negative", nameof(padding));
        }

        _stride = stride;
        _padding = padding;
    }

    /// <inheritdoc/>
    public OperationType Type => OperationType.Conv2D;

    /// <inheritdoc/>
    public string Name => "Conv2D";

    /// <inheritdoc/>
    public int RequiredInputTensors => 1;

    /// <inheritdoc/>
    public bool ValidateInputShapes(params long[][] inputShapes)
    {
        // For validation, we need both input shape and kernel shape
        // However, the spec suggests this should work with the registry
        // Let's assume inputShapes contains [inputShape, kernelShape] for validation purposes
        if (inputShapes.Length != 2)
        {
            return false;
        }

        var inputShape = inputShapes[0];
        var kernelShape = inputShapes[1];

        // Input should be 4D: [N, C, H, W]
        if (inputShape.Length != 4)
        {
            return false;
        }

        // Kernel should be 4D: [F, C, kH, kW]
        if (kernelShape.Length != 4)
        {
            return false;
        }

        // Input channels (C) must match kernel channels (C)
        long inputChannels = inputShape[1];
        long kernelChannels = kernelShape[1];

        return inputChannels == kernelChannels;
    }

    /// <inheritdoc/>
    public long[] InferOutputShape(params long[][] inputShapes)
    {
        // inputShapes should contain [inputShape, kernelShape]
        var inputShape = inputShapes[0];
        var kernelShape = inputShapes[1];

        long N = inputShape[0];  // Batch size
        long F = kernelShape[0];  // Number of filters
        long H = inputShape[2];   // Input height
        long W = inputShape[3];   // Input width
        long kH = kernelShape[2]; // Kernel height
        long kW = kernelShape[3]; // Kernel width

        // Calculate output dimensions
        long outH = ((H + 2 * _padding - kH) / _stride) + 1;
        long outW = ((W + 2 * _padding - kW) / _stride) + 1;

        if (outH <= 0 || outW <= 0)
        {
            throw new ArgumentException(
                $"Invalid output dimensions: [{outH}, {outW}]. Check kernel size, padding, and stride.");
        }

        return new long[] { N, F, outH, outW };
    }

    /// <summary>
    /// Gets the stride for this convolution.
    /// </summary>
    public int Stride => _stride;

    /// <summary>
    /// Gets the padding for this convolution.
    /// </summary>
    public int Padding => _padding;
}
