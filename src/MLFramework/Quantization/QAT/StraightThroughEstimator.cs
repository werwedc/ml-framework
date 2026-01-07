using RitterFramework.Core.Tensor;

namespace MLFramework.Quantization.QAT;

/// <summary>
/// Straight-Through Estimator (STE) for gradient computation in fake quantization.
/// Provides identity gradients during backpropagation to bypass the non-differentiable quantization operation.
/// </summary>
public static class StraightThroughEstimator
{
    /// <summary>
    /// Computes the backward pass using the straight-through estimator.
    /// The gradient is passed through unchanged (identity function) to allow gradients to flow through
    /// the non-differentiable quantization operation.
    /// </summary>
    /// <param name="upstreamGradient">The gradient from downstream operations.</param>
    /// <returns>The gradient passed through unchanged.</returns>
    public static Tensor Backward(Tensor upstreamGradient)
    {
        // STE: Identity function - pass through gradients directly
        // This bypasses the non-differentiable rounding operation during backpropagation
        return upstreamGradient.Clone();
    }

    /// <summary>
    /// Clips the gradient to prevent gradient explosion.
    /// Useful when working with quantized networks where gradients can become unstable.
    /// </summary>
    /// <param name="gradient">The gradient to clip.</param>
    /// <param name="maxNorm">The maximum gradient norm.</param>
    /// <returns>The clipped gradient.</returns>
    public static Tensor ClipGradient(Tensor gradient, float maxNorm = 1.0f)
    {
        var data = gradient.Data;
        var clippedData = new float[data.Length];

        for (int i = 0; i < data.Length; i++)
        {
            clippedData[i] = Math.Max(-maxNorm, Math.Min(maxNorm, data[i]));
        }

        return Tensor.FromArray(clippedData, gradient.Dtype);
    }

    /// <summary>
    /// Computes the straight-through gradient with optional clipping.
    /// </summary>
    /// <param name="upstreamGradient">The gradient from downstream operations.</param>
    /// <param name="enableClipping">Whether to enable gradient clipping.</param>
    /// <param name="maxNorm">The maximum gradient norm when clipping is enabled.</param>
    /// <returns>The gradient passed through with optional clipping.</returns>
    public static Tensor BackwardWithClipping(Tensor upstreamGradient, bool enableClipping = false, float maxNorm = 1.0f)
    {
        var gradient = Backward(upstreamGradient);

        if (enableClipping)
        {
            gradient = ClipGradient(gradient, maxNorm);
        }

        return gradient;
    }
}
