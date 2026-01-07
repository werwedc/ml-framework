namespace MLFramework.Quantization.QAT;

/// <summary>
/// Interface for layers in the ML framework.
/// </summary>
public interface ILayer
{
    /// <summary>
    /// Forward pass through the layer.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor.</returns>
    RitterFramework.Core.Tensor.Tensor Forward(RitterFramework.Core.Tensor.Tensor input);

    /// <summary>
    /// Backward pass through the layer.
    /// </summary>
    /// <param name="upstreamGradient">The gradient from downstream operations.</param>
    /// <returns>The gradient with respect to the input.</returns>
    RitterFramework.Core.Tensor.Tensor Backward(RitterFramework.Core.Tensor.Tensor upstreamGradient);
}
