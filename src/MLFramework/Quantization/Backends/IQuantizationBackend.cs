using RitterFramework.Core.Tensor;
using MLFramework.Quantization.DataStructures;
using MLFramework.Distributed.FSDP;

namespace MLFramework.Quantization.Backends
{
    /// <summary>
    /// Interface for quantization backend implementations.
    /// </summary>
    public interface IQuantizationBackend
    {
        /// <summary>
        /// Checks if the backend is available on the current system.
        /// </summary>
        /// <returns>True if the backend is available, false otherwise.</returns>
        bool IsAvailable();

        /// <summary>
        /// Gets the name of the backend.
        /// </summary>
        /// <returns>The backend name.</returns>
        string GetName();

        /// <summary>
        /// Gets the capabilities of the backend.
        /// </summary>
        /// <returns>The backend capabilities.</returns>
        BackendCapabilities GetCapabilities();

        /// <summary>
        /// Quantizes a float32 tensor to int8 tensor.
        /// </summary>
        /// <param name="input">The input float32 tensor.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The quantized int8 tensor.</returns>
        Tensor Quantize(Tensor input, QuantizationParameters parameters);

        /// <summary>
        /// Dequantizes an int8 tensor to float32 tensor.
        /// </summary>
        /// <param name="input">The input int8 tensor.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The dequantized float32 tensor.</returns>
        Tensor Dequantize(Tensor input, QuantizationParameters parameters);

        /// <summary>
        /// Performs int8 matrix multiplication.
        /// </summary>
        /// <param name="A">First matrix (int8).</param>
        /// <param name="B">Second matrix (int8).</param>
        /// <param name="outputScale">Scale factor for the output.</param>
        /// <returns>The result of the matrix multiplication (int32 or float32).</returns>
        Tensor MatMulInt8(Tensor A, Tensor B, float outputScale = 1.0f);

        /// <summary>
        /// Performs int8 2D convolution.
        /// </summary>
        /// <param name="input">Input tensor (int8).</param>
        /// <param name="weights">Weight tensor (int8).</param>
        /// <param name="bias">Bias tensor (float32 or int32).</param>
        /// <param name="stride">Stride for the convolution.</param>
        /// <param name="padding">Padding for the convolution.</param>
        /// <param name="dilation">Dilation for the convolution.</param>
        /// <param name="outputScale">Scale factor for the output.</param>
        /// <returns>The result of the convolution (int32 or float32).</returns>
        Tensor Conv2DInt8(
            Tensor input,
            Tensor weights,
            Tensor? bias,
            int[] stride,
            int[] padding,
            int[] dilation,
            float outputScale = 1.0f);

        /// <summary>
        /// Performs inference with the backend for a given model.
        /// </summary>
        /// <param name="model">The model to run inference on.</param>
        /// <param name="inputs">The input tensors.</param>
        /// <returns>The output tensors.</returns>
        Tensor Infer(IModel model, Tensor inputs);
    }
}
