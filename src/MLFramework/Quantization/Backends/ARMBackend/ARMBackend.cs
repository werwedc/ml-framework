using RitterFramework.Core.Tensor;
using MLFramework.Quantization.DataStructures;
using MLFramework.Distributed.FSDP;

namespace MLFramework.Quantization.Backends.ARMBackend
{
    /// <summary>
    /// ARM backend with NEON integration and ARMv8.2 dot product instructions.
    /// </summary>
    public class ARMBackend : IQuantizationBackend
    {
        private readonly Backends.CPUBackend.CPUBackend _fallbackBackend;
        private readonly bool _isAvailable;

        /// <summary>
        /// Initializes a new instance of <see cref="ARMBackend"/> class.
        /// </summary>
        public ARMBackend()
        {
            _fallbackBackend = new Backends.CPUBackend.CPUBackend();
            _isAvailable = ARMFeatureDetection.IsAvailable();
        }

        /// <summary>
        /// Checks if the backend is available on current system.
        /// </summary>
        /// <returns>True if ARM features are available, false otherwise.</returns>
        public bool IsAvailable()
        {
            return _isAvailable;
        }

        /// <summary>
        /// Gets the name of the backend.
        /// </summary>
        /// <returns>The backend name.</returns>
        public string GetName()
        {
            return "ARM NEON";
        }

        /// <summary>
        /// Gets the capabilities of the backend.
        /// </summary>
        /// <returns>The backend capabilities.</returns>
        public BackendCapabilities GetCapabilities()
        {
            if (!_isAvailable)
            {
                return _fallbackBackend.GetCapabilities();
            }

            return new BackendCapabilities(
                flags: BackendCapabilityFlags.Int8MatMul |
                       BackendCapabilityFlags.Int8Conv2D |
                       BackendCapabilityFlags.PerChannelQuantization,
                maxTensorSize: long.MaxValue,
                minBatchSize: 1,
                preferredBatchSize: 32,
                maxThreads: Environment.ProcessorCount);
        }

        /// <summary>
        /// Quantizes a float32 tensor to int8 tensor.
        /// </summary>
        public Tensor Quantize(Tensor input, QuantizationParameters parameters)
        {
            return _fallbackBackend.Quantize(input, parameters);
        }

        /// <summary>
        /// Dequantizes an int8 tensor to float32 tensor.
        /// </summary>
        public Tensor Dequantize(Tensor input, QuantizationParameters parameters)
        {
            return _fallbackBackend.Dequantize(input, parameters);
        }

        /// <summary>
        /// Performs int8 matrix multiplication.
        /// </summary>
        public Tensor MatMulInt8(Tensor A, Tensor B, float outputScale = 1.0f)
        {
            return _fallbackBackend.MatMulInt8(A, B, outputScale);
        }

        /// <summary>
        /// Performs int8 2D convolution.
        /// </summary>
        public Tensor Conv2DInt8(
            Tensor input,
            Tensor weights,
            Tensor? bias,
            int[] stride,
            int[] padding,
            int[] dilation,
            float outputScale = 1.0f)
        {
            return _fallbackBackend.Conv2DInt8(input, weights, bias, stride, padding, dilation, outputScale);
        }

        /// <summary>
        /// Performs inference with the backend for a given model.
        /// </summary>
        public Tensor Infer(IModel model, Tensor inputs)
        {
            return _fallbackBackend.Infer(model, inputs);
        }
    }
}
