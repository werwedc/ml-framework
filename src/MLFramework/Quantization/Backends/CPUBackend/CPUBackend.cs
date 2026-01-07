using RitterFramework.Core.Tensor;
using MLFramework.Quantization.DataStructures;
using MLFramework.Distributed.FSDP;
using RitterFramework.Core;

namespace MLFramework.Quantization.Backends.CPUBackend
{
    /// <summary>
    /// CPU-based quantization backend (pure C# implementation with SIMD optimization).
    /// This is fallback backend that is always available.
    /// </summary>
    public class CPUBackend : IQuantizationBackend
    {
        private readonly BackendCapabilities _capabilities;
        private readonly CPUInt8Operations _int8Operations;

        /// <summary>
        /// Initializes a new instance of the <see cref="CPUBackend"/> class.
        /// </summary>
        public CPUBackend()
        {
            _capabilities = new BackendCapabilities(
                flags: BackendCapabilityFlags.Int8MatMul |
                       BackendCapabilityFlags.Int8Conv2D |
                       BackendCapabilityFlags.PerChannelQuantization |
                       BackendCapabilityFlags.MixedPrecision |
                       BackendCapabilityFlags.DynamicQuantization |
                       BackendCapabilityFlags.StaticQuantization |
                       BackendCapabilityFlags.AsymmetricQuantization |
                       BackendCapabilityFlags.SymmetricQuantization,
                maxTensorSize: long.MaxValue,
                minBatchSize: 1,
                preferredBatchSize: 32,
                maxThreads: Environment.ProcessorCount
            );

            _int8Operations = new CPUInt8Operations();
        }

        /// <summary>
        /// Checks if the backend is available on the current system.
        /// CPU backend is always available.
        /// </summary>
        /// <returns>Always true.</returns>
        public bool IsAvailable()
        {
            return true;
        }

        /// <summary>
        /// Gets the name of the backend.
        /// </summary>
        /// <returns>The backend name.</returns>
        public string GetName()
        {
            return "CPU";
        }

        /// <summary>
        /// Gets the capabilities of the backend.
        /// </summary>
        /// <returns>The backend capabilities.</returns>
        public BackendCapabilities GetCapabilities()
        {
            return _capabilities;
        }

        /// <summary>
        /// Quantizes a float32 tensor to int8 tensor.
        /// </summary>
        /// <param name="input">The input float32 tensor.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The quantized int8 tensor.</returns>
        public Tensor Quantize(Tensor input, QuantizationParameters parameters)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (!parameters.Validate())
            {
                throw new ArgumentException("Invalid quantization parameters.", nameof(parameters));
            }

            var inputData = input.Data;
            var outputData = new float[inputData.Length];

            if (parameters.IsPerChannel && parameters.ChannelScales != null && parameters.ChannelZeroPoints != null)
            {
                // Per-channel quantization
                int channelCount = parameters.ChannelCount;
                int elementsPerChannel = inputData.Length / channelCount;

                Parallel.For(0, channelCount, channel =>
                {
                    float scale = parameters.ChannelScales[channel];
                    int zeroPoint = parameters.ChannelZeroPoints[channel];

                    for (int i = 0; i < elementsPerChannel; i++)
                    {
                        int idx = channel * elementsPerChannel + i;
                        float quantized = (float)Math.Round(inputData[idx] / scale) + zeroPoint;
                        outputData[idx] = (sbyte)Math.Clamp(quantized, sbyte.MinValue, sbyte.MaxValue);
                    }
                });
            }
            else
            {
                // Per-tensor quantization
                float scale = parameters.Scale;
                int zeroPoint = parameters.ZeroPoint;

                Parallel.For(0, inputData.Length, i =>
                {
                    float quantized = (float)Math.Round(inputData[i] / scale) + zeroPoint;
                    outputData[i] = (sbyte)Math.Clamp(quantized, sbyte.MinValue, sbyte.MaxValue);
                });
            }

            return new Tensor(outputData, input.Shape, false, RitterFramework.Core.DataType.Int8);
        }

        /// <summary>
        /// Dequantizes an int8 tensor to float32 tensor.
        /// </summary>
        /// <param name="input">The input int8 tensor.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The dequantized float32 tensor.</returns>
        public Tensor Dequantize(Tensor input, QuantizationParameters parameters)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (!parameters.Validate())
            {
                throw new ArgumentException("Invalid quantization parameters.", nameof(parameters));
            }

            var inputData = input.Data;
            var outputData = new float[inputData.Length];

            if (parameters.IsPerChannel && parameters.ChannelScales != null && parameters.ChannelZeroPoints != null)
            {
                // Per-channel dequantization
                int channelCount = parameters.ChannelCount;
                int elementsPerChannel = inputData.Length / channelCount;

                Parallel.For(0, channelCount, channel =>
                {
                    float scale = parameters.ChannelScales[channel];
                    int zeroPoint = parameters.ChannelZeroPoints[channel];

                    for (int i = 0; i < elementsPerChannel; i++)
                    {
                        int idx = channel * elementsPerChannel + i;
                        outputData[idx] = (inputData[idx] - zeroPoint) * scale;
                    }
                });
            }
            else
            {
                // Per-tensor dequantization
                float scale = parameters.Scale;
                int zeroPoint = parameters.ZeroPoint;

                Parallel.For(0, inputData.Length, i =>
                {
                    outputData[i] = (inputData[i] - zeroPoint) * scale;
                });
            }

            return new Tensor(outputData, input.Shape, false, RitterFramework.Core.DataType.Float32);
        }

        /// <summary>
        /// Performs int8 matrix multiplication.
        /// </summary>
        /// <param name="A">First matrix (int8).</param>
        /// <param name="B">Second matrix (int8).</param>
        /// <param name="outputScale">Scale factor for the output.</param>
        /// <returns>The result of the matrix multiplication (float32).</returns>
        public Tensor MatMulInt8(Tensor A, Tensor B, float outputScale = 1.0f)
        {
            if (A == null || B == null)
            {
                throw new ArgumentNullException(A == null ? nameof(A) : nameof(B));
            }

            if (A.Dimensions != 2 || B.Dimensions != 2)
            {
                throw new ArgumentException("Both matrices must be 2D.");
            }

            int m = A.Shape[0];
            int k = A.Shape[1];
            int n = B.Shape[1];

            if (B.Shape[0] != k)
            {
                throw new ArgumentException($"Matrix dimensions mismatch: A({m}x{k}) * B({B.Shape[0]}x{n}).");
            }

            var outputData = new float[m * n];

            // Use SIMD-optimized matrix multiplication
            _int8Operations.MatMulInt8(
                A.Data, B.Data, outputData,
                m, k, n, outputScale);

            return new Tensor(outputData, new int[] { m, n }, false, RitterFramework.Core.DataType.Float32);
        }

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
        /// <returns>The result of the convolution (float32).</returns>
        public Tensor Conv2DInt8(
            Tensor input,
            Tensor weights,
            Tensor? bias,
            int[] stride,
            int[] padding,
            int[] dilation,
            float outputScale = 1.0f)
        {
            if (input == null || weights == null)
            {
                throw new ArgumentNullException(input == null ? nameof(input) : nameof(weights));
            }

            if (input.Dimensions != 4 || weights.Dimensions != 4)
            {
                throw new ArgumentException("Input and weights must be 4D tensors (NCHW format).");
            }

            int batchSize = input.Shape[0];
            int inChannels = input.Shape[1];
            int inputHeight = input.Shape[2];
            int inputWidth = input.Shape[3];

            int outChannels = weights.Shape[0];
            int kernelHeight = weights.Shape[2];
            int kernelWidth = weights.Shape[3];

            if (weights.Shape[1] != inChannels)
            {
                throw new ArgumentException($"Weight channels ({weights.Shape[1]}) must match input channels ({inChannels}).");
            }

            if (stride.Length != 2 || padding.Length != 2 || dilation.Length != 2)
            {
                throw new ArgumentException("Stride, padding, and dilation must be arrays of length 2.");
            }

            int strideHeight = stride[0];
            int strideWidth = stride[1];
            int paddingHeight = padding[0];
            int paddingWidth = padding[1];
            int dilationHeight = dilation[0];
            int dilationWidth = dilation[1];

            // Calculate output dimensions
            int outputHeight = (inputHeight + 2 * paddingHeight - dilationHeight * (kernelHeight - 1) - 1) / strideHeight + 1;
            int outputWidth = (inputWidth + 2 * paddingWidth - dilationWidth * (kernelWidth - 1) - 1) / strideWidth + 1;

            var outputData = new float[batchSize * outChannels * outputHeight * outputWidth];

            // Use SIMD-optimized convolution
            _int8Operations.Conv2DInt8(
                input.Data, weights.Data, bias?.Data, outputData,
                batchSize, inChannels, inputHeight, inputWidth,
                outChannels, kernelHeight, kernelWidth,
                outputHeight, outputWidth,
                strideHeight, strideWidth,
                paddingHeight, paddingWidth,
                dilationHeight, dilationWidth,
                outputScale);

            return new Tensor(outputData,
                new int[] { batchSize, outChannels, outputHeight, outputWidth },
                false, RitterFramework.Core.DataType.Float32);
        }

        /// <summary>
        /// Performs inference with the backend for a given model.
        /// </summary>
        /// <param name="model">The model to run inference on.</param>
        /// <param name="inputs">The input tensors.</param>
        /// <returns>The output tensors.</returns>
        public Tensor Infer(IModel model, Tensor inputs)
        {
            if (model == null)
            {
                throw new ArgumentNullException(nameof(model));
            }

            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            // For CPU backend, we use the model's forward pass directly
            // In a real implementation, we would quantize inputs, run quantized operations,
            // and dequantize outputs. For now, we'll use the model's forward pass.
            return model.Forward(inputs);
        }
    }
}
