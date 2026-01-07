using System;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.Operations
{
    /// <summary>
    /// Per-channel quantization operations for N-dimensional tensors.
    /// </summary>
    public static class PerChannelOperations
    {
        /// <summary>
        /// Quantizes a tensor using per-channel quantization parameters.
        /// Assumes channels-first layout (e.g., [C, H, W] for Conv2D output).
        /// </summary>
        /// <param name="tensor">The tensor to quantize.</param>
        /// <param name="parameters">The per-channel quantization parameters.</param>
        /// <param name="channelAxis">The axis representing channels (default: 0).</param>
        /// <returns>The quantized tensor.</returns>
        public static sbyte[] QuantizeTensorPerChannel(float[] tensor, QuantizationParameters parameters, int channelAxis = 0)
        {
            if (tensor == null || tensor.Length == 0)
            {
                return Array.Empty<sbyte>();
            }

            if (!parameters.IsPerChannel)
            {
                throw new ArgumentException("Parameters must be per-channel for per-channel quantization.", nameof(parameters));
            }

            int channelCount = parameters.ChannelCount;
            if (channelCount <= 0)
            {
                throw new ArgumentException("Invalid channel count in parameters.", nameof(parameters));
            }

            var result = new sbyte[tensor.Length];
            var channelScales = parameters.ChannelScales!;
            var channelZeroPoints = parameters.ChannelZeroPoints!;

            // For per-channel quantization with channels-first layout,
            // we need to determine the size of each channel
            int channelSize = tensor.Length / channelCount;

            if (tensor.Length % channelCount != 0)
            {
                throw new ArgumentException("Tensor length must be divisible by channel count.", nameof(tensor));
            }

            // Quantize each channel independently
            for (int c = 0; c < channelCount; c++)
            {
                float scale = channelScales[c];
                int zeroPoint = channelZeroPoints[c];

                int channelStart = c * channelSize;
                int channelEnd = channelStart + channelSize;

                for (int i = channelStart; i < channelEnd; i++)
                {
                    result[i] = QuantizePerChannel(tensor[i], scale, zeroPoint, parameters.Type);
                }
            }

            return result;
        }

        /// <summary>
        /// Dequantizes a tensor using per-channel quantization parameters.
        /// </summary>
        /// <param name="quantizedTensor">The quantized tensor.</param>
        /// <param name="parameters">The per-channel quantization parameters.</param>
        /// <param name="channelAxis">The axis representing channels (default: 0).</param>
        /// <returns>The dequantized floating-point tensor.</returns>
        public static float[] DequantizeTensorPerChannel(sbyte[] quantizedTensor, QuantizationParameters parameters, int channelAxis = 0)
        {
            if (quantizedTensor == null || quantizedTensor.Length == 0)
            {
                return Array.Empty<float>();
            }

            if (!parameters.IsPerChannel)
            {
                throw new ArgumentException("Parameters must be per-channel for per-channel dequantization.", nameof(parameters));
            }

            int channelCount = parameters.ChannelCount;
            if (channelCount <= 0)
            {
                throw new ArgumentException("Invalid channel count in parameters.", nameof(parameters));
            }

            var result = new float[quantizedTensor.Length];
            var channelScales = parameters.ChannelScales!;
            var channelZeroPoints = parameters.ChannelZeroPoints!;

            int channelSize = quantizedTensor.Length / channelCount;

            if (quantizedTensor.Length % channelCount != 0)
            {
                throw new ArgumentException("Quantized tensor length must be divisible by channel count.", nameof(quantizedTensor));
            }

            // Dequantize each channel independently
            for (int c = 0; c < channelCount; c++)
            {
                float scale = channelScales[c];
                int zeroPoint = channelZeroPoints[c];

                int channelStart = c * channelSize;
                int channelEnd = channelStart + channelSize;

                for (int i = channelStart; i < channelEnd; i++)
                {
                    result[i] = DequantizePerChannel(quantizedTensor[i], scale, zeroPoint);
                }
            }

            return result;
        }

        /// <summary>
        /// Quantizes a tensor using per-channel quantization with custom shape.
        /// </summary>
        /// <param name="tensor">The tensor to quantize.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="parameters">The per-channel quantization parameters.</param>
        /// <param name="channelAxis">The axis representing channels.</param>
        /// <returns>The quantized tensor.</returns>
        public static sbyte[] QuantizeTensorPerChannel(float[] tensor, int[] shape, QuantizationParameters parameters, int channelAxis = 0)
        {
            if (tensor == null || tensor.Length == 0)
            {
                return Array.Empty<sbyte>();
            }

            if (!parameters.IsPerChannel)
            {
                throw new ArgumentException("Parameters must be per-channel for per-channel quantization.", nameof(parameters));
            }

            int channelCount = parameters.ChannelCount;
            if (shape.Length <= channelAxis || shape[channelAxis] != channelCount)
            {
                throw new ArgumentException($"Shape[{channelAxis}] must equal channel count ({channelCount}).", nameof(shape));
            }

            // Calculate strides for N-dimensional indexing
            int[] strides = new int[shape.Length];
            strides[shape.Length - 1] = 1;
            for (int i = shape.Length - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            var result = new sbyte[tensor.Length];
            var channelScales = parameters.ChannelScales!;
            var channelZeroPoints = parameters.ChannelZeroPoints!;

            // Iterate over all elements and quantize with channel-specific parameters
            for (int i = 0; i < tensor.Length; i++)
            {
                // Calculate multi-dimensional index
                int[] index = new int[shape.Length];
                int temp = i;
                for (int j = 0; j < shape.Length; j++)
                {
                    index[j] = temp / strides[j];
                    temp = temp % strides[j];
                }

                // Get channel index
                int channelIndex = index[channelAxis];

                // Quantize with channel-specific parameters
                result[i] = QuantizePerChannel(
                    tensor[i],
                    channelScales[channelIndex],
                    channelZeroPoints[channelIndex],
                    parameters.Type);
            }

            return result;
        }

        /// <summary>
        /// Dequantizes a tensor using per-channel quantization with custom shape.
        /// </summary>
        /// <param name="quantizedTensor">The quantized tensor.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="parameters">The per-channel quantization parameters.</param>
        /// <param name="channelAxis">The axis representing channels.</param>
        /// <returns>The dequantized floating-point tensor.</returns>
        public static float[] DequantizeTensorPerChannel(sbyte[] quantizedTensor, int[] shape, QuantizationParameters parameters, int channelAxis = 0)
        {
            if (quantizedTensor == null || quantizedTensor.Length == 0)
            {
                return Array.Empty<float>();
            }

            if (!parameters.IsPerChannel)
            {
                throw new ArgumentException("Parameters must be per-channel for per-channel dequantization.", nameof(parameters));
            }

            int channelCount = parameters.ChannelCount;
            if (shape.Length <= channelAxis || shape[channelAxis] != channelCount)
            {
                throw new ArgumentException($"Shape[{channelAxis}] must equal channel count ({channelCount}).", nameof(shape));
            }

            // Calculate strides for N-dimensional indexing
            int[] strides = new int[shape.Length];
            strides[shape.Length - 1] = 1;
            for (int i = shape.Length - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            var result = new float[quantizedTensor.Length];
            var channelScales = parameters.ChannelScales!;
            var channelZeroPoints = parameters.ChannelZeroPoints!;

            // Iterate over all elements and dequantize with channel-specific parameters
            for (int i = 0; i < quantizedTensor.Length; i++)
            {
                // Calculate multi-dimensional index
                int[] index = new int[shape.Length];
                int temp = i;
                for (int j = 0; j < shape.Length; j++)
                {
                    index[j] = temp / strides[j];
                    temp = temp % strides[j];
                }

                // Get channel index
                int channelIndex = index[channelAxis];

                // Dequantize with channel-specific parameters
                result[i] = DequantizePerChannel(
                    quantizedTensor[i],
                    channelScales[channelIndex],
                    channelZeroPoints[channelIndex]);
            }

            return result;
        }

        /// <summary>
        /// Quantizes a single value with per-channel parameters.
        /// </summary>
        /// <param name="value">The value to quantize.</param>
        /// <param name="scale">The channel-specific scale.</param>
        /// <param name="zeroPoint">The channel-specific zero-point.</param>
        /// <param name="type">The quantization type.</param>
        /// <returns>The quantized value.</returns>
        private static sbyte QuantizePerChannel(float value, float scale, int zeroPoint, QuantizationType type)
        {
            if (type != QuantizationType.Int8)
            {
                throw new ArgumentException("Only Int8 is supported for per-channel quantization.", nameof(type));
            }

            if (float.IsNaN(value))
            {
                return (sbyte)zeroPoint;
            }

            if (float.IsInfinity(value))
            {
                return float.IsPositiveInfinity(value) ? sbyte.MaxValue : sbyte.MinValue;
            }

            // Quantize: q = clamp(round(value / scale) + zeroPoint)
            float scaled = value / scale;
            int quantized = QuantizationUtils.RoundHalfAwayFromZero(scaled) + zeroPoint;

            // Clamp to Int8 range
            quantized = QuantizationUtils.Clamp(quantized, sbyte.MinValue, sbyte.MaxValue);

            return (sbyte)quantized;
        }

        /// <summary>
        /// Dequantizes a single value with per-channel parameters.
        /// </summary>
        /// <param name="quantizedValue">The quantized value.</param>
        /// <param name="scale">The channel-specific scale.</param>
        /// <param name="zeroPoint">The channel-specific zero-point.</param>
        /// <returns>The dequantized floating-point value.</returns>
        private static float DequantizePerChannel(sbyte quantizedValue, float scale, int zeroPoint)
        {
            // Dequantize: value = (q - zeroPoint) * scale
            return (quantizedValue - zeroPoint) * scale;
        }

        /// <summary>
        /// Computes per-channel quantization parameters from a tensor.
        /// </summary>
        /// <param name="tensor">The tensor to compute parameters for.</param>
        /// <param name="channelAxis">The axis representing channels.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="mode">The quantization mode.</param>
        /// <param name="type">The quantization type.</param>
        /// <returns>The per-channel quantization parameters.</returns>
        public static QuantizationParameters ComputePerChannelParameters(
            float[] tensor,
            int[] shape,
            int channelAxis,
            QuantizationMode mode,
            QuantizationType type)
        {
            if (tensor == null || tensor.Length == 0)
            {
                throw new ArgumentException("Tensor must not be null or empty.", nameof(tensor));
            }

            if (shape.Length <= channelAxis)
            {
                throw new ArgumentException($"Shape length must be greater than channelAxis ({channelAxis}).", nameof(shape));
            }

            int channelCount = shape[channelAxis];
            var channelScales = new float[channelCount];
            var channelZeroPoints = new int[channelCount];
            var channelMins = new float[channelCount];
            var channelMaxs = new float[channelCount];

            // Initialize min/max for each channel
            for (int c = 0; c < channelCount; c++)
            {
                channelMins[c] = float.MaxValue;
                channelMaxs[c] = float.MinValue;
            }

            // Calculate strides for N-dimensional indexing
            int[] strides = new int[shape.Length];
            strides[shape.Length - 1] = 1;
            for (int i = shape.Length - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Find min/max for each channel
            for (int i = 0; i < tensor.Length; i++)
            {
                // Calculate multi-dimensional index
                int[] index = new int[shape.Length];
                int temp = i;
                for (int j = 0; j < shape.Length; j++)
                {
                    index[j] = temp / strides[j];
                    temp = temp % strides[j];
                }

                // Get channel index
                int channelIndex = index[channelAxis];

                if (float.IsNaN(tensor[i]) || float.IsInfinity(tensor[i]))
                {
                    continue;
                }

                if (tensor[i] < channelMins[channelIndex])
                {
                    channelMins[channelIndex] = tensor[i];
                }

                if (tensor[i] > channelMaxs[channelIndex])
                {
                    channelMaxs[channelIndex] = tensor[i];
                }
            }

            // Compute parameters for each channel
            int quantMin = QuantizationUtils.GetQuantMin(type);
            int quantMax = QuantizationUtils.GetQuantMax(type);

            float globalMin = float.MaxValue;
            float globalMax = float.MinValue;

            for (int c = 0; c < channelCount; c++)
            {
                // Skip channels with no valid data
                if (channelMins[c] == float.MaxValue || channelMaxs[c] == float.MinValue)
                {
                    channelScales[c] = 1.0f;
                    channelZeroPoints[c] = (quantMin + quantMax) / 2;
                    channelMins[c] = 0f;
                    channelMaxs[c] = 0f;
                    continue;
                }

                // Handle symmetric quantization
                if (mode == QuantizationMode.PerChannelSymmetric)
                {
                    float absMax = Math.Max(Math.Abs(channelMins[c]), Math.Abs(channelMaxs[c]));
                    channelMins[c] = -absMax;
                    channelMaxs[c] = absMax;
                }

                // Compute scale and zero-point
                channelScales[c] = QuantizationUtils.CalculateScale(
                    channelMins[c], channelMaxs[c], quantMin, quantMax);

                if (mode == QuantizationMode.PerChannelSymmetric)
                {
                    channelZeroPoints[c] = (quantMin + quantMax) / 2;
                }
                else
                {
                    channelZeroPoints[c] = QuantizationUtils.CalculateZeroPoint(
                        channelMins[c], channelMaxs[c], channelScales[c], quantMin, quantMax);
                }

                // Track global min/max for the parameters struct
                if (channelMins[c] < globalMin) globalMin = channelMins[c];
                if (channelMaxs[c] > globalMax) globalMax = channelMaxs[c];
            }

            // Create per-channel parameters
            return new QuantizationParameters(
                channelScales,
                channelZeroPoints,
                globalMin,
                globalMax,
                mode,
                type);
        }
    }
}
