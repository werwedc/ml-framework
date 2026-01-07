using System;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.Operations
{
    /// <summary>
    /// Core quantization operations for converting FP32 to Int8/UInt8.
    /// </summary>
    public static class QuantizationOperations
    {
        /// <summary>
        /// Quantizes a single floating-point value to an integer.
        /// </summary>
        /// <param name="value">The floating-point value to quantize.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The quantized integer value.</returns>
        public static sbyte Quantize(float value, QuantizationParameters parameters)
        {
            if (parameters.Type != QuantizationType.Int8)
            {
                throw new ArgumentException("Use QuantizeUInt8 for UInt8 quantization.", nameof(parameters));
            }

            if (float.IsNaN(value))
            {
                return (sbyte)parameters.ZeroPoint;
            }

            if (float.IsInfinity(value))
            {
                return float.IsPositiveInfinity(value) ? sbyte.MaxValue : sbyte.MinValue;
            }

            // Quantize: q = clamp(round(value / scale) + zeroPoint)
            float scaled = value / parameters.Scale;
            int quantized = QuantizationUtils.RoundHalfAwayFromZero(scaled) + parameters.ZeroPoint;

            // Clamp to Int8 range
            quantized = QuantizationUtils.Clamp(quantized, sbyte.MinValue, sbyte.MaxValue);

            return (sbyte)quantized;
        }

        /// <summary>
        /// Quantizes a single floating-point value to an unsigned integer.
        /// </summary>
        /// <param name="value">The floating-point value to quantize.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The quantized unsigned integer value.</returns>
        public static byte QuantizeUInt8(float value, QuantizationParameters parameters)
        {
            if (parameters.Type != QuantizationType.UInt8)
            {
                throw new ArgumentException("Use Quantize for Int8 quantization.", nameof(parameters));
            }

            if (float.IsNaN(value))
            {
                return (byte)parameters.ZeroPoint;
            }

            if (float.IsInfinity(value))
            {
                return float.IsPositiveInfinity(value) ? byte.MaxValue : byte.MinValue;
            }

            // Quantize: q = clamp(round(value / scale) + zeroPoint)
            float scaled = value / parameters.Scale;
            int quantized = QuantizationUtils.RoundHalfAwayFromZero(scaled) + parameters.ZeroPoint;

            // Clamp to UInt8 range
            quantized = QuantizationUtils.Clamp(quantized, byte.MinValue, byte.MaxValue);

            return (byte)quantized;
        }

        /// <summary>
        /// Dequantizes a single integer value back to floating-point.
        /// </summary>
        /// <param name="quantizedValue">The quantized integer value.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The dequantized floating-point value.</returns>
        public static float Dequantize(sbyte quantizedValue, QuantizationParameters parameters)
        {
            // Dequantize: value = (q - zeroPoint) * scale
            float result = (quantizedValue - parameters.ZeroPoint) * parameters.Scale;
            return result;
        }

        /// <summary>
        /// Dequantizes a single unsigned integer value back to floating-point.
        /// </summary>
        /// <param name="quantizedValue">The quantized unsigned integer value.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The dequantized floating-point value.</returns>
        public static float DequantizeUInt8(byte quantizedValue, QuantizationParameters parameters)
        {
            // Dequantize: value = (q - zeroPoint) * scale
            float result = (quantizedValue - parameters.ZeroPoint) * parameters.Scale;
            return result;
        }

        /// <summary>
        /// Quantizes an array of floating-point values using SIMD for performance.
        /// </summary>
        /// <param name="tensor">The tensor to quantize.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The quantized tensor.</returns>
        public static sbyte[] QuantizeTensor(float[] tensor, QuantizationParameters parameters)
        {
            if (tensor == null || tensor.Length == 0)
            {
                return Array.Empty<sbyte>();
            }

            var result = new sbyte[tensor.Length];

            // Use scalar quantization for simplicity
            QuantizeTensorScalar(tensor, result, parameters);

            return result;
        }

        /// <summary>
        /// Quantizes an array of floating-point values to UInt8 using SIMD.
        /// </summary>
        /// <param name="tensor">The tensor to quantize.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The quantized tensor.</returns>
        public static byte[] QuantizeTensorUInt8(float[] tensor, QuantizationParameters parameters)
        {
            if (tensor == null || tensor.Length == 0)
            {
                return Array.Empty<byte>();
            }

            var result = new byte[tensor.Length];

            for (int i = 0; i < tensor.Length; i++)
            {
                result[i] = QuantizeUInt8(tensor[i], parameters);
            }

            return result;
        }

        /// <summary>
        /// Scalar-based tensor quantization.
        /// </summary>
        private static void QuantizeTensorScalar(float[] input, sbyte[] output, QuantizationParameters parameters)
        {
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Quantize(input[i], parameters);
            }
        }

        /// <summary>
        /// Quantizes a tensor in-place, modifying the original array.
        /// </summary>
        /// <param name="tensor">The tensor to quantize (modified in-place).</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The same array with quantized values cast to floats.</returns>
        public static float[] QuantizeTensorInPlace(float[] tensor, QuantizationParameters parameters)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            for (int i = 0; i < tensor.Length; i++)
            {
                var quantized = Quantize(tensor[i], parameters);
                tensor[i] = quantized; // Store as float for compatibility
            }

            return tensor;
        }

        /// <summary>
        /// Performs a symmetric quantization with zero-point at the center.
        /// </summary>
        /// <param name="value">The value to quantize.</param>
        /// <param name="scale">The scale factor.</param>
        /// <param name="quantMin">The minimum quantized value.</param>
        /// <param name="quantMax">The maximum quantized value.</param>
        /// <returns>The quantized value.</returns>
        public static sbyte QuantizeSymmetric(float value, float scale, int quantMin = sbyte.MinValue, int quantMax = sbyte.MaxValue)
        {
            if (float.IsNaN(value))
            {
                return (sbyte)((quantMin + quantMax) / 2);
            }

            // Symmetric quantization: q = clamp(round(value / scale))
            int quantized = QuantizationUtils.RoundHalfAwayFromZero(value / scale);
            quantized = QuantizationUtils.Clamp(quantized, quantMin, quantMax);

            return (sbyte)quantized;
        }

        /// <summary>
        /// Performs an asymmetric quantization with explicit zero-point.
        /// </summary>
        /// <param name="value">The value to quantize.</param>
        /// <param name="scale">The scale factor.</param>
        /// <param name="zeroPoint">The zero-point offset.</param>
        /// <param name="quantMin">The minimum quantized value.</param>
        /// <param name="quantMax">The maximum quantized value.</param>
        /// <returns>The quantized value.</returns>
        public static sbyte QuantizeAsymmetric(float value, float scale, int zeroPoint, int quantMin = sbyte.MinValue, int quantMax = sbyte.MaxValue)
        {
            if (float.IsNaN(value))
            {
                return (sbyte)zeroPoint;
            }

            // Asymmetric quantization: q = clamp(round(value / scale) + zeroPoint)
            int quantized = QuantizationUtils.RoundHalfAwayFromZero(value / scale) + zeroPoint;
            quantized = QuantizationUtils.Clamp(quantized, quantMin, quantMax);

            return (sbyte)quantized;
        }
    }
}
