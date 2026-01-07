using System;
using System.Numerics;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.Operations
{
    /// <summary>
    /// Dequantization operations for converting Int8/UInt8 back to FP32.
    /// </summary>
    public static class DequantizationOperations
    {
        /// <summary>
        /// Dequantizes an array of signed integer values back to floating-point.
        /// </summary>
        /// <param name="quantizedTensor">The quantized tensor.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The dequantized floating-point tensor.</returns>
        public static float[] DequantizeTensor(sbyte[] quantizedTensor, QuantizationParameters parameters)
        {
            if (quantizedTensor == null || quantizedTensor.Length == 0)
            {
                return Array.Empty<float>();
            }

            var result = new float[quantizedTensor.Length];

            // Use scalar dequantization for simplicity
            DequantizeTensorScalar(quantizedTensor, result, parameters);

            return result;
        }

        /// <summary>
        /// Dequantizes an array of unsigned integer values back to floating-point.
        /// </summary>
        /// <param name="quantizedTensor">The quantized tensor.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The dequantized floating-point tensor.</returns>
        public static float[] DequantizeTensorUInt8(byte[] quantizedTensor, QuantizationParameters parameters)
        {
            if (quantizedTensor == null || quantizedTensor.Length == 0)
            {
                return Array.Empty<float>();
            }

            var result = new float[quantizedTensor.Length];

            for (int i = 0; i < quantizedTensor.Length; i++)
            {
                result[i] = DequantizeUInt8(quantizedTensor[i], parameters);
            }

            return result;
        }

        /// <summary>
        /// Scalar-based tensor dequantization.
        /// </summary>
        private static void DequantizeTensorScalar(sbyte[] input, float[] output, QuantizationParameters parameters)
        {
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Dequantize(input[i], parameters);
            }
        }

        /// <summary>
        /// Dequantizes a single signed integer value back to floating-point.
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
        /// Dequantizes a tensor in-place, modifying the original array.
        /// </summary>
        /// <param name="tensor">The quantized tensor to dequantize (modified in-place).</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The same array with dequantized floating-point values.</returns>
        public static float[] DequantizeTensorInPlace(sbyte[] tensor, QuantizationParameters parameters)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            var result = new float[tensor.Length];

            for (int i = 0; i < tensor.Length; i++)
            {
                result[i] = Dequantize(tensor[i], parameters);
            }

            return result;
        }

        /// <summary>
        /// Performs a symmetric dequantization.
        /// </summary>
        /// <param name="quantizedValue">The quantized value.</param>
        /// <param name="scale">The scale factor.</param>
        /// <returns>The dequantized floating-point value.</returns>
        public static float DequantizeSymmetric(sbyte quantizedValue, float scale)
        {
            // Symmetric dequantization: value = q * scale
            return quantizedValue * scale;
        }

        /// <summary>
        /// Performs an asymmetric dequantization.
        /// </summary>
        /// <param name="quantizedValue">The quantized value.</param>
        /// <param name="scale">The scale factor.</param>
        /// <param name="zeroPoint">The zero-point offset.</param>
        /// <returns>The dequantized floating-point value.</returns>
        public static float DequantizeAsymmetric(sbyte quantizedValue, float scale, int zeroPoint)
        {
            // Asymmetric dequantization: value = (q - zeroPoint) * scale
            return (quantizedValue - zeroPoint) * scale;
        }

        /// <summary>
        /// Performs a round-trip quantization and dequantization to estimate error.
        /// </summary>
        /// <param name="tensor">The original tensor.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The error tensor (dequantized - original).</returns>
        public static float[] ComputeQuantizationError(float[] tensor, QuantizationParameters parameters)
        {
            if (tensor == null || tensor.Length == 0)
            {
                return Array.Empty<float>();
            }

            var quantized = QuantizationOperations.QuantizeTensor(tensor, parameters);
            var dequantized = DequantizeTensor(quantized, parameters);
            var error = new float[tensor.Length];

            for (int i = 0; i < tensor.Length; i++)
            {
                error[i] = dequantized[i] - tensor[i];
            }

            return error;
        }

        /// <summary>
        /// Computes the mean squared error (MSE) after quantization and dequantization.
        /// </summary>
        /// <param name="tensor">The original tensor.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The MSE.</returns>
        public static float ComputeMSE(float[] tensor, QuantizationParameters parameters)
        {
            var error = ComputeQuantizationError(tensor, parameters);
            float mse = 0f;

            for (int i = 0; i < error.Length; i++)
            {
                mse += error[i] * error[i];
            }

            return mse / error.Length;
        }

        /// <summary>
        /// Computes the signal-to-quantization-noise ratio (SQNR).
        /// </summary>
        /// <param name="tensor">The original tensor.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The SQNR in dB.</returns>
        public static float ComputeSQNR(float[] tensor, QuantizationParameters parameters)
        {
            if (tensor == null || tensor.Length == 0)
            {
                return float.NegativeInfinity;
            }

            var error = ComputeQuantizationError(tensor, parameters);

            float signalPower = 0f;
            float noisePower = 0f;

            for (int i = 0; i < tensor.Length; i++)
            {
                signalPower += tensor[i] * tensor[i];
                noisePower += error[i] * error[i];
            }

            if (noisePower == 0)
            {
                return float.PositiveInfinity;
            }

            float sqnr = 10f * (float)Math.Log10(signalPower / noisePower);
            return sqnr;
        }

        /// <summary>
        /// Computes the peak signal-to-noise ratio (PSNR) after quantization.
        /// </summary>
        /// <param name="tensor">The original tensor.</param>
        /// <param name="parameters">The quantization parameters.</param>
        /// <returns>The PSNR in dB.</returns>
        public static float ComputePSNR(float[] tensor, QuantizationParameters parameters)
        {
            if (tensor == null || tensor.Length == 0)
            {
                return float.NegativeInfinity;
            }

            var error = ComputeQuantizationError(tensor, parameters);

            // Find max value in original tensor
            float maxValue = 0f;
            for (int i = 0; i < tensor.Length; i++)
            {
                float absValue = Math.Abs(tensor[i]);
                if (absValue > maxValue)
                {
                    maxValue = absValue;
                }
            }

            if (maxValue == 0)
            {
                return float.NegativeInfinity;
            }

            float mse = ComputeMSE(tensor, parameters);

            if (mse == 0)
            {
                return float.PositiveInfinity;
            }

            float psnr = 10f * (float)Math.Log10((maxValue * maxValue) / mse);
            return psnr;
        }
    }
}
