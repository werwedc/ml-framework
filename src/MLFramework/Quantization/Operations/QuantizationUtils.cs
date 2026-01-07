using System;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.Operations
{
    /// <summary>
    /// Utility methods for quantization operations.
    /// </summary>
    public static class QuantizationUtils
    {
        /// <summary>
        /// Gets the minimum quantized value for a given quantization type.
        /// </summary>
        /// <param name="type">The quantization type.</param>
        /// <returns>The minimum quantized value.</returns>
        public static int GetQuantMin(QuantizationType type)
        {
            return type switch
            {
                QuantizationType.Int8 => sbyte.MinValue,
                QuantizationType.UInt8 => byte.MinValue,
                _ => throw new ArgumentException($"Unsupported quantization type: {type}")
            };
        }

        /// <summary>
        /// Gets the maximum quantized value for a given quantization type.
        /// </summary>
        /// <param name="type">The quantization type.</param>
        /// <returns>The maximum quantized value.</returns>
        public static int GetQuantMax(QuantizationType type)
        {
            return type switch
            {
                QuantizationType.Int8 => sbyte.MaxValue,
                QuantizationType.UInt8 => byte.MaxValue,
                _ => throw new ArgumentException($"Unsupported quantization type: {type}")
            };
        }

        /// <summary>
        /// Calculates the scale factor for quantization.
        /// </summary>
        /// <param name="min">The minimum floating-point value.</param>
        /// <param name="max">The maximum floating-point value.</param>
        /// <param name="quantMin">The minimum quantized value.</param>
        /// <param name="quantMax">The maximum quantized value.</param>
        /// <returns>The scale factor.</returns>
        public static float CalculateScale(float min, float max, int quantMin, int quantMax)
        {
            if (float.IsNaN(min) || float.IsNaN(max))
            {
                throw new ArgumentException("Min and max must not be NaN.");
            }

            if (float.IsInfinity(min) || float.IsInfinity(max))
            {
                throw new ArgumentException("Min and max must not be infinite.");
            }

            if (max <= min)
            {
                throw new ArgumentException("Max must be greater than min.");
            }

            // Scale = (max - min) / (quantMax - quantMin)
            float scale = (max - min) / (quantMax - quantMin);

            // Ensure scale is positive
            if (scale <= 0)
            {
                scale = 1e-6f;
            }

            return scale;
        }

        /// <summary>
        /// Calculates the zero-point for asymmetric quantization.
        /// </summary>
        /// <param name="min">The minimum floating-point value.</param>
        /// <param name="max">The maximum floating-point value.</param>
        /// <param name="scale">The scale factor.</param>
        /// <param name="quantMin">The minimum quantized value.</param>
        /// <param name="quantMax">The maximum quantized value.</param>
        /// <returns>The zero-point.</returns>
        public static int CalculateZeroPoint(float min, float max, float scale, int quantMin, int quantMax)
        {
            if (float.IsNaN(min) || float.IsNaN(max) || float.IsNaN(scale))
            {
                throw new ArgumentException("Min, max, and scale must not be NaN.");
            }

            // ZeroPoint = quantMin - min / scale
            float zeroPoint = quantMin - min / scale;

            // Clamp to valid range
            zeroPoint = Clamp((int)Math.Round(zeroPoint), quantMin, quantMax);

            return (int)zeroPoint;
        }

        /// <summary>
        /// Clamps a value to the specified range.
        /// </summary>
        /// <param name="value">The value to clamp.</param>
        /// <param name="min">The minimum value.</param>
        /// <param name="max">The maximum value.</param>
        /// <returns>The clamped value.</returns>
        public static int Clamp(int value, int min, int max)
        {
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }

        /// <summary>
        /// Clamps a floating-point value to the specified range.
        /// </summary>
        /// <param name="value">The value to clamp.</param>
        /// <param name="min">The minimum value.</param>
        /// <param name="max">The maximum value.</param>
        /// <returns>The clamped value.</returns>
        public static float Clamp(float value, float min, float max)
        {
            if (float.IsNaN(value))
            {
                return min;
            }

            if (float.IsInfinity(value))
            {
                return float.IsPositiveInfinity(value) ? max : min;
            }

            if (value < min) return min;
            if (value > max) return max;
            return value;
        }

        /// <summary>
        /// Rounds a float to the nearest integer, rounding half away from zero.
        /// </summary>
        /// <param name="value">The value to round.</param>
        /// <returns>The rounded integer.</returns>
        public static int RoundHalfAwayFromZero(float value)
        {
            if (float.IsNaN(value))
            {
                return 0;
            }

            if (float.IsInfinity(value))
            {
                return float.IsPositiveInfinity(value) ? int.MaxValue : int.MinValue;
            }

            return (int)Math.CopySign(Math.Floor(Math.Abs(value) + 0.5f), value);
        }

        /// <summary>
        /// Computes quantization parameters for a given range.
        /// </summary>
        /// <param name="min">The minimum floating-point value.</param>
        /// <param name="max">The maximum floating-point value.</param>
        /// <param name="mode">The quantization mode.</param>
        /// <param name="type">The quantization type.</param>
        /// <returns>The quantization parameters.</returns>
        public static QuantizationParameters ComputeParameters(
            float min,
            float max,
            QuantizationMode mode,
            QuantizationType type)
        {
            int quantMin = GetQuantMin(type);
            int quantMax = GetQuantMax(type);

            if (float.IsNaN(min) || float.IsNaN(max))
            {
                throw new ArgumentException("Min and max must not be NaN.");
            }

            // Ensure min < max
            if (max <= min)
            {
                max = min + 1e-6f;
            }

            // Handle symmetric quantization
            if (mode == QuantizationMode.PerTensorSymmetric ||
                mode == QuantizationMode.PerChannelSymmetric)
            {
                // For symmetric, we want zero to map to zero-point
                float absMax = Math.Max(Math.Abs(min), Math.Abs(max));
                min = -absMax;
                max = absMax;
            }

            float scale = CalculateScale(min, max, quantMin, quantMax);
            int zeroPoint;

            if (mode == QuantizationMode.PerTensorSymmetric ||
                mode == QuantizationMode.PerChannelSymmetric)
            {
                // For symmetric quantization, zero-point is at the center of the quantized range
                zeroPoint = (quantMin + quantMax) / 2;
            }
            else
            {
                zeroPoint = CalculateZeroPoint(min, max, scale, quantMin, quantMax);
            }

            return new QuantizationParameters(scale, zeroPoint, min, max, mode, type);
        }

        /// <summary>
        /// Validates if a floating-point value can be quantized.
        /// </summary>
        /// <param name="value">The value to validate.</param>
        /// <returns>True if valid, false otherwise.</returns>
        public static bool IsValidForQuantization(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        /// <summary>
        /// Validates if an array of floating-point values can be quantized.
        /// </summary>
        /// <param name="values">The values to validate.</param>
        /// <returns>True if all values are valid, false otherwise.</returns>
        public static bool IsValidForQuantization(float[] values)
        {
            if (values == null || values.Length == 0)
            {
                return false;
            }

            foreach (var value in values)
            {
                if (!IsValidForQuantization(value))
                {
                    return false;
                }
            }

            return true;
        }
    }
}
