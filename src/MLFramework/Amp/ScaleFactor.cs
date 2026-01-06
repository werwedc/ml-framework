using System;
using MLFramework.Core;

namespace MLFramework.Amp
{
    /// <summary>
    /// Utility class for common scale factor values
    /// </summary>
    public static class ScaleFactor
    {
        /// <summary>
        /// No scaling (scale = 1.0)
        /// </summary>
        public const float None = 1.0f;

        /// <summary>
        /// Conservative scale (2^8 = 256)
        /// </summary>
        public const float Conservative = 256.0f;

        /// <summary>
        /// Moderate scale (2^16 = 65536)
        /// </summary>
        public const float Moderate = 65536.0f;

        /// <summary>
        /// Aggressive scale (2^20 = 1048576)
        /// </summary>
        public const float Aggressive = 1048576.0f;

        /// <summary>
        /// Creates a power-of-two scale factor
        /// </summary>
        /// <param name="exponent">The exponent (scale = 2^exponent)</param>
        /// <returns>The scale factor</returns>
        public static float PowerOfTwo(int exponent)
        {
            if (exponent < 0)
                throw new ArgumentException("Exponent must be non-negative", nameof(exponent));

            if (exponent > 31)
                throw new ArgumentException("Exponent must be <= 31 to prevent overflow", nameof(exponent));

            return (float)Math.Pow(2.0, exponent);
        }

        /// <summary>
        /// Gets the recommended scale for a given precision
        /// </summary>
        /// <param name="precision">The target precision</param>
        /// <returns>Recommended scale factor</returns>
        public static float GetRecommendedScale(DataType precision)
        {
            return precision switch
            {
                DataType.Float16 => Moderate,      // FP16 typically needs 2^16
                DataType.BFloat16 => Conservative, // BF16 has better range, can use lower scale
                DataType.Float32 => None,          // FP32 doesn't need scaling
                _ => throw new ArgumentException($"Unsupported precision: {precision}", nameof(precision))
            };
        }

        /// <summary>
        /// Validates a scale factor value
        /// </summary>
        /// <param name="scale">The scale factor to validate</param>
        /// <returns>True if valid, false otherwise</returns>
        public static bool IsValidScale(float scale)
        {
            return !float.IsNaN(scale) &&
                   !float.IsInfinity(scale) &&
                   scale > 0 &&
                   scale < float.MaxValue;
        }

        /// <summary>
        /// Clamps a scale factor to be within valid range
        /// </summary>
        /// <param name="scale">The scale factor to clamp</param>
        /// <param name="minScale">Minimum allowed scale (default: 1.0)</param>
        /// <param name="maxScale">Maximum allowed scale (default: 2^24)</param>
        /// <returns>Clamped scale factor</returns>
        public static float ClampScale(float scale, float minScale = 1.0f, float maxScale = 16777216.0f)
        {
            if (float.IsNaN(scale))
                return minScale;

            if (float.IsInfinity(scale))
                return maxScale;

            return Math.Max(minScale, Math.Min(scale, maxScale));
        }
    }
}
