using System;

namespace MLFramework.Amp
{
    /// <summary>
    /// High-performance casting utilities for AMP
    /// </summary>
    public static class AmpCast
    {
        /// <summary>
        /// Cast float array to Half (zero-copy when possible)
        /// </summary>
        public static Half[] CastToHalf(float[] input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            Half[] result = new Half[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                result[i] = new Half(input[i]);
            }
            return result;
        }

        /// <summary>
        /// Cast float array to BFloat16 (zero-copy when possible)
        /// </summary>
        public static BFloat16[] CastToBFloat16(float[] input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            BFloat16[] result = new BFloat16[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                result[i] = new BFloat16(input[i]);
            }
            return result;
        }

        /// <summary>
        /// Cast Half array to float
        /// </summary>
        public static float[] CastToFloat(Half[] input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            float[] result = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                result[i] = (float)input[i];
            }
            return result;
        }

        /// <summary>
        /// Cast BFloat16 array to float
        /// </summary>
        public static float[] CastToFloat(BFloat16[] input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            float[] result = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                result[i] = (float)input[i];
            }
            return result;
        }

        /// <summary>
        /// In-place cast Half to float
        /// </summary>
        public static void CastInPlace(Half[] input, float[] output)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (output == null)
                throw new ArgumentNullException(nameof(output));
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output arrays must have the same length");

            for (int i = 0; i < input.Length; i++)
            {
                output[i] = (float)input[i];
            }
        }

        /// <summary>
        /// In-place cast BFloat16 to float
        /// </summary>
        public static void CastInPlace(BFloat16[] input, float[] output)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (output == null)
                throw new ArgumentNullException(nameof(output));
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output arrays must have the same length");

            for (int i = 0; i < input.Length; i++)
            {
                output[i] = (float)input[i];
            }
        }
    }
}
