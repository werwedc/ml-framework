using System.Runtime.CompilerServices;

namespace MLFramework.MobileRuntime.Backends.Cpu.Utils
{
    /// <summary>
    /// CPU vectorization utilities for SIMD operations.
    /// Provides optimized implementations for ARM NEON/SVE and Intel AVX.
    /// </summary>
    internal static class CpuVectorization
    {
        private static readonly bool IsArmArchitecture;
        private static readonly bool IsX86Architecture;

        static CpuVectorization()
        {
            // Detect architecture
            var arch = System.Runtime.InteropServices.RuntimeInformation.OSArchitecture;
            IsArmArchitecture = arch == System.Runtime.InteropServices.Architecture.Arm64 ||
                               arch == System.Runtime.InteropServices.Architecture.Arm;
            IsX86Architecture = arch == System.Runtime.InteropServices.Architecture.X64 ||
                               arch == System.Runtime.InteropServices.Architecture.X86;
        }

        #region ARM NEON Intrinsics (Simulated)

        /// <summary>
        /// NEON vectorized addition (simulated for cross-platform compatibility).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void NeonAdd(float* dst, float* src1, float* src2, int count)
        {
            // For ARM, we would use NEON intrinsics like vaddq_f32
            // For cross-platform compatibility, we simulate with unrolled loops
            int i = 0;
            int unrollCount = count - (count % 4);

            // Process 4 elements at a time (simulating 128-bit NEON)
            for (; i < unrollCount; i += 4)
            {
                dst[i] = src1[i] + src2[i];
                dst[i + 1] = src1[i + 1] + src2[i + 1];
                dst[i + 2] = src1[i + 2] + src2[i + 2];
                dst[i + 3] = src1[i + 3] + src2[i + 3];
            }

            // Process remaining elements
            for (; i < count; i++)
            {
                dst[i] = src1[i] + src2[i];
            }
        }

        /// <summary>
        /// NEON vectorized multiplication.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void NeonMultiply(float* dst, float* src1, float* src2, int count)
        {
            int i = 0;
            int unrollCount = count - (count % 4);

            for (; i < unrollCount; i += 4)
            {
                dst[i] = src1[i] * src2[i];
                dst[i + 1] = src1[i + 1] * src2[i + 1];
                dst[i + 2] = src1[i + 2] * src2[i + 2];
                dst[i + 3] = src1[i + 3] * src2[i + 3];
            }

            for (; i < count; i++)
            {
                dst[i] = src1[i] * src2[i];
            }
        }

        /// <summary>
        /// NEON vectorized ReLU: max(0, x).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void NeonRelu(float* dst, float* src, int count)
        {
            int i = 0;
            int unrollCount = count - (count % 4);

            for (; i < unrollCount; i += 4)
            {
                dst[i] = src[i] > 0 ? src[i] : 0;
                dst[i + 1] = src[i + 1] > 0 ? src[i + 1] : 0;
                dst[i + 2] = src[i + 2] > 0 ? src[i + 2] : 0;
                dst[i + 3] = src[i + 3] > 0 ? src[i + 3] : 0;
            }

            for (; i < count; i++)
            {
                dst[i] = src[i] > 0 ? src[i] : 0;
            }
        }

        /// <summary>
        /// NEON vectorized sigmoid: 1 / (1 + exp(-x)).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void NeonSigmoid(float* dst, float* src, int count)
        {
            int i = 0;
            int unrollCount = count - (count % 4);

            for (; i < unrollCount; i += 4)
            {
                dst[i] = 1.0f / (1.0f + (float)System.Math.Exp(-src[i]));
                dst[i + 1] = 1.0f / (1.0f + (float)System.Math.Exp(-src[i + 1]));
                dst[i + 2] = 1.0f / (1.0f + (float)System.Math.Exp(-src[i + 2]));
                dst[i + 3] = 1.0f / (1.0f + (float)System.Math.Exp(-src[i + 3]));
            }

            for (; i < count; i++)
            {
                dst[i] = 1.0f / (1.0f + (float)System.Math.Exp(-src[i]));
            }
        }

        #endregion

        #region Intel AVX Intrinsics (Simulated)

        /// <summary>
        /// AVX vectorized addition (simulated).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void AvxAdd(float* dst, float* src1, float* src2, int count)
        {
            int i = 0;
            int unrollCount = count - (count % 8);

            // Process 8 elements at a time (simulating 256-bit AVX)
            for (; i < unrollCount; i += 8)
            {
                dst[i] = src1[i] + src2[i];
                dst[i + 1] = src1[i + 1] + src2[i + 1];
                dst[i + 2] = src1[i + 2] + src2[i + 2];
                dst[i + 3] = src1[i + 3] + src2[i + 3];
                dst[i + 4] = src1[i + 4] + src2[i + 4];
                dst[i + 5] = src1[i + 5] + src2[i + 5];
                dst[i + 6] = src1[i + 6] + src2[i + 6];
                dst[i + 7] = src1[i + 7] + src2[i + 7];
            }

            for (; i < count; i++)
            {
                dst[i] = src1[i] + src2[i];
            }
        }

        /// <summary>
        /// AVX vectorized multiplication.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void AvxMultiply(float* dst, float* src1, float* src2, int count)
        {
            int i = 0;
            int unrollCount = count - (count % 8);

            for (; i < unrollCount; i += 8)
            {
                dst[i] = src1[i] * src2[i];
                dst[i + 1] = src1[i + 1] * src2[i + 1];
                dst[i + 2] = src1[i + 2] * src2[i + 2];
                dst[i + 3] = src1[i + 3] * src2[i + 3];
                dst[i + 4] = src1[i + 4] * src2[i + 4];
                dst[i + 5] = src1[i + 5] * src2[i + 5];
                dst[i + 6] = src1[i + 6] * src2[i + 6];
                dst[i + 7] = src1[i + 7] * src2[i + 7];
            }

            for (; i < count; i++)
            {
                dst[i] = src1[i] * src2[i];
            }
        }

        #endregion

        #region Scalar Fallbacks

        /// <summary>
        /// Scalar addition fallback.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ScalarAdd(float* dst, float* src1, float* src2, int count)
        {
            for (int i = 0; i < count; i++)
            {
                dst[i] = src1[i] + src2[i];
            }
        }

        /// <summary>
        /// Scalar multiplication fallback.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ScalarMultiply(float* dst, float* src1, float* src2, int count)
        {
            for (int i = 0; i < count; i++)
            {
                dst[i] = src1[i] * src2[i];
            }
        }

        #endregion

        #region Public Helper Methods

        /// <summary>
        /// Adds two arrays element-wise, using vectorization when available.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void Add(float* dst, float* src1, float* src2, int count, bool enableVectorization)
        {
            if (!enableVectorization || count < 8)
            {
                ScalarAdd(dst, src1, src2, count);
                return;
            }

            // Use appropriate vectorization based on architecture
            if (IsArmArchitecture)
            {
                NeonAdd(dst, src1, src2, count);
            }
            else if (IsX86Architecture)
            {
                AvxAdd(dst, src1, src2, count);
            }
            else
            {
                ScalarAdd(dst, src1, src2, count);
            }
        }

        /// <summary>
        /// Multiplies two arrays element-wise, using vectorization when available.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void Multiply(float* dst, float* src1, float* src2, int count, bool enableVectorization)
        {
            if (!enableVectorization || count < 8)
            {
                ScalarMultiply(dst, src1, src2, count);
                return;
            }

            if (IsArmArchitecture)
            {
                NeonMultiply(dst, src1, src2, count);
            }
            else if (IsX86Architecture)
            {
                AvxMultiply(dst, src1, src2, count);
            }
            else
            {
                ScalarMultiply(dst, src1, src2, count);
            }
        }

        /// <summary>
        /// Applies ReLU activation in-place.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void Relu(float* data, int count, bool enableVectorization)
        {
            if (!enableVectorization || count < 8)
            {
                for (int i = 0; i < count; i++)
                {
                    data[i] = data[i] > 0 ? data[i] : 0;
                }
                return;
            }

            NeonRelu(data, data, count);
        }

        /// <summary>
        /// Applies sigmoid activation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void Sigmoid(float* dst, float* src, int count, bool enableVectorization)
        {
            if (!enableVectorization || count < 8)
            {
                for (int i = 0; i < count; i++)
                {
                    dst[i] = 1.0f / (1.0f + (float)System.Math.Exp(-src[i]));
                }
                return;
            }

            NeonSigmoid(dst, src, count);
        }

        #endregion
    }
}
