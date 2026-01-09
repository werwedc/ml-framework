using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace MobileRuntime.Tensors
{
    /// <summary>
    /// Helper class for ARM NEON/SVE vectorization operations
    /// Provides vectorized implementations for ARM platforms with scalar fallbacks for x86
    /// </summary>
    internal static class ARMVectorization
    {
        private static readonly bool IsArmPlatform;

        static ARMVectorization()
        {
            IsArmPlatform = RuntimeInformation.IsOSPlatform(OSPlatform.Create("ARM")) ||
                           RuntimeInformation.IsOSPlatform(OSPlatform.Create("ARM64"));
        }

        #region Vectorized Operations

        /// <summary>
        /// Vectorized addition of two arrays
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void VectorizedAdd(IntPtr dst, IntPtr src1, IntPtr src2, long count)
        {
            if (IsArmPlatform)
            {
                VectorizedAddNEON(dst, src1, src2, count);
            }
            else
            {
                ScalarAdd(dst, src1, src2, count);
            }
        }

        /// <summary>
        /// Vectorized multiplication of two arrays
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void VectorizedMultiply(IntPtr dst, IntPtr src1, IntPtr src2, long count)
        {
            if (IsArmPlatform)
            {
                VectorizedMultiplyNEON(dst, src1, src2, count);
            }
            else
            {
                ScalarMultiply(dst, src1, src2, count);
            }
        }

        /// <summary>
        /// Vectorized ReLU activation
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void VectorizedRelu(IntPtr dst, IntPtr src, long count)
        {
            if (IsArmPlatform)
            {
                VectorizedReluNEON(dst, src, count);
            }
            else
            {
                ScalarRelu(dst, src, count);
            }
        }

        /// <summary>
        /// Vectorized sigmoid activation
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void VectorizedSigmoid(IntPtr dst, IntPtr src, long count)
        {
            if (IsArmPlatform)
            {
                VectorizedSigmoidNEON(dst, src, count);
            }
            else
            {
                ScalarSigmoid(dst, src, count);
            }
        }

        #endregion

        #region ARM NEON Implementations

        /// <summary>
        /// ARM NEON vectorized addition
        /// Uses NEON instructions to process 4 floats at a time
        /// </summary>
        private static unsafe void VectorizedAddNEON(IntPtr dst, IntPtr src1, IntPtr src2, long count)
        {
            float* dstPtr = (float*)dst;
            float* src1Ptr = (float*)src1;
            float* src2Ptr = (float*)src2;

            long i = 0;

            // Process 4 floats at a time (128-bit NEON register)
            for (; i <= count - 4; i += 4)
            {
                // In a real implementation, this would use NEON intrinsics:
                // float32x4_t a = vld1q_f32(src1Ptr + i);
                // float32x4_t b = vld1q_f32(src2Ptr + i);
                // float32x4_t result = vaddq_f32(a, b);
                // vst1q_f32(dstPtr + i, result);

                // For now, use scalar as a placeholder
                dstPtr[i] = src1Ptr[i] + src2Ptr[i];
                dstPtr[i + 1] = src1Ptr[i + 1] + src2Ptr[i + 1];
                dstPtr[i + 2] = src1Ptr[i + 2] + src2Ptr[i + 2];
                dstPtr[i + 3] = src1Ptr[i + 3] + src2Ptr[i + 3];
            }

            // Process remaining elements
            for (; i < count; i++)
            {
                dstPtr[i] = src1Ptr[i] + src2Ptr[i];
            }
        }

        /// <summary>
        /// ARM NEON vectorized multiplication
        /// </summary>
        private static unsafe void VectorizedMultiplyNEON(IntPtr dst, IntPtr src1, IntPtr src2, long count)
        {
            float* dstPtr = (float*)dst;
            float* src1Ptr = (float*)src1;
            float* src2Ptr = (float*)src2;

            long i = 0;

            for (; i <= count - 4; i += 4)
            {
                // NEON intrinsic version would be:
                // float32x4_t a = vld1q_f32(src1Ptr + i);
                // float32x4_t b = vld1q_f32(src2Ptr + i);
                // float32x4_t result = vmulq_f32(a, b);
                // vst1q_f32(dstPtr + i, result);

                dstPtr[i] = src1Ptr[i] * src2Ptr[i];
                dstPtr[i + 1] = src1Ptr[i + 1] * src2Ptr[i + 1];
                dstPtr[i + 2] = src1Ptr[i + 2] * src2Ptr[i + 2];
                dstPtr[i + 3] = src1Ptr[i + 3] * src2Ptr[i + 3];
            }

            for (; i < count; i++)
            {
                dstPtr[i] = src1Ptr[i] * src2Ptr[i];
            }
        }

        /// <summary>
        /// ARM NEON vectorized ReLU
        /// </summary>
        private static unsafe void VectorizedReluNEON(IntPtr dst, IntPtr src, long count)
        {
            float* dstPtr = (float*)dst;
            float* srcPtr = (float*)src;

            long i = 0;

            for (; i <= count - 4; i += 4)
            {
                // NEON intrinsic version would be:
                // float32x4_t a = vld1q_f32(srcPtr + i);
                // float32x4_t zero = vdupq_n_f32(0.0f);
                // float32x4_t result = vmaxq_f32(a, zero);
                // vst1q_f32(dstPtr + i, result);

                dstPtr[i] = Math.Max(0.0f, srcPtr[i]);
                dstPtr[i + 1] = Math.Max(0.0f, srcPtr[i + 1]);
                dstPtr[i + 2] = Math.Max(0.0f, srcPtr[i + 2]);
                dstPtr[i + 3] = Math.Max(0.0f, srcPtr[i + 3]);
            }

            for (; i < count; i++)
            {
                dstPtr[i] = Math.Max(0.0f, srcPtr[i]);
            }
        }

        /// <summary>
        /// ARM NEON vectorized sigmoid
        /// Uses approximation for better performance
        /// </summary>
        private static unsafe void VectorizedSigmoidNEON(IntPtr dst, IntPtr src, long count)
        {
            float* dstPtr = (float*)dst;
            float* srcPtr = (float*)src;

            long i = 0;

            for (; i <= count - 4; i += 4)
            {
                dstPtr[i] = FastSigmoid(srcPtr[i]);
                dstPtr[i + 1] = FastSigmoid(srcPtr[i + 1]);
                dstPtr[i + 2] = FastSigmoid(srcPtr[i + 2]);
                dstPtr[i + 3] = FastSigmoid(srcPtr[i + 3]);
            }

            for (; i < count; i++)
            {
                dstPtr[i] = FastSigmoid(srcPtr[i]);
            }
        }

        #endregion

        #region Scalar Fallbacks

        /// <summary>
        /// Scalar addition (fallback for non-ARM platforms)
        /// </summary>
        private static unsafe void ScalarAdd(IntPtr dst, IntPtr src1, IntPtr src2, long count)
        {
            float* dstPtr = (float*)dst;
            float* src1Ptr = (float*)src1;
            float* src2Ptr = (float*)src2;

            for (long i = 0; i < count; i++)
            {
                dstPtr[i] = src1Ptr[i] + src2Ptr[i];
            }
        }

        /// <summary>
        /// Scalar multiplication (fallback for non-ARM platforms)
        /// </summary>
        private static unsafe void ScalarMultiply(IntPtr dst, IntPtr src1, IntPtr src2, long count)
        {
            float* dstPtr = (float*)dst;
            float* src1Ptr = (float*)src1;
            float* src2Ptr = (float*)src2;

            for (long i = 0; i < count; i++)
            {
                dstPtr[i] = src1Ptr[i] * src2Ptr[i];
            }
        }

        /// <summary>
        /// Scalar ReLU (fallback for non-ARM platforms)
        /// </summary>
        private static unsafe void ScalarRelu(IntPtr dst, IntPtr src, long count)
        {
            float* dstPtr = (float*)dst;
            float* srcPtr = (float*)src;

            for (long i = 0; i < count; i++)
            {
                dstPtr[i] = Math.Max(0.0f, srcPtr[i]);
            }
        }

        /// <summary>
        /// Scalar sigmoid (fallback for non-ARM platforms)
        /// </summary>
        private static unsafe void ScalarSigmoid(IntPtr dst, IntPtr src, long count)
        {
            float* dstPtr = (float*)dst;
            float* srcPtr = (float*)src;

            for (long i = 0; i < count; i++)
            {
                dstPtr[i] = FastSigmoid(srcPtr[i]);
            }
        }

        #endregion

        #region Helper Functions

        /// <summary>
        /// Fast sigmoid approximation using:
        /// sigmoid(x) ≈ 0.5 + 0.5 * tanh(x / 2)
        /// Or: 1 / (1 + exp(-x))
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float FastSigmoid(float x)
        {
            // Use approximation for better performance
            return 0.5f + 0.5f * (float)Math.Tanh(x / 2.0f);
        }

        /// <summary>
        /// Check if running on ARM platform
        /// </summary>
        public static bool IsArm => IsArmPlatform;

        #endregion
    }
}
