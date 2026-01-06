using RitterFramework.Core.Tensor;
using System;

namespace MLFramework.Amp
{
    /// <summary>
    /// Convenience methods for creating AutoCast contexts
    /// </summary>
    public static class AutoCastContext
    {
        /// <summary>
        /// Creates an AutoCast context with FP16 mode
        /// </summary>
        public static AutoCast Fp16(AmpRegistry? registry = null)
        {
            return new AutoCast(AutoCastMode.Fp16, true, registry);
        }

        /// <summary>
        /// Creates an AutoCast context with BF16 mode (recommended)
        /// </summary>
        public static AutoCast Bf16(AmpRegistry? registry = null)
        {
            return new AutoCast(AutoCastMode.Bf16, true, registry);
        }

        /// <summary>
        /// Creates an AutoCast context with the specified mode
        /// </summary>
        /// <param name="mode">The AutoCast mode</param>
        /// <param name="registry">The operation precision registry</param>
        public static AutoCast Create(AutoCastMode mode, AmpRegistry? registry = null)
        {
            return new AutoCast(mode, true, registry);
        }

        /// <summary>
        /// Checks if AutoCast is currently active
        /// </summary>
        public static bool IsActive => AutoCast.Current != null;

        /// <summary>
        /// Gets the current AutoCast mode (returns None if not active)
        /// </summary>
        public static AutoCastMode CurrentMode => AutoCast.Current?.Mode ?? AutoCastMode.None;

        /// <summary>
        /// Casts a tensor to the appropriate precision (uses current context)
        /// </summary>
        /// <param name="tensor">The tensor to cast</param>
        /// <param name="operationType">The type of operation being performed</param>
        /// <returns>Casted tensor</returns>
        public static Tensor Cast(Tensor tensor, Type operationType)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (operationType == null)
                throw new ArgumentNullException(nameof(operationType));

            var autocast = AutoCast.Current;
            if (autocast == null)
            {
                return tensor;
            }

            return autocast.Cast(tensor, operationType);
        }

        /// <summary>
        /// Casts a tensor to a specific dtype (uses current context)
        /// </summary>
        /// <param name="tensor">The tensor to cast</param>
        /// <param name="dtype">The target data type</param>
        /// <returns>Casted tensor</returns>
        public static Tensor Cast(Tensor tensor, MLFramework.Core.DataType dtype)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            var autocast = AutoCast.Current;
            if (autocast == null)
            {
                return tensor;
            }

            return autocast.Cast(tensor, dtype);
        }
    }
}
