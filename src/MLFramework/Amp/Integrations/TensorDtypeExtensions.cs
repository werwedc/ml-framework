using System;
using MLFramework.Core;
using RitterFramework.Core.Tensor;

namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Extension methods for Tensor to support AMP dtype operations
    /// </summary>
    public static class TensorDtypeExtensions
    {
        // Use a thread-safe dictionary to store dtype information for tensors
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<int, DataType> _tensorDtypes =
            new System.Collections.Concurrent.ConcurrentDictionary<int, DataType>();

        /// <summary>
        /// Gets or sets the data type for a tensor
        /// </summary>
        /// <param name="tensor">The tensor</param>
        /// <returns>The data type of the tensor</returns>
        public static DataType GetDtype(this Tensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            int hash = tensor.GetHashCode();
            if (_tensorDtypes.TryGetValue(hash, out var dtype))
            {
                return dtype;
            }

            // Default to Float32 if not set
            return DataType.Float32;
        }

        /// <summary>
        /// Sets the data type for a tensor
        /// </summary>
        /// <param name="tensor">The tensor</param>
        /// <param name="dtype">The data type to set</param>
        public static void SetDtype(this Tensor tensor, DataType dtype)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            int hash = tensor.GetHashCode();
            _tensorDtypes[hash] = dtype;
        }

        /// <summary>
        /// Casts a tensor to a different data type
        /// </summary>
        /// <param name="tensor">The tensor to cast</param>
        /// <param name="dtype">The target data type</param>
        /// <returns>A new tensor with the specified data type</returns>
        public static Tensor Cast(this Tensor tensor, DataType dtype)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            // For now, since we only have float tensors, we just return a clone
            // In a real implementation with multiple dtypes, we would convert the data
            var result = tensor.Clone();
            result.SetDtype(dtype);
            return result;
        }

        /// <summary>
        /// Checks if the tensor is of a specific dtype
        /// </summary>
        /// <param name="tensor">The tensor to check</param>
        /// <param name="dtype">The dtype to compare against</param>
        /// <returns>True if the tensor is of the specified dtype</returns>
        public static bool IsDtype(this Tensor tensor, DataType dtype)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            return tensor.GetDtype() == dtype;
        }

        /// <summary>
        /// Checks if the tensor is a low precision type (Float16 or BFloat16)
        /// </summary>
        /// <param name="tensor">The tensor to check</param>
        /// <returns>True if the tensor is low precision</returns>
        public static bool IsLowPrecision(this Tensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            return tensor.GetDtype().IsLowPrecision();
        }
    }
}
