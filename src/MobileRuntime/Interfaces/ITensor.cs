using System;

namespace MobileRuntime
{
    /// <summary>
    /// Interface for tensor
    /// </summary>
    public interface ITensor : IDisposable
    {
        /// <summary>
        /// Gets the tensor shape
        /// </summary>
        int[] Shape { get; }

        /// <summary>
        /// Gets the data type
        /// </summary>
        DataType DataType { get; }

        /// <summary>
        /// Gets the total number of elements
        /// </summary>
        long Size { get; }

        /// <summary>
        /// Gets the total number of elements (legacy property)
        /// </summary>
        [Obsolete("Use Size instead")]
        int Length { get; }

        /// <summary>
        /// Gets the total number of bytes
        /// </summary>
        long ByteCount { get; }

        /// <summary>
        /// Gets the raw data pointer
        /// </summary>
        IntPtr DataPointer { get; }

        /// <summary>
        /// Gets or sets tensor data (legacy property)
        /// </summary>
        [Obsolete("Use DataPointer and ToArray() instead")]
        float[] Data { get; set; }
    }
}
