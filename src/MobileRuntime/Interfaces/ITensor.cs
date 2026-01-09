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
        int Length { get; }

        /// <summary>
        /// Gets or sets tensor data
        /// </summary>
        float[] Data { get; set; }
    }
}
