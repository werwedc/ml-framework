using System;

namespace MLFramework.MobileRuntime
{
    /// <summary>
    /// Interface for a tensor in the mobile runtime.
    /// </summary>
    public interface ITensor : IDisposable
    {
        /// <summary>
        /// Shape of the tensor.
        /// </summary>
        int[] Shape { get; }

        /// <summary>
        /// Data type of the tensor.
        /// </summary>
        DataType DataType { get; }

        /// <summary>
        /// Total number of elements in the tensor.
        /// </summary>
        long Size { get; }

        /// <summary>
        /// Total number of bytes occupied by the tensor.
        /// </summary>
        long ByteCount { get; }

        /// <summary>
        /// Gets the value at the specified indices.
        /// </summary>
        /// <typeparam name="T">Type to return.</typeparam>
        /// <param name="indices">Indices into the tensor.</param>
        /// <returns>Value at the specified position.</returns>
        T GetData<T>(params int[] indices);

        /// <summary>
        /// Converts the tensor to a flat array.
        /// </summary>
        /// <typeparam name="T">Type of array elements.</typeparam>
        /// <returns>Flat array of tensor data.</returns>
        T[] ToArray<T>();
    }
}
