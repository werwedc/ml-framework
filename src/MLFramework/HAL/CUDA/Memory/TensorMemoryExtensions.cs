using RitterFramework.Core.Tensor;
using RitterFramework.Core;
using System;

namespace MLFramework.HAL.CUDA;

/// <summary>
/// Extension methods for tensor memory operations with CUDA graph support.
/// </summary>
public static class TensorMemoryExtensions
{
    /// <summary>
    /// Gets the element size in bytes based on the data type.
    /// </summary>
    /// <param name="dtype">The data type</param>
    /// <returns>The size in bytes</returns>
    private static int GetElementSize(DataType dtype)
    {
        return dtype switch
        {
            DataType.Float32 => 4,
            DataType.Float64 => 8,
            DataType.Int32 => 4,
            DataType.Int64 => 8,
            DataType.Int16 => 2,
            DataType.Int8 => 1,
            DataType.UInt8 => 1,
            DataType.Bool => 1,
            DataType.Float16 => 2,
            DataType.BFloat16 => 2,
            _ => throw new ArgumentException($"Unsupported data type: {dtype}")
        };
    }

    /// <summary>
    /// Creates a tensor that uses graph-compatible memory.
    /// Note: This is a simplified implementation. A full implementation would need
    /// to support CUDA tensors with device memory pointers.
    /// </summary>
    /// <param name="tensor">The tensor to allocate memory for</param>
    /// <param name="pool">The graph memory pool to use</param>
    /// <returns>A new tensor with graph-compatible memory</returns>
    public static Tensor WithGraphMemory(
        this Tensor tensor,
        CUDAGraphMemoryPool pool)
    {
        // Calculate the size needed
        var elementSize = GetElementSize(tensor.Dtype);
        var size = tensor.Size * elementSize;

        // Allocate memory from the graph pool
        var block = pool.Allocate((ulong)size);

        // Note: In a full implementation, we would create a CUDA tensor
        // that uses the device memory pointer. For now, we return the original tensor
        // as the current Tensor class only supports CPU memory.

        // Return a clone of the tensor for demonstration
        // In a real implementation, this would be a CUDA tensor using block.Ptr
        return tensor.Clone();
    }

    /// <summary>
    /// Ensures tensor memory is allocated for graph execution.
    /// </summary>
    /// <param name="tensor">The tensor to check</param>
    /// <param name="allocator">The memory allocator to use</param>
    /// <returns>A tensor with graph-compatible memory if needed</returns>
    public static Tensor EnsureGraphCompatible(
        this Tensor tensor,
        ICUDAMemoryAllocator allocator)
    {
        // Check if graph mode is enabled and if the tensor needs reallocation
        // For now, since our Tensor class doesn't track CUDA memory allocation,
        // we'll use the graph pool if in graph mode

        if (allocator.IsGraphMode && allocator.GraphPool != null)
        {
            // Reallocate with graph-compatible memory
            return tensor.WithGraphMemory(allocator.GraphPool);
        }

        // Return the original tensor if not in graph mode
        return tensor;
    }
}
