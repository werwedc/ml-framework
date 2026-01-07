using MLFramework.Core;
using RitterFramework.Core.Tensor;
using System.Runtime.InteropServices;

namespace MLFramework.HAL;

/// <summary>
/// HAL-specific extensions for Tensor to work with Hardware Abstraction Layer
/// </summary>
public static class TensorHALExtensions
{
    /// <summary>
    /// Creates a zero-filled tensor on the specified device
    /// </summary>
    /// <param name="shape">Shape of the tensor</param>
    /// <param name="device">Device to create the tensor on</param>
    /// <returns>A new zero-filled tensor</returns>
    public static Tensor Zeros(int[] shape, IDevice device)
    {
        // For now, we only support CPU devices
        if (device.DeviceType != DeviceType.CPU)
        {
            throw new NotSupportedException($"Device type {device.DeviceType} is not yet supported");
        }

        return Tensor.Zeros(shape);
    }

    /// <summary>
    /// Creates a tensor from an array on the specified device
    /// </summary>
    /// <param name="data">Data array</param>
    /// <param name="device">Device to create the tensor on</param>
    /// <returns>A new tensor with the given data</returns>
    public static Tensor FromArray(float[] data, IDevice device)
    {
        // For now, we only support CPU devices
        if (device.DeviceType != DeviceType.CPU)
        {
            throw new NotSupportedException($"Device type {device.DeviceType} is not yet supported");
        }

        return Tensor.FromArray(data);
    }

    /// <summary>
    /// Gets a pinned pointer to the tensor data for unsafe operations
    /// This pins the data in memory to prevent garbage collection during the operation
    /// </summary>
    /// <param name="tensor">The tensor to get data pointer for</param>
    /// <param name="action">Action to execute with the pinned pointer</param>
    public static void WithDataPointer(this Tensor tensor, Action<IntPtr> action)
    {
        var handle = GCHandle.Alloc(tensor.Data, GCHandleType.Pinned);
        try
        {
            action(handle.AddrOfPinnedObject());
        }
        finally
        {
            handle.Free();
        }
    }

    /// <summary>
    /// Converts shape from int[] to long[] for compatibility
    /// </summary>
    /// <param name="shape">Shape as int array</param>
    /// <returns>Shape as long array</returns>
    public static long[] ToLongArray(this int[] shape)
    {
        var result = new long[shape.Length];
        for (int i = 0; i < shape.Length; i++)
        {
            result[i] = shape[i];
        }
        return result;
    }

    /// <summary>
    /// Converts shape from long[] to int[] for compatibility
    /// </summary>
    /// <param name="shape">Shape as long array</param>
    /// <returns>Shape as int array</returns>
    public static int[] ToIntArray(this long[] shape)
    {
        var result = new int[shape.Length];
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] > int.MaxValue || shape[i] < int.MinValue)
            {
                throw new OverflowException($"Shape dimension {shape[i]} at index {i} is too large for int");
            }
            result[i] = (int)shape[i];
        }
        return result;
    }
}
