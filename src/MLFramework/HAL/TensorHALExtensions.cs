using MLFramework.Core;
using MLFramework.HAL.CUDA;
using RitterFramework.Core.Tensor;
using System.Runtime.InteropServices;

namespace MLFramework.HAL;

/// <summary>
/// HAL-specific extensions for Tensor to work with Hardware Abstraction Layer
/// </summary>
public static class TensorHALExtensions
{
    // Dictionary to track device allocations for tensors
    private static readonly Dictionary<int, IntPtr> _cudaDeviceAllocations = new();
    private static readonly object _allocationsLock = new();
    private static int _nextTensorId = 0;
    private static readonly Dictionary<Tensor, int> _tensorToIdMap = new();

    /// <summary>
    /// Allocate CUDA memory for a tensor
    /// </summary>
    private static void AllocateOnCudaDevice(Tensor tensor, CudaDevice device)
    {
        var size = (ulong)(tensor.Size * sizeof(float));

        lock (_allocationsLock)
        {
            var error = CudaApi.CudaMalloc(out IntPtr devicePtr, size);
            if (error != CudaError.Success)
            {
                throw new InvalidOperationException($"CUDA malloc failed with error: {error}");
            }

            var tensorId = _nextTensorId++;
            _cudaDeviceAllocations[tensorId] = devicePtr;
            _tensorToIdMap[tensor] = tensorId;
        }
    }

    /// <summary>
    /// Get the CUDA device pointer for a tensor
    /// </summary>
    public static IntPtr GetCudaDevicePointer(Tensor tensor)
    {
        lock (_allocationsLock)
        {
            if (_tensorToIdMap.TryGetValue(tensor, out int tensorId))
            {
                return _cudaDeviceAllocations[tensorId];
            }
            throw new InvalidOperationException("Tensor does not have CUDA memory allocated");
        }
    }

    /// <summary>
    /// Free CUDA memory for a tensor
    /// </summary>
    public static void FreeCudaMemory(Tensor tensor)
    {
        lock (_allocationsLock)
        {
            if (_tensorToIdMap.TryGetValue(tensor, out int tensorId))
            {
                var devicePtr = _cudaDeviceAllocations[tensorId];
                CudaApi.CudaFree(devicePtr);
                _cudaDeviceAllocations.Remove(tensorId);
                _tensorToIdMap.Remove(tensor);
            }
        }
    }
    /// <summary>
    /// Creates a zero-filled tensor on the specified device
    /// </summary>
    /// <param name="shape">Shape of the tensor</param>
    /// <param name="device">Device to create the tensor on</param>
    /// <returns>A new zero-filled tensor</returns>
    public static Tensor Zeros(int[] shape, IDevice device)
    {
        if (device.DeviceType == DeviceType.CPU)
        {
            return Tensor.Zeros(shape);
        }
        else if (device.DeviceType == DeviceType.CUDA)
        {
            // Create a CPU tensor first, then allocate memory on CUDA device
            var tensor = Tensor.Zeros(shape);
            AllocateOnCudaDevice(tensor, (CudaDevice)device);
            return tensor;
        }

        throw new NotSupportedException($"Device type {device.DeviceType} is not yet supported");
    }

    /// <summary>
    /// Creates a tensor from an array on the specified device
    /// </summary>
    /// <param name="data">Data array</param>
    /// <param name="device">Device to create the tensor on</param>
    /// <returns>A new tensor with the given data</returns>
    public static Tensor FromArray(float[] data, IDevice device)
    {
        if (device.DeviceType == DeviceType.CPU)
        {
            return Tensor.FromArray(data);
        }
        else if (device.DeviceType == DeviceType.CUDA)
        {
            // Create a CPU tensor first, then allocate and copy to CUDA device
            var tensor = Tensor.FromArray(data);
            AllocateOnCudaDevice(tensor, (CudaDevice)device);
            return tensor;
        }

        throw new NotSupportedException($"Device type {device.DeviceType} is not yet supported");
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

    /// <summary>
    /// Transfer tensor to specified device
    /// </summary>
    /// <param name="tensor">Source tensor</param>
    /// <param name="device">Target device</param>
    /// <returns>New tensor on target device with copied data, or same tensor if already on target device</returns>
    public static Tensor To(this Tensor tensor, IDevice device)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (device == null)
            throw new ArgumentNullException(nameof(device));

        // CPU to CPU: no-op, return same tensor
        if (device.DeviceType == DeviceType.CPU && !_tensorToIdMap.ContainsKey(tensor))
        {
            return tensor;
        }

        // CPU to CUDA
        if (device.DeviceType == DeviceType.CUDA && !_tensorToIdMap.ContainsKey(tensor))
        {
            var cudaDevice = (CudaDevice)device;
            var result = Tensor.FromArray(tensor.Data);
            AllocateOnCudaDevice(result, cudaDevice);

            // Copy data from host to device
            result.WithDataPointer(hostPtr =>
            {
                var devicePtr = GetCudaDevicePointer(result);
                var error = CudaApi.CudaMemcpy(
                    devicePtr,
                    hostPtr,
                    (ulong)(result.Size * sizeof(float)),
                    CudaMemcpyKind.HostToDevice);

                if (error != CudaError.Success)
                {
                    throw new InvalidOperationException($"CUDA memcpy failed with error: {error}");
                }
            });

            return result;
        }

        // CUDA to CPU
        if (device.DeviceType == DeviceType.CPU && _tensorToIdMap.ContainsKey(tensor))
        {
            var result = Tensor.Zeros(tensor.Shape);

            result.WithDataPointer(hostPtr =>
            {
                var devicePtr = GetCudaDevicePointer(tensor);
                var error = CudaApi.CudaMemcpy(
                    hostPtr,
                    devicePtr,
                    (ulong)(tensor.Size * sizeof(float)),
                    CudaMemcpyKind.DeviceToHost);

                if (error != CudaError.Success)
                {
                    throw new InvalidOperationException($"CUDA memcpy failed with error: {error}");
                }
            });

            return result;
        }

        // CUDA to CUDA with same device: no-op
        if (device.DeviceType == DeviceType.CUDA && _tensorToIdMap.ContainsKey(tensor))
        {
            return tensor;
        }

        throw new NotSupportedException($"Device transfer from current device to {device.DeviceType} is not yet supported");
    }
}
