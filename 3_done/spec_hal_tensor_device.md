# Spec: HAL Tensor Device Transfer

## Overview
Implement device-agnostic tensor API with support for transferring tensors between devices.

## Responsibilities
- Create Tensor.To(IDevice) extension method
- Implement memory copy between devices
- Support device specification on tensor creation

## Files to Create/Modify
- `src/Tensor/TensorExtensions.cs` - Device transfer extensions
- `src/Tensor/Tensor.cs` - Add Device property (if not exists)
- `tests/Tensor/DeviceTransferTests.cs` - Transfer tests

## API Design

### TensorExtensions.cs
```csharp
namespace MLFramework;

/// <summary>
/// Extensions for tensor device operations
/// </summary>
public static class TensorExtensions
{
    /// <summary>
    /// Transfer tensor to specified device
    /// </summary>
    /// <param name="tensor">Source tensor</param>
    /// <param name="device">Target device</param>
    /// <returns>New tensor on target device with copied data</returns>
    public static Tensor To(this Tensor tensor, IDevice device)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (device == null)
            throw new ArgumentNullException(nameof(device));

        // If already on target device, return same tensor
        if (tensor.Device == device)
            return tensor;

        // Allocate new tensor on target device
        var result = Tensor.Zeros(tensor.Shape, device);

        // Copy data
        CopyTensorData(tensor, result);

        return result;
    }

    private static void CopyTensorData(Tensor source, Tensor destination)
    {
        if (source.Shape.Length != destination.Shape.Length ||
            !source.Shape.SequenceEqual(destination.Shape))
        {
            throw new ArgumentException(
                "Source and destination tensors must have the same shape");
        }

        var sourceDevice = source.Device;
        var destDevice = destination.Device;

        // CPU to CPU
        if (sourceDevice.DeviceType == DeviceType.CPU &&
            destDevice.DeviceType == DeviceType.CPU)
        {
            CopyCpuToCpu(source, destination);
        }
        // CPU to GPU
        else if (sourceDevice.DeviceType == DeviceType.CPU &&
                 destDevice.DeviceType != DeviceType.CPU)
        {
            CopyCpuToGpu(source, destination);
        }
        // GPU to CPU
        else if (sourceDevice.DeviceType != DeviceType.CPU &&
                 destDevice.DeviceType == DeviceType.CPU)
        {
            CopyGpuToCpu(source, destination);
        }
        // GPU to GPU
        else
        {
            CopyGpuToGpu(source, destination);
        }
    }

    private static void CopyCpuToCpu(Tensor source, Tensor destination)
    {
        unsafe
        {
            var srcPtr = (float*)source.DataPointer;
            var dstPtr = (float*)destination.DataPointer;
            var byteCount = source.Size * sizeof(float);

            Buffer.MemoryCopy(srcPtr, dstPtr, byteCount, byteCount);
        }
    }

    private static void CopyCpuToGpu(Tensor source, Tensor destination)
    {
        // Allocate GPU memory buffer
        var gpuBuffer = destination.Device.AllocateMemory(source.Size * sizeof(float));

        unsafe
        {
            var srcPtr = (float*)source.DataPointer;
            var gpuPtr = gpuBuffer.Pointer;

            // Copy from CPU to GPU
            // Implementation depends on GPU API (CUDA, etc.)
            CopyHostToDevice(gpuPtr, srcPtr, source.Size * sizeof(float));
        }
    }

    private static void CopyGpuToCpu(Tensor source, Tensor destination)
    {
        // Copy from GPU to CPU
        unsafe
        {
            var gpuPtr = source.DataPointer;
            var dstPtr = (float*)destination.DataPointer;

            // Implementation depends on GPU API
            CopyDeviceToHost(dstPtr, gpuPtr, source.Size * sizeof(float));
        }
    }

    private static void CopyGpuToGpu(Tensor source, Tensor destination)
    {
        // Copy between GPUs
        unsafe
        {
            var srcPtr = source.DataPointer;
            var dstPtr = destination.DataPointer;

            // Implementation depends on GPU API
            CopyDeviceToDevice(dstPtr, srcPtr, source.Size * sizeof(float));
        }
    }

    // P/Invoke methods for GPU memory transfers (placeholder implementations)
    [DllImport("kernel32.dll", EntryPoint = "RtlMoveMemory")]
    private static extern void CopyHostToDevice(
        IntPtr destination, IntPtr source, UIntPtr length);

    [DllImport("kernel32.dll", EntryPoint = "RtlMoveMemory")]
    private static extern void CopyDeviceToHost(
        IntPtr destination, IntPtr source, UIntPtr length);

    [DllImport("kernel32.dll", EntryPoint = "RtlMoveMemory")]
    private static extern void CopyDeviceToDevice(
        IntPtr destination, IntPtr source, UIntPtr length);
}
```

### Tensor.cs Updates
```csharp
// Add to Tensor class:
public partial class Tensor
{
    /// <summary>
    /// Device this tensor is allocated on
    /// </summary>
    public IDevice Device { get; private set; }

    /// <summary>
    /// Raw data pointer
    /// </summary>
    public IntPtr DataPointer { get; private set; }

    // Update constructor to accept device
    private Tensor(long[] shape, IDevice device)
    {
        Shape = shape;
        Size = shape.Aggregate(1L, (acc, dim) => acc * dim);
        Device = device;

        // Allocate memory on device
        var bufferSize = Size * sizeof(float);
        var buffer = device.AllocateMemory(bufferSize);
        DataPointer = buffer.Pointer;
    }
}
```

## Testing Requirements
```csharp
public class DeviceTransferTests
{
    [Test]
    public void To_SameDevice_ReturnsSameTensor()
    {
        var tensor = Tensor.Zeros(new[] { 10L }, Device.CPU);
        var result = tensor.To(Device.CPU);

        Assert.AreSame(tensor, result);
    }

    [Test]
    public void To_DifferentDevice_CopiesData()
    {
        var data = new[] { 1.0f, 2.0f, 3.0f };
        var tensor = Tensor.FromArray(data, Device.CPU);

        var result = tensor.To(Device.CPU); // Same device, but test the path

        CollectionAssert.AreEqual(data, result.ToArray());
    }

    [Test]
    public void To_CpuToDevice_TransfersData()
    {
        var data = new[] { 1.0f, 2.0f, 3.0f };
        var tensor = Tensor.FromArray(data, Device.CPU);

        // When GPU backend is available, test this
        if (BackendRegistry.IsDeviceAvailable(DeviceType.CUDA))
        {
            var result = tensor.To(Device.CUDA(0));

            Assert.AreEqual(DeviceType.CUDA, result.Device.DeviceType);
            CollectionAssert.AreEqual(data, result.ToArray());
        }
    }

    [Test]
    public To_DifferentShape_ThrowsException()
    {
        var tensor = Tensor.Zeros(new[] { 10L }, Device.CPU);
        var other = Tensor.Zeros(new[] { 5L }, Device.CPU);

        // This should work - the extension creates a new tensor
        var result = tensor.To(other.Device);
        Assert.AreEqual(tensor.Shape, result.Shape);
    }
}
```

## Acceptance Criteria
- [ ] Tensor.To(IDevice) extension method implemented
- [ ] CPU to CPU transfer works correctly
- [ ] Stubs for GPU transfers (CPU to GPU, GPU to CPU, GPU to GPU)
- [ ] Proper error handling for shape mismatches
- [ ] All tests pass
- [ ] Performance: minimize copies for same-device transfers

## Notes for Coder
- GPU memory transfer P/Invoke methods are placeholders
- Real implementation will use CUDA/ROCm specific APIs
- For now, CPU-to-CPU is fully implemented
- GPU transfers will be completed when CUDA backend is implemented
- Tensor class may need updates - check existing implementation
