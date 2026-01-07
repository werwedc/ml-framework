# Spec: HAL CUDA Backend Operations

## Overview
Implement the IBackend interface for CUDA using cuBLAS and custom kernels.

## Responsibilities
- Create CudaBackend class implementing IBackend
- Implement basic tensor operations using CUDA
- Register backend with BackendRegistry

## Files to Create/Modify
- `src/HAL/CUDA/CudaBackend.cs` - CUDA backend implementation
- `src/HAL/CUDA/CudaKernels.cs` - P/Invoke declarations for CUDA kernels
- `tests/HAL/CUDA/CudaBackendTests.cs` - Backend tests

## API Design

### CudaKernels.cs
```csharp
using System.Runtime.InteropServices;

namespace MLFramework.HAL.CUDA;

/// <summary>
/// P/Invoke declarations for CUDA kernels and cuBLAS
/// </summary>
public static class CudaKernels
{
    private const string CudaLibrary = "nvcuda.dll";

    #region Element-wise Kernels

    /// <summary>
    /// Add two tensors element-wise
    /// </summary>
    [DllImport("mlframework_cuda.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaAdd(
        IntPtr result,
        IntPtr a,
        IntPtr b,
        long size);

    /// <summary>
    /// Subtract two tensors element-wise
    /// </summary>
    [DllImport("mlframework_cuda.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaSubtract(
        IntPtr result,
        IntPtr a,
        IntPtr b,
        long size);

    /// <summary>
    /// Multiply two tensors element-wise
    /// </summary>
    [DllImport("mlframework_cuda.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaMultiply(
        IntPtr result,
        IntPtr a,
        IntPtr b,
        long size);

    /// <summary>
    /// Divide two tensors element-wise
    /// </summary>
    [DllImport("mlframework_cuda.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaDivide(
        IntPtr result,
        IntPtr a,
        IntPtr b,
        long size);

    #endregion

    #region Activation Kernels

    /// <summary>
    /// ReLU activation
    /// </summary>
    [DllImport("mlframework_cuda.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaReLU(
        IntPtr result,
        IntPtr input,
        long size);

    /// <summary>
    /// Sigmoid activation
    /// </summary>
    [DllImport("mlframework_cuda.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaSigmoid(
        IntPtr result,
        IntPtr input,
        long size);

    /// <summary>
    /// Tanh activation
    /// </summary>
    [DllImport("mlframework_cuda.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaTanh(
        IntPtr result,
        IntPtr input,
        long size);

    #endregion

    #region Reduction Kernels

    /// <summary>
    /// Sum reduction
    /// </summary>
    [DllImport("mlframework_cuda.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaSum(
        IntPtr result,
        IntPtr input,
        long size);

    #endregion
}

// Note: The actual CUDA kernels will be implemented in CUDA C++ (.cu files)
// and compiled into mlframework_cuda.dll
```

### CudaBackend.cs
```csharp
namespace MLFramework.HAL.CUDA;

/// <summary>
/// CUDA backend using cuBLAS and custom kernels
/// </summary>
public class CudaBackend : IBackend
{
    public string Name => "CUDA";
    public DeviceType Type => DeviceType.CUDA;

    private bool _isAvailable;

    public bool IsAvailable
    {
        get
        {
            if (_isAvailable)
                return true;

            // Check if CUDA is available
            var result = CudaApi.CudaGetDeviceCount(out int count);
            _isAvailable = (result == CudaError.Success) && (count > 0);

            return _isAvailable;
        }
    }

    public bool SupportsOperation(Operation operation)
    {
        return operation switch
        {
            Operation.Add or
            Operation.Subtract or
            Operation.Multiply or
            Operation.Divide or
            Operation.Sum or
            Operation.Mean or
            Operation.ReLU or
            Operation.Sigmoid or
            Operation.Tanh or
            Operation.Copy or
            Operation.Cast or
            Operation.Reshape => true,
            _ => false // More complex operations not yet implemented
        };
    }

    public Tensor ExecuteOperation(Operation operation, Tensor[] inputs)
    {
        if (inputs == null || inputs.Length == 0)
            throw new ArgumentException("At least one input tensor required");

        if (!SupportsOperation(operation))
            throw new NotSupportedException($"Operation {operation} not supported");

        return operation switch
        {
            Operation.Add => ExecuteAdd(inputs),
            Operation.Subtract => ExecuteSubtract(inputs),
            Operation.Multiply => ExecuteMultiply(inputs),
            Operation.Divide => ExecuteDivide(inputs),
            Operation.Sum => ExecuteSum(inputs),
            Operation.Mean => ExecuteMean(inputs),
            Operation.ReLU => ExecuteReLU(inputs),
            Operation.Sigmoid => ExecuteSigmoid(inputs),
            Operation.Tanh => ExecuteTanh(inputs),
            Operation.Copy => ExecuteCopy(inputs),
            _ => throw new NotSupportedException($"Operation {operation}")
        };
    }

    public void Initialize()
    {
        // Initialize CUDA resources
        // Load libraries, create contexts, etc.
    }

    private Tensor ExecuteAdd(Tensor[] inputs)
    {
        if (inputs.Length != 2)
            throw new ArgumentException("Add requires exactly 2 inputs");

        var a = inputs[0];
        var b = inputs[1];

        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Tensors must have same shape");

        var result = Tensor.Zeros(a.Shape, Device.CUDA);

        CudaKernels.CudaAdd(
            result.DataPointer,
            a.DataPointer,
            b.DataPointer,
            a.Size);

        return result;
    }

    private Tensor ExecuteSubtract(Tensor[] inputs)
    {
        if (inputs.Length != 2)
            throw new ArgumentException("Subtract requires exactly 2 inputs");

        var a = inputs[0];
        var b = inputs[1];

        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Tensors must have same shape");

        var result = Tensor.Zeros(a.Shape, Device.CUDA);

        CudaKernels.CudaSubtract(
            result.DataPointer,
            a.DataPointer,
            b.DataPointer,
            a.Size);

        return result;
    }

    private Tensor ExecuteMultiply(Tensor[] inputs)
    {
        if (inputs.Length != 2)
            throw new ArgumentException("Multiply requires exactly 2 inputs");

        var a = inputs[0];
        var b = inputs[1];

        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Tensors must have same shape");

        var result = Tensor.Zeros(a.Shape, Device.CUDA);

        CudaKernels.CudaMultiply(
            result.DataPointer,
            a.DataPointer,
            b.DataPointer,
            a.Size);

        return result;
    }

    private Tensor ExecuteDivide(Tensor[] inputs)
    {
        if (inputs.Length != 2)
            throw new ArgumentException("Divide requires exactly 2 inputs");

        var a = inputs[0];
        var b = inputs[1];

        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Tensors must have same shape");

        var result = Tensor.Zeros(a.Shape, Device.CUDA);

        CudaKernels.CudaDivide(
            result.DataPointer,
            a.DataPointer,
            b.DataPointer,
            a.Size);

        return result;
    }

    private Tensor ExecuteSum(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = Tensor.Zeros(new long[] { 1 }, Device.CUDA);

        CudaKernels.CudaSum(
            result.DataPointer,
            a.DataPointer,
            a.Size);

        return result;
    }

    private Tensor ExecuteMean(Tensor[] inputs)
    {
        var sum = ExecuteSum(inputs);
        var count = inputs[0].Size;

        // Divide by count
        var result = Tensor.Zeros(new long[] { 1 }, Device.CUDA);
        var scalar = Tensor.FromArray(new[] { 1.0f / count }, Device.CUDA);

        CudaKernels.CudaMultiply(
            result.DataPointer,
            sum.DataPointer,
            scalar.DataPointer,
            1);

        return result;
    }

    private Tensor ExecuteReLU(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = Tensor.Zeros(a.Shape, Device.CUDA);

        CudaKernels.CudaReLU(
            result.DataPointer,
            a.DataPointer,
            a.Size);

        return result;
    }

    private Tensor ExecuteSigmoid(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = Tensor.Zeros(a.Shape, Device.CUDA);

        CudaKernels.CudaSigmoid(
            result.DataPointer,
            a.DataPointer,
            a.Size);

        return result;
    }

    private Tensor ExecuteTanh(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = Tensor.Zeros(a.Shape, Device.CUDA);

        CudaKernels.CudaTanh(
            result.DataPointer,
            a.DataPointer,
            a.Size);

        return result;
    }

    private Tensor ExecuteCopy(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = Tensor.Zeros(a.Shape, Device.CUDA);

        unsafe
        {
            var ptrA = (float*)a.DataPointer;
            var ptrResult = (float*)result.DataPointer;
            var count = a.Size;

            for (int i = 0; i < count; i++)
            {
                ptrResult[i] = ptrA[i];
            }
        }

        return result;
    }
}
```

## Testing Requirements
```csharp
public class CudaBackendTests
{
    private CudaBackend? _backend;

    [SetUp]
    public void Setup()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        _backend = new CudaBackend();
        _backend.Initialize();
    }

    [Test]
    public void ExecuteOperation_Add_WorksCorrectly()
    {
        var a = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CUDA(0));
        var b = Tensor.FromArray(new[] { 4.0f, 5.0f, 6.0f }, Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.Add, new[] { a, b });

        var resultCpu = result.To(Device.CPU);
        var expected = new[] { 5.0f, 7.0f, 9.0f };
        Assert.AreEqual(expected, resultCpu.ToArray());
    }

    [Test]
    public void ExecuteOperation_Sum_WorksCorrectly()
    {
        var a = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.Sum, new[] { a });
        var resultCpu = result.To(Device.CPU);

        Assert.AreEqual(6.0f, resultCpu.ToArray()[0]);
    }

    private bool CudaAvailable()
    {
        var result = CudaApi.CudaGetDeviceCount(out int count);
        return result == CudaError.Success && count > 0;
    }
}
```

## Acceptance Criteria
- [ ] CudaBackend implements IBackend
- [ ] Supports all basic arithmetic operations
- [ ] Supports activation functions
- [ ] P/Invoke declarations for CUDA kernels
- [ ] Backend can be registered with BackendRegistry
- [ ] All tests pass (when CUDA hardware available)

## Notes for Coder
- CUDA kernels are referenced but not implemented in C#
- Real CUDA kernels will be written in CUDA C++ (.cu files)
- mlframework_cuda.dll will contain compiled CUDA kernels
- Tests require CUDA hardware to run
- This spec defines the C# interface only - kernel implementation is separate
- Consider adding cuBLAS integration for matrix operations later
