# Spec: HAL CPU Backend Operations

## Overview
Implement the IBackend interface for CPU using basic .NET math libraries.

## Responsibilities
- Create CpuBackend class implementing IBackend
- Implement basic tensor operations using managed code
- Register backend with BackendRegistry

## Files to Create/Modify
- `src/HAL/CpuBackend.cs` - CPU backend implementation
- `tests/HAL/CpuBackendTests.cs` - Backend tests

## API Design

### CpuBackend.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// CPU backend using managed .NET libraries
/// </summary>
public class CpuBackend : IBackend
{
    public string Name => "ManagedCPU";
    public DeviceType Type => DeviceType.CPU;
    public bool IsAvailable => true;

    public bool SupportsOperation(Operation operation)
    {
        // CPU supports all operations in this implementation
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
        // No initialization needed for CPU backend
    }

    private Tensor ExecuteAdd(Tensor[] inputs)
    {
        if (inputs.Length != 2)
            throw new ArgumentException("Add requires exactly 2 inputs");

        var a = inputs[0];
        var b = inputs[1];

        // Broadcasting or element-wise add would be here
        // For now, assume same shape
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Tensors must have same shape");

        var result = Tensor.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            var ptrA = (float*)a.DataPointer;
            var ptrB = (float*)b.DataPointer;
            var ptrResult = (float*)result.DataPointer;
            var count = a.Size;

            for (int i = 0; i < count; i++)
            {
                ptrResult[i] = ptrA[i] + ptrB[i];
            }
        }

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

        var result = Tensor.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            var ptrA = (float*)a.DataPointer;
            var ptrB = (float*)b.DataPointer;
            var ptrResult = (float*)result.DataPointer;
            var count = a.Size;

            for (int i = 0; i < count; i++)
            {
                ptrResult[i] = ptrA[i] - ptrB[i];
            }
        }

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

        var result = Tensor.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            var ptrA = (float*)a.DataPointer;
            var ptrB = (float*)b.DataPointer;
            var ptrResult = (float*)result.DataPointer;
            var count = a.Size;

            for (int i = 0; i < count; i++)
            {
                ptrResult[i] = ptrA[i] * ptrB[i];
            }
        }

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

        var result = Tensor.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            var ptrA = (float*)a.DataPointer;
            var ptrB = (float*)b.DataPointer;
            var ptrResult = (float*)result.DataPointer;
            var count = a.Size;

            for (int i = 0; i < count; i++)
            {
                ptrResult[i] = ptrA[i] / ptrB[i];
            }
        }

        return result;
    }

    private Tensor ExecuteSum(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = Tensor.Zeros(new long[] { 1 }, Device.CPU);

        unsafe
        {
            var ptrA = (float*)a.DataPointer;
            var ptrResult = (float*)result.DataPointer;
            var count = a.Size;

            float sum = 0;
            for (int i = 0; i < count; i++)
            {
                sum += ptrA[i];
            }
            ptrResult[0] = sum;
        }

        return result;
    }

    private Tensor ExecuteMean(Tensor[] inputs)
    {
        var sum = ExecuteSum(inputs);
        var count = inputs[0].Size;

        unsafe
        {
            var ptr = (float*)sum.DataPointer;
            ptr[0] /= count;
        }

        return sum;
    }

    private Tensor ExecuteReLU(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = Tensor.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            var ptrA = (float*)a.DataPointer;
            var ptrResult = (float*)result.DataPointer;
            var count = a.Size;

            for (int i = 0; i < count; i++)
            {
                ptrResult[i] = Math.Max(0, ptrA[i]);
            }
        }

        return result;
    }

    private Tensor ExecuteSigmoid(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = Tensor.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            var ptrA = (float*)a.DataPointer;
            var ptrResult = (float*)result.DataPointer;
            var count = a.Size;

            for (int i = 0; i < count; i++)
            {
                ptrResult[i] = 1.0f / (1.0f + (float)Math.Exp(-ptrA[i]));
            }
        }

        return result;
    }

    private Tensor ExecuteTanh(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = Tensor.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            var ptrA = (float*)a.DataPointer;
            var ptrResult = (float*)result.DataPointer;
            var count = a.Size;

            for (int i = 0; i < count; i++)
            {
                ptrResult[i] = (float)Math.Tanh(ptrA[i]);
            }
        }

        return result;
    }

    private Tensor ExecuteCopy(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = Tensor.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            var ptrA = (float*)a.DataPointer;
            var ptrResult = (float*)result.DataPointer;
            var count = a.Size;

            Buffer.MemoryCopy(ptrA, ptrResult, count * sizeof(float), count * sizeof(float));
        }

        return result;
    }
}
```

### BackendRegistry Registration
```csharp
// In Program.cs or initialization code:
BackendRegistry.Register(new CpuBackend());
```

## Testing Requirements
```csharp
public class CpuBackendTests
{
    private CpuBackend _backend;

    [SetUp]
    public void Setup()
    {
        _backend = new CpuBackend();
    }

    [Test]
    public void SupportsOperation_SupportedOperations_ReturnsTrue()
    {
        Assert.IsTrue(_backend.SupportsOperation(Operation.Add));
        Assert.IsTrue(_backend.SupportsOperation(Operation.ReLU));
    }

    [Test]
    public void ExecuteOperation_Add_WorksCorrectly()
    {
        var a = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);
        var b = Tensor.FromArray(new[] { 4.0f, 5.0f, 6.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.Add, new[] { a, b });

        var expected = new[] { 5.0f, 7.0f, 9.0f };
        Assert.AreEqual(expected, result.ToArray());
    }

    [Test]
    public void ExecuteOperation_Sum_WorksCorrectly()
    {
        var a = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.Sum, new[] { a });

        Assert.AreEqual(6.0f, result.ToArray()[0]);
    }

    [Test]
    public void ExecuteOperation_ReLU_WorksCorrectly()
    {
        var a = Tensor.FromArray(new[] { -1.0f, 0.0f, 1.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.ReLU, new[] { a });

        var expected = new[] { 0.0f, 0.0f, 1.0f };
        Assert.AreEqual(expected, result.ToArray());
    }
}
```

## Acceptance Criteria
- [ ] CpuBackend implements IBackend
- [ ] Supports arithmetic operations (Add, Subtract, Multiply, Divide)
- [ ] Supports reduction operations (Sum, Mean)
- [ ] Supports activation functions (ReLU, Sigmoid, Tanh)
- [ ] Operations use unsafe code for performance
- [ ] Backend can be registered with BackendRegistry
- [ ] All tests pass

## Notes for Coder
- This is a basic implementation using unsafe pointers
- Tensor class is referenced - assumed to exist or be implemented separately
- Shape checking is simplified - full implementation would support broadcasting
- Performance optimization using SIMD could be added later
- Focus on correctness over optimization for now
