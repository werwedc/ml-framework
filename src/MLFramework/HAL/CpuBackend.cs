using RitterFramework.Core.Tensor;

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
            Operation.Cast => ExecuteCast(inputs),
            Operation.Reshape => ExecuteReshape(inputs),
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

        var result = TensorHALExtensions.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            fixed (float* ptrA = a.Data)
            fixed (float* ptrB = b.Data)
            fixed (float* ptrResult = result.Data)
            {
                var count = a.Size;

                for (int i = 0; i < count; i++)
                {
                    ptrResult[i] = ptrA[i] + ptrB[i];
                }
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

        var result = TensorHALExtensions.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            fixed (float* ptrA = a.Data)
            fixed (float* ptrB = b.Data)
            fixed (float* ptrResult = result.Data)
            {
                var count = a.Size;

                for (int i = 0; i < count; i++)
                {
                    ptrResult[i] = ptrA[i] - ptrB[i];
                }
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

        var result = TensorHALExtensions.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            fixed (float* ptrA = a.Data)
            fixed (float* ptrB = b.Data)
            fixed (float* ptrResult = result.Data)
            {
                var count = a.Size;

                for (int i = 0; i < count; i++)
                {
                    ptrResult[i] = ptrA[i] * ptrB[i];
                }
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

        var result = TensorHALExtensions.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            fixed (float* ptrA = a.Data)
            fixed (float* ptrB = b.Data)
            fixed (float* ptrResult = result.Data)
            {
                var count = a.Size;

                for (int i = 0; i < count; i++)
                {
                    ptrResult[i] = ptrA[i] / ptrB[i];
                }
            }
        }

        return result;
    }

    private Tensor ExecuteSum(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = TensorHALExtensions.Zeros(new int[] { 1 }, Device.CPU);

        unsafe
        {
            fixed (float* ptrA = a.Data)
            fixed (float* ptrResult = result.Data)
            {
                var count = a.Size;

                float sum = 0;
                for (int i = 0; i < count; i++)
                {
                    sum += ptrA[i];
                }
                ptrResult[0] = sum;
            }
        }

        return result;
    }

    private Tensor ExecuteMean(Tensor[] inputs)
    {
        var sum = ExecuteSum(inputs);
        var count = inputs[0].Size;

        unsafe
        {
            fixed (float* ptr = sum.Data)
            {
                ptr[0] /= count;
            }
        }

        return sum;
    }

    private Tensor ExecuteReLU(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = TensorHALExtensions.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            fixed (float* ptrA = a.Data)
            fixed (float* ptrResult = result.Data)
            {
                var count = a.Size;

                for (int i = 0; i < count; i++)
                {
                    ptrResult[i] = Math.Max(0, ptrA[i]);
                }
            }
        }

        return result;
    }

    private Tensor ExecuteSigmoid(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = TensorHALExtensions.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            fixed (float* ptrA = a.Data)
            fixed (float* ptrResult = result.Data)
            {
                var count = a.Size;

                for (int i = 0; i < count; i++)
                {
                    ptrResult[i] = 1.0f / (1.0f + (float)Math.Exp(-ptrA[i]));
                }
            }
        }

        return result;
    }

    private Tensor ExecuteTanh(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = TensorHALExtensions.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            fixed (float* ptrA = a.Data)
            fixed (float* ptrResult = result.Data)
            {
                var count = a.Size;

                for (int i = 0; i < count; i++)
                {
                    ptrResult[i] = (float)Math.Tanh(ptrA[i]);
                }
            }
        }

        return result;
    }

    private Tensor ExecuteCopy(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = TensorHALExtensions.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            fixed (float* ptrA = a.Data)
            fixed (float* ptrResult = result.Data)
            {
                var count = a.Size;

                Buffer.MemoryCopy(ptrA, ptrResult, count * sizeof(float), count * sizeof(float));
            }
        }

        return result;
    }

    private Tensor ExecuteCast(Tensor[] inputs)
    {
        // For now, casting is a no-op as we only support float32
        var a = inputs[0];
        var result = TensorHALExtensions.Zeros(a.Shape, Device.CPU);

        unsafe
        {
            fixed (float* ptrA = a.Data)
            fixed (float* ptrResult = result.Data)
            {
                var count = a.Size;

                for (int i = 0; i < count; i++)
                {
                    ptrResult[i] = ptrA[i];
                }
            }
        }

        return result;
    }

    private Tensor ExecuteReshape(Tensor[] inputs)
    {
        // Reshape expects 2 inputs: the tensor and the target shape
        if (inputs.Length != 2)
            throw new ArgumentException("Reshape requires exactly 2 inputs (tensor and shape)");

        var a = inputs[0];
        var shapeTensor = inputs[1];

        // Convert shape tensor to int array
        var newShape = new int[shapeTensor.Size];
        for (int i = 0; i < newShape.Length; i++)
        {
            newShape[i] = (int)shapeTensor.Data[i];
        }

        return a.Reshape(newShape);
    }
}
