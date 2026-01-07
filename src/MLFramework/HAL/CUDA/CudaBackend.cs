using RitterFramework.Core.Tensor;
using MLFramework.HAL.CUDA;

namespace MLFramework.HAL.CUDA;

/// <summary>
/// CUDA backend using custom CUDA kernels
/// </summary>
public class CudaBackend : IBackend
{
    public string Name => "CUDA";
    public DeviceType Type => DeviceType.CUDA;

    private bool _isAvailable;
    private bool _isInitialized;

    public bool IsAvailable
    {
        get
        {
            if (_isInitialized)
                return _isAvailable;

            // Check if CUDA is available
            var result = CudaApi.CudaGetDeviceCount(out int count);
            _isAvailable = (result == CudaError.Success) && (count > 0);
            _isInitialized = true;

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
            Operation.Max or
            Operation.Min or
            Operation.ReLU or
            Operation.Sigmoid or
            Operation.Tanh or
            Operation.Copy or
            Operation.Fill => true,
            _ => false // More complex operations not yet implemented
        };
    }

    public Tensor ExecuteOperation(Operation operation, Tensor[] inputs)
    {
        if (inputs == null || inputs.Length == 0)
            throw new ArgumentException("At least one input tensor required");

        if (!SupportsOperation(operation))
            throw new NotSupportedException($"Operation {operation} not supported");

        if (!IsAvailable)
            throw new InvalidOperationException("CUDA is not available on this system");

        return operation switch
        {
            Operation.Add => ExecuteAdd(inputs),
            Operation.Subtract => ExecuteSubtract(inputs),
            Operation.Multiply => ExecuteMultiply(inputs),
            Operation.Divide => ExecuteDivide(inputs),
            Operation.Sum => ExecuteSum(inputs),
            Operation.Mean => ExecuteMean(inputs),
            Operation.Max => ExecuteMax(inputs),
            Operation.Min => ExecuteMin(inputs),
            Operation.ReLU => ExecuteReLU(inputs),
            Operation.Sigmoid => ExecuteSigmoid(inputs),
            Operation.Tanh => ExecuteTanh(inputs),
            Operation.Copy => ExecuteCopy(inputs),
            Operation.Fill => ExecuteFill(inputs),
            _ => throw new NotSupportedException($"Operation {operation}")
        };
    }

    public void Initialize()
    {
        // Initialize CUDA resources if needed
        // Load libraries, create contexts, etc.
        _isInitialized = true;
    }

    private Tensor ExecuteAdd(Tensor[] inputs)
    {
        if (inputs.Length != 2)
            throw new ArgumentException("Add requires exactly 2 inputs");

        var a = inputs[0];
        var b = inputs[1];

        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Tensors must have same shape");

        var result = TensorHALExtensions.Zeros(a.Shape, Device.CUDA(0));

        a.WithDataPointer(ptrA =>
        {
            b.WithDataPointer(ptrB =>
            {
                result.WithDataPointer(ptrResult =>
                {
                    CudaKernels.CudaAdd(ptrResult, ptrA, ptrB, a.Size);
                });
            });
        });

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

        var result = TensorHALExtensions.Zeros(a.Shape, Device.CUDA(0));

        a.WithDataPointer(ptrA =>
        {
            b.WithDataPointer(ptrB =>
            {
                result.WithDataPointer(ptrResult =>
                {
                    CudaKernels.CudaSubtract(ptrResult, ptrA, ptrB, a.Size);
                });
            });
        });

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

        var result = TensorHALExtensions.Zeros(a.Shape, Device.CUDA(0));

        a.WithDataPointer(ptrA =>
        {
            b.WithDataPointer(ptrB =>
            {
                result.WithDataPointer(ptrResult =>
                {
                    CudaKernels.CudaMultiply(ptrResult, ptrA, ptrB, a.Size);
                });
            });
        });

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

        var result = TensorHALExtensions.Zeros(a.Shape, Device.CUDA(0));

        a.WithDataPointer(ptrA =>
        {
            b.WithDataPointer(ptrB =>
            {
                result.WithDataPointer(ptrResult =>
                {
                    CudaKernels.CudaDivide(ptrResult, ptrA, ptrB, a.Size);
                });
            });
        });

        return result;
    }

    private Tensor ExecuteSum(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = TensorHALExtensions.Zeros(new int[] { 1 }, Device.CUDA(0));

        a.WithDataPointer(ptrA =>
        {
            result.WithDataPointer(ptrResult =>
            {
                CudaKernels.CudaSum(ptrResult, ptrA, a.Size);
            });
        });

        return result;
    }

    private Tensor ExecuteMean(Tensor[] inputs)
    {
        var sum = ExecuteSum(inputs);
        var count = inputs[0].Size;

        // Divide by count
        var result = TensorHALExtensions.Zeros(new int[] { 1 }, Device.CUDA(0));
        var scalar = TensorHALExtensions.FromArray(new[] { 1.0f / count }, Device.CUDA(0));

        sum.WithDataPointer(ptrSum =>
        {
            scalar.WithDataPointer(ptrScalar =>
            {
                result.WithDataPointer(ptrResult =>
                {
                    CudaKernels.CudaMultiply(ptrResult, ptrSum, ptrScalar, 1);
                });
            });
        });

        return result;
    }

    private Tensor ExecuteMax(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = TensorHALExtensions.Zeros(new int[] { 1 }, Device.CUDA(0));

        a.WithDataPointer(ptrA =>
        {
            result.WithDataPointer(ptrResult =>
            {
                CudaKernels.CudaMax(ptrResult, ptrA, a.Size);
            });
        });

        return result;
    }

    private Tensor ExecuteMin(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = TensorHALExtensions.Zeros(new int[] { 1 }, Device.CUDA(0));

        a.WithDataPointer(ptrA =>
        {
            result.WithDataPointer(ptrResult =>
            {
                CudaKernels.CudaMin(ptrResult, ptrA, a.Size);
            });
        });

        return result;
    }

    private Tensor ExecuteReLU(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = TensorHALExtensions.Zeros(a.Shape, Device.CUDA(0));

        a.WithDataPointer(ptrA =>
        {
            result.WithDataPointer(ptrResult =>
            {
                CudaKernels.CudaReLU(ptrResult, ptrA, a.Size);
            });
        });

        return result;
    }

    private Tensor ExecuteSigmoid(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = TensorHALExtensions.Zeros(a.Shape, Device.CUDA(0));

        a.WithDataPointer(ptrA =>
        {
            result.WithDataPointer(ptrResult =>
            {
                CudaKernels.CudaSigmoid(ptrResult, ptrA, a.Size);
            });
        });

        return result;
    }

    private Tensor ExecuteTanh(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = TensorHALExtensions.Zeros(a.Shape, Device.CUDA(0));

        a.WithDataPointer(ptrA =>
        {
            result.WithDataPointer(ptrResult =>
            {
                CudaKernels.CudaTanh(ptrResult, ptrA, a.Size);
            });
        });

        return result;
    }

    private Tensor ExecuteCopy(Tensor[] inputs)
    {
        var a = inputs[0];
        var result = TensorHALExtensions.Zeros(a.Shape, Device.CUDA(0));

        a.WithDataPointer(ptrA =>
        {
            result.WithDataPointer(ptrResult =>
            {
                CudaKernels.CudaCopy(ptrResult, ptrA, a.Size);
            });
        });

        return result;
    }

    private Tensor ExecuteFill(Tensor[] inputs)
    {
        if (inputs.Length != 2)
            throw new ArgumentException("Fill requires exactly 2 inputs (tensor and scalar value)");

        var a = inputs[0];
        var valueTensor = inputs[1];
        var value = valueTensor.Data[0];

        var result = TensorHALExtensions.Zeros(a.Shape, Device.CUDA(0));

        result.WithDataPointer(ptrResult =>
        {
            CudaKernels.CudaFill(ptrResult, value, a.Size);
        });

        return result;
    }
}
