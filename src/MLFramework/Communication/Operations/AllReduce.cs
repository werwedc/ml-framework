namespace MLFramework.Communication.Operations;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System.Collections.Generic;

/// <summary>
/// Synchronous all-reduce operation
/// </summary>
public static class AllReduce
{
    /// <summary>
    /// All-reduce tensor across all ranks
    /// </summary>
    /// <param name="backend">Communication backend</param>
    /// <param name="tensor">Tensor to all-reduce</param>
    /// <param name="operation">Reduction operation</param>
    /// <param name="group">Process group (default: world)</param>
    /// <returns>All-reduced tensor</returns>
    public static Tensor AllReduceTensor(
        ICommunicationBackend backend,
        Tensor tensor,
        ReduceOp operation,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Validate operation is supported for type
        Reduce.ValidateReduceOperation(operation, tensor.Dtype);

        return backend.AllReduce(tensor, operation);
    }

    /// <summary>
    /// All-reduce with automatic division by world size (common pattern)
    /// </summary>
    public static Tensor AverageGradients(
        ICommunicationBackend backend,
        Tensor gradients)
    {
        var summed = AllReduceTensor(backend, gradients, ReduceOp.Sum);
        return DivideByScalar(summed, backend.WorldSize);
    }

    /// <summary>
    /// Divide tensor by a scalar value
    /// </summary>
    private static Tensor DivideByScalar(Tensor tensor, int divisor)
    {
        // Create a new tensor with divided values
        var resultData = new float[tensor.Size];
        for (int i = 0; i < tensor.Size; i++)
        {
            resultData[i] = tensor.Data[i] / divisor;
        }
        return new Tensor(resultData, tensor.Shape, tensor.RequiresGrad, tensor.Dtype);
    }

    /// <summary>
    /// All-reduce multiple tensors
    /// </summary>
    public static List<Tensor> AllReduceTensors(
        ICommunicationBackend backend,
        IEnumerable<Tensor> tensors,
        ReduceOp operation,
        ProcessGroup? group = null)
    {
        var result = new List<Tensor>();
        foreach (var tensor in tensors)
        {
            result.Add(AllReduceTensor(backend, tensor, operation, group));
        }
        return result;
    }
}
