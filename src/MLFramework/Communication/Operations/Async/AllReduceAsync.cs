namespace MLFramework.Communication.Operations.Async;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System;

/// <summary>
/// Asynchronous all-reduce operation
/// </summary>
public static class AllReduceAsync
{
    /// <summary>
    /// All-reduce tensor asynchronously across all ranks
    /// </summary>
    /// <param name="backend">Async communication backend</param>
    /// <param name="tensor">Tensor to all-reduce</param>
    /// <param name="operation">Reduction operation</param>
    /// <param name="group">Process group (default: world)</param>
    /// <returns>Handle to the ongoing all-reduce operation</returns>
    public static ICommunicationHandle AllReduceTensorAsync(
        IAsyncCommunicationBackend backend,
        Tensor tensor,
        ReduceOp operation,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        Operations.Reduce.ValidateReduceOperation(operation, tensor.Dtype);

        return backend.AllReduceAsync(tensor, operation);
    }

    /// <summary>
    /// Wait for all-reduce to complete and divide by world size
    /// </summary>
    public static Tensor AverageGradientsAsync(
        IAsyncCommunicationBackend backend,
        Tensor gradients)
    {
        var handle = AllReduceTensorAsync(backend, gradients, ReduceOp.Sum);
        handle.Wait();

        var summed = handle.GetResult();
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
}
