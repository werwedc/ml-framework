namespace MLFramework.Communication.Operations;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System.Collections.Generic;

/// <summary>
/// Synchronous reduce-scatter operation
/// </summary>
public static class ReduceScatter
{
    /// <summary>
    /// Reduce tensor from all ranks and scatter chunks to different ranks
    /// </summary>
    /// <param name="backend">Communication backend</param>
    /// <param name="tensor">Tensor to reduce and scatter</param>
    /// <param name="operation">Reduction operation</param>
    /// <param name="group">Process group (default: world)</param>
    /// <returns>Reduced chunk for this rank</returns>
    public static Tensor ReduceScatterTensor(
        ICommunicationBackend backend,
        Tensor tensor,
        ReduceOp operation,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        Reduce.ValidateReduceOperation(operation, tensor.Dtype);

        return backend.ReduceScatter(tensor, operation);
    }

    /// <summary>
    /// Reduce-scatter multiple tensors
    /// </summary>
    public static List<Tensor> ReduceScatterTensors(
        ICommunicationBackend backend,
        IEnumerable<Tensor> tensors,
        ReduceOp operation,
        ProcessGroup? group = null)
    {
        var result = new List<Tensor>();
        foreach (var tensor in tensors)
        {
            result.Add(ReduceScatterTensor(backend, tensor, operation, group));
        }
        return result;
    }
}
