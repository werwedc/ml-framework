namespace MLFramework.Communication.Operations;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using RitterFramework.Core;
using System.Collections.Generic;
using System;

/// <summary>
/// Synchronous reduce operation
/// </summary>
public static class Reduce
{
    /// <summary>
    /// Reduce tensor from all ranks to root rank
    /// </summary>
    /// <param name="backend">Communication backend</param>
    /// <param name="tensor">Tensor to reduce</param>
    /// <param name="operation">Reduction operation</param>
    /// <param name="rootRank">Rank that will receive the result</param>
    /// <param name="group">Process group (default: world)</param>
    /// <returns>Reduced tensor on root rank, null on other ranks</returns>
    public static Tensor ReduceTensor(
        ICommunicationBackend backend,
        Tensor tensor,
        ReduceOp operation,
        int rootRank,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (rootRank < 0 || rootRank >= backend.WorldSize)
            throw new ArgumentOutOfRangeException(nameof(rootRank));

        if (group != null)
        {
            // Check if rank is in the group
            // Note: For the existing ProcessGroup, we'll just do basic validation
        }

        // Validate operation is supported for type
        ValidateReduceOperation(operation, tensor.Dtype);

        return backend.Reduce(tensor, operation, rootRank);
    }

    /// <summary>
    /// Validate that the reduction operation is valid for the tensor data type
    /// </summary>
    internal static void ValidateReduceOperation(ReduceOp operation, DataType dtype)
    {
        // Validate that the operation is valid for the data type
        // Some operations like Product may not make sense for all types
        // This is a placeholder - implement proper validation based on data type
        if (operation == ReduceOp.Product && dtype == DataType.Bool)
        {
            throw new ArgumentException($"ReduceOp.Product is not supported for type {dtype}");
        }
    }

    /// <summary>
    /// Reduce multiple tensors from all ranks to root rank
    /// </summary>
    public static List<Tensor> ReduceTensors(
        ICommunicationBackend backend,
        IEnumerable<Tensor> tensors,
        ReduceOp operation,
        int rootRank,
        ProcessGroup? group = null)
    {
        var result = new List<Tensor>();
        foreach (var tensor in tensors)
        {
            result.Add(ReduceTensor(backend, tensor, operation, rootRank, group));
        }
        return result;
    }
}
