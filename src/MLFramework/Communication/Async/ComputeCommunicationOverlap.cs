namespace MLFramework.Communication.Async;

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;
using MLFramework.Communication;

/// <summary>
/// Helper for overlapping computation with communication
/// </summary>
public static class ComputeCommunicationOverlap
{
    /// <summary>
    /// Start async communication and immediately return handle
    /// </summary>
    public static ICommunicationHandle StartAllReduce(
        IAsyncCommunicationBackend backend,
        Tensor tensor,
        ReduceOp operation)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var task = Task.Run(() => backend.AllReduce(tensor, operation));
        return new AsyncCommunicationHandle(task);
    }

    /// <summary>
    /// Pattern: Do computation while communication is in progress
    /// </summary>
    public static Tensor ComputeWhileCommunicating(
        IAsyncCommunicationBackend backend,
        Tensor tensorToSync,
        Func<Tensor> computeFunc,
        ReduceOp operation = ReduceOp.Sum)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensorToSync == null)
            throw new ArgumentNullException(nameof(tensorToSync));

        if (computeFunc == null)
            throw new ArgumentNullException(nameof(computeFunc));

        // Start async communication
        var commHandle = StartAllReduce(backend, tensorToSync, operation);

        // Do computation while communicating
        var computed = computeFunc();

        // Wait for communication to finish
        commHandle.Wait();

        return commHandle.GetResult();
    }

    /// <summary>
    /// Pattern: Pipeline multiple compute-communication stages
    /// </summary>
    public static List<Tensor> PipelineComputeCommunicate(
        IAsyncCommunicationBackend backend,
        List<Tensor> tensors,
        Func<Tensor, Tensor> computeFunc,
        ReduceOp operation = ReduceOp.Sum)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        if (computeFunc == null)
            throw new ArgumentNullException(nameof(computeFunc));

        var results = new List<Tensor>();
        var queue = new CommunicationOperationQueue();

        // Stage 1: Start all communications
        foreach (var tensor in tensors)
        {
            var handle = StartAllReduce(backend, tensor, operation);
            queue.Enqueue(handle);
        }

        // Stage 2: Compute while waiting
        foreach (var tensor in tensors)
        {
            // Do computation
            var computed = computeFunc(tensor);
            results.Add(computed);
        }

        // Stage 3: Wait for all communications
        queue.WaitForAll();

        // Return computation results (not communication results, as that's the typical pattern)
        return results;
    }
}
