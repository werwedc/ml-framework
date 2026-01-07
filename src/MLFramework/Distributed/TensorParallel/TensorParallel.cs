namespace MLFramework.Distributed.TensorParallel;

using MLFramework.Distributed.Communication;
using System;

/// <summary>
/// Simplified static API wrapper for Tensor Parallelism context management.
/// Provides convenient access to the current TP context and common operations.
/// </summary>
public static class TensorParallel
{
    /// <summary>
    /// Initializes TP with default settings (for common use cases).
    /// </summary>
    /// <param name="worldSize">Total number of processes/ranks</param>
    /// <param name="rank">Rank of this process (0 to worldSize-1)</param>
    /// <param name="backend">Backend type ("mock", "nccl", etc.)</param>
    /// <returns>New TensorParallelContext instance</returns>
    public static TensorParallelContext Initialize(int worldSize, int rank, string backend = "mock")
    {
        return TensorParallelContext.Initialize(worldSize, rank, backend);
    }

    /// <summary>
    /// Gets the current TP context (throws if not initialized).
    /// </summary>
    /// <returns>Current TensorParallelContext instance</returns>
    /// <exception cref="InvalidOperationException">Thrown when TP context is not initialized</exception>
    public static TensorParallelContext GetContext()
    {
        return TensorParallelContext.Current
            ?? throw new InvalidOperationException(
                "TensorParallel context not initialized. Call TensorParallel.Initialize() first.");
    }

    /// <summary>
    /// Tries to get the current TP context (returns null if not initialized).
    /// </summary>
    /// <returns>Current TensorParallelContext instance or null if not initialized</returns>
    public static TensorParallelContext? TryGetContext()
    {
        return TensorParallelContext.Current;
    }

    /// <summary>
    /// Checks if TP is currently active (context is initialized).
    /// </summary>
    public static bool IsInitialized => TensorParallelContext.Current != null;

    /// <summary>
    /// Gets the world size from current context.
    /// </summary>
    /// <returns>Total number of processes/ranks</returns>
    public static int GetWorldSize() => GetContext().WorldSize;

    /// <summary>
    /// Gets the rank from current context.
    /// </summary>
    /// <returns>Rank of this process (0 to worldSize-1)</returns>
    public static int GetRank() => GetContext().Rank;

    /// <summary>
    /// Gets the communicator from current context.
    /// </summary>
    /// <returns>Communicator instance</returns>
    public static ICommunicator GetCommunicator() => GetContext().Communicator;
}
