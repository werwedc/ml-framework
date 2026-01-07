namespace MLFramework.Distributed.TensorParallel;

using System;
using System.Linq;
using System.Threading.Tasks;

/// <summary>
/// Helper utilities for common Tensor Parallelism state management operations.
/// </summary>
public static class TPState
{
    /// <summary>
    /// Helper method to safely execute code only on a specific rank.
    /// </summary>
    /// <param name="targetRank">Rank on which to execute the action</param>
    /// <param name="action">Action to execute</param>
    /// <returns>Completed task</returns>
    public static Task ExecuteOnRankAsync(int targetRank, Func<Task> action)
    {
        var rank = TensorParallel.GetRank();
        if (rank == targetRank)
        {
            return action();
        }
        return Task.CompletedTask;
    }

    /// <summary>
    /// Helper method to execute code only on a specific rank (synchronous version).
    /// </summary>
    /// <param name="targetRank">Rank on which to execute the action</param>
    /// <param name="action">Action to execute</param>
    public static void ExecuteOnRank(int targetRank, Action action)
    {
        var rank = TensorParallel.GetRank();
        if (rank == targetRank)
        {
            action();
        }
    }

    /// <summary>
    /// Helper method to execute code only on the master (rank 0).
    /// </summary>
    /// <param name="action">Action to execute</param>
    /// <returns>Completed task</returns>
    public static Task ExecuteOnMasterAsync(Func<Task> action)
    {
        return ExecuteOnRankAsync(0, action);
    }

    /// <summary>
    /// Helper method to execute code only on the master (rank 0) - synchronous version.
    /// </summary>
    /// <param name="action">Action to execute</param>
    public static void ExecuteOnMaster(Action action)
    {
        ExecuteOnRank(0, action);
    }

    /// <summary>
    /// Executes code on all ranks, but waits for all to complete.
    /// </summary>
    /// <param name="action">Action to execute on all ranks</param>
    /// <returns>Completed task</returns>
    public static async Task ExecuteOnAllAsync(Func<Task> action)
    {
        await Task.WhenAll(
            Enumerable.Range(0, TensorParallel.GetWorldSize())
                     .Select(rank => TPState.ExecuteOnRankAsync(rank, action))
        );

        // Barrier to ensure all ranks have completed
        await TensorParallel.GetCommunicator().BarrierAsync();
    }

    /// <summary>
    /// Executes code on all ranks, but waits for all to complete - synchronous version.
    /// Note: This will block until all ranks complete.
    /// </summary>
    /// <param name="action">Action to execute on all ranks</param>
    public static void ExecuteOnAll(Action action)
    {
        // Execute action on all ranks (in a real distributed setting, this would happen on each rank)
        action();

        // Barrier to ensure all ranks have completed
        TensorParallel.GetCommunicator().BarrierAsync().GetAwaiter().GetResult();
    }

    /// <summary>
    /// Gets whether the current rank is the master (rank 0).
    /// </summary>
    public static bool IsMaster => TensorParallel.GetRank() == 0;

    /// <summary>
    /// Gets whether the current rank is the last rank (worldSize - 1).
    /// </summary>
    public static bool IsLastRank => TensorParallel.GetRank() == TensorParallel.GetWorldSize() - 1;
}
