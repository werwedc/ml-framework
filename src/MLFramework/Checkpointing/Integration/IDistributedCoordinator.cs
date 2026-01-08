namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface for distributed coordination across multiple processes
/// </summary>
public interface IDistributedCoordinator
{
    /// <summary>
    /// Gets the total number of processes in the distributed group
    /// </summary>
    int WorldSize { get; }

    /// <summary>
    /// Gets the rank of the current process in the distributed group
    /// </summary>
    int Rank { get; }

    /// <summary>
    /// Barrier: blocks until all processes have reached this point
    /// </summary>
    Task BarrierAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Broadcast data from rank 0 to all other ranks
    /// </summary>
    /// <typeparam name="T">Type of data to broadcast (must be a class)</typeparam>
    /// <param name="data">Data to broadcast (only used by rank 0)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>The broadcasted data (same for all ranks)</returns>
    Task<T> BroadcastAsync<T>(T data, CancellationToken cancellationToken = default) where T : class;

    /// <summary>
    /// Perform all-reduce operation across all ranks
    /// </summary>
    /// <typeparam name="T">Type of data to reduce (must be a class)</typeparam>
    /// <param name="data">Data to reduce from each rank</param>
    /// <param name="reducer">Function to combine data from two ranks</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>The reduced result (same for all ranks)</returns>
    Task<T> AllReduceAsync<T>(T data, Func<T, T, T> reducer, CancellationToken cancellationToken = default) where T : class;

    /// <summary>
    /// Gather data from all ranks to rank 0
    /// </summary>
    /// <typeparam name="T">Type of data to gather (must be a class)</typeparam>
    /// <param name="data">Data from this rank</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>
    /// On rank 0: list of data from all ranks
    /// On other ranks: null
    /// </returns>
    Task<IList<T>?> GatherAsync<T>(T data, CancellationToken cancellationToken = default) where T : class;
}
