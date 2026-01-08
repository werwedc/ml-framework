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
}
