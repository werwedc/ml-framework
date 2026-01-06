namespace MLFramework.Data;

/// <summary>
/// Interface for distributed samplers that partition data across multiple training processes/nodes.
/// </summary>
public interface IDistributedSampler : ISampler
{
    /// <summary>
    /// Gets the total number of replicas (processes/nodes) participating in distributed training.
    /// </summary>
    int NumReplicas { get; }

    /// <summary>
    /// Gets the rank of the current process/node within the distributed training setup.
    /// </summary>
    int Rank { get; }

    /// <summary>
    /// Gets the current epoch number used for shuffling.
    /// </summary>
    int Epoch { get; }

    /// <summary>
    /// Sets the epoch number for the sampler to ensure different shuffling across epochs.
    /// </summary>
    /// <param name="epoch">The epoch number (must be non-negative).</param>
    void SetEpoch(int epoch);
}
