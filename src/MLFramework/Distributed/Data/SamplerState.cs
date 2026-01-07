namespace MachineLearning.Distributed.Data;

/// <summary>
/// Represents the state of a sampler for checkpointing
/// </summary>
public class SamplerState
{
    public int WorkerCount { get; set; }
    public int WorkerRank { get; set; }
    public int CurrentPosition { get; set; }
    public int Seed { get; set; }
    public List<int> Indices { get; set; } = new();
}

/// <summary>
/// Helper class for sampler serialization/deserialization
/// </summary>
public static class SamplerSerializer
{
    /// <summary>
    /// Capture the current state of a sampler
    /// </summary>
    public static SamplerState CaptureState(ElasticDistributedSampler sampler)
    {
        return new SamplerState
        {
            WorkerCount = sampler.DatasetSize,
            WorkerRank = 0,
            CurrentPosition = 0,
            Seed = 0,
            Indices = new List<int>()
        };
    }

    /// <summary>
    /// Restore a sampler from a captured state
    /// </summary>
    public static ElasticDistributedSampler RestoreState(SamplerState state)
    {
        var sampler = new ElasticDistributedSampler(
            state.WorkerCount,
            state.WorkerCount,
            state.WorkerRank,
            seed: state.Seed);

        return sampler;
    }
}
