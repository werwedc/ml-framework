namespace MachineLearning.Distributed.Models;

/// <summary>
/// Plan for redistributing data during a topology change
/// </summary>
public class DataRedistributionPlan
{
    /// <summary>
    /// List of transfers to execute
    /// </summary>
    public List<DataTransfer> Transfers { get; set; } = new();

    /// <summary>
    /// Worker-specific redistribution assignments
    /// </summary>
    public Dictionary<WorkerId, List<DataShard>> WorkerAssignments { get; set; } = new();

    /// <summary>
    /// Total number of shards in the redistribution
    /// </summary>
    public int TotalShards { get; set; }
}

/// <summary>
/// Represents a single data transfer between workers
/// </summary>
public record DataTransfer
{
    public WorkerId SourceWorker { get; init; } = null!;
    public WorkerId DestinationWorker { get; init; } = null!;
    public DataShard Shard { get; init; } = null!;
    public int Priority { get; init; }
    public DateTime EstimatedCompletionTime { get; init; }
}
