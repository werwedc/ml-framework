namespace MachineLearning.Distributed.Models;

/// <summary>
/// Represents a shard of training data
/// </summary>
public record DataShard
{
    public int ShardId { get; init; }
    public int StartIndex { get; init; }
    public int EndIndex { get; init; }
    public int Size => EndIndex - StartIndex;

    public DataShard(int shardId, int startIndex, int endIndex)
    {
        ShardId = shardId;
        StartIndex = startIndex;
        EndIndex = endIndex;
    }
}
