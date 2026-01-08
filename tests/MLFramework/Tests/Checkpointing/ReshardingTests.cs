namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for SimpleReshardingStrategy
/// </summary>
public class SimpleReshardingStrategyTests
{
    private SimpleReshardingStrategy _strategy = null!;

    public SimpleReshardingStrategyTests()
    {
        _strategy = new SimpleReshardingStrategy();
    }

    [Fact]
    public void CreatePlan_WithDifferentWorldSizes_CreatesValidPlan()
    {
        // Arrange
        var metadata = CreateMetadata(sourceWorldSize: 4);
        var targetWorldSize = 2;

        // Act
        var plan = _strategy.CreatePlan(metadata, targetWorldSize);

        // Assert
        Assert.Equal(4, plan.SourceWorldSize);
        Assert.Equal(2, plan.TargetWorldSize);
        Assert.NotEmpty(plan.TensorRedistributions);
    }

    [Fact]
    public async Task ExecuteAsync_WithValidPlan_ReturnsReshardedData()
    {
        // Arrange
        var plan = new ReshardingPlan
        {
            SourceWorldSize = 2,
            TargetWorldSize = 4,
            TensorRedistributions = new List<TensorRedistribution>()
        };
        var sourceShards = new List<ShardData>
        {
            new ShardData
            {
                Rank = 0,
                Data = System.Text.Encoding.UTF8.GetBytes("shard0"),
                TensorInfo = new List<TensorMetadata>()
            },
            new ShardData
            {
                Rank = 1,
                Data = System.Text.Encoding.UTF8.GetBytes("shard1"),
                TensorInfo = new List<TensorMetadata>()
            }
        };

        // Act
        var result = await _strategy.ExecuteAsync(plan, sourceShards);

        // Assert
        Assert.True(result.Success);
        Assert.Equal(4, result.TargetShards.Count);
    }

    private static CheckpointMetadata CreateMetadata(int sourceWorldSize)
    {
        var metadata = new CheckpointMetadata
        {
            Sharding = new ShardingMetadata
            {
                ShardCount = sourceWorldSize
            },
            Shards = new List<ShardMetadata>()
        };

        for (int rank = 0; rank < sourceWorldSize; rank++)
        {
            metadata.Shards.Add(new ShardMetadata
            {
                Rank = rank,
                Tensors = new List<TensorMetadata>
                {
                    new TensorMetadata
                    {
                        Name = $"tensor_{rank}",
                        Shape = new long[] { 10 }
                    }
                }
            });
        }

        return metadata;
    }
}
