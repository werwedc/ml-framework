namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for resharding strategies
/// </summary>
public class ReshardingStrategyTests
{
    /// <summary>
    /// Mock distributed coordinator for testing
    /// </summary>
    private class MockDistributedCoordinator : IDistributedCoordinator
    {
        public int WorldSize => 4;
        public int Rank => 0;

        public Task BarrierAsync(CancellationToken cancellationToken = default)
        {
            return Task.CompletedTask;
        }

        public Task<T> BroadcastAsync<T>(T data, CancellationToken cancellationToken = default) where T : class
        {
            return Task.FromResult(data);
        }

        public Task<T> AllReduceAsync<T>(T data, Func<T, T, T> reducer, CancellationToken cancellationToken = default) where T : class
        {
            return Task.FromResult(data);
        }

        public Task<IList<T>?> GatherAsync<T>(T data, CancellationToken cancellationToken = default) where T : class
        {
            return Task.FromResult<IList<T>?>(new List<T> { data });
        }
    }

    // Simple Resharding Strategy Tests

    [Fact]
    public void SimpleReshardingStrategy_CreatePlan_WithDifferentWorldSizes_CreatesValidPlan()
    {
        // Arrange
        var strategy = new SimpleReshardingStrategy();
        var metadata = CreateMetadata(sourceWorldSize: 4);
        var targetWorldSize = 2;

        // Act
        var plan = strategy.CreatePlan(metadata, targetWorldSize);

        // Assert
        Assert.Equal(4, plan.SourceWorldSize);
        Assert.Equal(2, plan.TargetWorldSize);
        Assert.NotEmpty(plan.TensorRedistributions);
    }

    [Fact]
    public void SimpleReshardingStrategy_CreatePlan_RoundRobinDistributesTensorsEvenly()
    {
        // Arrange
        var strategy = new SimpleReshardingStrategy();
        var metadata = CreateMetadataWithMultipleTensors(sourceWorldSize: 2, tensorsPerShard: 4);
        var targetWorldSize = 2;

        // Act
        var plan = strategy.CreatePlan(metadata, targetWorldSize);

        // Assert
        Assert.Equal(8, plan.TensorRedistributions.Count); // 2 shards * 4 tensors
        var rank0Count = plan.TensorRedistributions.Count(r => r.TargetRanks.Contains(0));
        var rank1Count = plan.TensorRedistributions.Count(r => r.TargetRanks.Contains(1));
        Assert.Equal(4, rank0Count);
        Assert.Equal(4, rank1Count);
    }

    [Fact]
    public async Task SimpleReshardingStrategy_ExecuteAsync_WithValidPlan_ReturnsReshardedData()
    {
        // Arrange
        var strategy = new SimpleReshardingStrategy();
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
        var result = await strategy.ExecuteAsync(plan, sourceShards);

        // Assert
        Assert.True(result.Success);
        Assert.Equal(4, result.TargetShards.Count);
        Assert.True(result.Duration.TotalMilliseconds > 0);
    }

    [Fact]
    public void SimpleReshardingStrategy_CreatePlan_FewerToMoreGPUs()
    {
        // Arrange
        var strategy = new SimpleReshardingStrategy();
        var metadata = CreateMetadata(sourceWorldSize: 2);
        var targetWorldSize = 4;

        // Act
        var plan = strategy.CreatePlan(metadata, targetWorldSize);

        // Assert
        Assert.Equal(2, plan.SourceWorldSize);
        Assert.Equal(4, plan.TargetWorldSize);
        Assert.All(plan.TensorRedistributions, r =>
        {
            Assert.Single(r.TargetRanks);
            Assert.InRange(r.TargetRanks[0], 0, 3);
        });
    }

    [Fact]
    public void SimpleReshardingStrategy_CreatePlan_MoreToFewerGPUs()
    {
        // Arrange
        var strategy = new SimpleReshardingStrategy();
        var metadata = CreateMetadata(sourceWorldSize: 4);
        var targetWorldSize = 2;

        // Act
        var plan = strategy.CreatePlan(metadata, targetWorldSize);

        // Assert
        Assert.Equal(4, plan.SourceWorldSize);
        Assert.Equal(2, plan.TargetWorldSize);
        Assert.All(plan.TensorRedistributions, r =>
        {
            Assert.Single(r.TargetRanks);
            Assert.InRange(r.TargetRanks[0], 0, 1);
        });
    }

    [Fact]
    public void ReshardingPlan_GetSourceRankForTensor_ReturnsCorrectRank()
    {
        // Arrange
        var plan = new ReshardingPlan
        {
            TensorRedistributions = new List<TensorRedistribution>
            {
                new TensorRedistribution
                {
                    TensorName = "layer1.weight",
                    SourceRank = 0
                },
                new TensorRedistribution
                {
                    TensorName = "layer2.weight",
                    SourceRank = 1
                }
            }
        };

        // Act
        var rank1 = plan.GetSourceRankForTensor("layer1.weight");
        var rank2 = plan.GetSourceRankForTensor("layer2.weight");
        var rank3 = plan.GetSourceRankForTensor("layer3.weight");

        // Assert
        Assert.Equal(0, rank1);
        Assert.Equal(1, rank2);
        Assert.Equal(-1, rank3); // Not found
    }

    [Fact]
    public void ReshardingPlan_GetTargetRanksForTensor_ReturnsCorrectRanks()
    {
        // Arrange
        var plan = new ReshardingPlan
        {
            TensorRedistributions = new List<TensorRedistribution>
            {
                new TensorRedistribution
                {
                    TensorName = "layer1.weight",
                    TargetRanks = new List<int> { 0, 1, 2 }
                }
            }
        };

        // Act
        var ranks = plan.GetTargetRanksForTensor("layer1.weight");
        var emptyRanks = plan.GetTargetRanksForTensor("unknown");

        // Assert
        Assert.Equal(3, ranks.Count);
        Assert.Equal(new List<int> { 0, 1, 2 }, ranks);
        Assert.Empty(emptyRanks);
    }

    [Fact]
    public void SimpleReshardingStrategy_CreatePlan_WithVariousTensorShapes_HandlesAllShapes()
    {
        // Arrange
        var strategy = new SimpleReshardingStrategy();
        var metadata = CreateMetadataWithVariousShapes(sourceWorldSize: 2);
        var targetWorldSize = 2;

        // Act
        var plan = strategy.CreatePlan(metadata, targetWorldSize);

        // Assert
        Assert.NotNull(plan);
        Assert.Equal(2, plan.SourceWorldSize);
        Assert.All(plan.TensorRedistributions, r =>
        {
            Assert.NotEmpty(r.SourceShape);
            Assert.NotEmpty(r.Slices);
            Assert.Single(r.Slices);
        });
    }

    // Parallel Resharding Strategy Tests

    [Fact]
    public void ParallelReshardingStrategy_CreatePlan_WithShardedTensors_DistributesCorrectly()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator();
        var strategy = new ParallelReshardingStrategy(coordinator);
        var metadata = CreateMetadataWithShardedTensors(sourceWorldSize: 4);
        var targetWorldSize = 2;

        // Act
        var plan = strategy.CreatePlan(metadata, targetWorldSize);

        // Assert
        Assert.Equal(4, plan.SourceWorldSize);
        Assert.Equal(2, plan.TargetWorldSize);
        Assert.NotEmpty(plan.TensorRedistributions);
    }

    [Fact]
    public void ParallelReshardingStrategy_CreatePlan_ScalarTensor_GoesToRank0()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator();
        var strategy = new ParallelReshardingStrategy(coordinator);
        var metadata = CreateMetadataWithScalarTensor(sourceWorldSize: 2);
        var targetWorldSize = 4;

        // Act
        var plan = strategy.CreatePlan(metadata, targetWorldSize);

        // Assert
        var scalarRedistribution = plan.TensorRedistributions
            .FirstOrDefault(r => r.TensorName == "learning_rate");
        Assert.NotNull(scalarRedistribution);
        Assert.Single(scalarRedistribution.TargetRanks);
        Assert.Equal(0, scalarRedistribution.TargetRanks[0]);
    }

    [Fact]
    public async Task ParallelReshardingStrategy_ExecuteAsync_ParallelExecution_Succeeds()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator();
        var strategy = new ParallelReshardingStrategy(coordinator);
        var metadata = CreateMetadata(sourceWorldSize: 2);
        var plan = strategy.CreatePlan(metadata, targetWorldSize: 4);
        var sourceShards = CreateSourceShards(2);

        // Act
        var result = await strategy.ExecuteAsync(plan, sourceShards);

        // Assert
        Assert.True(result.Success);
        Assert.Equal(4, result.TargetShards.Count);
        Assert.True(result.Duration.TotalMilliseconds > 0);
    }

    [Fact]
    public void ParallelReshardingStrategy_CreatePlan_ComputesCorrectSlices()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator();
        var strategy = new ParallelReshardingStrategy(coordinator);
        var metadata = new CheckpointMetadata
        {
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata
                {
                    Rank = 0,
                    Tensors = new List<TensorMetadata>
                    {
                        new TensorMetadata
                        {
                            Name = "layer.weight",
                            Shape = new long[] { 100, 50 }
                        }
                    }
                }
            }
        };

        // Act
        var plan = strategy.CreatePlan(metadata, targetWorldSize: 2);

        // Assert
        var redistribution = plan.TensorRedistributions.FirstOrDefault();
        Assert.NotNull(redistribution);
        Assert.Equal(2, redistribution.TargetRanks.Count); // Should split across 2 targets
        Assert.All(redistribution.Slices, s =>
        {
            Assert.NotNull(s.Start);
            Assert.NotNull(s.End);
            Assert.NotNull(s.Shape);
        });
    }

    [Fact]
    public async Task ParallelReshardingStrategy_ExecuteAsync_FromFewerToMoreGPUs()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator();
        var strategy = new ParallelReshardingStrategy(coordinator);
        var metadata = CreateMetadataWithShardedTensors(sourceWorldSize: 2);
        var plan = strategy.CreatePlan(metadata, targetWorldSize: 4);
        var sourceShards = CreateSourceShards(2);

        // Act
        var result = await strategy.ExecuteAsync(plan, sourceShards);

        // Assert
        Assert.True(result.Success);
        Assert.Equal(4, result.TargetShards.Count);
        Assert.True(result.Duration.TotalMilliseconds > 0);
    }

    [Fact]
    public async Task ParallelReshardingStrategy_ExecuteAsync_FromMoreToFewerGPUs()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator();
        var strategy = new ParallelReshardingStrategy(coordinator);
        var metadata = CreateMetadataWithShardedTensors(sourceWorldSize: 4);
        var plan = strategy.CreatePlan(metadata, targetWorldSize: 2);
        var sourceShards = CreateSourceShards(4);

        // Act
        var result = await strategy.ExecuteAsync(plan, sourceShards);

        // Assert
        Assert.True(result.Success);
        Assert.Equal(2, result.TargetShards.Count);
        Assert.True(result.Duration.TotalMilliseconds > 0);
    }

    // Factory Tests

    [Fact]
    public void ReshardingStrategyFactory_Create_Simple_ReturnsSimpleStrategy()
    {
        // Arrange & Act
        var strategy = ReshardingStrategyFactory.Create("simple");

        // Assert
        Assert.IsType<SimpleReshardingStrategy>(strategy);
    }

    [Fact]
    public void ReshardingStrategyFactory_Create_Parallel_ReturnsParallelStrategy()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator();

        // Act
        var strategy = ReshardingStrategyFactory.Create("parallel", coordinator);

        // Assert
        Assert.IsType<ParallelReshardingStrategy>(strategy);
    }

    [Fact]
    public void ReshardingStrategyFactory_Create_UnknownStrategy_ThrowsException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentException>(() =>
            ReshardingStrategyFactory.Create("unknown"));
    }

    [Fact]
    public void ReshardingStrategyFactory_Create_ParallelWithoutCoordinator_ThrowsException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ReshardingStrategyFactory.Create("parallel", null));
    }

    // Helper Methods

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

    private static CheckpointMetadata CreateMetadataWithMultipleTensors(int sourceWorldSize, int tensorsPerShard)
    {
        var metadata = new CheckpointMetadata
        {
            Shards = new List<ShardMetadata>()
        };

        for (int rank = 0; rank < sourceWorldSize; rank++)
        {
            var tensors = new List<TensorMetadata>();
            for (int i = 0; i < tensorsPerShard; i++)
            {
                tensors.Add(new TensorMetadata
                {
                    Name = $"tensor_{rank}_{i}",
                    Shape = new long[] { 10 }
                });
            }
            metadata.Shards.Add(new ShardMetadata
            {
                Rank = rank,
                Tensors = tensors
            });
        }

        return metadata;
    }

    private static CheckpointMetadata CreateMetadataWithVariousShapes(int sourceWorldSize)
    {
        var metadata = new CheckpointMetadata
        {
            Shards = new List<ShardMetadata>()
        };

        for (int rank = 0; rank < sourceWorldSize; rank++)
        {
            metadata.Shards.Add(new ShardMetadata
            {
                Rank = rank,
                Tensors = new List<TensorMetadata>
                {
                    new TensorMetadata { Name = $"tensor_1d_{rank}", Shape = new long[] { 100 } },
                    new TensorMetadata { Name = $"tensor_2d_{rank}", Shape = new long[] { 100, 50 } },
                    new TensorMetadata { Name = $"tensor_3d_{rank}", Shape = new long[] { 100, 50, 25 } }
                }
            });
        }

        return metadata;
    }

    private static CheckpointMetadata CreateMetadataWithShardedTensors(int sourceWorldSize)
    {
        var metadata = new CheckpointMetadata
        {
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
                        Name = "layer.weight",
                        Shape = new long[] { 100, 50 }
                    },
                    new TensorMetadata
                    {
                        Name = "layer.bias",
                        Shape = new long[] { 50 }
                    }
                }
            });
        }

        return metadata;
    }

    private static CheckpointMetadata CreateMetadataWithScalarTensor(int sourceWorldSize)
    {
        var metadata = new CheckpointMetadata
        {
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
                        Name = "learning_rate",
                        Shape = Array.Empty<long>() // Scalar
                    }
                }
            });
        }

        return metadata;
    }

    private static List<ShardData> CreateSourceShards(int count)
    {
        var shards = new List<ShardData>();
        for (int rank = 0; rank < count; rank++)
        {
            shards.Add(new ShardData
            {
                Rank = rank,
                Data = System.Text.Encoding.UTF8.GetBytes($"shard_{rank}"),
                TensorInfo = new List<TensorMetadata>()
            });
        }
        return shards;
    }
}
