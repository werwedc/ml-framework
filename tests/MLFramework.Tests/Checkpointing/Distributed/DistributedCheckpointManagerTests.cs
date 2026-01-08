using MLFramework.Checkpointing;
using MLFramework.Checkpointing.Distributed;
using MLFramework.Distributed;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Checkpointing.Distributed;

/// <summary>
/// Tests for DistributedCheckpointManager
/// </summary>
public class DistributedCheckpointManagerTests
{
    private MockProcessGroup CreateMockProcessGroup(int rank, int worldSize)
    {
        return new MockProcessGroup(rank, worldSize);
    }

    [Fact]
    public void Constructor_InitializesCorrectly()
    {
        // Arrange & Act
        var processGroup = CreateMockProcessGroup(0, 4);
        var manager = new DistributedCheckpointManager(0, 4, processGroup);

        // Assert
        Assert.Equal(0, manager.Rank);
        Assert.Equal(4, manager.WorldSize);
        Assert.NotNull(manager.LocalManager);
    }

    [Fact]
    public void RegisterCheckpoint_StoresLocally()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var manager = new DistributedCheckpointManager(0, 4, processGroup);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });

        // Act
        manager.RegisterCheckpoint("layer1", tensor);

        // Assert
        Assert.True(manager.LocalManager.HasCheckpoint("layer1"));
    }

    [Fact]
    public void RegisterCheckpoint_NullLayerId_ThrowsArgumentException()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var manager = new DistributedCheckpointManager(0, 4, processGroup);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
        {
            manager.RegisterCheckpoint(null!, tensor);
        });
    }

    [Fact]
    public void RegisterCheckpoint_NullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var manager = new DistributedCheckpointManager(0, 4, processGroup);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
        {
            manager.RegisterCheckpoint("layer1", null!);
        });
    }

    [Fact]
    public void RegisterCheckpointBroadcast_StoresLocally()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var manager = new DistributedCheckpointManager(0, 4, processGroup);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });

        // Act
        manager.RegisterCheckpointBroadcast("layer1", tensor);

        // Assert
        Assert.True(manager.LocalManager.HasCheckpoint("layer1"));
    }

    [Fact]
    public void RetrieveOrFetch_Local_ReturnsCorrectTensor()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var manager = new DistributedCheckpointManager(0, 4, processGroup);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });
        manager.RegisterCheckpoint("layer1", tensor);

        // Act
        var retrieved = manager.RetrieveOrFetch("layer1", sourceRank: null);

        // Assert
        Assert.NotNull(retrieved);
    }

    [Fact]
    public void RetrieveOrFetch_NonExistent_ThrowsKeyNotFoundException()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var manager = new DistributedCheckpointManager(0, 4, processGroup);

        // Act & Assert
        Assert.Throws<KeyNotFoundException>(() =>
        {
            manager.RetrieveOrFetch("nonexistent_layer", sourceRank: null);
        });
    }

    [Fact]
    public void RetrieveOrFetch_NullLayerId_ThrowsArgumentException()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var manager = new DistributedCheckpointManager(0, 4, processGroup);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
        {
            manager.RetrieveOrFetch(null!);
        });
    }

    [Fact]
    public void ClearCheckpointsDistributed_ClearsAllCheckpoints()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var manager = new DistributedCheckpointManager(0, 4, processGroup);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });
        manager.RegisterCheckpoint("layer1", tensor);

        // Act
        manager.ClearCheckpointsDistributed();

        // Assert
        Assert.False(manager.LocalManager.HasCheckpoint("layer1"));
    }

    [Fact]
    public void GetAggregatedMemoryStats_ReturnsCorrectStatistics()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var manager = new DistributedCheckpointManager(0, 4, processGroup);

        // Act
        var stats = manager.GetAggregatedMemoryStats();

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(0, stats.TotalCheckpointCount);
    }

    [Fact]
    public void Dispose_ReleasesResources()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var manager = new DistributedCheckpointManager(0, 4, processGroup);

        // Act
        manager.Dispose();

        // Assert - Should be able to dispose without throwing
        Assert.True(true);
    }

    [Fact]
    public void Dispose_AfterDispose_ThrowsObjectDisposedException()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var manager = new DistributedCheckpointManager(0, 4, processGroup);
        manager.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() =>
        {
            manager.RegisterCheckpoint("layer1", Tensor.Zeros(new int[] { 10, 10 }));
        });
    }

    // Mock ProcessGroup for testing
    private class MockProcessGroup : ProcessGroup
    {
        private readonly int _rank;
        private readonly int _worldSize;

        public MockProcessGroup(int rank, int worldSize) : base(null!, null!, rank, worldSize)
        {
            _rank = rank;
            _worldSize = worldSize;
        }

        public override int Rank => _rank;
        public override int WorldSize => _worldSize;

        public override void AllReduce(Tensor tensor, ReduceOp op = ReduceOp.Sum)
        {
            // No-op for mock
        }

        public override void Broadcast(Tensor tensor, int root = 0)
        {
            // No-op for mock
        }

        public override void Barrier()
        {
            // No-op for mock
        }

        public override Task AllReduceAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum)
        {
            return Task.CompletedTask;
        }

        public override Task BroadcastAsync(Tensor tensor, int root = 0)
        {
            return Task.CompletedTask;
        }

        public override Task BarrierAsync()
        {
            return Task.CompletedTask;
        }

        public override void Send(Tensor tensor, int dst)
        {
            // No-op for mock
        }

        public override void Recv(Tensor tensor, int src)
        {
            // No-op for mock
        }

        public override Task SendAsync(Tensor tensor, int dst)
        {
            return Task.CompletedTask;
        }

        public override Task RecvAsync(Tensor tensor, int src)
        {
            return Task.CompletedTask;
        }
    }
}
