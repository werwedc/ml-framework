using MLFramework.Checkpointing.Distributed;
using MLFramework.Distributed;
using RitterFramework.Core.Tensor;
using System.Runtime.CompilerServices;

namespace MLFramework.Tests.Checkpointing.Distributed;

/// <summary>
/// Tests for DistributedCommunication
/// </summary>
public class DistributedCommunicationTests
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
        var communication = new DistributedCommunication(processGroup);

        // Assert
        Assert.NotNull(communication);
    }

    [Fact]
    public void Constructor_NullProcessGroup_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
        {
            new DistributedCommunication(null!);
        });
    }

    [Fact]
    public void Broadcast_CallsProcessGroupBroadcast()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });

        // Act & Assert - Should not throw
        communication.Broadcast(tensor, 0);
    }

    [Fact]
    public void Broadcast_AfterDispose_ThrowsObjectDisposedException()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);
        communication.Dispose();

        var tensor = Tensor.Zeros(new int[] { 10, 10 });

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() =>
        {
            communication.Broadcast(tensor, 0);
        });
    }

    [Fact]
    public void Send_CallsProcessGroupSend()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);
        var tensor = Tensor.Zeros(new int[] { 10, 10 });

        // Act & Assert - Should not throw
        communication.Send(tensor, 1, tag: 0);
    }

    [Fact]
    public void Receive_CallsProcessGroupReceive()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);

        // Act
        var received = communication.Receive(1, tag: 0);

        // Assert
        Assert.NotNull(received);
    }

    [Fact]
    public void ReceiveTensor_UsesCorrectTag()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);

        // Act
        var received = communication.ReceiveTensor(1, "layer1");

        // Assert
        Assert.NotNull(received);
    }

    [Fact]
    public void AllGather_ReturnsCorrectSize()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);

        // Act
        var gathered = communication.AllGather("test_data");

        // Assert
        Assert.Equal(4, gathered.Count);
    }

    [Fact]
    public void AllGather_WithInt_ReturnsIntegers()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);

        // Act
        var gathered = communication.AllGather(42);

        // Assert
        Assert.Equal(4, gathered.Count);
        Assert.Equal(42, gathered[0]);
    }

    [Fact]
    public void AllGather_WithMemoryStats_ReturnsStats()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);
        var stats = new MLFramework.Checkpointing.MemoryStats
        {
            CurrentMemoryUsed = 1024,
            CheckpointCount = 5
        };

        // Act
        var gathered = communication.AllGather(stats);

        // Assert
        Assert.Equal(4, gathered.Count);
        Assert.NotNull(gathered[0]);
    }

    [Fact]
    public void Barrier_CallsProcessGroupBarrier()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);

        // Act & Assert - Should not throw
        communication.Barrier();
    }

    [Fact]
    public void Reduce_WithInt_ReturnsInt()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);

        // Act
        var reduced = communication.Reduce(42);

        // Assert
        Assert.Equal(42, reduced);
    }

    [Fact]
    public void Reduce_WithLong_ReturnsLong()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);

        // Act
        var reduced = communication.Reduce(42L);

        // Assert
        Assert.Equal(42L, reduced);
    }

    [Fact]
    public void Reduce_WithFloat_ReturnsFloat()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);

        // Act
        var reduced = communication.Reduce(42.0f);

        // Assert
        Assert.Equal(42.0f, reduced);
    }

    [Fact]
    public void Reduce_WithDouble_ReturnsDouble()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);

        // Act
        var reduced = communication.Reduce(42.0);

        // Assert
        Assert.Equal(42.0, reduced);
    }

    [Fact]
    public void AllReduce_CallsReduce()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);

        // Act
        var reduced = communication.AllReduce(42);

        // Assert
        Assert.Equal(42, reduced);
    }

    [Fact]
    public void Dispose_AfterDispose_ThrowsObjectDisposedException()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);
        communication.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() =>
        {
            communication.AllGather("test");
        });
    }

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var processGroup = CreateMockProcessGroup(0, 4);
        var communication = new DistributedCommunication(processGroup);

        // Act & Assert - Should not throw
        communication.Dispose();
        communication.Dispose();
        Assert.True(true);
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
