using System;
using System.Linq;
using Xunit;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Distributed.FSDP;

namespace MLFramework.Tests.Distributed.FSDP
{
    /// <summary>
    /// Mock implementation of IProcessGroup for testing.
    /// </summary>
    internal class MockProcessGroup : MLFramework.Distributed.IProcessGroup
    {
        public int Rank { get; private set; }
        public int WorldSize { get; private set; }
        public MLFramework.Distributed.ICommunicationBackend Backend { get; private set; }
        private bool _disposed;

        public MockProcessGroup(int rank, int worldSize)
        {
            Rank = rank;
            WorldSize = worldSize;
            Backend = new MockCommunicationBackend();
        }

        public void AllReduce(Tensor tensor, MLFramework.Distributed.ReduceOp op = MLFramework.Distributed.ReduceOp.Sum)
        {
            // Mock implementation - does nothing
        }

        public void Broadcast(Tensor tensor, int root = 0)
        {
            // Mock implementation - does nothing
        }

        public void Barrier()
        {
            // Mock implementation - does nothing
        }

        public System.Threading.Tasks.Task AllReduceAsync(Tensor tensor, MLFramework.Distributed.ReduceOp op = MLFramework.Distributed.ReduceOp.Sum)
        {
            return System.Threading.Tasks.Task.CompletedTask;
        }

        public System.Threading.Tasks.Task BroadcastAsync(Tensor tensor, int root = 0)
        {
            return System.Threading.Tasks.Task.CompletedTask;
        }

        public System.Threading.Tasks.Task BarrierAsync()
        {
            return System.Threading.Tasks.Task.CompletedTask;
        }

        public void Send(Tensor tensor, int dst)
        {
            // Mock implementation - does nothing
        }

        public void Recv(Tensor tensor, int src)
        {
            // Mock implementation - does nothing
        }

        public System.Threading.Tasks.Task SendAsync(Tensor tensor, int dst)
        {
            return System.Threading.Tasks.Task.CompletedTask;
        }

        public System.Threading.Tasks.Task RecvAsync(Tensor tensor, int src)
        {
            return System.Threading.Tasks.Task.CompletedTask;
        }

        public void Destroy()
        {
            // Mock implementation - does nothing
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Mock implementation of ICommunicationBackend for testing.
    /// </summary>
    internal class MockCommunicationBackend : MLFramework.Distributed.ICommunicationBackend
    {
        public string Name => "MockBackend";
        public bool IsAvailable => true;
        public int DeviceCount => 1;
        public bool SupportsAsync => true;
        public bool SupportsGPUDirect => false;

        public long GetBufferSizeLimit()
        {
            return long.MaxValue;
        }
    }

    /// <summary>
    /// Unit tests for FSDPShardingUnit.
    /// </summary>
    public class FSDPShardingUnitTests
    {
        [Fact]
        public void Constructor_WithValidParameters_CreatesShardingUnit()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape = new[] { 4 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);
            var processGroup = new MockProcessGroup(0, 2);

            // Act
            var shardingUnit = new FSDPShardingUnit("param1", fullParameter, processGroup);

            // Assert
            Assert.NotNull(shardingUnit);
            Assert.Equal("param1", shardingUnit.ParameterName);
            Assert.Equal(shape, shardingUnit.Shape);
            Assert.Equal(DataType.Float32, shardingUnit.DataType);
            Assert.NotNull(shardingUnit.ShardedParameter);
            Assert.Null(shardingUnit.GatheredParameter);
            Assert.Null(shardingUnit.LocalGradient);
            Assert.Null(shardingUnit.LocalOptimizerState);
            Assert.Equal(0, shardingUnit.State.OwnerRank);
            Assert.Equal(2, shardingUnit.State.NumShards);
            Assert.Equal(0, shardingUnit.State.ShardIndex);
            Assert.False(shardingUnit.State.IsGathered);
            Assert.False(shardingUnit.State.IsOffloaded);
        }

        [Fact]
        public void Constructor_WithEmptyParameterName_ThrowsArgumentException()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape = new[] { 4 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);
            var processGroup = new MockProcessGroup(0, 2);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new FSDPShardingUnit("", fullParameter, processGroup));
        }

        [Fact]
        public void Constructor_WithNullParameterName_ThrowsArgumentException()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape = new[] { 4 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);
            var processGroup = new MockProcessGroup(0, 2);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new FSDPShardingUnit(null!, fullParameter, processGroup));
        }

        [Fact]
        public void Constructor_WithNullProcessGroup_ThrowsArgumentNullException()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape = new[] { 4 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new FSDPShardingUnit("param1", fullParameter, null!));
        }

        [Fact]
        public void Constructor_ShardingAcrossTwoRanks_CreatesCorrectShards()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
            var shape = new[] { 8 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);
            var worldSize = 2;

            // Act - Create sharding unit for rank 0
            var shardingUnitRank0 = new FSDPShardingUnit("param1", fullParameter, new MockProcessGroup(0, worldSize));

            // Assert - Rank 0 should have first 4 elements
            Assert.NotNull(shardingUnitRank0.ShardedParameter);
            Assert.Equal(4, shardingUnitRank0.ShardedParameter.Size);
            Assert.Equal(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, shardingUnitRank0.ShardedParameter.Data);

            // Act - Create sharding unit for rank 1
            var shardingUnitRank1 = new FSDPShardingUnit("param1", fullParameter, new MockProcessGroup(1, worldSize));

            // Assert - Rank 1 should have last 4 elements
            Assert.NotNull(shardingUnitRank1.ShardedParameter);
            Assert.Equal(4, shardingUnitRank1.ShardedParameter.Size);
            Assert.Equal(new float[] { 5.0f, 6.0f, 7.0f, 8.0f }, shardingUnitRank1.ShardedParameter.Data);
        }

        [Fact]
        public void Constructor_ShardingAcrossThreeRanks_CreatesCorrectShards()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
            var shape = new[] { 8 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);
            var worldSize = 3;

            // Act - Create sharding units for all ranks
            var shardingUnitRank0 = new FSDPShardingUnit("param1", fullParameter, new MockProcessGroup(0, worldSize));
            var shardingUnitRank1 = new FSDPShardingUnit("param1", fullParameter, new MockProcessGroup(1, worldSize));
            var shardingUnitRank2 = new FSDPShardingUnit("param1", fullParameter, new MockProcessGroup(2, worldSize));

            // Assert - Rank 0 should have first 3 elements (ceil(8/3) = 3)
            Assert.Equal(3, shardingUnitRank0.ShardedParameter!.Size);
            Assert.Equal(new float[] { 1.0f, 2.0f, 3.0f }, shardingUnitRank0.ShardedParameter.Data);

            // Assert - Rank 1 should have next 3 elements
            Assert.Equal(3, shardingUnitRank1.ShardedParameter!.Size);
            Assert.Equal(new float[] { 4.0f, 5.0f, 6.0f }, shardingUnitRank1.ShardedParameter.Data);

            // Assert - Rank 2 should have last 2 elements
            Assert.Equal(2, shardingUnitRank2.ShardedParameter!.Size);
            Assert.Equal(new float[] { 7.0f, 8.0f }, shardingUnitRank2.ShardedParameter.Data);
        }

        [Fact]
        public void Constructor_ShardingWithSingleRank_CopiesAllData()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape = new[] { 4 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);

            // Act
            var shardingUnit = new FSDPShardingUnit("param1", fullParameter, new MockProcessGroup(0, 1));

            // Assert
            Assert.NotNull(shardingUnit.ShardedParameter);
            Assert.Equal(4, shardingUnit.ShardedParameter.Size);
            Assert.Equal(data, shardingUnit.ShardedParameter.Data);
        }

        [Fact]
        public void Constructor_WithMultiDimensionalTensor_CreatesCorrectShard()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
            var shape = new[] { 2, 4 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);

            // Act
            var shardingUnit = new FSDPShardingUnit("param1", fullParameter, new MockProcessGroup(0, 2));

            // Assert - Should shard across flattened tensor
            Assert.NotNull(shardingUnit.ShardedParameter);
            Assert.Equal(4, shardingUnit.ShardedParameter.Size);
            Assert.Equal(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, shardingUnit.ShardedParameter.Data);
            Assert.Equal(shape, shardingUnit.Shape);
        }

        [Fact]
        public void ReleaseGatheredParameters_WhenGatheredParameterExists_DisposesAndSetsToNull()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape = new[] { 4 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);
            var processGroup = new MockProcessGroup(0, 1);
            var shardingUnit = new FSDPShardingUnit("param1", fullParameter, processGroup);

            // Create a gathered parameter manually for testing
            shardingUnit.GatheredParameter = Tensor.Zeros(shape, DataType.Float32);

            // Act
            shardingUnit.ReleaseGatheredParameters();

            // Assert
            Assert.Null(shardingUnit.GatheredParameter);
            Assert.False(shardingUnit.State.IsGathered);
        }

        [Fact]
        public void ReleaseGatheredParameters_WhenNoGatheredParameter_DoesNotThrow()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape = new[] { 4 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);
            var processGroup = new MockProcessGroup(0, 1);
            var shardingUnit = new FSDPShardingUnit("param1", fullParameter, processGroup);

            // Act & Assert - Should not throw
            shardingUnit.ReleaseGatheredParameters();
            Assert.Null(shardingUnit.GatheredParameter);
        }

        [Fact]
        public void GatherParameters_ThrowsNotImplementedException()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape = new[] { 4 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);
            var processGroup = new MockProcessGroup(0, 2);
            var shardingUnit = new FSDPShardingUnit("param1", fullParameter, processGroup);

            // Act & Assert
            Assert.Throws<NotImplementedException>(() => shardingUnit.GatherParameters());
        }

        [Fact]
        public void ScatterGradients_WithNoGradient_ThrowsInvalidOperationException()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape = new[] { 4 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);
            var processGroup = new MockProcessGroup(0, 2);
            var shardingUnit = new FSDPShardingUnit("param1", fullParameter, processGroup);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => shardingUnit.ScatterGradients());
        }

        [Fact]
        public void ScatterGradients_WithGradientSet_ThrowsNotImplementedException()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape = new[] { 4 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);
            var processGroup = new MockProcessGroup(0, 2);
            var shardingUnit = new FSDPShardingUnit("param1", fullParameter, processGroup);
            shardingUnit.LocalGradient = Tensor.Zeros(new[] { 2 }, DataType.Float32);

            // Act & Assert
            Assert.Throws<NotImplementedException>(() => shardingUnit.ScatterGradients());
        }

        [Fact]
        public void Dispose_CleansUpResources()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape = new[] { 4 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);
            var processGroup = new MockProcessGroup(0, 1);
            var shardingUnit = new FSDPShardingUnit("param1", fullParameter, processGroup);

            // Act
            shardingUnit.Dispose();

            // Assert - After dispose, all tensors should be null
            Assert.Null(shardingUnit.ShardedParameter);
            Assert.Null(shardingUnit.GatheredParameter);
            Assert.Null(shardingUnit.LocalGradient);
        }

        [Fact]
        public void Dispose_CanBeCalledMultipleTimes()
        {
            // Arrange
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape = new[] { 4 };
            var fullParameter = new Tensor(data, shape, false, DataType.Float32);
            var processGroup = new MockProcessGroup(0, 1);
            var shardingUnit = new FSDPShardingUnit("param1", fullParameter, processGroup);

            // Act & Assert - Should not throw
            shardingUnit.Dispose();
            shardingUnit.Dispose();
        }
    }
}
