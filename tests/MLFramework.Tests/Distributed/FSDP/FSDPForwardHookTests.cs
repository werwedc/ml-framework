using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Distributed.FSDP;

namespace MLFramework.Tests.Distributed.FSDP
{
    /// <summary>
    /// Unit tests for FSDPForwardHook.
    /// </summary>
    public class FSDPForwardHookTests
    {
        [Fact]
        public void Constructor_WithValidFSDP_CreatesForwardHook()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();

            // Act
            var forwardHook = new FSDPForwardHook(mockFSDP);

            // Assert
            Assert.NotNull(forwardHook);
        }

        [Fact]
        public void Constructor_WithNullFSDP_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new FSDPForwardHook(null!));
        }

        [Fact]
        public void RegisterHooks_WithEmptyShardingUnits_DoesNotThrow()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var emptyUnits = new Dictionary<string, FSDPShardingUnit>();

            // Act & Assert - Should not throw
            forwardHook.RegisterHooks(emptyUnits);
        }

        [Fact]
        public void RegisterHooks_WithNullShardingUnits_DoesNotThrow()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);

            // Act & Assert - Should not throw
            forwardHook.RegisterHooks(null!);
        }

        [Fact]
        public void RegisterHooks_WithSingleShardingUnit_RegistersHooks()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);

            // Act
            forwardHook.RegisterHooks(shardingUnits);

            // Assert
            Assert.True(forwardHook.HasPreForwardHook("param0"));
            Assert.True(forwardHook.HasPostForwardHook("param0"));
        }

        [Fact]
        public void RegisterHooks_WithMultipleShardingUnits_RegistersHooks()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(3, mockFSDP.ProcessGroup);

            // Act
            forwardHook.RegisterHooks(shardingUnits);

            // Assert - All parameters should have hooks registered
            for (int i = 0; i < 3; i++)
            {
                var paramName = $"param{i}";
                Assert.True(forwardHook.HasPreForwardHook(paramName));
                Assert.True(forwardHook.HasPostForwardHook(paramName));
            }
        }

        [Fact]
        public void GetPreForwardHook_WithRegisteredParameter_ReturnsHook()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);
            forwardHook.RegisterHooks(shardingUnits);

            // Act
            var hook = forwardHook.GetPreForwardHook("param0");

            // Assert
            Assert.NotNull(hook);
        }

        [Fact]
        public void GetPreForwardHook_WithUnregisteredParameter_ThrowsArgumentException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => forwardHook.GetPreForwardHook("nonexistent"));
        }

        [Fact]
        public void GetPostForwardHook_WithRegisteredParameter_ReturnsHook()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);
            forwardHook.RegisterHooks(shardingUnits);

            // Act
            var hook = forwardHook.GetPostForwardHook("param0");

            // Assert
            Assert.NotNull(hook);
        }

        [Fact]
        public void GetPostForwardHook_WithUnregisteredParameter_ThrowsArgumentException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => forwardHook.GetPostForwardHook("nonexistent"));
        }

        [Fact]
        public void GetGatherOperations_WithRegisteredUnits_ReturnsAllOperations()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(3, mockFSDP.ProcessGroup);
            forwardHook.RegisterHooks(shardingUnits);

            // Act
            var operations = forwardHook.GetGatherOperations();

            // Assert
            Assert.Equal(3, operations.Count);
            Assert.True(operations.ContainsKey("param0"));
            Assert.True(operations.ContainsKey("param1"));
            Assert.True(operations.ContainsKey("param2"));
        }

        [Fact]
        public void GetGatherOperations_WithNoUnits_ReturnsEmptyDictionary()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);

            // Act
            var operations = forwardHook.GetGatherOperations();

            // Assert
            Assert.NotNull(operations);
            Assert.Empty(operations);
        }

        [Fact]
        public async Task GatherMultipleAsync_WithSingleShardingUnit_GathersParameter()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);
            forwardHook.RegisterHooks(shardingUnits);

            // Act
            await forwardHook.GatherMultipleAsync(shardingUnits);

            // Assert - Parameter should be marked as gathered
            Assert.True(shardingUnits["param0"].State.IsGathered);
        }

        [Fact]
        public async Task GatherMultipleAsync_WithMultipleShardingUnits_GathersAllParameters()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(3, mockFSDP.ProcessGroup);
            forwardHook.RegisterHooks(shardingUnits);

            // Act
            await forwardHook.GatherMultipleAsync(shardingUnits);

            // Assert - All parameters should be marked as gathered
            foreach (var kvp in shardingUnits)
            {
                Assert.True(kvp.Value.State.IsGathered);
            }
        }

        [Fact]
        public async Task GatherMultipleAsync_WithNullShardingUnits_DoesNotThrow()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);

            // Act & Assert - Should not throw
            await forwardHook.GatherMultipleAsync(null!);
        }

        [Fact]
        public async Task GatherMultipleAsync_WithEmptyShardingUnits_DoesNotThrow()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var emptyUnits = new Dictionary<string, FSDPShardingUnit>();

            // Act & Assert - Should not throw
            await forwardHook.GatherMultipleAsync(emptyUnits);
        }

        [Fact]
        public void ReleaseMultiple_WithGatheredParameters_ReleasesAll()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(3, mockFSDP.ProcessGroup);
            forwardHook.RegisterHooks(shardingUnits);

            // Manually set gathered parameters
            foreach (var unit in shardingUnits.Values)
            {
                unit.GatheredParameter = Tensor.Zeros(unit.Shape, DataType.Float32);
                unit.State.IsGathered = true;
            }

            // Act
            forwardHook.ReleaseMultiple(shardingUnits);

            // Assert - All gathered parameters should be null
            foreach (var unit in shardingUnits.Values)
            {
                Assert.Null(unit.GatheredParameter);
                Assert.False(unit.State.IsGathered);
            }
        }

        [Fact]
        public void ReleaseMultiple_WithNullShardingUnits_DoesNotThrow()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);

            // Act & Assert - Should not throw
            forwardHook.ReleaseMultiple(null!);
        }

        [Fact]
        public void ReleaseMultiple_WithEmptyShardingUnits_DoesNotThrow()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var emptyUnits = new Dictionary<string, FSDPShardingUnit>();

            // Act & Assert - Should not throw
            forwardHook.ReleaseMultiple(emptyUnits);
        }

        [Fact]
        public void ReleaseMultiple_WithNoGatheredParameters_DoesNotThrow()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(2, mockFSDP.ProcessGroup);
            forwardHook.RegisterHooks(shardingUnits);

            // Act & Assert - Should not throw even if no parameters are gathered
            forwardHook.ReleaseMultiple(shardingUnits);
        }

        [Fact]
        public void ClearHooks_RemovesAllRegisteredHooks()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(2, mockFSDP.ProcessGroup);
            forwardHook.RegisterHooks(shardingUnits);

            // Act
            forwardHook.ClearHooks();

            // Assert - All hooks should be removed
            Assert.False(forwardHook.HasPreForwardHook("param0"));
            Assert.False(forwardHook.HasPostForwardHook("param0"));
            Assert.False(forwardHook.HasPreForwardHook("param1"));
            Assert.False(forwardHook.HasPostForwardHook("param1"));
        }

        [Fact]
        public void Dispose_CanBeCalledMultipleTimes()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(2, mockFSDP.ProcessGroup);
            forwardHook.RegisterHooks(shardingUnits);

            // Act & Assert - Should not throw
            forwardHook.Dispose();
            forwardHook.Dispose();
        }

        [Fact]
        public void HasPreForwardHook_WithRegisteredParameter_ReturnsTrue()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);
            forwardHook.RegisterHooks(shardingUnits);

            // Act
            var hasHook = forwardHook.HasPreForwardHook("param0");

            // Assert
            Assert.True(hasHook);
        }

        [Fact]
        public void HasPreForwardHook_WithUnregisteredParameter_ReturnsFalse()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);

            // Act
            var hasHook = forwardHook.HasPreForwardHook("nonexistent");

            // Assert
            Assert.False(hasHook);
        }

        [Fact]
        public void HasPostForwardHook_WithRegisteredParameter_ReturnsTrue()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);
            forwardHook.RegisterHooks(shardingUnits);

            // Act
            var hasHook = forwardHook.HasPostForwardHook("param0");

            // Assert
            Assert.True(hasHook);
        }

        [Fact]
        public void HasPostForwardHook_WithUnregisteredParameter_ReturnsFalse()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var forwardHook = new FSDPForwardHook(mockFSDP);

            // Act
            var hasHook = forwardHook.HasPostForwardHook("nonexistent");

            // Assert
            Assert.False(hasHook);
        }

        #region Helper Methods

        private FSDP CreateMockFSDP()
        {
            var mockModel = new MockModel();
            var config = new FSDPConfig();
            var processGroup = new MockProcessGroup(0, 2);
            return new FSDP(mockModel, config, processGroup);
        }

        private Dictionary<string, FSDPShardingUnit> CreateMockShardingUnits(int count, MLFramework.Distributed.IProcessGroup processGroup)
        {
            var units = new Dictionary<string, FSDPShardingUnit>();

            for (int i = 0; i < count; i++)
            {
                var paramName = $"param{i}";
                var data = Enumerable.Range(0, 8).Select(x => (float)(i + x)).ToArray();
                var shape = new[] { 8 };
                var fullParameter = new Tensor(data, shape, false, DataType.Float32);

                var unit = new FSDPShardingUnit(paramName, fullParameter, processGroup);
                units[paramName] = unit;
            }

            return units;
        }

        /// <summary>
        /// Mock implementation of IModel for testing.
        /// </summary>
        private class MockModel : IModel
        {
            public string Name => "MockModel";
        }

        /// <summary>
        /// Mock implementation of IProcessGroup for testing.
        /// </summary>
        private class MockProcessGroup : MLFramework.Distributed.IProcessGroup
        {
            public int Rank { get; private set; }
            public int WorldSize { get; private set; }
            public MLFramework.Distributed.ICommunicationBackend Backend { get; private set; }

            public MockProcessGroup(int rank, int worldSize)
            {
                Rank = rank;
                WorldSize = worldSize;
                Backend = new MockCommunicationBackend();
            }

            public void AllReduce(Tensor tensor, MLFramework.Distributed.ReduceOp op = MLFramework.Distributed.ReduceOp.Sum) { }
            public void Broadcast(Tensor tensor, int root = 0) { }
            public void Barrier() { }
            public System.Threading.Tasks.Task AllReduceAsync(Tensor tensor, MLFramework.Distributed.ReduceOp op = MLFramework.Distributed.ReduceOp.Sum) => System.Threading.Tasks.Task.CompletedTask;
            public System.Threading.Tasks.Task BroadcastAsync(Tensor tensor, int root = 0) => System.Threading.Tasks.Task.CompletedTask;
            public System.Threading.Tasks.Task BarrierAsync() => System.Threading.Tasks.Task.CompletedTask;
            public void Send(Tensor tensor, int dst) { }
            public void Recv(Tensor tensor, int src) { }
            public System.Threading.Tasks.Task SendAsync(Tensor tensor, int dst) => System.Threading.Tasks.Task.CompletedTask;
            public System.Threading.Tasks.Task RecvAsync(Tensor tensor, int src) => System.Threading.Tasks.Task.CompletedTask;
            public void Destroy() { }
            public void Dispose() { }
        }

        private class MockCommunicationBackend : MLFramework.Distributed.ICommunicationBackend
        {
            public string Name => "MockBackend";
            public bool IsAvailable => true;
            public int DeviceCount => 1;
            public bool SupportsAsync => true;
            public bool SupportsGPUDirect => false;
            public long GetBufferSizeLimit() => long.MaxValue;
        }

        #endregion
    }
}
