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
    /// Unit tests for FSDPBackwardHook.
    /// </summary>
    public class FSDPBackwardHookTests
    {
        [Fact]
        public void Constructor_WithValidFSDP_CreatesBackwardHook()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();

            // Act
            var backwardHook = new FSDPBackwardHook(mockFSDP);

            // Assert
            Assert.NotNull(backwardHook);
        }

        [Fact]
        public void Constructor_WithNullFSDP_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new FSDPBackwardHook(null!));
        }

        [Fact]
        public void RegisterHooks_WithEmptyShardingUnits_DoesNotThrow()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var mockModel = new MockModel();
            var emptyUnits = new Dictionary<string, FSDPShardingUnit>();

            // Act & Assert - Should not throw
            backwardHook.RegisterHooks(mockModel, emptyUnits);
        }

        [Fact]
        public void RegisterHooks_WithNullShardingUnits_DoesNotThrow()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var mockModel = new MockModel();

            // Act & Assert - Should not throw
            backwardHook.RegisterHooks(mockModel, null!);
        }

        [Fact]
        public void RegisterHooks_WithNullModel_ThrowsArgumentNullException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => backwardHook.RegisterHooks(null!, shardingUnits));
        }

        [Fact]
        public void RegisterHooks_WithSingleShardingUnit_RegistersHooks()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var mockModel = new MockModel();
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);

            // Act
            backwardHook.RegisterHooks(mockModel, shardingUnits);

            // Assert
            Assert.True(backwardHook.HasBackwardHook("param0"));
        }

        [Fact]
        public void RegisterHooks_WithMultipleShardingUnits_RegistersHooks()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var mockModel = new MockModel();
            var shardingUnits = CreateMockShardingUnits(3, mockFSDP.ProcessGroup);

            // Act
            backwardHook.RegisterHooks(mockModel, shardingUnits);

            // Assert - All parameters should have hooks registered
            for (int i = 0; i < 3; i++)
            {
                var paramName = $"param{i}";
                Assert.True(backwardHook.HasBackwardHook(paramName));
            }
        }

        [Fact]
        public void GetBackwardHook_WithRegisteredParameter_ReturnsHook()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var mockModel = new MockModel();
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);
            backwardHook.RegisterHooks(mockModel, shardingUnits);

            // Act
            var hook = backwardHook.GetBackwardHook("param0");

            // Assert
            Assert.NotNull(hook);
        }

        [Fact]
        public void GetBackwardHook_WithUnregisteredParameter_ReturnsNull()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var mockModel = new MockModel();
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);
            backwardHook.RegisterHooks(mockModel, shardingUnits);

            // Act
            var hook = backwardHook.GetBackwardHook("nonexistent");

            // Assert
            Assert.Null(hook);
        }

        [Fact]
        public void GetScatterOperations_WithRegisteredUnits_ReturnsAllOperations()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var mockModel = new MockModel();
            var shardingUnits = CreateMockShardingUnits(3, mockFSDP.ProcessGroup);
            backwardHook.RegisterHooks(mockModel, shardingUnits);

            // Act
            var operations = backwardHook.GetScatterOperations();

            // Assert
            Assert.Equal(3, operations.Count);
            Assert.True(operations.ContainsKey("param0"));
            Assert.True(operations.ContainsKey("param1"));
            Assert.True(operations.ContainsKey("param2"));
        }

        [Fact]
        public void GetScatterOperations_WithNoUnits_ReturnsEmptyDictionary()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);

            // Act
            var operations = backwardHook.GetScatterOperations();

            // Assert
            Assert.NotNull(operations);
            Assert.Empty(operations);
        }

        [Fact]
        public void HasBackwardHook_WithRegisteredParameter_ReturnsTrue()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var mockModel = new MockModel();
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);
            backwardHook.RegisterHooks(mockModel, shardingUnits);

            // Act
            var hasHook = backwardHook.HasBackwardHook("param0");

            // Assert
            Assert.True(hasHook);
        }

        [Fact]
        public void HasBackwardHook_WithUnregisteredParameter_ReturnsFalse()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);

            // Act
            var hasHook = backwardHook.HasBackwardHook("nonexistent");

            // Assert
            Assert.False(hasHook);
        }

        [Fact]
        public void AccumulateGradients_WithNullShardingUnits_DoesNotThrow()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);

            // Act & Assert - Should not throw
            backwardHook.AccumulateGradients(null!, new Dictionary<string, Tensor>());
        }

        [Fact]
        public void AccumulateGradients_WithNullGradients_DoesNotThrow()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);

            // Act & Assert - Should not throw
            backwardHook.AccumulateGradients(shardingUnits, null!);
        }

        [Fact]
        public void AccumulateGradients_WithFirstGradient_AssignsToLocalGradient()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);
            var newGradients = new Dictionary<string, Tensor>();
            var gradData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            newGradients["param0"] = new Tensor(gradData, new[] { 4 }, false, DataType.Float32);

            // Act
            backwardHook.AccumulateGradients(shardingUnits, newGradients);

            // Assert
            Assert.NotNull(shardingUnits["param0"].LocalGradient);
            Assert.Equal(4, shardingUnits["param0"].LocalGradient.Size);
            Assert.Equal(1.0f, shardingUnits["param0"].LocalGradient.Data[0]);
        }

        [Fact]
        public void AccumulateGradients_WithMultipleGradients_AccumulatesCorrectly()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);

            // First gradient
            var firstGrad = new Dictionary<string, Tensor>();
            firstGrad["param0"] = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32);
            backwardHook.AccumulateGradients(shardingUnits, firstGrad);

            // Second gradient
            var secondGrad = new Dictionary<string, Tensor>();
            secondGrad["param0"] = new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 }, false, DataType.Float32);

            // Act
            backwardHook.AccumulateGradients(shardingUnits, secondGrad);

            // Assert
            Assert.NotNull(shardingUnits["param0"].LocalGradient);
            Assert.Equal(4.0f, shardingUnits["param0"].LocalGradient.Data[0]); // 1.0 + 3.0
            Assert.Equal(6.0f, shardingUnits["param0"].LocalGradient.Data[1]); // 2.0 + 4.0
        }

        [Fact]
        public void ClearGradients_WithNullShardingUnits_DoesNotThrow()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);

            // Act & Assert - Should not throw
            backwardHook.ClearGradients(null!);
        }

        [Fact]
        public void ClearGradients_WithGradients_ZerosOutGradients()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);
            var newGradients = new Dictionary<string, Tensor>();
            newGradients["param0"] = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, false, DataType.Float32);
            backwardHook.AccumulateGradients(shardingUnits, newGradients);

            // Act
            backwardHook.ClearGradients(shardingUnits);

            // Assert
            Assert.NotNull(shardingUnits["param0"].LocalGradient);
            Assert.Equal(0.0f, shardingUnits["param0"].LocalGradient.Data[0]);
            Assert.Equal(0.0f, shardingUnits["param0"].LocalGradient.Data[1]);
            Assert.Equal(0.0f, shardingUnits["param0"].LocalGradient.Data[2]);
        }

        [Fact]
        public void VerifyGradients_WithNullShardingUnits_ReturnsTrue()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);

            // Act
            var result = backwardHook.VerifyGradients(null!, new Dictionary<string, Tensor>());

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void VerifyGradients_WithNullFullGradients_ReturnsTrue()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);

            // Act
            var result = backwardHook.VerifyGradients(shardingUnits, null!);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void VerifyGradients_WithNoLocalGradient_ReturnsFalse()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);
            var fullGradients = new Dictionary<string, Tensor>();
            fullGradients["param0"] = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, false, DataType.Float32);

            // Act
            var result = backwardHook.VerifyGradients(shardingUnits, fullGradients);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void VerifyGradients_WithCorrectGradients_ReturnsTrue()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var shardingUnits = CreateMockShardingUnits(1, mockFSDP.ProcessGroup);

            // Set up local gradient (shard)
            var shardData = new float[] { 1.0f, 2.0f };
            shardingUnits["param0"].LocalGradient = new Tensor(shardData, new[] { 2 }, false, DataType.Float32);

            // Set up full gradient
            var fullGradients = new Dictionary<string, Tensor>();
            fullGradients["param0"] = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 }, false, DataType.Float32);

            // Act
            var result = backwardHook.VerifyGradients(shardingUnits, fullGradients);

            // Assert - Should return true for single device (shard index 0, world size 2)
            Assert.True(result);
        }

        [Fact]
        public void ClearHooks_RemovesAllRegisteredHooks()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var mockModel = new MockModel();
            var shardingUnits = CreateMockShardingUnits(2, mockFSDP.ProcessGroup);
            backwardHook.RegisterHooks(mockModel, shardingUnits);

            // Act
            backwardHook.ClearHooks();

            // Assert - All hooks should be removed
            Assert.False(backwardHook.HasBackwardHook("param0"));
            Assert.False(backwardHook.HasBackwardHook("param1"));
        }

        [Fact]
        public void Dispose_CanBeCalledMultipleTimes()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var backwardHook = new FSDPBackwardHook(mockFSDP);
            var mockModel = new MockModel();
            var shardingUnits = CreateMockShardingUnits(2, mockFSDP.ProcessGroup);
            backwardHook.RegisterHooks(mockModel, shardingUnits);

            // Act & Assert - Should not throw
            backwardHook.Dispose();
            backwardHook.Dispose();
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
            public Tensor Forward(Tensor input) => input;
            public void Backward() { }
            public List<RitterFramework.Core.Tensor.NamedTensor> GetParameters() => new();
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
