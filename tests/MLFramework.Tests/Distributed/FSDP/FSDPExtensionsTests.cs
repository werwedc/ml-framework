using System;
using System.Collections.Generic;
using Xunit;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Distributed.FSDP;

namespace MLFramework.Tests.Distributed.FSDP
{
    /// <summary>
    /// Unit tests for FSDPExtensions.
    /// </summary>
    public class FSDPExtensionsTests
    {
        [Fact]
        public void ReplaceParameter_WithNullFSDP_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => FSDPExtensions.ReplaceParameter(null!, "param1", Tensor.Zeros(new[] { 4 }, DataType.Float32)));
        }

        [Fact]
        public void ReplaceParameter_WithEmptyParameterName_ThrowsArgumentException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var tensor = Tensor.Zeros(new[] { 4 }, DataType.Float32);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => mockFSDP.ReplaceParameter("", tensor));
        }

        [Fact]
        public void ReplaceParameter_WithNullParameterName_ThrowsArgumentException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var tensor = Tensor.Zeros(new[] { 4 }, DataType.Float32);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => mockFSDP.ReplaceParameter(null!, tensor));
        }

        [Fact]
        public void ReplaceParameter_WithNullTensor_ThrowsArgumentNullException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => mockFSDP.ReplaceParameter("param1", null!));
        }

        [Fact]
        public void ReplaceParameter_ThrowsNotImplementedException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();
            var tensor = Tensor.Zeros(new[] { 4 }, DataType.Float32);

            // Act & Assert
            Assert.Throws<NotImplementedException>(() => mockFSDP.ReplaceParameter("param1", tensor));
        }

        [Fact]
        public void GetParameter_WithNullFSDP_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => FSDPExtensions.GetParameter(null!, "param1"));
        }

        [Fact]
        public void GetParameter_WithEmptyParameterName_ThrowsArgumentException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => mockFSDP.GetParameter(""));
        }

        [Fact]
        public void GetParameter_WithNullParameterName_ThrowsArgumentException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => mockFSDP.GetParameter(null!));
        }

        [Fact]
        public void GetParameter_WithExistingParameter_ReturnsShardedParameter()
        {
            // Arrange
            var mockFSDP = CreateMockFSDPWithShardingUnits(1);
            var paramName = "param0";

            // Act
            var param = mockFSDP.GetParameter(paramName);

            // Assert
            Assert.NotNull(param);
            Assert.Equal(8, param.Size);
        }

        [Fact]
        public void GetParameter_WithNonexistentParameter_ThrowsArgumentException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDPWithShardingUnits(1);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => mockFSDP.GetParameter("nonexistent"));
        }

        [Fact]
        public void GetParameter_WithGatheredParameter_ReturnsGatheredParameter()
        {
            // Arrange
            var mockFSDP = CreateMockFSDPWithShardingUnits(1);
            var paramName = "param0";

            // Manually set gathered parameter
            var shardingUnits = (List<FSDPShardingUnit>)mockFSDP.GetShardingUnits();
            shardingUnits[0].GatheredParameter = Tensor.Zeros(new[] { 8 }, DataType.Float32);
            shardingUnits[0].State.IsGathered = true;

            // Act
            var param = mockFSDP.GetParameter(paramName);

            // Assert
            Assert.NotNull(param);
            Assert.True(shardingUnits[0].State.IsGathered);
        }

        [Fact]
        public void GetAllParameters_WithNullFSDP_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => FSDPExtensions.GetAllParameters(null!));
        }

        [Fact]
        public void GetAllParameters_WithMultipleParameters_ReturnsAllParameters()
        {
            // Arrange
            var mockFSDP = CreateMockFSDPWithShardingUnits(3);

            // Act
            var parameters = mockFSDP.GetAllParameters();

            // Assert
            Assert.NotNull(parameters);
            Assert.Equal(3, parameters.Count);
            Assert.True(parameters.ContainsKey("param0"));
            Assert.True(parameters.ContainsKey("param1"));
            Assert.True(parameters.ContainsKey("param2"));
        }

        [Fact]
        public void GetAllParameters_WithNoParameters_ReturnsEmptyDictionary()
        {
            // Arrange - Create FSDP with no sharding units
            var mockModel = new MockModel();
            var config = new FSDPConfig();
            var processGroup = new MockProcessGroup(0, 2);
            var mockFSDP = new FSDP(mockModel, config, processGroup);

            // Act
            var parameters = mockFSDP.GetAllParameters();

            // Assert
            Assert.NotNull(parameters);
            Assert.Empty(parameters);
        }

        [Fact]
        public void GetAllParameters_WithGatheredParameters_ReturnsGatheredParameters()
        {
            // Arrange
            var mockFSDP = CreateMockFSDPWithShardingUnits(2);

            // Manually set gathered parameters
            var shardingUnits = (List<FSDPShardingUnit>)mockFSDP.GetShardingUnits();
            shardingUnits[0].GatheredParameter = Tensor.Zeros(new[] { 8 }, DataType.Float32);
            shardingUnits[0].State.IsGathered = true;
            shardingUnits[1].GatheredParameter = Tensor.Zeros(new[] { 8 }, DataType.Float32);
            shardingUnits[1].State.IsGathered = true;

            // Act
            var parameters = mockFSDP.GetAllParameters();

            // Assert
            Assert.Equal(2, parameters.Count);
            Assert.NotNull(parameters["param0"]);
            Assert.NotNull(parameters["param1"]);
        }

        [Fact]
        public void IsParameterGathered_WithNullFSDP_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => FSDPExtensions.IsParameterGathered(null!, "param1"));
        }

        [Fact]
        public void IsParameterGathered_WithEmptyParameterName_ThrowsArgumentException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => mockFSDP.IsParameterGathered(""));
        }

        [Fact]
        public void IsParameterGathered_WithNullParameterName_ThrowsArgumentException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => mockFSDP.IsParameterGathered(null!));
        }

        [Fact]
        public void IsParameterGathered_WithGatheredParameter_ReturnsTrue()
        {
            // Arrange
            var mockFSDP = CreateMockFSDPWithShardingUnits(1);
            var paramName = "param0";

            // Manually set gathered parameter
            var shardingUnits = (List<FSDPShardingUnit>)mockFSDP.GetShardingUnits();
            shardingUnits[0].GatheredParameter = Tensor.Zeros(new[] { 8 }, DataType.Float32);
            shardingUnits[0].State.IsGathered = true;

            // Act
            var isGathered = mockFSDP.IsParameterGathered(paramName);

            // Assert
            Assert.True(isGathered);
        }

        [Fact]
        public void IsParameterGathered_WithUngatheredParameter_ReturnsFalse()
        {
            // Arrange
            var mockFSDP = CreateMockFSDPWithShardingUnits(1);
            var paramName = "param0";

            // Act
            var isGathered = mockFSDP.IsParameterGathered(paramName);

            // Assert
            Assert.False(isGathered);
        }

        [Fact]
        public void IsParameterGathered_WithNonexistentParameter_ReturnsFalse()
        {
            // Arrange
            var mockFSDP = CreateMockFSDPWithShardingUnits(1);

            // Act
            var isGathered = mockFSDP.IsParameterGathered("nonexistent");

            // Assert
            Assert.False(isGathered);
        }

        [Fact]
        public void IsParameterOffloaded_WithNullFSDP_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => FSDPExtensions.IsParameterOffloaded(null!, "param1"));
        }

        [Fact]
        public void IsParameterOffloaded_WithEmptyParameterName_ThrowsArgumentException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => mockFSDP.IsParameterOffloaded(""));
        }

        [Fact]
        public void IsParameterOffloaded_WithNullParameterName_ThrowsArgumentException()
        {
            // Arrange
            var mockFSDP = CreateMockFSDP();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => mockFSDP.IsParameterOffloaded(null!));
        }

        [Fact]
        public void IsParameterOffloaded_WithOffloadedParameter_ReturnsTrue()
        {
            // Arrange
            var mockFSDP = CreateMockFSDPWithShardingUnits(1);
            var paramName = "param0";

            // Manually set offloaded state
            var shardingUnits = (List<FSDPShardingUnit>)mockFSDP.GetShardingUnits();
            shardingUnits[0].State.IsOffloaded = true;

            // Act
            var isOffloaded = mockFSDP.IsParameterOffloaded(paramName);

            // Assert
            Assert.True(isOffloaded);
        }

        [Fact]
        public void IsParameterOffloaded_WithNotOffloadedParameter_ReturnsFalse()
        {
            // Arrange
            var mockFSDP = CreateMockFSDPWithShardingUnits(1);
            var paramName = "param0";

            // Act
            var isOffloaded = mockFSDP.IsParameterOffloaded(paramName);

            // Assert
            Assert.False(isOffloaded);
        }

        [Fact]
        public void IsParameterOffloaded_WithNonexistentParameter_ReturnsFalse()
        {
            // Arrange
            var mockFSDP = CreateMockFSDPWithShardingUnits(1);

            // Act
            var isOffloaded = mockFSDP.IsParameterOffloaded("nonexistent");

            // Assert
            Assert.False(isOffloaded);
        }

        #region Helper Methods

        private FSDP CreateMockFSDP()
        {
            var mockModel = new MockModel();
            var config = new FSDPConfig();
            var processGroup = new MockProcessGroup(0, 2);
            return new FSDP(mockModel, config, processGroup);
        }

        private FSDP CreateMockFSDPWithShardingUnits(int count)
        {
            var mockModel = new MockModel();
            var config = new FSDPConfig();
            var processGroup = new MockProcessGroup(0, 2);
            var fsdp = new FSDP(mockModel, config, processGroup);

            // Create mock sharding units
            var shardingUnits = new List<FSDPShardingUnit>();
            for (int i = 0; i < count; i++)
            {
                var paramName = $"param{i}";
                var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
                var shape = new[] { 8 };
                var fullParameter = new Tensor(data, shape, false, DataType.Float32);

                var unit = new FSDPShardingUnit(paramName, fullParameter, processGroup);
                shardingUnits.Add(unit);
            }

            // Use reflection to set private _shardingUnits field
            var fieldInfo = typeof(FSDP).GetField("_shardingUnits",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            fieldInfo.SetValue(fsdp, shardingUnits);

            return fsdp;
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
