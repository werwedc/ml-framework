using System;
using Xunit;
using MLFramework.Communication.Optimizations;
using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Communication.Optimizations
{
    public class CommunicationOptimizerTests : IDisposable
    {
        private readonly MockCommunicationBackend _backend;
        private readonly CommunicationConfig _config;
        private readonly CommunicationOptimizer _optimizer;

        public CommunicationOptimizerTests()
        {
            _backend = new MockCommunicationBackend(4);
            _config = new CommunicationConfig
            {
                UsePinnedMemory = false,
                EnableLogging = false
            };
            _optimizer = new CommunicationOptimizer(_backend, _config);
        }

        [Fact]
        public void Constructor_NullBackend_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => new CommunicationOptimizer(null, _config));
        }

        [Fact]
        public void Constructor_NullConfig_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => new CommunicationOptimizer(_backend, null));
        }

        [Fact]
        public void Constructor_ValidParameters_CreatesInstance()
        {
            var optimizer = new CommunicationOptimizer(_backend, _config);
            Assert.NotNull(optimizer);
        }

        [Fact]
        public void Constructor_WithPinnedMemory_CreatesPinnedMemoryManager()
        {
            var configWithPinnedMemory = new CommunicationConfig
            {
                UsePinnedMemory = true,
                EnableLogging = false
            };

            using var optimizer = new CommunicationOptimizer(_backend, configWithPinnedMemory);
            Assert.NotNull(optimizer);
        }

        [Fact]
        public void AllReduceOptimized_ValidTensor_ReturnsResult()
        {
            var data = new float[100];
            var shape = new int[] { 10, 10 };
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = 1.0f;
            }
            var tensor = new Tensor(data, shape);

            var result = _optimizer.AllReduceOptimized(tensor, ReduceOp.Sum);

            Assert.NotNull(result);
            Assert.Equal(tensor.Shape, result.Shape);
        }

        [Fact]
        public void AllReduceOptimized_ProfilesOperation()
        {
            var data = new float[100];
            var shape = new int[] { 10, 10 };
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = 1.0f;
            }
            var tensor = new Tensor(data, shape);

            _optimizer.AllReduceOptimized(tensor, ReduceOp.Sum);

            var profiles = _optimizer.Profiler.Profiles;
            Assert.True(profiles.Count > 0);
            Assert.Equal("AllReduce", profiles[0].Operation);
        }

        [Fact]
        public void AllReduceOptimized_SelectsAlgorithm()
        {
            var data = new float[100];
            var shape = new int[] { 10, 10 };
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = 1.0f;
            }
            var tensor = new Tensor(data, shape);

            _optimizer.AllReduceOptimized(tensor, ReduceOp.Sum);

            var profiles = _optimizer.Profiler.Profiles;
            Assert.False(string.IsNullOrEmpty(profiles[0].Algorithm));
        }

        [Fact]
        public void AllReduceOptimizedAsync_ValidTensor_ReturnsHandle()
        {
            var data = new float[100];
            var shape = new int[] { 10, 10 };
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = 1.0f;
            }
            var tensor = new Tensor(data, shape);

            var handle = _optimizer.AllReduceOptimizedAsync(tensor, ReduceOp.Sum);

            Assert.NotNull(handle);
        }

        [Fact]
        public void Profiler_ReturnsProfiler()
        {
            var profiler = _optimizer.Profiler;
            Assert.NotNull(profiler);
        }

        [Fact]
        public void AlgorithmSelector_ReturnsAlgorithmSelector()
        {
            var selector = _optimizer.AlgorithmSelector;
            Assert.NotNull(selector);
        }

        [Fact]
        public void Dispose_CleanupResources()
        {
            var optimizer = new CommunicationOptimizer(_backend, _config);
            Assert.NotNull(optimizer);

            optimizer.Dispose();
            // Should not throw
        }

        public void Dispose()
        {
            _optimizer.Dispose();
        }
    }
}
