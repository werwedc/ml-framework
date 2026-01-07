using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.Pipeline;

namespace MLFramework.Tests.Pipeline
{
    public class PipelineOptimizerTests : IDisposable
    {
        private readonly IDevice _device;

        public PipelineOptimizerTests()
        {
            _device = DeviceManager.DefaultDevice;
        }

        public void Dispose()
        {
        }

        private PipelineOptimizer CreateOptimizer(int numStages)
        {
            var stages = new List<PipelineStage>();
            for (int i = 0; i < numStages; i++)
            {
                var module = new Linear(10, 10, $"linear_{i}");
                var stage = new PipelineStage(module, i, numStages, _device);
                stages.Add(stage);
            }

            var communicator = new LocalPipelineCommunicator(0, numStages);
            return new PipelineOptimizer(stages, communicator, learningRate: 0.001f);
        }

        [Fact]
        public void Constructor_WithValidInputs_CreatesOptimizer()
        {
            // Act
            var optimizer = CreateOptimizer(2);

            // Assert
            Assert.NotNull(optimizer);
            Assert.Equal(2, optimizer.NumStages);
            Assert.Equal(0.001f, optimizer.LearningRate);
        }

        [Fact]
        public async Task StepAsync_ExecutesSuccessfully()
        {
            // Arrange
            var optimizer = CreateOptimizer(2);

            // Act & Assert - Should not throw
            await optimizer.StepAsync();
        }

        [Fact]
        public async Task SynchronizeGradientsAsync_CompletesSuccessfully()
        {
            // Arrange
            var optimizer = CreateOptimizer(2);

            // Act & Assert - Should not throw
            await optimizer.SynchronizeGradientsAsync();
        }

        [Fact]
        public async Task BroadcastParametersAsync_WithValidRoot_ExecutesSuccessfully()
        {
            // Arrange
            var optimizer = CreateOptimizer(4);

            // Act & Assert - Should not throw
            await optimizer.BroadcastParametersAsync(rootStage: 0);
        }

        [Fact]
        public void BroadcastParametersAsync_WithInvalidRoot_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var optimizer = CreateOptimizer(2);

            // Act & Assert
            Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
                optimizer.BroadcastParametersAsync(2)); // Invalid root
        }

        [Fact]
        public void ZeroGradients_ClearsGradients()
        {
            // Arrange
            var optimizer = CreateOptimizer(2);

            // Act & Assert - Should not throw
            optimizer.ZeroGradients();
        }

        [Fact]
        public void LearningRate_CanBeSetAndGet()
        {
            // Arrange
            var optimizer = CreateOptimizer(2);

            // Act
            optimizer.LearningRate = 0.01f;

            // Assert
            Assert.Equal(0.01f, optimizer.LearningRate);
        }

        [Fact]
        public void LearningRate_WithZeroValue_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var optimizer = CreateOptimizer(2);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                optimizer.LearningRate = 0.0f);
        }

        [Fact]
        public void LearningRate_WithNegativeValue_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var optimizer = CreateOptimizer(2);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                optimizer.LearningRate = -0.001f);
        }

        [Fact]
        public void SetLearningRate_WithValidValue_SetsCorrectly()
        {
            // Arrange
            var optimizer = CreateOptimizer(2);

            // Act
            optimizer.SetLearningRate(0.01f);

            // Assert
            Assert.Equal(0.01f, optimizer.GetLearningRate());
        }

        [Fact]
        public void SetLearningRate_WithZeroValue_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var optimizer = CreateOptimizer(2);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                optimizer.SetLearningRate(0.0f));
        }

        [Fact]
        public void GetLearningRate_ReturnsCorrectValue()
        {
            // Arrange
            var optimizer = CreateOptimizer(2);
            var expectedLR = 0.01f;
            optimizer.LearningRate = expectedLR;

            // Act
            var actualLR = optimizer.GetLearningRate();

            // Assert
            Assert.Equal(expectedLR, actualLR);
        }

        [Fact]
        public void Dispose_CanBeCalledMultipleTimes()
        {
            // Arrange
            var optimizer = CreateOptimizer(2);

            // Act - Should not throw
            optimizer.Dispose();
            optimizer.Dispose();

            // Assert - No exception
        }
    }
}
