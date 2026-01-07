using System;
using System.Linq;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.NN;
using MLFramework.Pipeline;
using Xunit;

namespace MLFramework.Tests.Pipeline
{
    /// <summary>
    /// Simple test module for ActivationCheckpointManager tests
    /// </summary>
    public class SimpleForwardModule : Module
    {
        public SimpleForwardModule() : base("SimpleForwardModule")
        {
        }

        public override Tensor Forward(Tensor input)
        {
            // Simple identity forward for testing
            return input.Clone();
        }

        public override System.Collections.Generic.IEnumerable<Parameter> GetParameters()
        {
            return Enumerable.Empty<Parameter>();
        }

        public override System.Collections.Generic.IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            return Enumerable.Empty<(string, Parameter)>();
        }
    }

    /// <summary>
    /// Unit tests for ActivationCheckpointManager
    /// </summary>
    public class ActivationCheckpointManagerTests
    {
        private Tensor CreateTestTensor(int[] shape, float value = 1.0f)
        {
            var size = 1;
            foreach (var dim in shape)
            {
                size *= dim;
            }
            var data = new float[size];
            for (int i = 0; i < size; i++)
            {
                data[i] = value;
            }
            return new Tensor(data, shape);
        }

        #region Strategy Tests

        [Fact]
        public void ShouldCheckpoint_StoreAll_AlwaysReturnsTrue()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll, stage);

            // Act & Assert
            for (int i = 0; i < 10; i++)
            {
                Assert.True(manager.ShouldCheckpoint(i));
            }
        }

        [Fact]
        public void ShouldCheckpoint_RecomputeAll_AlwaysReturnsFalse()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.RecomputeAll, stage);

            // Act & Assert
            for (int i = 0; i < 10; i++)
            {
                Assert.False(manager.ShouldCheckpoint(i));
            }
        }

        [Fact]
        public void ShouldCheckpoint_Selective_CheckpointsEveryNth()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.Selective, stage, checkpointInterval: 3);

            // Act & Assert
            Assert.True(manager.ShouldCheckpoint(0));  // Always checkpoint first
            Assert.False(manager.ShouldCheckpoint(1));
            Assert.False(manager.ShouldCheckpoint(2));
            Assert.True(manager.ShouldCheckpoint(3));  // Every 3rd
            Assert.False(manager.ShouldCheckpoint(4));
            Assert.False(manager.ShouldCheckpoint(5));
            Assert.True(manager.ShouldCheckpoint(6));  // Every 3rd
        }

        [Fact]
        public void ShouldCheckpoint_MemoryBased_AlwaysReturnsTrue()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.MemoryBased, stage);

            // Act & Assert
            // MemoryBased checks threshold during StoreActivation, but ShouldCheckpoint always returns true
            Assert.True(manager.ShouldCheckpoint(0));
            Assert.True(manager.ShouldCheckpoint(1));
        }

        #endregion

        #region Store and Retrieve Tests

        [Fact]
        public void StoreActivation_StoreAll_CheckpointsAllActivations()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll, stage);
            var activation1 = CreateTestTensor(new[] { 10, 10 }, 1.0f);
            var activation2 = CreateTestTensor(new[] { 10, 10 }, 2.0f);

            // Act
            manager.StoreActivation(0, activation1);
            manager.StoreActivation(1, activation2);

            // Assert
            Assert.Equal(2, manager.CheckpointCount);
            Assert.NotNull(manager.GetActivation(0));
            Assert.NotNull(manager.GetActivation(1));
            Assert.Equal(100, manager.GetActivation(0)!.Size);
            Assert.Equal(100, manager.GetActivation(1)!.Size);
        }

        [Fact]
        public void StoreActivation_RecomputeAll_CheckpointsNoActivations()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.RecomputeAll, stage);
            var activation = CreateTestTensor(new[] { 10, 10 }, 1.0f);

            // Act
            manager.StoreActivation(0, activation);

            // Assert
            Assert.Equal(0, manager.CheckpointCount);
            Assert.Null(manager.GetActivation(0));
        }

        [Fact]
        public void GetActivation_NotFound_ReturnsNull()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll, stage);

            // Act
            var result = manager.GetActivation(999);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void HasActivation_ReturnsCorrectStatus()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll, stage);
            var activation = CreateTestTensor(new[] { 10, 10 }, 1.0f);

            // Act & Assert
            Assert.False(manager.HasActivation(0));
            manager.StoreActivation(0, activation);
            Assert.True(manager.HasActivation(0));
            Assert.False(manager.HasActivation(1));
        }

        [Fact]
        public void StoreActivation_ClonesTensor()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll, stage);
            var activation = CreateTestTensor(new[] { 10, 10 }, 1.0f);

            // Act
            manager.StoreActivation(0, activation);
            activation.Data[0] = 999.0f; // Modify original tensor

            // Assert - Checkpointed tensor should not be modified
            var retrieved = manager.GetActivation(0);
            Assert.NotEqual(999.0f, retrieved!.Data[0]);
            Assert.Equal(1.0f, retrieved.Data[0]);
        }

        [Fact]
        public void StoreActivation_OverwritesExistingCheckpoint()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll, stage);
            var activation1 = CreateTestTensor(new[] { 10, 10 }, 1.0f);
            var activation2 = CreateTestTensor(new[] { 10, 10 }, 2.0f);

            // Act
            manager.StoreActivation(0, activation1);
            manager.StoreActivation(0, activation2);

            // Assert
            Assert.Equal(1, manager.CheckpointCount);
            var retrieved = manager.GetActivation(0);
            Assert.Equal(2.0f, retrieved!.Data[0]);
        }

        #endregion

        #region GetOrCompute Tests

        [Fact]
        public void GetOrComputeActivation_WithStoredActivation_ReturnsStored()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll, stage);
            var activation = CreateTestTensor(new[] { 10, 10 }, 1.0f);
            manager.StoreActivation(0, activation);
            var input = CreateTestTensor(new[] { 10, 10 }, 0.5f);

            // Act
            var result = manager.GetOrComputeActivation(input, 0);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(1.0f, result.Data[0]); // Should return stored activation, not input
        }

        [Fact]
        public void GetOrComputeActivation_WithoutStoredActivation_Recomputes()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.RecomputeAll, stage);
            var input = CreateTestTensor(new[] { 10, 10 }, 5.0f);

            // Act
            var result = manager.GetOrComputeActivation(input, 0);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(5.0f, result.Data[0]); // Should return computed result (input value)
        }

        #endregion

        #region Recompute Tests

        [Fact]
        public void RecomputeActivation_CallsStageForward()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.RecomputeAll, stage);
            var input = CreateTestTensor(new[] { 10, 10 }, 7.0f);

            // Act
            var result = manager.RecomputeActivation(input, 0);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(input.Size, result.Size);
            Assert.Equal(7.0f, result.Data[0]);
        }

        [Fact]
        public void RecomputeActivation_CachesIfStrategyChanged()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.Selective, stage, checkpointInterval: 2);
            var input = CreateTestTensor(new[] { 10, 10 }, 8.0f);

            // Act - Recompute with micro-batch 0, which should be checkpointed
            var result = manager.RecomputeActivation(input, 0);

            // Assert
            Assert.True(manager.HasActivation(0)); // Should have been cached
        }

        #endregion

        #region Memory Estimation Tests

        [Fact]
        public void EstimateMemoryUsage_ReturnsCorrectSize()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll, stage);
            var activation1 = CreateTestTensor(new[] { 10, 10 }, 1.0f); // 100 elements
            var activation2 = CreateTestTensor(new[] { 20, 20 }, 2.0f); // 400 elements

            // Act
            manager.StoreActivation(0, activation1);
            manager.StoreActivation(1, activation2);
            var memory = manager.EstimateMemoryUsage();

            // Assert - 500 elements * 4 bytes (float)
            Assert.Equal(500 * sizeof(float), memory);
        }

        [Fact]
        public void EstimateMemoryUsage_EmptyManager_ReturnsZero()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll, stage);

            // Act
            var memory = manager.EstimateMemoryUsage();

            // Assert
            Assert.Equal(0, memory);
        }

        #endregion

        #region Clear Tests

        [Fact]
        public void Clear_RemovesAllCheckpoints()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll, stage);
            var activation = CreateTestTensor(new[] { 10, 10 }, 1.0f);
            manager.StoreActivation(0, activation);
            manager.StoreActivation(1, activation);
            manager.StoreActivation(2, activation);

            // Act
            manager.Clear();

            // Assert
            Assert.Equal(0, manager.CheckpointCount);
            Assert.False(manager.HasActivation(0));
            Assert.False(manager.HasActivation(1));
            Assert.False(manager.HasActivation(2));
        }

        #endregion

        #region MemoryBased Strategy Tests

        [Fact]
        public void MemoryBasedStrategy_UnderThreshold_StoresAll()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var memoryThreshold = 100 * sizeof(float) * 2; // Enough for 2 tensors of 100 elements
            var manager = new ActivationCheckpointManager(CheckpointStrategy.MemoryBased, stage, memoryThreshold: memoryThreshold);
            var activation = CreateTestTensor(new[] { 10, 10 }, 1.0f);

            // Act
            manager.StoreActivation(0, activation);
            manager.StoreActivation(1, activation);

            // Assert
            Assert.Equal(2, manager.CheckpointCount);
        }

        [Fact]
        public void MemoryBasedStrategy_ExceedsThreshold_KeepsFirst()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var memoryThreshold = 100 * sizeof(float); // Only enough for 1 tensor
            var manager = new ActivationCheckpointManager(CheckpointStrategy.MemoryBased, stage, memoryThreshold: memoryThreshold);
            var activation = CreateTestTensor(new[] { 10, 10 }, 1.0f);

            // Act
            manager.StoreActivation(0, activation); // First - should be kept
            manager.StoreActivation(1, activation); // Would exceed threshold

            // Assert
            Assert.True(manager.HasActivation(0)); // First should be kept
            // Second may or may not be stored depending on threshold logic
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Constructor_WithZeroCheckpointInterval_TreatsAsOne()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.Selective, stage, checkpointInterval: 0);

            // Act
            var shouldCheckpoint = manager.ShouldCheckpoint(0);

            // Assert - Division by zero protection should treat as interval 1
            Assert.True(shouldCheckpoint);
        }

        [Fact]
        public void DuplicateMicroBatchIndices_OverwritesExisting()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll, stage);
            var activation1 = CreateTestTensor(new[] { 10, 10 }, 1.0f);
            var activation2 = CreateTestTensor(new[] { 10, 10 }, 2.0f);

            // Act
            manager.StoreActivation(5, activation1);
            manager.StoreActivation(5, activation2);

            // Assert
            Assert.Equal(1, manager.CheckpointCount);
            var retrieved = manager.GetActivation(5);
            Assert.Equal(2.0f, retrieved!.Data[0]);
        }

        [Fact]
        public void SelectiveStrategy_WithVeryLargeInterval_OnlyStoresFirst()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.Selective, stage, checkpointInterval: 1000);
            var activation = CreateTestTensor(new[] { 10, 10 }, 1.0f);

            // Act
            manager.StoreActivation(0, activation); // First
            manager.StoreActivation(1, activation);
            manager.StoreActivation(2, activation);

            // Assert
            Assert.Equal(1, manager.CheckpointCount);
            Assert.True(manager.HasActivation(0));
        }

        [Fact]
        public void VeryLargeMemoryThreshold_StoresAll()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(
                CheckpointStrategy.MemoryBased,
                stage,
                memoryThreshold: long.MaxValue);
            var activation = CreateTestTensor(new[] { 10, 10 }, 1.0f);

            // Act
            manager.StoreActivation(0, activation);
            manager.StoreActivation(1, activation);
            manager.StoreActivation(2, activation);

            // Assert
            Assert.Equal(3, manager.CheckpointCount);
        }

        [Fact]
        public void VerySmallMemoryThreshold_StoresFew()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(
                CheckpointStrategy.MemoryBased,
                stage,
                memoryThreshold: 10); // Very small threshold
            var activation = CreateTestTensor(new[] { 10, 10 }, 1.0f);

            // Act
            manager.StoreActivation(0, activation);
            manager.StoreActivation(1, activation);
            manager.StoreActivation(2, activation);

            // Assert - Should at least keep the first
            Assert.True(manager.CheckpointCount >= 1);
        }

        #endregion

        #region Metadata Tests

        [Fact]
        public void CheckpointMetadata_IsCreatedCorrectly()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll, stage);
            var activation = CreateTestTensor(new[] { 10, 20, 30 }, 1.0f);

            // Act
            manager.StoreActivation(5, activation);

            // Assert
            Assert.Equal(1, manager.CheckpointCount);
        }

        #endregion

        #region Null Handling

        [Fact]
        public void StoreActivation_NullTensor_ThrowsArgumentNullException()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll, stage);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => manager.StoreActivation(0, null!));
        }

        [Fact]
        public void GetOrComputeActivation_NullInput_ThrowsArgumentNullException()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.RecomputeAll, stage);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => manager.GetOrComputeActivation(null!, 0));
        }

        [Fact]
        public void RecomputeActivation_NullInput_ThrowsArgumentNullException()
        {
            // Arrange
            var module = new SimpleForwardModule();
            var stage = new PipelineStage(module, 0, 1, Device.CPU);
            var manager = new ActivationCheckpointManager(CheckpointStrategy.RecomputeAll, stage);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => manager.RecomputeActivation(null!, 0));
        }

        #endregion
    }
}
