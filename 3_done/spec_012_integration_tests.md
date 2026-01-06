# Spec: Integration Tests

## Overview
Implement end-to-end integration tests for the mixed-precision optimizer to verify all components work together correctly.

## Dependencies
- All previous specs (001-011)

## Implementation Details

### Test File Structure
Create the test file in `tests/MLFramework.Tests/Optimizers/MixedPrecision/MixedPrecisionIntegrationTests.cs`:

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using MLFramework.Optimizers;
using MLFramework.Optimizers.MixedPrecision;

namespace MLFramework.Tests.Optimizers.MixedPrecision
{
    /// <summary>
    /// Integration tests for mixed-precision training with a simple model
    /// </summary>
    public class MixedPrecisionIntegrationTests
    {
        #region Basic Training Loop Tests

        [Fact]
        public void FullTrainingLoop_CompletesSuccessfully()
        {
            // Arrange
            var model = CreateSimpleModel();
            var optimizer = CreateAdamOptimizer();
            var mpOptimizer = new MixedPrecisionOptimizer(optimizer);
            var dataLoader = CreateMockDataLoader(numBatches: 10);

            mpOptimizer.SetParameters(model.Parameters);

            // Act
            foreach (var batch in dataLoader)
            {
                // Forward pass
                var loss = model.Forward(batch);

                // Scale loss
                var scaledLoss = mpOptimizer.ScaleLoss(loss);

                // Backward pass
                var gradients = model.Backward(scaledLoss);

                // Optimizer step
                mpOptimizer.Step(gradients);
            }

            // Assert
            Assert.Equal(10, mpOptimizer.StepCount);
            Assert.True(mpOptimizer.LossScale > 0);
            Assert.False(mpOptimizer.HasFallback);
        }

        [Fact]
        public void TrainingLoopWithOverflow_HandlesGracefully()
        {
            // Arrange
            var model = CreateSimpleModel();
            var optimizer = CreateAdamOptimizer();
            var options = new MixedPrecisionOptions
            {
                InitialLossScale = 1000.0f,
                MaxConsecutiveOverflows = 10
            };
            var mpOptimizer = new MixedPrecisionOptimizer(optimizer, options);
            var dataLoader = CreateMockDataLoader(numBatches: 20);

            mpOptimizer.SetParameters(model.Parameters);

            // Act
            foreach (var batch in dataLoader)
            {
                var loss = model.Forward(batch);
                var scaledLoss = mpOptimizer.ScaleLoss(loss);

                // Simulate overflow on some batches
                var gradients = batch.Index % 5 == 0
                    ? CreateOverflowGradients(model)
                    : model.Backward(scaledLoss);

                mpOptimizer.Step(gradients);
            }

            // Assert
            Assert.Equal(20, mpOptimizer.StepCount);
            Assert.True(mpOptimizer.SkippedSteps > 0);  // Some steps should be skipped
            Assert.False(mpOptimizer.HasFallback);  // Should not fallback
        }

        [Fact]
        public void TrainingLoopWithSevereOverflow_TriggersFallback()
        {
            // Arrange
            var model = CreateSimpleModel();
            var optimizer = CreateAdamOptimizer();
            var options = new MixedPrecisionOptions
            {
                InitialLossScale = 1000.0f,
                MaxConsecutiveOverflows = 3,
                EnableAutoFallback = true
            };
            var mpOptimizer = new MixedPrecisionOptimizer(optimizer, options);
            var dataLoader = CreateMockDataLoader(numBatches: 20);

            mpOptimizer.SetParameters(model.Parameters);

            // Act
            foreach (var batch in dataLoader)
            {
                var loss = model.Forward(batch);
                var scaledLoss = mpOptimizer.ScaleLoss(loss);

                // Simulate overflow on all batches
                var gradients = CreateOverflowGradients(model);

                mpOptimizer.Step(gradients);
            }

            // Assert
            Assert.Equal(20, mpOptimizer.StepCount);
            Assert.True(mpOptimizer.HasFallback);  // Should fallback
        }

        #endregion

        #region Checkpointing Tests

        [Fact]
        public void SaveAndLoadCheckpoint_RestoresCorrectState()
        {
            // Arrange
            var model1 = CreateSimpleModel();
            var optimizer1 = CreateAdamOptimizer();
            var mpOptimizer1 = new MixedPrecisionOptimizer(optimizer1);
            var dataLoader = CreateMockDataLoader(numBatches: 5);

            mpOptimizer1.SetParameters(model1.Parameters);

            // Train for 5 steps
            foreach (var batch in dataLoader)
            {
                var loss = model1.Forward(batch);
                var scaledLoss = mpOptimizer1.ScaleLoss(loss);
                var gradients = model1.Backward(scaledLoss);
                mpOptimizer1.Step(gradients);
            }

            // Save checkpoint
            var checkpoint = MixedPrecisionCheckpointManager.CreateCheckpoint(mpOptimizer1);

            // Act - Create new optimizer and restore
            var model2 = CreateSimpleModel();
            var optimizer2 = CreateAdamOptimizer();
            var mpOptimizer2 = new MixedPrecisionOptimizer(optimizer2);
            mpOptimizer2.SetParameters(model2.Parameters);

            MixedPrecisionCheckpointManager.RestoreCheckpoint(mpOptimizer2, checkpoint);

            // Assert
            Assert.Equal(mpOptimizer1.StepCount, mpOptimizer2.StepCount);
            Assert.Equal(mpOptimizer1.SkippedSteps, mpOptimizer2.SkippedSteps);
            Assert.Equal(mpOptimizer1.LossScale, mpOptimizer2.LossScale);
        }

        #endregion

        #region MultiOptimizer Tests

        [Fact]
        public void MultipleOptimizersWithMixedPrecision_WorkIndependently()
        {
            // Arrange
            var model1 = CreateSimpleModel();
            var model2 = CreateSimpleModel();

            var optimizer1 = new MixedPrecisionOptimizer(CreateAdamOptimizer());
            var optimizer2 = new MixedPrecisionOptimizer(CreateAdamOptimizer());

            optimizer1.SetParameters(model1.Parameters);
            optimizer2.SetParameters(model2.Parameters);

            var dataLoader = CreateMockDataLoader(numBatches: 5);

            // Act
            foreach (var batch in dataLoader)
            {
                // Train model 1
                var loss1 = model1.Forward(batch);
                var scaledLoss1 = optimizer1.ScaleLoss(loss1);
                var grads1 = model1.Backward(scaledLoss1);
                optimizer1.Step(grads1);

                // Train model 2
                var loss2 = model2.Forward(batch);
                var scaledLoss2 = optimizer2.ScaleLoss(loss2);
                var grads2 = model2.Backward(scaledLoss2);
                optimizer2.Step(grads2);
            }

            // Assert
            Assert.Equal(5, optimizer1.StepCount);
            Assert.Equal(5, optimizer2.StepCount);
            // Loss scales may differ slightly
            Assert.True(optimizer1.LossScale > 0);
            Assert.True(optimizer2.LossScale > 0);
        }

        #endregion

        #region Performance Monitoring Tests

        [Fact]
        public void PerformanceMonitoring_TracksMetricsCorrectly()
        {
            // Arrange
            var model = CreateSimpleModel();
            var optimizer = CreateAdamOptimizer();
            var options = new MixedPrecisionOptions
            {
                EnablePerformanceMonitoring = true,
                PerformanceLogInterval = 3
            };
            var mpOptimizer = new MixedPrecisionOptimizer(optimizer, options);
            var dataLoader = CreateMockDataLoader(numBatches: 10);

            mpOptimizer.SetParameters(model.Parameters);

            // Act
            foreach (var batch in dataLoader)
            {
                var loss = model.Forward(batch);
                var scaledLoss = mpOptimizer.ScaleLoss(loss);
                var gradients = model.Backward(scaledLoss);
                mpOptimizer.Step(gradients);
            }

            // Assert
            var stats = mpOptimizer.GetStats();
            Assert.NotNull(stats.PerformanceStats);
            Assert.Equal(10, stats.PerformanceStats.StepCount);
            Assert.True(stats.PerformanceStats.AverageStepTimeMs > 0);
        }

        #endregion

        #region Edge Case Tests

        [Fact]
        public void TrainingWithZeroGrad_HandlesCorrectly()
        {
            // Arrange
            var model = CreateSimpleModel();
            var optimizer = CreateAdamOptimizer();
            var mpOptimizer = new MixedPrecisionOptimizer(optimizer);

            mpOptimizer.SetParameters(model.Parameters);

            // Act
            mpOptimizer.ZeroGrad();
            var zeroGrads = CreateZeroGradients(model);
            mpOptimizer.Step(zeroGrads);

            // Assert
            Assert.Equal(1, mpOptimizer.StepCount);
        }

        [Fact]
        public void TrainingWithLearningRateSchedule_UpdatesCorrectly()
        {
            // Arrange
            var model = CreateSimpleModel();
            var optimizer = new MixedPrecisionOptimizer(CreateAdamOptimizer());
            var dataLoader = CreateMockDataLoader(numBatches: 10);

            optimizer.SetParameters(model.Parameters);
            float initialLR = optimizer.LearningRate;

            // Act
            foreach (var batch in dataLoader)
            {
                var loss = model.Forward(batch);
                var scaledLoss = optimizer.ScaleLoss(loss);
                var gradients = model.Backward(scaledLoss);
                optimizer.Step(gradients);

                // Decay learning rate
                optimizer.SetLearningRate(optimizer.LearningRate * 0.95f);
            }

            // Assert
            Assert.True(optimizer.LearningRate < initialLR);
        }

        #endregion

        #region Realistic Scenario Tests

        [Fact]
        public void TransformerModelTraining_DoesNotCrash()
        {
            // Arrange
            var model = CreateTransformerModel();
            var optimizer = new MixedPrecisionOptimizer(CreateAdamOptimizer());
            var dataLoader = CreateMockDataLoader(numBatches: 5);

            optimizer.SetParameters(model.Parameters);

            // Act
            foreach (var batch in dataLoader)
            {
                var loss = model.Forward(batch);
                var scaledLoss = optimizer.ScaleLoss(loss);
                var gradients = model.Backward(scaledLoss);
                optimizer.Step(gradients);
            }

            // Assert
            Assert.Equal(5, optimizer.StepCount);
            Assert.False(optimizer.HasFallback);
        }

        [Fact]
        public void CNNModelTraining_DoesNotCrash()
        {
            // Arrange
            var model = CreateCNNModel();
            var optimizer = new MixedPrecisionOptimizer(CreateAdamOptimizer());
            var dataLoader = CreateMockDataLoader(numBatches: 5);

            optimizer.SetParameters(model.Parameters);

            // Act
            foreach (var batch in dataLoader)
            {
                var loss = model.Forward(batch);
                var scaledLoss = optimizer.ScaleLoss(loss);
                var gradients = model.Backward(scaledLoss);
                optimizer.Step(gradients);
            }

            // Assert
            Assert.Equal(5, optimizer.StepCount);
            Assert.False(optimizer.HasFallback);
        }

        #endregion

        #region Helper Methods

        private SimpleModel CreateSimpleModel()
        {
            // Create a simple 2-layer model for testing
            var model = new SimpleModel();
            model.Initialize();
            return model;
        }

        private TransformerModel CreateTransformerModel()
        {
            // Create a simple transformer model
            var model = new TransformerModel();
            model.Initialize();
            return model;
        }

        private CNNModel CreateCNNModel()
        {
            // Create a simple CNN model
            var model = new CNNModel();
            model.Initialize();
            return model;
        }

        private IOptimizer CreateAdamOptimizer()
        {
            return new AdamOptimizer(learningRate: 0.001f);
        }

        private MockDataLoader CreateMockDataLoader(int numBatches)
        {
            return new MockDataLoader(numBatches);
        }

        private Dictionary<string, ITensor> CreateOverflowGradients(IModel model)
        {
            var grads = new Dictionary<string, ITensor>();
            foreach (var paramName in model.Parameters.Keys)
            {
                grads[paramName] = new MockTensor(float.NaN);
            }
            return grads;
        }

        private Dictionary<string, ITensor> CreateZeroGradients(IModel model)
        {
            var grads = new Dictionary<string, ITensor>();
            foreach (var paramName in model.Parameters.Keys)
            {
                grads[paramName] = new MockTensor(0.0f);
            }
            return grads;
        }

        #endregion

        #region Mock Classes

        private class SimpleModel : IModel
        {
            public Dictionary<string, ITensor> Parameters { get; private set; }

            public void Initialize()
            {
                Parameters = new Dictionary<string, ITensor>
                {
                    { "layer1.weight", new MockTensor(0.1f) },
                    { "layer1.bias", new MockTensor(0.0f) },
                    { "layer2.weight", new MockTensor(0.1f) },
                    { "layer2.bias", new MockTensor(0.0f) }
                };
            }

            public ITensor Forward(object batch)
            {
                return new MockTensor(1.0f);  // Dummy loss
            }

            public Dictionary<string, ITensor> Backward(ITensor loss)
            {
                var grads = new Dictionary<string, ITensor>();
                foreach (var paramName in Parameters.Keys)
                {
                    grads[paramName] = new MockTensor(0.01f);
                }
                return grads;
            }
        }

        private class TransformerModel : IModel
        {
            public Dictionary<string, ITensor> Parameters { get; private set; }

            public void Initialize()
            {
                Parameters = new Dictionary<string, ITensor>
                {
                    { "embedding.weight", new MockTensor(0.1f) },
                    { "encoder.layer1.weight", new MockTensor(0.1f) },
                    { "encoder.layer1.norm.weight", new MockTensor(0.1f) },  // LayerNorm - should be excluded
                    { "output.weight", new MockTensor(0.1f) }
                };
            }

            public ITensor Forward(object batch)
            {
                return new MockTensor(1.0f);
            }

            public Dictionary<string, ITensor> Backward(ITensor loss)
            {
                var grads = new Dictionary<string, ITensor>();
                foreach (var paramName in Parameters.Keys)
                {
                    grads[paramName] = new MockTensor(0.01f);
                }
                return grads;
            }
        }

        private class CNNModel : IModel
        {
            public Dictionary<string, ITensor> Parameters { get; private set; }

            public void Initialize()
            {
                Parameters = new Dictionary<string, ITensor>
                {
                    { "conv1.weight", new MockTensor(0.1f) },
                    { "bn1.weight", new MockTensor(0.1f) },  // BatchNorm - should be excluded
                    { "fc1.weight", new MockTensor(0.1f) }
                };
            }

            public ITensor Forward(object batch)
            {
                return new MockTensor(1.0f);
            }

            public Dictionary<string, ITensor> Backward(ITensor loss)
            {
                var grads = new Dictionary<string, ITensor>();
                foreach (var paramName in Parameters.Keys)
                {
                    grads[paramName] = new MockTensor(0.01f);
                }
                return grads;
            }
        }

        private class MockDataLoader
        {
            private readonly int _numBatches;

            public MockDataLoader(int numBatches)
            {
                _numBatches = numBatches;
            }

            public IEnumerable<MockBatch> GetBatches()
            {
                for (int i = 0; i < _numBatches; i++)
                {
                    yield return new MockBatch { Index = i };
                }
            }
        }

        private class MockBatch
        {
            public int Index { get; set; }
        }

        private class MockTensor : ITensor
        {
            private readonly float _value;

            public MockTensor(float value)
            {
                _value = value;
            }

            public void Dispose() { }
        }

        #endregion
    }
}
```

## Requirements

### Test Coverage
1. **Basic Training Loop**: Full forward-backward-update cycle
2. **Overflow Handling**: Graceful handling of gradient overflow
3. **Fallback Mechanism**: Auto-fallback to FP32
4. **Checkpointing**: Save and restore optimizer state
5. **Multi-Optimizer**: Multiple independent optimizers
6. **Performance Monitoring**: Verify metrics tracking
7. **Edge Cases**: Zero gradients, learning rate schedules
8. **Realistic Scenarios**: Transformer and CNN models

### Test Categories
- **Happy Path**: Normal training scenarios
- **Edge Cases**: Overflow, zero gradients, learning rate schedules
- **Integration**: Multiple components working together
- **Realistic Models**: Transformer and CNN architectures

## Deliverables

### Test Files
1. `tests/MLFramework.Tests/Optimizers/MixedPrecision/MixedPrecisionIntegrationTests.cs`

## Notes for Coder
- Mock implementations for models, optimizers, and tensors are needed
- Tests should be realistic and mirror actual training loops
- Verify layer exclusions work correctly (BatchNorm, LayerNorm)
- Test checkpoint save/load with full state restoration
- Performance monitoring should track actual timing
- Integration tests are critical for catching component interaction bugs
- These tests may take longer to run than unit tests
- Consider marking them as [Fact] or [Theory] based on needs
- Mock implementations should be as realistic as possible
