# Spec: MixedPrecisionOptimizer Unit Tests

## Overview
Implement comprehensive unit tests for the MixedPrecisionOptimizer component.

## Dependencies
- Spec 001-008: All core mixed-precision components
- Spec 002: MixedPrecisionOptions
- Spec 006: MixedPrecisionOptimizer

## Implementation Details

### Test File Structure
Create the test file in `tests/MLFramework.Tests/Optimizers/MixedPrecision/MixedPrecisionOptimizerTests.cs`:

```csharp
using System;
using System.Collections.Generic;
using Xunit;
using MLFramework.Optimizers;
using MLFramework.Optimizers.MixedPrecision;

namespace MLFramework.Tests.Optimizers.MixedPrecision
{
    public class MixedPrecisionOptimizerTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithNullBaseOptimizer_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new MixedPrecisionOptimizer(null));
        }

        [Fact]
        public void Constructor_WithValidOptimizer_InitializesCorrectly()
        {
            // Arrange
            var baseOptimizer = CreateMockOptimizer();

            // Act
            var mpOptimizer = new MixedPrecisionOptimizer(baseOptimizer);

            // Assert
            Assert.Same(baseOptimizer, mpOptimizer.BaseOptimizer);
            Assert.Equal(Precision.FP16, mpOptimizer.TargetPrecision);
            Assert.Equal(0, mpOptimizer.StepCount);
            Assert.Equal(0, mpOptimizer.SkippedSteps);
            Assert.False(mpOptimizer.HasFallback);
        }

        [Fact]
        public void Constructor_WithCustomOptions_UsesOptions()
        {
            // Arrange
            var baseOptimizer = CreateMockOptimizer();
            var options = MixedPrecisionOptions.ForBF16();

            // Act
            var mpOptimizer = new MixedPrecisionOptimizer(baseOptimizer, options);

            // Assert
            Assert.Equal(Precision.BF16, mpOptimizer.TargetPrecision);
            Assert.Same(options, mpOptimizer.Options);
        }

        #endregion

        #region SetParameters Tests

        [Fact]
        public void SetParameters_WithNullParameters_ThrowsArgumentNullException()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => mpOptimizer.SetParameters(null));
        }

        [Fact]
        public void SetParameters_WithFallbackMode_PassesThroughToBase()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            mpOptimizer.FallbackToFP32();
            var parameters = CreateMockParameters();

            // Act
            mpOptimizer.SetParameters(parameters);

            // Assert
            Assert.Same(parameters, mpOptimizer.MasterWeights);
            Assert.Same(parameters, mpOptimizer.TrainingWeights);
        }

        [Fact]
        public void SetParameters_CreatesMasterAndTrainingWeights()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            var parameters = CreateMockParameters();

            // Act
            mpOptimizer.SetParameters(parameters);

            // Assert
            Assert.NotNull(mpOptimizer.MasterWeights);
            Assert.NotNull(mpOptimizer.TrainingWeights);
            Assert.Equal(parameters.Count, mpOptimizer.MasterWeights.Count);
            Assert.Equal(parameters.Count, mpOptimizer.TrainingWeights.Count);
        }

        #endregion

        #region Step Tests

        [Fact]
        public void Step_WithNullGradients_ThrowsArgumentNullException()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => mpOptimizer.Step(null));
        }

        [Fact]
        public void Step_WithFallbackMode_PassesThroughToBase()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            mpOptimizer.SetParameters(CreateMockParameters());
            mpOptimizer.FallbackToFP32();
            var gradients = CreateMockGradients();
            int initialStepCount = mpOptimizer.StepCount;

            // Act
            mpOptimizer.Step(gradients);

            // Assert
            Assert.Equal(initialStepCount + 1, mpOptimizer.StepCount);
            Assert.Equal(0, mpOptimizer.SkippedSteps);
        }

        [Fact]
        public void Step_WithValidGradients_IncrementsStepCount()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            mpOptimizer.SetParameters(CreateMockParameters());
            var gradients = CreateMockGradients();
            int initialStepCount = mpOptimizer.StepCount;

            // Act
            mpOptimizer.Step(gradients);

            // Assert
            Assert.Equal(initialStepCount + 1, mpOptimizer.StepCount);
        }

        [Fact]
        public void Step_WithOverflow_SkipsStep()
        {
            // Arrange
            var options = new MixedPrecisionOptions
            {
                InitialLossScale = 1000.0f,
                BackoffFactor = 0.5f,
                EnableDynamicLossScaling = true
            };
            var mpOptimizer = new MixedPrecisionOptimizer(CreateMockOptimizer(), options);
            mpOptimizer.SetParameters(CreateMockParameters());
            var gradients = CreateMockGradientsWithNaN();

            // Act
            mpOptimizer.Step(gradients);

            // Assert
            Assert.Equal(1, mpOptimizer.StepCount);
            Assert.Equal(1, mpOptimizer.SkippedSteps);
        }

        [Fact]
        public void Step_WithConsecutiveOverflows_TriggersFallback()
        {
            // Arrange
            var options = new MixedPrecisionOptions
            {
                MaxConsecutiveOverflows = 2,
                EnableAutoFallback = true,
                EnableDynamicLossScaling = true
            };
            var mpOptimizer = new MixedPrecisionOptimizer(CreateMockOptimizer(), options);
            mpOptimizer.SetParameters(CreateMockParameters());
            var gradients = CreateMockGradientsWithNaN();

            // Act
            mpOptimizer.Step(gradients);  // Skip 1
            mpOptimizer.Step(gradients);  // Skip 2 - should trigger fallback
            mpOptimizer.Step(gradients);  // In fallback mode now

            // Assert
            Assert.True(mpOptimizer.HasFallback);
            Assert.Equal(3, mpOptimizer.StepCount);
        }

        #endregion

        #region StepParameter Tests

        [Fact]
        public void StepParameter_WithEmptyName_ThrowsArgumentException()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => mpOptimizer.StepParameter("", CreateMockTensor()));
        }

        [Fact]
        public void StepParameter_WithNullGradient_ThrowsArgumentNullException()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => mpOptimizer.StepParameter("param1", null));
        }

        [Fact]
        public void StepParameter_WithValidInput_ExecutesStep()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            mpOptimizer.SetParameters(CreateMockParameters());
            int initialStepCount = mpOptimizer.StepCount;

            // Act
            mpOptimizer.StepParameter("param1", CreateMockTensor());

            // Assert
            Assert.Equal(initialStepCount + 1, mpOptimizer.StepCount);
        }

        #endregion

        #region ScaleLoss Tests

        [Fact]
        public void ScaleLoss_WithFallback_ReturnsUnscaledLoss()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            mpOptimizer.FallbackToFP32();
            var loss = CreateMockTensor();

            // Act
            var scaled = mpOptimizer.ScaleLoss(loss);

            // Assert
            Assert.Same(loss, scaled);  // Should return same tensor
        }

        [Fact]
        public void ScaleLoss_WithoutFallback_ScalesLoss()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            var loss = CreateMockTensor();

            // Act
            var scaled = mpOptimizer.ScaleLoss(loss);

            // Assert
            Assert.NotNull(scaled);
            // Verify scaling occurred (implementation dependent)
        }

        #endregion

        #region Fallback Tests

        [Fact]
        public void FallbackToFP32_WhenNotFallback_SwitchesMode()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            mpOptimizer.SetParameters(CreateMockParameters());

            // Act
            mpOptimizer.FallbackToFP32();

            // Assert
            Assert.True(mpOptimizer.HasFallback);
        }

        [Fact]
        public void FallbackToFP32_WhenAlreadyFallback_DoesNothing()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            mpOptimizer.FallbackToFP32();
            var initialWeights = mpOptimizer.MasterWeights;

            // Act
            mpOptimizer.FallbackToFP32();

            // Assert
            Assert.Same(initialWeights, mpOptimizer.MasterWeights);
        }

        [Fact]
        public void FallbackToFP32_UpdatesWeightsToFP32()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            mpOptimizer.SetParameters(CreateMockParameters());

            // Act
            mpOptimizer.FallbackToFP32();

            // Assert
            // Master weights should now be the same as training weights
            Assert.Same(mpOptimizer.MasterWeights, mpOptimizer.TrainingWeights);
        }

        #endregion

        #region GetStats Tests

        [Fact]
        public void GetStats_ReturnsCorrectStatistics()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            mpOptimizer.SetParameters(CreateMockParameters());

            // Act
            var stats = mpOptimizer.GetStats();

            // Assert
            Assert.NotNull(stats);
            Assert.Equal(Precision.FP16, stats.TargetPrecision);
            Assert.Equal(0, stats.StepCount);
            Assert.Equal(0, stats.SkippedSteps);
            Assert.NotNull(stats.GradientStats);
            Assert.NotNull(stats.LossScalerStats);
        }

        [Fact]
        public void GetStats_AfterSteps_ReturnsCorrectStepCount()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            mpOptimizer.SetParameters(CreateMockParameters());
            mpOptimizer.Step(CreateMockGradients());
            mpOptimizer.Step(CreateMockGradients());

            // Act
            var stats = mpOptimizer.GetStats();

            // Assert
            Assert.Equal(2, stats.StepCount);
            Assert.Equal(0.0f, stats.SkipRate);
        }

        [Fact]
        public void GetStats_WithSkips_ReturnsCorrectSkipRate()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            mpOptimizer.SetParameters(CreateMockParameters());
            mpOptimizer.Step(CreateMockGradients());      // Success
            mpOptimizer.Step(CreateMockGradientsWithNaN()); // Skip

            // Act
            var stats = mpOptimizer.GetStats();

            // Assert
            Assert.Equal(2, stats.StepCount);
            Assert.Equal(1, stats.SkippedSteps);
            Assert.Equal(0.5f, stats.SkipRate);
        }

        #endregion

        #region Reset Tests

        [Fact]
        public void Reset_ClearsAllCounters()
        {
            // Arrange
            var mpOptimizer = CreateMixedPrecisionOptimizer();
            mpOptimizer.SetParameters(CreateMockParameters());
            mpOptimizer.Step(CreateMockGradients());
            mpOptimizer.Step(CreateMockGradients());

            // Act
            mpOptimizer.Reset();

            // Assert
            Assert.Equal(0, mpOptimizer.StepCount);
            Assert.Equal(0, mpOptimizer.SkippedSteps);
            // Loss scaler should also be reset
        }

        #endregion

        #region IOptimizer Interface Tests

        [Fact]
        public void LearningRate_PassesThroughToBaseOptimizer()
        {
            // Arrange
            var baseOptimizer = CreateMockOptimizer();
            var mpOptimizer = new MixedPrecisionOptimizer(baseOptimizer);

            // Act
            var lr = mpOptimizer.LearningRate;

            // Assert
            Assert.Equal(0.001f, lr);  // Mock default value
        }

        [Fact]
        public void SetLearningRate_PassesThroughToBaseOptimizer()
        {
            // Arrange
            var baseOptimizer = CreateMockOptimizer();
            var mpOptimizer = new MixedPrecisionOptimizer(baseOptimizer);

            // Act
            mpOptimizer.SetLearningRate(0.01f);

            // Assert
            Assert.Equal(0.01f, baseOptimizer.LearningRate);
        }

        [Fact]
        public void ZeroGrad_PassesThroughToBaseOptimizer()
        {
            // Arrange
            var baseOptimizer = CreateMockOptimizer();
            var mpOptimizer = new MixedPrecisionOptimizer(baseOptimizer);

            // Act
            mpOptimizer.ZeroGrad();

            // Assert
            // Verify ZeroGrad was called on base optimizer (implementation dependent)
        }

        #endregion

        #region Helper Methods

        private MixedPrecisionOptimizer CreateMixedPrecisionOptimizer()
        {
            return new MixedPrecisionOptimizer(CreateMockOptimizer());
        }

        private IOptimizer CreateMockOptimizer()
        {
            // TODO: Create mock optimizer implementation
            return new MockOptimizer();
        }

        private Dictionary<string, ITensor> CreateMockParameters()
        {
            return new Dictionary<string, ITensor>
            {
                { "param1", CreateMockTensor() },
                { "param2", CreateMockTensor() }
            };
        }

        private Dictionary<string, ITensor> CreateMockGradients()
        {
            return new Dictionary<string, ITensor>
            {
                { "param1", CreateMockTensor() },
                { "param2", CreateMockTensor() }
            };
        }

        private Dictionary<string, ITensor> CreateMockGradientsWithNaN()
        {
            return new Dictionary<string, ITensor>
            {
                { "param1", CreateMockTensor(float.NaN) }
            };
        }

        private ITensor CreateMockTensor(float value = 0.1f)
        {
            // TODO: Create mock tensor implementation
            return new MockTensor(value);
        }

        #endregion

        #region Mock Classes

        private class MockOptimizer : IOptimizer
        {
            public float LearningRate { get; private set; } = 0.001f;
            private Dictionary<string, ITensor> _parameters;

            public void SetParameters(Dictionary<string, ITensor> parameters)
            {
                _parameters = parameters;
            }

            public void Step(Dictionary<string, ITensor> gradients)
            {
                // Mock implementation
            }

            public void StepParameter(string parameterName, ITensor gradient)
            {
                // Mock implementation
            }

            public void ZeroGrad()
            {
                // Mock implementation
            }

            public void SetLearningRate(float lr)
            {
                LearningRate = lr;
            }
        }

        private class MockTensor : ITensor
        {
            private float _value;

            public MockTensor(float value)
            {
                _value = value;
            }

            // ITensor implementation stubs
            public void Dispose() { }
        }

        #endregion
    }
}
```

## Requirements

### Test Coverage
1. **Constructor Tests**: Test initialization with various configurations
2. **SetParameters Tests**: Test parameter setting with/without fallback
3. **Step Tests**: Test optimizer step execution and overflow handling
4. **StepParameter Tests**: Test single parameter updates
5. **ScaleLoss Tests**: Test loss scaling
6. **Fallback Tests**: Test fallback mode activation
7. **GetStats Tests**: Test statistics reporting
8. **Reset Tests**: Test state reset
9. **IOptimizer Interface Tests**: Test interface implementation

### Test Categories
- **Happy Path**: Normal operation scenarios
- **Edge Cases**: Fallback mode, consecutive overflows
- **Error Handling**: Null inputs, invalid configurations
- **Interface Compliance**: Verify IOptimizer interface implementation

## Deliverables

### Test Files
1. `tests/MLFramework.Tests/Optimizers/MixedPrecision/MixedPrecisionOptimizerTests.cs`

## Notes for Coder
- Mock implementations for IOptimizer and ITensor are needed
- Test both normal mode and fallback mode extensively
- Verify all IOptimizer interface methods are implemented correctly
- Test auto-fallback mechanism with consecutive overflow threshold
- Statistics should accurately reflect optimizer state
- Learning rate management should pass through to base optimizer
- Consider mocking base optimizer to verify method calls
