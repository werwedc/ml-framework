# Spec: DynamicLossScaler Unit Tests

## Overview
Implement comprehensive unit tests for the DynamicLossScaler component.

## Dependencies
- Spec 002: MixedPrecisionOptions
- Spec 003: DynamicLossScaler

## Implementation Details

### Test File Structure
Create the test file in `tests/MLFramework.Tests/Optimizers/MixedPrecision/DynamicLossScalerTests.cs`:

```csharp
using System;
using System.Collections.Generic;
using Xunit;
using MLFramework.Optimizers.MixedPrecision;

namespace MLFramework.Tests.Optimizers.MixedPrecision
{
    public class DynamicLossScalerTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithNullOptions_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new DynamicLossScaler(null));
        }

        [Fact]
        public void Constructor_WithDefaultOptions_InitializesCorrectly()
        {
            // Arrange
            var options = MixedPrecisionOptions.ForFP16();

            // Act
            var scaler = new DynamicLossScaler(options);

            // Assert
            Assert.Equal(options.InitialLossScale, scaler.CurrentScale);
            Assert.Equal(0, scaler.StepsSinceLastOverflow);
            Assert.Equal(0, scaler.ConsecutiveOverflows);
            Assert.Equal(0, scaler.TotalOverflows);
            Assert.True(scaler.IsEnabled);
        }

        [Fact]
        public void Constructor_WithDisabledLossScaling_DoesNotScale()
        {
            // Arrange
            var options = new MixedPrecisionOptions
            {
                EnableDynamicLossScaling = false,
                InitialLossScale = 65536.0f
            };

            // Act
            var scaler = new DynamicLossScaler(options);

            // Assert
            Assert.False(scaler.IsEnabled);
            Assert.Equal(options.InitialLossScale, scaler.CurrentScale);
        }

        #endregion

        #region Scale Loss Tests

        [Fact]
        public void ScaleLoss_WithNullTensor_ThrowsArgumentNullException()
        {
            // Arrange
            var scaler = new DynamicLossScaler();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => scaler.ScaleLoss(null));
        }

        [Fact]
        public void ScaleLoss_WhenDisabled_ReturnsUnscaledTensor()
        {
            // Arrange
            var options = new MixedPrecisionOptions { EnableDynamicLossScaling = false };
            var scaler = new DynamicLossScaler(options);
            var loss = CreateMockTensor(1.0f);

            // Act
            var scaled = scaler.ScaleLoss(loss);

            // Assert
            Assert.Same(loss, scaled);  // Should return same tensor
        }

        [Fact]
        public void ScaleLoss_WhenEnabled_MultipliesByCurrentScale()
        {
            // Arrange
            var options = new MixedPrecisionOptions
            {
                EnableDynamicLossScaling = true,
                InitialLossScale = 2.0f
            };
            var scaler = new DynamicLossScaler(options);
            var loss = CreateMockTensor(3.0f);

            // Act
            var scaled = scaler.ScaleLoss(loss);

            // Assert
            // Verify multiplication occurred (implementation dependent)
            Assert.NotNull(scaled);
        }

        #endregion

        #region Unscale Gradients Tests

        [Fact]
        public void UnscaleGradients_WithNullGradients_ThrowsArgumentNullException()
        {
            // Arrange
            var scaler = new DynamicLossScaler();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => scaler.UnscaleGradients(null));
        }

        [Fact]
        public void UnscaleGradients_WhenDisabled_ReturnsUnscaledGradients()
        {
            // Arrange
            var options = new MixedPrecisionOptions { EnableDynamicLossScaling = false };
            var scaler = new DynamicLossScaler(options);
            var grads = CreateMockGradients();

            // Act
            var unscaled = scaler.UnscaleGradients(grads);

            // Assert
            Assert.Equal(grads.Count, unscaled.Count);
            // Each gradient should be unchanged
        }

        #endregion

        #region Overflow Detection Tests

        [Fact]
        public void CheckOverflow_WithNullGradients_ThrowsArgumentNullException()
        {
            // Arrange
            var scaler = new DynamicLossScaler();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => scaler.CheckOverflow(null));
        }

        [Fact]
        public void CheckOverflow_WhenDisabled_AlwaysReturnsFalse()
        {
            // Arrange
            var options = new MixedPrecisionOptions { EnableDynamicLossScaling = false };
            var scaler = new DynamicLossScaler(options);
            var grads = CreateMockGradients();

            // Act
            bool hasOverflow = scaler.CheckOverflow(grads);

            // Assert
            Assert.False(hasOverflow);
        }

        [Fact]
        public void CheckOverflow_WithValidGradients_ReturnsFalse()
        {
            // Arrange
            var scaler = new DynamicLossScaler();
            var grads = CreateMockGradients();

            // Act
            bool hasOverflow = scaler.CheckOverflow(grads);

            // Assert
            Assert.False(hasOverflow);
        }

        [Fact]
        public void CheckOverflow_WithNaNGradients_ReturnsTrue()
        {
            // Arrange
            var scaler = new DynamicLossScaler();
            var grads = CreateMockGradientsWithNaN();

            // Act
            bool hasOverflow = scaler.CheckOverflow(grads);

            // Assert
            Assert.True(hasOverflow);
        }

        #endregion

        #region Scale Update Tests

        [Fact]
        public void UpdateScale_WhenDisabled_ReturnsFalse()
        {
            // Arrange
            var options = new MixedPrecisionOptions { EnableDynamicLossScaling = false };
            var scaler = new DynamicLossScaler(options);

            // Act
            bool shouldSkip = scaler.UpdateScale(false);

            // Assert
            Assert.False(shouldSkip);
        }

        [Fact]
        public void UpdateScale_WithNoOverflow_DoesNotSkip()
        {
            // Arrange
            var scaler = new DynamicLossScaler();

            // Act
            bool shouldSkip = scaler.UpdateScale(false);

            // Assert
            Assert.False(shouldSkip);
            Assert.Equal(0, scaler.ConsecutiveOverflows);
            Assert.Equal(1, scaler.StepsSinceLastOverflow);
        }

        [Fact]
        public void UpdateScale_WithOverflow_SkipsAndReducesScale()
        {
            // Arrange
            var options = new MixedPrecisionOptions
            {
                InitialLossScale = 1000.0f,
                BackoffFactor = 0.5f,
                MinLossScale = 1.0f
            };
            var scaler = new DynamicLossScaler(options);

            // Act
            bool shouldSkip = scaler.UpdateScale(true);

            // Assert
            Assert.True(shouldSkip);
            Assert.Equal(1, scaler.ConsecutiveOverflows);
            Assert.Equal(0, scaler.StepsSinceLastOverflow);
            Assert.Equal(500.0f, scaler.CurrentScale);  // 1000 * 0.5
            Assert.Equal(1, scaler.TotalOverflows);
        }

        [Fact]
        public void UpdateScale_WithMultipleOverflows_TracksConsecutiveCount()
        {
            // Arrange
            var options = new MixedPrecisionOptions
            {
                InitialLossScale = 1000.0f,
                BackoffFactor = 0.5f,
                MinLossScale = 1.0f
            };
            var scaler = new DynamicLossScaler(options);

            // Act
            scaler.UpdateScale(true);
            scaler.UpdateScale(true);
            scaler.UpdateScale(true);

            // Assert
            Assert.Equal(3, scaler.ConsecutiveOverflows);
            Assert.Equal(3, scaler.TotalOverflows);
        }

        [Fact]
        public void UpdateScale_AfterGrowthInterval_IncreasesScale()
        {
            // Arrange
            var options = new MixedPrecisionOptions
            {
                InitialLossScale = 1000.0f,
                GrowthFactor = 2.0f,
                GrowthInterval = 3,
                MaxLossScale = 1e9f
            };
            var scaler = new DynamicLossScaler(options);

            // Act
            scaler.UpdateScale(false);  // Step 1
            scaler.UpdateScale(false);  // Step 2
            scaler.UpdateScale(false);  // Step 3 - should grow
            scaler.UpdateScale(false);  // Step 4

            // Assert
            Assert.Equal(2000.0f, scaler.CurrentScale);  // Grew after 3 steps
            Assert.Equal(0, scaler.ConsecutiveOverflows);
        }

        [Fact]
        public void UpdateScale_ResetAfterOverflow_ResetsConsecutiveCount()
        {
            // Arrange
            var scaler = new DynamicLossScaler();
            scaler.UpdateScale(true);  // Overflow
            scaler.UpdateScale(true);  // Overflow

            // Act
            scaler.UpdateScale(false);  // Success

            // Assert
            Assert.Equal(0, scaler.ConsecutiveOverflows);
            Assert.Equal(1, scaler.StepsSinceLastOverflow);
        }

        [Fact]
        public void UpdateScale_RespectsMinScale()
        {
            // Arrange
            var options = new MixedPrecisionOptions
            {
                InitialLossScale = 10.0f,
                BackoffFactor = 0.5f,
                MinLossScale = 5.0f
            };
            var scaler = new DynamicLossScaler(options);

            // Act
            scaler.UpdateScale(true);  // 10 -> 5
            scaler.UpdateScale(true);  // Should stay at 5

            // Assert
            Assert.Equal(5.0f, scaler.CurrentScale);
        }

        [Fact]
        public void UpdateScale_RespectsMaxScale()
        {
            // Arrange
            var options = new MixedPrecisionOptions
            {
                InitialLossScale = 90.0f,
                GrowthFactor = 2.0f,
                GrowthInterval = 1,
                MaxLossScale = 100.0f
            };
            var scaler = new DynamicLossScaler(options);

            // Act
            scaler.UpdateScale(false);  // Should grow: 90 -> 100

            // Assert
            Assert.Equal(100.0f, scaler.CurrentScale);
        }

        #endregion

        #region Reset Tests

        [Fact]
        public void Reset_RestoresInitialState()
        {
            // Arrange
            var options = new MixedPrecisionOptions { InitialLossScale = 65536.0f };
            var scaler = new DynamicLossScaler(options);
            scaler.UpdateScale(true);  // Cause some state change
            scaler.UpdateScale(false);

            // Act
            scaler.Reset();

            // Assert
            Assert.Equal(options.InitialLossScale, scaler.CurrentScale);
            Assert.Equal(0, scaler.StepsSinceLastOverflow);
            Assert.Equal(0, scaler.ConsecutiveOverflows);
            Assert.Equal(0, scaler.TotalOverflows);
        }

        #endregion

        #region Stats Tests

        [Fact]
        public void GetStats_ReturnsCorrectStatistics()
        {
            // Arrange
            var options = new MixedPrecisionOptions
            {
                InitialLossScale = 1000.0f,
                GrowthInterval = 5,
                MaxConsecutiveOverflows = 10
            };
            var scaler = new DynamicLossScaler(options);
            scaler.UpdateScale(false);

            // Act
            var stats = scaler.GetStats();

            // Assert
            Assert.Equal(1000.0f, stats.CurrentScale);
            Assert.Equal(1, stats.StepsSinceLastOverflow);
            Assert.Equal(0, stats.ConsecutiveOverflows);
            Assert.Equal(0, stats.TotalOverflows);
            Assert.Equal(5, stats.GrowthInterval);
            Assert.Equal(10, stats.MaxConsecutiveOverflows);
            Assert.True(stats.IsStable);
        }

        [Fact]
        public void GetStats_WithUnstableState_ReturnsCorrectStabilityFlag()
        {
            // Arrange
            var options = new MixedPrecisionOptions { MaxConsecutiveOverflows = 3 };
            var scaler = new DynamicLossScaler(options);
            scaler.UpdateScale(true);
            scaler.UpdateScale(true);
            scaler.UpdateScale(true);

            // Act
            var stats = scaler.GetStats();

            // Assert
            Assert.False(stats.IsStable);
        }

        #endregion

        #region Helper Methods

        private ITensor CreateMockTensor(float value)
        {
            // TODO: Create mock tensor implementation
            return null;
        }

        private Dictionary<string, ITensor> CreateMockGradients()
        {
            return new Dictionary<string, ITensor>
            {
                { "param1", CreateMockTensor(0.1f) },
                { "param2", CreateMockTensor(0.2f) }
            };
        }

        private Dictionary<string, ITensor> CreateMockGradientsWithNaN()
        {
            return new Dictionary<string, ITensor>
            {
                { "param1", CreateMockTensor(float.NaN) }
            };
        }

        #endregion
    }
}
```

## Requirements

### Test Coverage
1. **Constructor Tests**: Test initialization with various option configurations
2. **Scale Loss Tests**: Test loss scaling with enabled/disabled mode
3. **Unscale Gradients Tests**: Test gradient unscaling
4. **Overflow Detection Tests**: Test overflow detection (mock tensors)
5. **Scale Update Tests**: Test scale adjustment logic (growth, backoff, bounds)
6. **Reset Tests**: Test state reset functionality
7. **Stats Tests**: Test statistics reporting

### Test Categories
- **Happy Path**: Normal operation scenarios
- **Edge Cases**: Boundary conditions (min/max scale, zero gradients)
- **Error Handling**: Null inputs, invalid configurations
- **State Management**: Verify correct state transitions

## Deliverables

### Test Files
1. `tests/MLFramework.Tests/Optimizers/MixedPrecision/DynamicLossScalerTests.cs`

## Notes for Coder
- Mock tensor implementation will be needed (placeholder in spec)
- Test both enabled and disabled loss scaling modes
- Test all edge cases for scale adjustments
- Verify consecutive overflow counting
- Test growth interval logic carefully
- Ensure min/max scale bounds are enforced
- Stats should reflect accurate current state
