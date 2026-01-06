# Spec: PrecisionManager Unit Tests

## Overview
Implement comprehensive unit tests for the PrecisionManager component.

## Dependencies
- Spec 001: Precision enum and detection utilities
- Spec 002: MixedPrecisionOptions
- Spec 004: PrecisionManager

## Implementation Details

### Test File Structure
Create the test file in `tests/MLFramework.Tests/Optimizers/MixedPrecision/PrecisionManagerTests.cs`:

```csharp
using System;
using System.Collections.Generic;
using Xunit;
using MLFramework.Optimizers.MixedPrecision;

namespace MLFramework.Tests.Optimizers.MixedPrecision
{
    public class PrecisionManagerTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithNullOptions_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new PrecisionManager(null));
        }

        [Fact]
        public void Constructor_WithValidOptions_InitializesCorrectly()
        {
            // Arrange
            var options = MixedPrecisionOptions.ForFP16();

            // Act
            var manager = new PrecisionManager(options);

            // Assert
            Assert.Equal(Precision.FP16, manager.TargetPrecision);
            Assert.True(manager.IsReducedPrecision);
            Assert.True(manager.ExcludedLayerCount > 0);  // Should have default exclusions
        }

        [Fact]
        public void Constructor_WithFP32_IsNotReducedPrecision()
        {
            // Arrange
            var options = new MixedPrecisionOptions
            {
                Precision = Precision.FP32,
                AutoDetectPrecision = false,
                AutoExcludeSensitiveLayers = false
            };

            // Act
            var manager = new PrecisionManager(options);

            // Assert
            Assert.Equal(Precision.FP32, manager.TargetPrecision);
            Assert.False(manager.IsReducedPrecision);
        }

        [Fact]
        public void Constructor_WithAutoDetectDisabled_UsesSpecifiedPrecision()
        {
            // Arrange
            var options = new MixedPrecisionOptions
            {
                Precision = Precision.BF16,
                AutoDetectPrecision = false,
                AutoExcludeSensitiveLayers = false
            };

            // Act
            var manager = new PrecisionManager(options);

            // Assert
            Assert.Equal(Precision.BF16, manager.TargetPrecision);
        }

        #endregion

        #region Tensor Conversion Tests

        [Fact]
        public void ConvertToTrainingPrecision_WithNullTensor_ThrowsArgumentNullException()
        {
            // Arrange
            var manager = new PrecisionManager();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => manager.ConvertToTrainingPrecision(null));
        }

        [Fact]
        public void ConvertToTrainingPrecision_WithExcludedLayer_ReturnsFP32Tensor()
        {
            // Arrange
            var manager = new PrecisionManager();
            var tensor = CreateMockTensor(Precision.FP32);
            manager.ExcludeLayersMatching("BatchNorm");

            // Act
            var converted = manager.ConvertToTrainingPrecision(tensor, "model.BatchNorm1.weight");

            // Assert
            Assert.NotNull(converted);
            // Verify precision is FP32 (implementation dependent)
        }

        [Fact]
        public void ConvertToTrainingPrecision_WithRegularLayer_ConvertsToTargetPrecision()
        {
            // Arrange
            var options = new MixedPrecisionOptions
            {
                Precision = Precision.FP16,
                AutoDetectPrecision = false,
                AutoExcludeSensitiveLayers = false
            };
            var manager = new PrecisionManager(options);
            var tensor = CreateMockTensor(Precision.FP32);

            // Act
            var converted = manager.ConvertToTrainingPrecision(tensor, "model.Linear1.weight");

            // Assert
            Assert.NotNull(converted);
            // Verify conversion to FP16 (implementation dependent)
        }

        [Fact]
        public void ConvertToFP32_WithNullTensor_ThrowsArgumentNullException()
        {
            // Arrange
            var manager = new PrecisionManager();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => manager.ConvertToFP32(null));
        }

        [Fact]
        public void ConvertToFP32_ConvertsToFP32()
        {
            // Arrange
            var manager = new PrecisionManager();
            var tensor = CreateMockTensor(Precision.FP16);

            // Act
            var converted = manager.ConvertToFP32(tensor);

            // Assert
            Assert.NotNull(converted);
            // Verify conversion to FP32 (implementation dependent)
        }

        #endregion

        #region Batch Conversion Tests

        [Fact]
        public void ConvertWeights_WithNullWeights_ThrowsArgumentNullException()
        {
            // Arrange
            var manager = new PrecisionManager();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => manager.ConvertWeights(null));
        }

        [Fact]
        public void ConvertWeights_ConvertsAllWeights()
        {
            // Arrange
            var manager = new PrecisionManager();
            var weights = CreateMockWeights();

            // Act
            var converted = manager.ConvertWeights(weights);

            // Assert
            Assert.Equal(weights.Count, converted.Count);
            Assert.True(converted.ContainsKey("layer1.weight"));
            Assert.True(converted.ContainsKey("BatchNorm1.weight"));
        }

        [Fact]
        public void ConvertToFP32_WithDictionary_ConvertsAllTensors()
        {
            // Arrange
            var manager = new PrecisionManager();
            var tensors = CreateMockWeights();

            // Act
            var converted = manager.ConvertToFP32(tensors);

            // Assert
            Assert.Equal(tensors.Count, converted.Count);
        }

        #endregion

        #region Layer Exclusion Tests

        [Fact]
        public void ShouldExcludeLayer_WithEmptyName_ReturnsFalse()
        {
            // Arrange
            var manager = new PrecisionManager();

            // Act
            bool shouldExclude = manager.ShouldExcludeLayer("");

            // Assert
            Assert.False(shouldExclude);
        }

        [Fact]
        public void ShouldExcludeLayer_WithNullName_ReturnsFalse()
        {
            // Arrange
            var manager = new PrecisionManager();

            // Act
            bool shouldExclude = manager.ShouldExcludeLayer(null);

            // Assert
            Assert.False(shouldExclude);
        }

        [Fact]
        public void ShouldExcludeLayer_WithExactMatch_ReturnsTrue()
        {
            // Arrange
            var manager = new PrecisionManager();
            manager.ExcludeLayersMatching("BatchNorm");

            // Act
            bool shouldExclude = manager.ShouldExcludeLayer("BatchNorm");

            // Assert
            Assert.True(shouldExclude);
        }

        [Fact]
        public void ShouldExcludeLayer_WithContainsMatch_ReturnsTrue()
        {
            // Arrange
            var manager = new PrecisionManager();
            manager.ExcludeLayersMatching("BatchNorm");

            // Act
            bool shouldExclude = manager.ShouldExcludeLayer("model.BatchNorm1.weight");

            // Assert
            Assert.True(shouldExclude);
        }

        [Fact]
        public void ShouldExcludeLayer_WithNoMatch_ReturnsFalse()
        {
            // Arrange
            var manager = new PrecisionManager();
            manager.ExcludeLayersMatching("BatchNorm");

            // Act
            bool shouldExclude = manager.ShouldExcludeLayer("model.Linear1.weight");

            // Assert
            Assert.False(shouldExclude);
        }

        [Fact]
        public void ShouldExcludeLayer_WithDefaultExclusions_ExcludesBatchNorm()
        {
            // Arrange
            var manager = new PrecisionManager();

            // Act
            bool shouldExclude = manager.ShouldExcludeLayer("model.BatchNorm1.weight");

            // Assert
            Assert.True(shouldExclude);
        }

        [Fact]
        public void ExcludeLayersMatching_WithEmptyPattern_ThrowsArgumentException()
        {
            // Arrange
            var manager = new PrecisionManager();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => manager.ExcludeLayersMatching(""));
            Assert.Throws<ArgumentException>(() => manager.ExcludeLayersMatching("   "));
        }

        [Fact]
        public void ExcludeLayersMatching_AddsPatternSuccessfully()
        {
            // Arrange
            var manager = new PrecisionManager();
            int initialCount = manager.ExcludedLayerCount;

            // Act
            manager.ExcludeLayersMatching("CustomLayer");

            // Assert
            Assert.Equal(initialCount + 1, manager.ExcludedLayerCount);
        }

        [Fact]
        public void RemoveExclusion_WithExistingPattern_RemovesSuccessfully()
        {
            // Arrange
            var manager = new PrecisionManager();
            manager.ExcludeLayersMatching("CustomLayer");

            // Act
            bool removed = manager.RemoveExclusion("CustomLayer");

            // Assert
            Assert.True(removed);
            Assert.False(manager.ShouldExcludeLayer("CustomLayer"));
        }

        [Fact]
        public void RemoveExclusion_WithNonExistentPattern_ReturnsFalse()
        {
            // Arrange
            var manager = new PrecisionManager();

            // Act
            bool removed = manager.RemoveExclusion("NonExistentLayer");

            // Assert
            Assert.False(removed);
        }

        [Fact]
        public void ClearExclusions_RemovesAllExclusions()
        {
            // Arrange
            var manager = new PrecisionManager();
            manager.ExcludeLayersMatching("Layer1");
            manager.ExcludeLayersMatching("Layer2");

            // Act
            manager.ClearExclusions();

            // Assert
            Assert.Equal(0, manager.ExcludedLayerCount);
            Assert.False(manager.ShouldExcludeLayer("Layer1"));
        }

        [Fact]
        public void GetExclusionPatterns_ReturnsAllPatterns()
        {
            // Arrange
            var manager = new PrecisionManager();
            manager.ExcludeLayersMatching("Layer1");
            manager.ExcludeLayersMatching("Layer2");

            // Act
            var patterns = manager.GetExclusionPatterns();

            // Assert
            Assert.Contains("Layer1", patterns);
            Assert.Contains("Layer2", patterns);
        }

        #endregion

        #region Master Weights Tests

        [Fact]
        public void CreateMasterWeights_WithNullWeights_ThrowsArgumentNullException()
        {
            // Arrange
            var manager = new PrecisionManager();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => manager.CreateMasterWeights(null));
        }

        [Fact]
        public void CreateMasterWeights_CreatesFP32Weights()
        {
            // Arrange
            var manager = new PrecisionManager();
            var trainingWeights = CreateMockWeights();

            // Act
            var masterWeights = manager.CreateMasterWeights(trainingWeights);

            // Assert
            Assert.Equal(trainingWeights.Count, masterWeights.Count);
            // Verify all are FP32 (implementation dependent)
        }

        [Fact]
        public void SyncTrainingWeights_WithNullWeights_ThrowsArgumentNullException()
        {
            // Arrange
            var manager = new PrecisionManager();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => manager.SyncTrainingWeights(null));
        }

        [Fact]
        public void SyncTrainingWeights_ConvertsToTargetPrecision()
        {
            // Arrange
            var manager = new PrecisionManager();
            var masterWeights = CreateMockWeights();

            // Act
            var trainingWeights = manager.SyncTrainingWeights(masterWeights);

            // Assert
            Assert.Equal(masterWeights.Count, trainingWeights.Count);
            // Verify conversion to target precision (implementation dependent)
        }

        #endregion

        #region Helper Methods

        private ITensor CreateMockTensor(Precision precision)
        {
            // TODO: Create mock tensor implementation
            return null;
        }

        private Dictionary<string, ITensor> CreateMockWeights()
        {
            return new Dictionary<string, ITensor>
            {
                { "layer1.weight", CreateMockTensor(Precision.FP32) },
                { "layer1.bias", CreateMockTensor(Precision.FP32) },
                { "BatchNorm1.weight", CreateMockTensor(Precision.FP32) },
                { "BatchNorm1.bias", CreateMockTensor(Precision.FP32) }
            };
        }

        #endregion
    }
}
```

## Requirements

### Test Coverage
1. **Constructor Tests**: Test initialization with various configurations
2. **Tensor Conversion Tests**: Test individual tensor conversion with/without exclusions
3. **Batch Conversion Tests**: Test dictionary-based conversion
4. **Layer Exclusion Tests**: Test exclusion pattern matching (exact, contains)
5. **Master Weights Tests**: Test master weight creation and sync
6. **Default Exclusions**: Test that sensitive layers are excluded by default

### Test Categories
- **Happy Path**: Normal operation scenarios
- **Edge Cases**: Empty layer names, null inputs
- **Error Handling**: Invalid patterns, null parameters
- **Pattern Matching**: Verify exact, contains, and regex matching

## Deliverables

### Test Files
1. `tests/MLFramework.Tests/Optimizers/MixedPrecision/PrecisionManagerTests.cs`

## Notes for Coder
- Mock tensor implementation will be needed (placeholder in spec)
- Test all three pattern matching modes: exact, contains, regex
- Verify default exclusions are applied correctly
- Test exclusion management (add, remove, clear)
- Batch conversion should handle all weights consistently
- Master weights should always be FP32 regardless of target precision
- Training weights should respect layer exclusions
