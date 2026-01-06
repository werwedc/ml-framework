using System;
using Xunit;
using MLFramework.Optimizers.MixedPrecision;

namespace MLFramework.Tests.Optimizers.MixedPrecision;

public class MixedPrecisionOptionsTests
{
    #region Validation Tests

    [Fact]
    public void Validate_WithValidOptions_DoesNotThrow()
    {
        // Arrange
        var options = new MixedPrecisionOptions();

        // Act & Assert
        options.Validate(); // Should not throw
    }

    [Fact]
    public void Validate_WithInvalidInitialLossScale_ThrowsArgumentException()
    {
        // Arrange
        var options = new MixedPrecisionOptions
        {
            InitialLossScale = -1.0f
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithZeroInitialLossScale_ThrowsArgumentException()
    {
        // Arrange
        var options = new MixedPrecisionOptions
        {
            InitialLossScale = 0.0f
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithGrowthFactorLessThanOrEqualToOne_ThrowsArgumentException()
    {
        // Arrange
        var options1 = new MixedPrecisionOptions { GrowthFactor = 1.0f };
        var options2 = new MixedPrecisionOptions { GrowthFactor = 0.5f };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options1.Validate());
        Assert.Throws<ArgumentException>(() => options2.Validate());
    }

    [Fact]
    public void Validate_WithBackoffFactorZeroOrNegative_ThrowsArgumentException()
    {
        // Arrange
        var options1 = new MixedPrecisionOptions { BackoffFactor = 0.0f };
        var options2 = new MixedPrecisionOptions { BackoffFactor = -0.5f };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options1.Validate());
        Assert.Throws<ArgumentException>(() => options2.Validate());
    }

    [Fact]
    public void Validate_WithBackoffFactorGreaterThanOrEqualToOne_ThrowsArgumentException()
    {
        // Arrange
        var options1 = new MixedPrecisionOptions { BackoffFactor = 1.0f };
        var options2 = new MixedPrecisionOptions { BackoffFactor = 1.5f };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options1.Validate());
        Assert.Throws<ArgumentException>(() => options2.Validate());
    }

    [Fact]
    public void Validate_WithMaxLossScaleLessThanMinLossScale_ThrowsArgumentException()
    {
        // Arrange
        var options = new MixedPrecisionOptions
        {
            MaxLossScale = 100.0f,
            MinLossScale = 200.0f
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithMaxLossScaleEqualToMinLossScale_ThrowsArgumentException()
    {
        // Arrange
        var options = new MixedPrecisionOptions
        {
            MaxLossScale = 100.0f,
            MinLossScale = 100.0f
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithNegativeGrowthInterval_ThrowsArgumentException()
    {
        // Arrange
        var options = new MixedPrecisionOptions
        {
            GrowthInterval = -1
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithZeroGrowthInterval_ThrowsArgumentException()
    {
        // Arrange
        var options = new MixedPrecisionOptions
        {
            GrowthInterval = 0
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithNegativeMaxGradNorm_ThrowsArgumentException()
    {
        // Arrange
        var options = new MixedPrecisionOptions
        {
            MaxGradNorm = -1.0f
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithNegativeMaxConsecutiveOverflows_ThrowsArgumentException()
    {
        // Arrange
        var options = new MixedPrecisionOptions
        {
            MaxConsecutiveOverflows = -1
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithZeroMaxConsecutiveOverflows_ThrowsArgumentException()
    {
        // Arrange
        var options = new MixedPrecisionOptions
        {
            MaxConsecutiveOverflows = 0
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithNegativePerformanceLogInterval_ThrowsArgumentException()
    {
        // Arrange
        var options = new MixedPrecisionOptions
        {
            PerformanceLogInterval = -1
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithZeroPerformanceLogInterval_ThrowsArgumentException()
    {
        // Arrange
        var options = new MixedPrecisionOptions
        {
            PerformanceLogInterval = 0
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    #endregion

    #region Clone Tests

    [Fact]
    public void Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new MixedPrecisionOptions
        {
            Precision = Precision.FP16,
            InitialLossScale = 12345.0f,
            GrowthFactor = 3.0f
        };

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotSame(original, clone);
        Assert.Equal(original.Precision, clone.Precision);
        Assert.Equal(original.InitialLossScale, clone.InitialLossScale);
        Assert.Equal(original.GrowthFactor, clone.GrowthFactor);
    }

    [Fact]
    public void Clone_ModificationDoesNotAffectOriginal()
    {
        // Arrange
        var original = new MixedPrecisionOptions();
        var clone = original.Clone();

        // Act
        clone.InitialLossScale = 9999.0f;
        clone.GrowthFactor = 5.0f;
        clone.EnableDynamicLossScaling = false;

        // Assert
        Assert.Equal(65536.0f, original.InitialLossScale);
        Assert.Equal(2.0f, original.GrowthFactor);
        Assert.True(original.EnableDynamicLossScaling);
    }

    [Fact]
    public void Clone_CreatesDeepCopyOfLayerPatterns()
    {
        // Arrange
        var original = new MixedPrecisionOptions();
        var clone = original.Clone();

        // Act
        clone.Fp32LayerPatterns.Add("CustomLayer");

        // Assert
        Assert.DoesNotContain("CustomLayer", original.Fp32LayerPatterns);
        Assert.Contains("CustomLayer", clone.Fp32LayerPatterns);
    }

    [Fact]
    public void Clone_PreservesDefaultLayerPatterns()
    {
        // Arrange
        var original = new MixedPrecisionOptions();

        // Act
        var clone = original.Clone();

        // Assert
        Assert.Equal(4, clone.Fp32LayerPatterns.Count);
        Assert.Contains("BatchNorm", clone.Fp32LayerPatterns);
        Assert.Contains("LayerNorm", clone.Fp32LayerPatterns);
        Assert.Contains("InstanceNorm", clone.Fp32LayerPatterns);
        Assert.Contains("GroupNorm", clone.Fp32LayerPatterns);
    }

    #endregion

    #region Factory Methods Tests

    [Fact]
    public void ForFP16_ReturnsValidOptions()
    {
        // Act
        var options = MixedPrecisionOptions.ForFP16();

        // Assert
        Assert.NotNull(options);
        Assert.Equal(Precision.FP16, options.Precision);
        Assert.False(options.AutoDetectPrecision);
        Assert.Equal(65536.0f, options.InitialLossScale);
        Assert.Equal(2.0f, options.GrowthFactor);
        Assert.Equal(0.5f, options.BackoffFactor);

        // Should validate successfully
        options.Validate();
    }

    [Fact]
    public void ForBF16_ReturnsValidOptions()
    {
        // Act
        var options = MixedPrecisionOptions.ForBF16();

        // Assert
        Assert.NotNull(options);
        Assert.Equal(Precision.BF16, options.Precision);
        Assert.False(options.AutoDetectPrecision);
        Assert.Equal(1.0f, options.InitialLossScale); // BF16 has wider range
        Assert.Equal(2.0f, options.GrowthFactor);
        Assert.Equal(0.5f, options.BackoffFactor);
        Assert.Equal(1000, options.GrowthInterval);

        // Should validate successfully
        options.Validate();
    }

    [Fact]
    public void Conservative_ReturnsValidOptions()
    {
        // Act
        var options = MixedPrecisionOptions.Conservative();

        // Assert
        Assert.NotNull(options);
        Assert.Equal(Precision.FP16, options.Precision);
        Assert.False(options.AutoDetectPrecision);
        Assert.Equal(8192.0f, options.InitialLossScale); // Lower initial scale
        Assert.Equal(1.5f, options.GrowthFactor); // Slower growth
        Assert.Equal(0.75f, options.BackoffFactor); // Slower backoff
        Assert.Equal(4000, options.GrowthInterval); // Longer interval
        Assert.True(options.AutoExcludeSensitiveLayers);
        Assert.Equal(0.5f, options.MaxGradNorm); // Stricter clipping
        Assert.Equal(5, options.MaxConsecutiveOverflows); // Faster fallback

        // Should validate successfully
        options.Validate();
    }

    [Fact]
    public void FactoryMethods_ReturnNewInstances()
    {
        // Act
        var fp16Options1 = MixedPrecisionOptions.ForFP16();
        var fp16Options2 = MixedPrecisionOptions.ForFP16();

        // Assert
        Assert.NotSame(fp16Options1, fp16Options2);
    }

    #endregion

    #region Default Values Tests

    [Fact]
    public void Constructor_SetsDefaultPrecisionToFP16()
    {
        // Arrange & Act
        var options = new MixedPrecisionOptions();

        // Assert
        Assert.Equal(Precision.FP16, options.Precision);
    }

    [Fact]
    public void Constructor_SetsDefaultAutoDetectPrecisionToTrue()
    {
        // Arrange & Act
        var options = new MixedPrecisionOptions();

        // Assert
        Assert.True(options.AutoDetectPrecision);
    }

    [Fact]
    public void Constructor_SetsDefaultLossScalingValues()
    {
        // Arrange & Act
        var options = new MixedPrecisionOptions();

        // Assert
        Assert.Equal(65536.0f, options.InitialLossScale);
        Assert.Equal(2.0f, options.GrowthFactor);
        Assert.Equal(0.5f, options.BackoffFactor);
        Assert.Equal(1e9f, options.MaxLossScale);
        Assert.Equal(1.0f, options.MinLossScale);
        Assert.Equal(2000, options.GrowthInterval);
        Assert.True(options.EnableDynamicLossScaling);
    }

    [Fact]
    public void Constructor_SetsDefaultLayerExclusionValues()
    {
        // Arrange & Act
        var options = new MixedPrecisionOptions();

        // Assert
        Assert.True(options.AutoExcludeSensitiveLayers);
        Assert.Equal(4, options.Fp32LayerPatterns.Count);
        Assert.Contains("BatchNorm", options.Fp32LayerPatterns);
    }

    [Fact]
    public void Constructor_SetsDefaultGradientClippingValues()
    {
        // Arrange & Act
        var options = new MixedPrecisionOptions();

        // Assert
        Assert.Equal(1.0f, options.MaxGradNorm);
        Assert.True(options.EnableGradientClipping);
    }

    [Fact]
    public void Constructor_SetsDefaultFallbackValues()
    {
        // Arrange & Act
        var options = new MixedPrecisionOptions();

        // Assert
        Assert.Equal(10, options.MaxConsecutiveOverflows);
        Assert.True(options.EnableAutoFallback);
        Assert.True(options.LogFallbackEvents);
    }

    [Fact]
    public void Constructor_SetsDefaultPerformanceValues()
    {
        // Arrange & Act
        var options = new MixedPrecisionOptions();

        // Assert
        Assert.False(options.EnablePerformanceMonitoring);
        Assert.Equal(100, options.PerformanceLogInterval);
    }

    #endregion

    #region Layer Patterns Tests

    [Fact]
    public void Fp32LayerPatterns_InitializedWithDefaultPatterns()
    {
        // Arrange & Act
        var options = new MixedPrecisionOptions();

        // Assert
        Assert.NotNull(options.Fp32LayerPatterns);
        Assert.Equal(4, options.Fp32LayerPatterns.Count);
    }

    [Fact]
    public void Fp32LayerPatterns_CanBeModified()
    {
        // Arrange
        var options = new MixedPrecisionOptions();

        // Act
        options.Fp32LayerPatterns.Add("CustomPattern");

        // Assert
        Assert.Equal(5, options.Fp32LayerPatterns.Count);
        Assert.Contains("CustomPattern", options.Fp32LayerPatterns);
    }

    [Fact]
    public void Fp32LayerPatterns_CanBeCleared()
    {
        // Arrange
        var options = new MixedPrecisionOptions();

        // Act
        options.Fp32LayerPatterns.Clear();

        // Assert
        Assert.Empty(options.Fp32LayerPatterns);
    }

    #endregion

    #region All Properties Test

    [Fact]
    public void AllProperties_AreSettable()
    {
        // Arrange & Act
        var options = new MixedPrecisionOptions
        {
            Precision = Precision.BF16,
            AutoDetectPrecision = false,
            InitialLossScale = 128.0f,
            GrowthFactor = 1.2f,
            BackoffFactor = 0.8f,
            MaxLossScale = 1e8f,
            MinLossScale = 0.5f,
            GrowthInterval = 500,
            EnableDynamicLossScaling = false,
            AutoExcludeSensitiveLayers = false,
            MaxGradNorm = 2.0f,
            EnableGradientClipping = false,
            MaxConsecutiveOverflows = 20,
            EnableAutoFallback = false,
            LogFallbackEvents = false,
            EnablePerformanceMonitoring = true,
            PerformanceLogInterval = 200
        };

        // Assert
        Assert.Equal(Precision.BF16, options.Precision);
        Assert.False(options.AutoDetectPrecision);
        Assert.Equal(128.0f, options.InitialLossScale);
        Assert.Equal(1.2f, options.GrowthFactor);
        Assert.Equal(0.8f, options.BackoffFactor);
        Assert.Equal(1e8f, options.MaxLossScale);
        Assert.Equal(0.5f, options.MinLossScale);
        Assert.Equal(500, options.GrowthInterval);
        Assert.False(options.EnableDynamicLossScaling);
        Assert.False(options.AutoExcludeSensitiveLayers);
        Assert.Equal(2.0f, options.MaxGradNorm);
        Assert.False(options.EnableGradientClipping);
        Assert.Equal(20, options.MaxConsecutiveOverflows);
        Assert.False(options.EnableAutoFallback);
        Assert.False(options.LogFallbackEvents);
        Assert.True(options.EnablePerformanceMonitoring);
        Assert.Equal(200, options.PerformanceLogInterval);

        // Should validate successfully
        options.Validate();
    }

    #endregion
}
