using MLFramework.Optimizers.MixedPrecision;
using Xunit;

namespace MLFramework.Tests.Amp.Scalers;

/// <summary>
/// Tests for MixedPrecisionOptions class
/// </summary>
public class MixedPrecisionOptionsTests
{
    [Fact]
    public void Constructor_DefaultValues_AreCorrect()
    {
        var options = new MixedPrecisionOptions();

        Assert.Equal(Precision.FP16, options.Precision);
        Assert.True(options.AutoDetectPrecision);
        Assert.Equal(65536.0f, options.InitialLossScale);
        Assert.Equal(2.0f, options.GrowthFactor);
        Assert.Equal(0.5f, options.BackoffFactor);
        Assert.Equal(1e9f, options.MaxLossScale);
        Assert.Equal(1.0f, options.MinLossScale);
        Assert.Equal(2000, options.GrowthInterval);
        Assert.True(options.EnableDynamicLossScaling);
        Assert.True(options.AutoExcludeSensitiveLayers);
        Assert.Equal(1.0f, options.MaxGradNorm);
        Assert.True(options.EnableGradientClipping);
        Assert.Equal(10, options.MaxConsecutiveOverflows);
        Assert.True(options.EnableAutoFallback);
        Assert.True(options.LogFallbackEvents);
        Assert.False(options.EnablePerformanceMonitoring);
        Assert.Equal(100, options.PerformanceLogInterval);
    }

    [Fact]
    public void Validate_WithValidOptions_DoesNotThrow()
    {
        var options = new MixedPrecisionOptions();

        var exception = Record.Exception(() => options.Validate());

        Assert.Null(exception);
    }

    [Fact]
    public void Validate_WithInvalidInitialLossScale_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            InitialLossScale = 0
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithNegativeInitialLossScale_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            InitialLossScale = -100.0f
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithGrowthFactorLessThanOne_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            GrowthFactor = 0.9f
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithGrowthFactorEqualsOne_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            GrowthFactor = 1.0f
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithInvalidBackoffFactor_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            BackoffFactor = 1.5f
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithBackoffFactorZero_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            BackoffFactor = 0
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithBackoffFactorNegative_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            BackoffFactor = -0.5f
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithMaxLessThanMinLossScale_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            MaxLossScale = 100.0f,
            MinLossScale = 200.0f
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithMaxEqualsMinLossScale_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            MaxLossScale = 100.0f,
            MinLossScale = 100.0f
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithNegativeGrowthInterval_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            GrowthInterval = -100
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithZeroGrowthInterval_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            GrowthInterval = 0
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithNegativeMaxGradNorm_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            MaxGradNorm = -1.0f
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithNegativeMaxConsecutiveOverflows_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            MaxConsecutiveOverflows = -10
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithZeroMaxConsecutiveOverflows_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            MaxConsecutiveOverflows = 0
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithNegativePerformanceLogInterval_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            PerformanceLogInterval = -50
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithZeroPerformanceLogInterval_ThrowsArgumentException()
    {
        var options = new MixedPrecisionOptions
        {
            PerformanceLogInterval = 0
        };

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Clone_CreatesDeepCopy()
    {
        var options = new MixedPrecisionOptions
        {
            InitialLossScale = 1000.0f,
            MaxGradNorm = 0.5f
        };
        options.Fp32LayerPatterns.Add("CustomLayer");

        var clone = options.Clone();

        Assert.Equal(options.InitialLossScale, clone.InitialLossScale);
        Assert.Equal(options.MaxGradNorm, clone.MaxGradNorm);
        Assert.Equal(options.Fp32LayerPatterns.Count, clone.Fp32LayerPatterns.Count);
        Assert.Contains("CustomLayer", clone.Fp32LayerPatterns);
    }

    [Fact]
    public void Clone_ModifyingOriginal_DoesNotAffectClone()
    {
        var options = new MixedPrecisionOptions
        {
            InitialLossScale = 1000.0f
        };
        var clone = options.Clone();

        options.InitialLossScale = 2000.0f;
        options.Fp32LayerPatterns.Add("NewLayer");

        Assert.Equal(1000.0f, clone.InitialLossScale);
        Assert.DoesNotContain("NewLayer", clone.Fp32LayerPatterns);
    }

    [Fact]
    public void Clone_ModifyingClone_DoesNotAffectOriginal()
    {
        var options = new MixedPrecisionOptions
        {
            InitialLossScale = 1000.0f
        };
        var clone = options.Clone();

        clone.InitialLossScale = 2000.0f;
        clone.Fp32LayerPatterns.Add("NewLayer");

        Assert.Equal(1000.0f, options.InitialLossScale);
        Assert.DoesNotContain("NewLayer", options.Fp32LayerPatterns);
    }

    [Fact]
    public void ForFP16_CreatesCorrectOptions()
    {
        var options = MixedPrecisionOptions.ForFP16();

        Assert.Equal(Precision.FP16, options.Precision);
        Assert.False(options.AutoDetectPrecision);
        Assert.Equal(65536.0f, options.InitialLossScale);
        Assert.Equal(2.0f, options.GrowthFactor);
        Assert.Equal(0.5f, options.BackoffFactor);
    }

    [Fact]
    public void ForBF16_CreatesCorrectOptions()
    {
        var options = MixedPrecisionOptions.ForBF16();

        Assert.Equal(Precision.BF16, options.Precision);
        Assert.False(options.AutoDetectPrecision);
        Assert.Equal(1.0f, options.InitialLossScale);
        Assert.Equal(2.0f, options.GrowthFactor);
        Assert.Equal(0.5f, options.BackoffFactor);
        Assert.Equal(1000, options.GrowthInterval);
    }

    [Fact]
    public void Conservative_CreatesCorrectOptions()
    {
        var options = MixedPrecisionOptions.Conservative();

        Assert.Equal(Precision.FP16, options.Precision);
        Assert.False(options.AutoDetectPrecision);
        Assert.Equal(8192.0f, options.InitialLossScale);
        Assert.Equal(1.5f, options.GrowthFactor);
        Assert.Equal(0.75f, options.BackoffFactor);
        Assert.Equal(4000, options.GrowthInterval);
        Assert.True(options.AutoExcludeSensitiveLayers);
        Assert.Equal(0.5f, options.MaxGradNorm);
        Assert.Equal(5, options.MaxConsecutiveOverflows);
    }

    [Theory]
    [InlineData(1.0f)]
    [InlineData(100.0f)]
    [InlineData(1000.0f)]
    [InlineData(65536.0f)]
    public void InitialLossScale_AcceptsValidValues(float value)
    {
        var options = new MixedPrecisionOptions
        {
            InitialLossScale = value
        };

        Assert.Equal(value, options.InitialLossScale);
    }

    [Theory]
    [InlineData(1.1f)]
    [InlineData(1.5f)]
    [InlineData(2.0f)]
    [InlineData(10.0f)]
    public void GrowthFactor_AcceptsValidValues(float value)
    {
        var options = new MixedPrecisionOptions
        {
            GrowthFactor = value
        };

        Assert.Equal(value, options.GrowthFactor);
    }

    [Theory]
    [InlineData(0.1f)]
    [InlineData(0.25f)]
    [InlineData(0.5f)]
    [InlineData(0.9f)]
    public void BackoffFactor_AcceptsValidValues(float value)
    {
        var options = new MixedPrecisionOptions
        {
            BackoffFactor = value
        };

        Assert.Equal(value, options.BackoffFactor);
    }
}
