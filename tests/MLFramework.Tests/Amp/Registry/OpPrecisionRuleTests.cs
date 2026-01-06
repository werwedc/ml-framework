using MLFramework.Amp;
using MLFramework.Core;
using Xunit;

namespace MLFramework.Tests.Amp.Registry;

/// <summary>
/// Tests for OpPrecisionRule class
/// </summary>
public class OpPrecisionRuleTests
{
    private class TestOperation { }

    [Fact]
    public void Constructor_WithValidParameters_CreatesRule()
    {
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Lower,
            OpPrecision.Keep,
            10
        );

        Assert.Equal(typeof(TestOperation), rule.OperationType);
        Assert.Equal(OpPrecision.Lower, rule.ForwardPrecision);
        Assert.Equal(OpPrecision.Keep, rule.BackwardPrecision);
        Assert.Equal(10, rule.Priority);
    }

    [Fact]
    public void Constructor_WithNullOperationType_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new OpPrecisionRule(null!, OpPrecision.Lower, OpPrecision.Keep));
    }

    [Fact]
    public void Priority_DefaultIsZero()
    {
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Lower,
            OpPrecision.Keep
        );

        Assert.Equal(0, rule.Priority);
    }

    [Fact]
    public void Priority_CanBeSet()
    {
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Lower,
            OpPrecision.Keep
        );

        rule.Priority = 100;
        Assert.Equal(100, rule.Priority);
    }

    [Fact]
    public void CustomForwardDtype_CanBeSet()
    {
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Custom,
            OpPrecision.Keep
        );

        rule.CustomForwardDtype = DataType.Float16;
        Assert.Equal(DataType.Float16, rule.CustomForwardDtype);
    }

    [Fact]
    public void CustomBackwardDtype_CanBeSet()
    {
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Keep,
            OpPrecision.Custom
        );

        rule.CustomBackwardDtype = DataType.Float32;
        Assert.Equal(DataType.Float32, rule.CustomBackwardDtype);
    }

    [Fact]
    public void GetForwardDtype_WithHigherPrecision_ReturnsHigherPrecisionFromConfig()
    {
        var config = AmpConfig.CreateFp16();
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Higher,
            OpPrecision.Keep
        );

        var dtype = rule.GetForwardDtype(config);
        Assert.Equal(DataType.Float32, dtype);
    }

    [Fact]
    public void GetForwardDtype_WithLowerPrecision_ReturnsTargetPrecisionFromConfig()
    {
        var config = AmpConfig.CreateFp16();
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Lower,
            OpPrecision.Keep
        );

        var dtype = rule.GetForwardDtype(config);
        Assert.Equal(DataType.Float16, dtype);
    }

    [Fact]
    public void GetForwardDtype_WithCustomPrecision_ReturnsCustomDtype()
    {
        var config = AmpConfig.CreateFp16();
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Custom,
            OpPrecision.Keep
        )
        {
            CustomForwardDtype = DataType.BFloat16
        };

        var dtype = rule.GetForwardDtype(config);
        Assert.Equal(DataType.BFloat16, dtype);
    }

    [Fact]
    public void GetForwardDtype_WithNullCustomDtype_ReturnsTargetPrecision()
    {
        var config = AmpConfig.CreateFp16();
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Custom,
            OpPrecision.Keep
        );

        var dtype = rule.GetForwardDtype(config);
        Assert.Equal(DataType.Float16, dtype);
    }

    [Fact]
    public void GetForwardDtype_WithKeepPrecision_ReturnsTargetPrecision()
    {
        var config = AmpConfig.CreateBf16();
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Keep,
            OpPrecision.Keep
        );

        var dtype = rule.GetForwardDtype(config);
        Assert.Equal(DataType.BFloat16, dtype);
    }

    [Fact]
    public void GetForwardDtype_WithNullConfig_ThrowsArgumentNullException()
    {
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Lower,
            OpPrecision.Keep
        );

        Assert.Throws<ArgumentNullException>(() =>
            rule.GetForwardDtype(null!));
    }

    [Fact]
    public void GetBackwardDtype_WithHigherPrecision_ReturnsHigherPrecisionFromConfig()
    {
        var config = AmpConfig.CreateFp16();
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Keep,
            OpPrecision.Higher
        );

        var dtype = rule.GetBackwardDtype(config);
        Assert.Equal(DataType.Float32, dtype);
    }

    [Fact]
    public void GetBackwardDtype_WithLowerPrecision_ReturnsTargetPrecisionFromConfig()
    {
        var config = AmpConfig.CreateFp16();
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Keep,
            OpPrecision.Lower
        );

        var dtype = rule.GetBackwardDtype(config);
        Assert.Equal(DataType.Float16, dtype);
    }

    [Fact]
    public void GetBackwardDtype_WithCustomPrecision_ReturnsCustomDtype()
    {
        var config = AmpConfig.CreateFp16();
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Keep,
            OpPrecision.Custom
        )
        {
            CustomBackwardDtype = DataType.BFloat16
        };

        var dtype = rule.GetBackwardDtype(config);
        Assert.Equal(DataType.BFloat16, dtype);
    }

    [Fact]
    public void GetBackwardDtype_WithNullCustomDtype_ReturnsTargetPrecision()
    {
        var config = AmpConfig.CreateFp16();
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Keep,
            OpPrecision.Custom
        );

        var dtype = rule.GetBackwardDtype(config);
        Assert.Equal(DataType.Float16, dtype);
    }

    [Fact]
    public void GetBackwardDtype_WithNullConfig_ThrowsArgumentNullException()
    {
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Lower,
            OpPrecision.Keep
        );

        Assert.Throws<ArgumentNullException>(() =>
            rule.GetBackwardDtype(null!));
    }

    [Fact]
    public void BothDirections_CanUseDifferentPrecisions()
    {
        var config = AmpConfig.CreateFp16();
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Lower,
            OpPrecision.Higher
        );

        var forwardDtype = rule.GetForwardDtype(config);
        var backwardDtype = rule.GetBackwardDtype(config);

        Assert.Equal(DataType.Float16, forwardDtype);
        Assert.Equal(DataType.Float32, backwardDtype);
    }

    [Theory]
    [InlineData(OpPrecision.Lower)]
    [InlineData(OpPrecision.Higher)]
    [InlineData(OpPrecision.Custom)]
    [InlineData(OpPrecision.Keep)]
    public void GetForwardDtype_AllPrecisionModes_Work(OpPrecision precision)
    {
        var config = AmpConfig.CreateBf16();
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            precision,
            OpPrecision.Keep
        );

        // Should not throw
        var dtype = rule.GetForwardDtype(config);
        Assert.NotNull(dtype);
    }
}
