using MLFramework.Amp;
using MLFramework.Core;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Amp.AutoCast;

/// <summary>
/// Tests for AutoCast class
/// </summary>
public class AutoCastTests
{
    private class TestOperation { }

    [Fact]
    public void Constructor_WithDefaults_CreatesEnabledBf16Context()
    {
        var autoCast = new AutoCast();

        Assert.True(autoCast.Enabled);
        Assert.Equal(AutoCastMode.Bf16, autoCast.Mode);
    }

    [Fact]
    public void Constructor_WithDisabled_CreatesDisabledContext()
    {
        var autoCast = new AutoCast(false);

        Assert.False(autoCast.Enabled);
        Assert.Equal(AutoCastMode.Bf16, autoCast.Mode);
    }

    [Fact]
    public void Constructor_WithMode_SetsCorrectMode()
    {
        var fp16Cast = new AutoCast(AutoCastMode.Fp16);
        var bf16Cast = new AutoCast(AutoCastMode.Bf16);
        var noneCast = new AutoCast(AutoCastMode.None);

        Assert.Equal(AutoCastMode.Fp16, fp16Cast.Mode);
        Assert.Equal(AutoCastMode.Bf16, bf16Cast.Mode);
        Assert.Equal(AutoCastMode.None, noneCast.Mode);
    }

    [Fact]
    public void Constructor_WithCustomRegistry_UsesProvidedRegistry()
    {
        var config = AmpConfig.CreateFp16();
        var registry = new AmpRegistry(config);

        var autoCast = new AutoCast(enabled: true, registry: registry);

        Assert.NotNull(autoCast);
    }

    [Fact]
    public void Cast_WithEnabled_ReturnsTensor()
    {
        var autoCast = new AutoCast();
        var tensor = new Tensor(new[] { 2, 3 });

        var result = autoCast.Cast(tensor, typeof(TestOperation));

        Assert.NotNull(result);
    }

    [Fact]
    public void Cast_WithDisabled_ReturnsOriginalTensor()
    {
        var autoCast = new AutoCast(false);
        var tensor = new Tensor(new[] { 2, 3 });

        var result = autoCast.Cast(tensor, typeof(TestOperation));

        Assert.Equal(tensor, result);
    }

    [Fact]
    public void Cast_WithNoneMode_ReturnsOriginalTensor()
    {
        var autoCast = new AutoCast(AutoCastMode.None);
        var tensor = new Tensor(new[] { 2, 3 });

        var result = autoCast.Cast(tensor, typeof(TestOperation));

        Assert.Equal(tensor, result);
    }

    [Fact]
    public void Cast_WithNullTensor_ThrowsArgumentNullException()
    {
        var autoCast = new AutoCast();

        Assert.Throws<ArgumentNullException>(() =>
            autoCast.Cast(null!, typeof(TestOperation)));
    }

    [Fact]
    public void Cast_WithNullOperationType_ThrowsArgumentNullException()
    {
        var autoCast = new AutoCast();
        var tensor = new Tensor(new[] { 2, 3 });

        Assert.Throws<ArgumentNullException>(() =>
            autoCast.Cast(tensor, null!));
    }

    [Fact]
    public void Cast_ToDataType_WithEnabled_ReturnsTensor()
    {
        var autoCast = new AutoCast();
        var tensor = new Tensor(new[] { 2, 3 });

        var result = autoCast.Cast(tensor, DataType.Float16);

        Assert.NotNull(result);
    }

    [Fact]
    public void Cast_ToDataType_WithDisabled_ReturnsOriginalTensor()
    {
        var autoCast = new AutoCast(false);
        var tensor = new Tensor(new[] { 2, 3 });

        var result = autoCast.Cast(tensor, DataType.Float16);

        Assert.Equal(tensor, result);
    }

    [Fact]
    public void Cast_ToDataType_WithNullTensor_ThrowsArgumentNullException()
    {
        var autoCast = new AutoCast();

        Assert.Throws<ArgumentNullException>(() =>
            autoCast.Cast(null!, DataType.Float16));
    }

    [Fact]
    public void GetForwardDtype_WithWhitelistedOp_ReturnsTargetPrecision()
    {
        var config = AmpConfig.CreateFp16();
        var registry = new AmpRegistry(config);
        registry.RegisterWhitelist(typeof(TestOperation));

        var autoCast = new AutoCast(AutoCastMode.Fp16, true, registry);

        var dtype = autoCast.GetForwardDtype(typeof(TestOperation), DataType.Float32);

        Assert.Equal(DataType.Float16, dtype);
    }

    [Fact]
    public void GetForwardDtype_WithBlacklistedOp_ReturnsHigherPrecision()
    {
        var config = AmpConfig.CreateFp16();
        var registry = new AmpRegistry(config);
        registry.RegisterBlacklist(typeof(TestOperation));

        var autoCast = new AutoCast(AutoCastMode.Fp16, true, registry);

        var dtype = autoCast.GetForwardDtype(typeof(TestOperation), DataType.Float32);

        Assert.Equal(DataType.Float32, dtype);
    }

    [Fact]
    public void GetForwardDtype_WithoutRule_ReturnsInputDtype()
    {
        var autoCast = new AutoCast();

        var dtype = autoCast.GetForwardDtype(typeof(TestOperation), DataType.BFloat16);

        Assert.Equal(DataType.BFloat16, dtype);
    }

    [Fact]
    public void GetBackwardDtype_WithWhitelistedOp_ReturnsTargetPrecision()
    {
        var config = AmpConfig.CreateFp16();
        var registry = new AmpRegistry(config);
        registry.RegisterWhitelist(typeof(TestOperation));

        var autoCast = new AutoCast(AutoCastMode.Fp16, true, registry);

        var dtype = autoCast.GetBackwardDtype(typeof(TestOperation), DataType.Float32);

        Assert.Equal(DataType.Float16, dtype);
    }

    [Fact]
    public void GetBackwardDtype_WithBlacklistedOp_ReturnsHigherPrecision()
    {
        var config = AmpConfig.CreateFp16();
        var registry = new AmpRegistry(config);
        registry.RegisterBlacklist(typeof(TestOperation));

        var autoCast = new AutoCast(AutoCastMode.Fp16, true, registry);

        var dtype = autoCast.GetBackwardDtype(typeof(TestOperation), DataType.Float32);

        Assert.Equal(DataType.Float32, dtype);
    }

    [Fact]
    public void GetBackwardDtype_WithoutRule_ReturnsInputDtype()
    {
        var autoCast = new AutoCast();

        var dtype = autoCast.GetBackwardDtype(typeof(TestOperation), DataType.BFloat16);

        Assert.Equal(DataType.BFloat16, dtype);
    }

    [Fact]
    public void Enter_SetsCurrentContext()
    {
        var autoCast = new AutoCast();

        Assert.Null(AutoCast.Current);

        autoCast.Enter();

        Assert.Equal(autoCast, AutoCast.Current);
    }

    [Fact]
    public void Exit_RestoresPreviousContext()
    {
        var autoCast1 = new AutoCast(AutoCastMode.Fp16);
        var autoCast2 = new AutoCast(AutoCastMode.Bf16);

        autoCast1.Enter();
        Assert.Equal(autoCast1, AutoCast.Current);

        autoCast2.Enter();
        Assert.Equal(autoCast2, AutoCast.Current);

        autoCast2.Exit();
        Assert.Equal(autoCast1, AutoCast.Current);

        autoCast1.Exit();
        Assert.Null(AutoCast.Current);
    }

    [Fact]
    public void Dispose_RestoresPreviousContext()
    {
        var autoCast1 = new AutoCast(AutoCastMode.Fp16);
        var autoCast2 = new AutoCast(AutoCastMode.Bf16);

        autoCast1.Enter();
        autoCast2.Enter();

        autoCast2.Dispose();

        Assert.Equal(autoCast1, AutoCast.Current);

        autoCast1.Dispose();

        Assert.Null(AutoCast.Current);
    }

    [Fact]
    public void MultipleEnterExitPairs_WorkCorrectly()
    {
        var autoCast = new AutoCast();

        autoCast.Enter();
        Assert.NotNull(AutoCast.Current);

        autoCast.Exit();
        Assert.Null(AutoCast.Current);

        autoCast.Enter();
        Assert.NotNull(AutoCast.Current);

        autoCast.Exit();
        Assert.Null(AutoCast.Current);
    }
}
