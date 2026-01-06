using MLFramework.Amp;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Amp.Scalers;

/// <summary>
/// Tests for GradScaler class
/// </summary>
public class GradScalerTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_StaticScaler_WithDefaultValues_CreatesScaler()
    {
        var scaler = new GradScaler();

        Assert.NotNull(scaler);
        Assert.Equal(65536.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    [Fact]
    public void Constructor_StaticScaler_WithCustomScale_CreatesScaler()
    {
        var scaler = new GradScaler(scale: 256.0f);

        Assert.NotNull(scaler);
        Assert.Equal(256.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    [Fact]
    public void Constructor_DynamicScaler_WithDefaultValues_CreatesScaler()
    {
        var scaler = new GradScaler(
            initialScale: 65536.0f,
            growthFactor: 2.0f,
            backoffFactor: 0.5f,
            growthInterval: 2000,
            minScale: 1.0f,
            maxScale: 16777216.0f,
            enabled: true);

        Assert.NotNull(scaler);
        Assert.Equal(65536.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    [Fact]
    public void Constructor_DynamicScaler_WithCustomParameters_CreatesScaler()
    {
        var scaler = new GradScaler(
            initialScale: 32768.0f,
            growthFactor: 1.5f,
            backoffFactor: 0.75f,
            growthInterval: 5000,
            minScale: 0.5f,
            maxScale: 1000000.0f,
            enabled: true);

        Assert.NotNull(scaler);
        Assert.Equal(32768.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    [Fact]
    public void Constructor_WithCustomScaler_WrapsScaler()
    {
        ILossScaler innerScaler = MLFramework.Optimizers.MixedPrecision.DynamicLossScaler.CreateDefault();
        var scaler = new GradScaler(innerScaler);

        Assert.NotNull(scaler);
        Assert.Same(innerScaler, scaler.Scaler);
    }

    [Fact]
    public void Constructor_WithNullScaler_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new GradScaler((ILossScaler)null!));
    }

    #endregion

    #region Scale Method Tests

    [Fact]
    public void ScaleLoss_WithValidLoss_ReturnsScaledLoss()
    {
        var scaler = new GradScaler(scale: 1000.0f);
        var loss = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        var scaledLoss = scaler.ScaleLoss(loss);

        Assert.NotNull(scaledLoss);
        Assert.Equal(1000.0f, scaledLoss[0]);
    }

    [Fact]
    public void ScaleLoss_WithNullLoss_ThrowsArgumentNullException()
    {
        var scaler = new GradScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.ScaleLoss(null!));
    }

    #endregion

    #region Unscale Method Tests

    [Fact]
    public void Unscale_WithValidGradients_ReturnsUnscaledGradients()
    {
        var scaler = new GradScaler(scale: 1000.0f);
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1000.0f }, new int[] { 1 }) },
            { "param2", new Tensor(new float[] { 2000.0f }, new int[] { 1 }) }
        };

        var unscaledGradients = scaler.Unscale(gradients);

        Assert.NotNull(unscaledGradients);
        Assert.Equal(2, unscaledGradients.Count);
        Assert.Equal(1.0f, unscaledGradients["param1"][0], precision: 4);
        Assert.Equal(2.0f, unscaledGradients["param2"][0], precision: 4);
    }

    [Fact]
    public void Unscale_WithNullGradients_ThrowsArgumentNullException()
    {
        var scaler = new GradScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.Unscale(null!));
    }

    #endregion

    #region CheckOverflow Method Tests

    [Fact]
    public void CheckOverflow_WithNormalGradients_ReturnsFalse()
    {
        var scaler = new GradScaler();
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }) }
        };

        bool hasOverflow = scaler.CheckOverflow(gradients);

        Assert.False(hasOverflow);
    }

    [Fact]
    public void CheckOverflow_WithNaNGradients_ReturnsTrue()
    {
        var scaler = new GradScaler();
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { float.NaN, 1.0f }, new int[] { 2 }) }
        };

        bool hasOverflow = scaler.CheckOverflow(gradients);

        Assert.True(hasOverflow);
    }

    [Fact]
    public void CheckOverflow_WithInfGradients_ReturnsTrue()
    {
        var scaler = new GradScaler();
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { float.PositiveInfinity, 1.0f }, new int[] { 2 }) }
        };

        bool hasOverflow = scaler.CheckOverflow(gradients);

        Assert.True(hasOverflow);
    }

    [Fact]
    public void CheckOverflow_WithSingleTensor_WorksCorrectly()
    {
        var scaler = new GradScaler();
        var gradient = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        bool hasOverflow = scaler.CheckOverflow(gradient);

        Assert.False(hasOverflow);
    }

    #endregion

    #region PrepareGradients Method Tests

    [Fact]
    public void PrepareGradients_WithValidGradients_ReturnsTrue()
    {
        var scaler = new GradScaler();
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f }, new int[] { 1 }) }
        };

        bool prepared = scaler.PrepareGradients(gradients);

        Assert.True(prepared);
    }

    [Fact]
    public void PrepareGradients_WithOverflow_ReturnsFalse()
    {
        var scaler = new GradScaler();
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { float.NaN }, new int[] { 1 }) }
        };

        bool prepared = scaler.PrepareGradients(gradients);

        Assert.False(prepared);
    }

    #endregion

    #region PrepareAndGetGradients Method Tests

    [Fact]
    public void PrepareAndGetGradients_WithValidGradients_ReturnsUnscaledGradients()
    {
        var scaler = new GradScaler(scale: 1000.0f);
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1000.0f }, new int[] { 1 }) }
        };

        var unscaledGradients = scaler.PrepareAndGetGradients(gradients);

        Assert.NotNull(unscaledGradients);
        Assert.Equal(1.0f, unscaledGradients["param1"][0], precision: 4);
    }

    [Fact]
    public void PrepareAndGetGradients_WithOverflow_ReturnsNull()
    {
        var scaler = new GradScaler();
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { float.NaN }, new int[] { 1 }) }
        };

        var unscaledGradients = scaler.PrepareAndGetGradients(gradients);

        Assert.Null(unscaledGradients);
    }

    #endregion

    #region Reset Method Tests

    [Fact]
    public void Reset_AfterOverflow_ResetsScale()
    {
        var scaler = new GradScaler(
            initialScale: 1000.0f,
            growthFactor: 2.0f,
            backoffFactor: 0.5f,
            growthInterval: 100);
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { float.NaN }, new int[] { 1 }) }
        };

        // Trigger overflow
        scaler.PrepareGradients(gradients);

        // Scale should have decreased
        float scaleAfterOverflow = scaler.Scale;
        Assert.True(scaleAfterOverflow < 1000.0f);

        // Reset
        scaler.Reset();

        // Scale should be back to initial
        Assert.Equal(1000.0f, scaler.Scale);
    }

    #endregion

    #region GetScaleTensor Method Tests

    [Fact]
    public void GetScaleTensor_ReturnsTensorWithScaleValue()
    {
        var scaler = new GradScaler(scale: 1000.0f);

        var scaleTensor = scaler.GetScaleTensor();

        Assert.NotNull(scaleTensor);
        Assert.Equal(1000.0f, scaleTensor[0]);
    }

    #endregion

    #region GetStats Method Tests

    [Fact]
    public void GetStats_ReturnsStatistics()
    {
        var scaler = new GradScaler();

        var stats = scaler.GetStats();

        Assert.NotNull(stats);
        Assert.True(stats.CurrentScale > 0);
    }

    #endregion

    #region Enable/Disable Method Tests

    [Fact]
    public void Enable_ThrowsNotImplementedException()
    {
        var scaler = new GradScaler();

        Assert.Throws<NotImplementedException>(() =>
            scaler.Enable());
    }

    [Fact]
    public void Disable_ThrowsNotImplementedException()
    {
        var scaler = new GradScaler();

        Assert.Throws<NotImplementedException>(() =>
            scaler.Disable());
    }

    #endregion
}
