using MLFramework.Amp;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Amp.Scalers;

/// <summary>
/// Tests for StaticLossScaler class
/// </summary>
public class StaticLossScalerTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultValues_CreatesScaler()
    {
        var scaler = new StaticLossScaler();

        Assert.NotNull(scaler);
        Assert.Equal(65536.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    [Fact]
    public void Constructor_WithCustomScale_CreatesScaler()
    {
        var scaler = new StaticLossScaler(scale: 256.0f);

        Assert.NotNull(scaler);
        Assert.Equal(256.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    [Fact]
    public void Constructor_WithDisabledFlag_CreatesDisabledScaler()
    {
        var scaler = new StaticLossScaler(scale: 65536.0f, enabled: false);

        Assert.NotNull(scaler);
        Assert.Equal(65536.0f, scaler.Scale);
        Assert.False(scaler.Enabled);
    }

    [Fact]
    public void Constructor_WithZeroScale_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new StaticLossScaler(scale: 0.0f));
    }

    [Fact]
    public void Constructor_WithNegativeScale_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new StaticLossScaler(scale: -1.0f));
    }

    [Fact]
    public void Constructor_WithScale1_CreatesScaler()
    {
        var scaler = new StaticLossScaler(scale: 1.0f);

        Assert.NotNull(scaler);
        Assert.Equal(1.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    #endregion

    #region ScaleLoss Tests

    [Fact]
    public void ScaleLoss_WithEnabled_ScalesLoss()
    {
        var scaler = new StaticLossScaler(scale: 2.0f);
        var loss = new Tensor(new[] { 1 });
        loss[0] = 10.0f;

        var scaledLoss = scaler.ScaleLoss(loss);

        Assert.NotNull(scaledLoss);
        Assert.Equal(20.0f, scaledLoss[0]);
    }

    [Fact]
    public void ScaleLoss_WithDisabled_ReturnsUnscaledLoss()
    {
        var scaler = new StaticLossScaler(scale: 2.0f, enabled: false);
        var loss = new Tensor(new[] { 1 });
        loss[0] = 10.0f;

        var scaledLoss = scaler.ScaleLoss(loss);

        Assert.Equal(loss, scaledLoss);
        Assert.Equal(10.0f, scaledLoss[0]);
    }

    [Fact]
    public void ScaleLoss_WithScale1_ReturnsUnscaledLoss()
    {
        var scaler = new StaticLossScaler(scale: 1.0f);
        var loss = new Tensor(new[] { 1 });
        loss[0] = 10.0f;

        var scaledLoss = scaler.ScaleLoss(loss);

        Assert.Equal(loss, scaledLoss);
        Assert.Equal(10.0f, scaledLoss[0]);
    }

    [Fact]
    public void ScaleLoss_WithNullLoss_ThrowsArgumentNullException()
    {
        var scaler = new StaticLossScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.ScaleLoss(null!));
    }

    [Fact]
    public void ScaleLoss_WithLargeScale_ScalesCorrectly()
    {
        var scaler = new StaticLossScaler(scale: 65536.0f);
        var loss = new Tensor(new[] { 1 });
        loss[0] = 0.01f;

        var scaledLoss = scaler.ScaleLoss(loss);

        Assert.Equal(655.36f, scaledLoss[0], precision: 2);
    }

    #endregion

    #region UnscaleGradient Tests

    [Fact]
    public void UnscaleGradient_WithEnabled_UnscalesGradient()
    {
        var scaler = new StaticLossScaler(scale: 2.0f);
        var gradient = new Tensor(new[] { 1 });
        gradient[0] = 20.0f;

        var unscaledGradient = scaler.UnscaleGradient(gradient);

        Assert.NotNull(unscaledGradient);
        Assert.Equal(10.0f, unscaledGradient[0]);
    }

    [Fact]
    public void UnscaleGradient_WithDisabled_ReturnsUnscaledGradient()
    {
        var scaler = new StaticLossScaler(scale: 2.0f, enabled: false);
        var gradient = new Tensor(new[] { 1 });
        gradient[0] = 20.0f;

        var unscaledGradient = scaler.UnscaleGradient(gradient);

        Assert.Equal(gradient, unscaledGradient);
        Assert.Equal(20.0f, unscaledGradient[0]);
    }

    [Fact]
    public void UnscaleGradient_WithScale1_ReturnsOriginalGradient()
    {
        var scaler = new StaticLossScaler(scale: 1.0f);
        var gradient = new Tensor(new[] { 1 });
        gradient[0] = 20.0f;

        var unscaledGradient = scaler.UnscaleGradient(gradient);

        Assert.Equal(gradient, unscaledGradient);
        Assert.Equal(20.0f, unscaledGradient[0]);
    }

    [Fact]
    public void UnscaleGradient_WithNullGradient_ThrowsArgumentNullException()
    {
        var scaler = new StaticLossScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.UnscaleGradient(null!));
    }

    #endregion

    #region UnscaleGradients Tests

    [Fact]
    public void UnscaleGradients_WithEnabled_UnscalesAllGradients()
    {
        var scaler = new StaticLossScaler(scale: 2.0f);
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", CreateTensor(new[] { 20.0f }) },
            { "param2", CreateTensor(new[] { 40.0f }) }
        };

        var unscaledGradients = scaler.UnscaleGradients(gradients);

        Assert.Equal(2, unscaledGradients.Count);
        Assert.Equal(10.0f, unscaledGradients["param1"][0]);
        Assert.Equal(20.0f, unscaledGradients["param2"][0]);
    }

    [Fact]
    public void UnscaleGradients_WithDisabled_ReturnsOriginalGradients()
    {
        var scaler = new StaticLossScaler(scale: 2.0f, enabled: false);
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", CreateTensor(new[] { 20.0f }) }
        };

        var unscaledGradients = scaler.UnscaleGradients(gradients);

        Assert.Same(gradients, unscaledGradients);
    }

    [Fact]
    public void UnscaleGradients_WithEmptyDictionary_ReturnsEmptyDictionary()
    {
        var scaler = new StaticLossScaler();
        var gradients = new Dictionary<string, Tensor>();

        var unscaledGradients = scaler.UnscaleGradients(gradients);

        Assert.Empty(unscaledGradients);
    }

    [Fact]
    public void UnscaleGradients_WithNullGradients_ThrowsArgumentNullException()
    {
        var scaler = new StaticLossScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.UnscaleGradients(null!));
    }

    #endregion

    #region CheckOverflow Tests (Single Tensor)

    [Fact]
    public void CheckOverflow_WithNaN_ReturnsTrue()
    {
        var scaler = new StaticLossScaler();
        var gradient = new Tensor(new[] { 1 });
        gradient[0] = float.NaN;

        var hasOverflow = scaler.CheckOverflow(gradient);

        Assert.True(hasOverflow);
    }

    [Fact]
    public void CheckOverflow_WithPositiveInfinity_ReturnsTrue()
    {
        var scaler = new StaticLossScaler();
        var gradient = new Tensor(new[] { 1 });
        gradient[0] = float.PositiveInfinity;

        var hasOverflow = scaler.CheckOverflow(gradient);

        Assert.True(hasOverflow);
    }

    [Fact]
    public void CheckOverflow_WithNegativeInfinity_ReturnsTrue()
    {
        var scaler = new StaticLossScaler();
        var gradient = new Tensor(new[] { 1 });
        gradient[0] = float.NegativeInfinity;

        var hasOverflow = scaler.CheckOverflow(gradient);

        Assert.True(hasOverflow);
    }

    [Fact]
    public void CheckOverflow_WithValidValues_ReturnsFalse()
    {
        var scaler = new StaticLossScaler();
        var gradient = new Tensor(new[] { 1 });
        gradient[0] = 10.0f;

        var hasOverflow = scaler.CheckOverflow(gradient);

        Assert.False(hasOverflow);
    }

    [Fact]
    public void CheckOverflow_WithDisabled_ReturnsFalse()
    {
        var scaler = new StaticLossScaler(enabled: false);
        var gradient = new Tensor(new[] { 1 });
        gradient[0] = float.NaN;

        var hasOverflow = scaler.CheckOverflow(gradient);

        Assert.False(hasOverflow);
    }

    [Fact]
    public void CheckOverflow_WithNullGradient_ThrowsArgumentNullException()
    {
        var scaler = new StaticLossScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.CheckOverflow((Tensor)null!));
    }

    [Fact]
    public void CheckOverflow_WithMultipleValues_DetectsOverflow()
    {
        var scaler = new StaticLossScaler();
        var gradient = new Tensor(new[] { 3 });
        gradient[0] = 10.0f;
        gradient[1] = float.NaN;
        gradient[2] = 20.0f;

        var hasOverflow = scaler.CheckOverflow(gradient);

        Assert.True(hasOverflow);
    }

    #endregion

    #region CheckOverflow Tests (Dictionary)

    [Fact]
    public void CheckOverflowDictionary_WithNaNGradient_ReturnsTrue()
    {
        var scaler = new StaticLossScaler();
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", CreateTensor(new[] { 10.0f }) },
            { "param2", CreateTensor(new[] { float.NaN }) }
        };

        var hasOverflow = scaler.CheckOverflow(gradients);

        Assert.True(hasOverflow);
    }

    [Fact]
    public void CheckOverflowDictionary_WithAllValid_ReturnsFalse()
    {
        var scaler = new StaticLossScaler();
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", CreateTensor(new[] { 10.0f }) },
            { "param2", CreateTensor(new[] { 20.0f }) }
        };

        var hasOverflow = scaler.CheckOverflow(gradients);

        Assert.False(hasOverflow);
    }

    [Fact]
    public void CheckOverflowDictionary_WithDisabled_ReturnsFalse()
    {
        var scaler = new StaticLossScaler(enabled: false);
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", CreateTensor(new[] { float.NaN }) }
        };

        var hasOverflow = scaler.CheckOverflow(gradients);

        Assert.False(hasOverflow);
    }

    [Fact]
    public void CheckOverflowDictionary_WithEmptyDictionary_ReturnsFalse()
    {
        var scaler = new StaticLossScaler();
        var gradients = new Dictionary<string, Tensor>();

        var hasOverflow = scaler.CheckOverflow(gradients);

        Assert.False(hasOverflow);
    }

    [Fact]
    public void CheckOverflowDictionary_WithNullGradients_ThrowsArgumentNullException()
    {
        var scaler = new StaticLossScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.CheckOverflow((Dictionary<string, Tensor>)null!));
    }

    #endregion

    #region UpdateScale Tests

    [Fact]
    public void UpdateScale_WithOverflow_ReturnsFalse()
    {
        var scaler = new StaticLossScaler(scale: 65536.0f);

        var shouldSkip = scaler.UpdateScale(overflow: true);

        Assert.False(shouldSkip);
        Assert.Equal(65536.0f, scaler.Scale); // Scale should not change
    }

    [Fact]
    public void UpdateScale_WithoutOverflow_ReturnsFalse()
    {
        var scaler = new StaticLossScaler(scale: 65536.0f);

        var shouldSkip = scaler.UpdateScale(overflow: false);

        Assert.False(shouldSkip);
        Assert.Equal(65536.0f, scaler.Scale); // Scale should not change
    }

    [Fact]
    public void UpdateScale_MultipleCalls_ScaleRemainsConstant()
    {
        var scaler = new StaticLossScaler(scale: 65536.0f);
        var initialScale = scaler.Scale;

        scaler.UpdateScale(overflow: true);
        scaler.UpdateScale(overflow: false);
        scaler.UpdateScale(overflow: true);

        Assert.Equal(initialScale, scaler.Scale);
    }

    #endregion

    #region Reset Tests

    [Fact]
    public void Reset_DoesNothing()
    {
        var scaler = new StaticLossScaler(scale: 65536.0f);
        var initialScale = scaler.Scale;

        scaler.Reset();

        Assert.Equal(initialScale, scaler.Scale);
    }

    #endregion

    #region GetScaleTensor Tests

    [Fact]
    public void GetScaleTensor_ReturnsTensorWithCorrectValue()
    {
        var scaler = new StaticLossScaler(scale: 100.0f);

        var scaleTensor = scaler.GetScaleTensor();

        Assert.NotNull(scaleTensor);
        Assert.Single(scaleTensor.Shape);
        Assert.Equal(100.0f, scaleTensor[0]);
    }

    [Fact]
    public void GetScaleTensor_WithDisabled_ThrowsInvalidOperationException()
    {
        var scaler = new StaticLossScaler(enabled: false);

        Assert.Throws<InvalidOperationException>(() =>
            scaler.GetScaleTensor());
    }

    [Fact]
    public void GetScaleTensor_ReturnsNewInstance()
    {
        var scaler = new StaticLossScaler();
        var tensor1 = scaler.GetScaleTensor();
        var tensor2 = scaler.GetScaleTensor();

        Assert.NotSame(tensor1, tensor2);
    }

    #endregion

    #region GetInverseScaleTensor Tests

    [Fact]
    public void GetInverseScaleTensor_ReturnsTensorWithCorrectValue()
    {
        var scaler = new StaticLossScaler(scale: 2.0f);

        var inverseScaleTensor = scaler.GetInverseScaleTensor();

        Assert.NotNull(inverseScaleTensor);
        Assert.Single(inverseScaleTensor.Shape);
        Assert.Equal(0.5f, inverseScaleTensor[0]);
    }

    [Fact]
    public void GetInverseScaleTensor_WithDisabled_ThrowsInvalidOperationException()
    {
        var scaler = new StaticLossScaler(enabled: false);

        Assert.Throws<InvalidOperationException>(() =>
            scaler.GetInverseScaleTensor());
    }

    [Fact]
    public void GetInverseScaleTensor_ReturnsNewInstance()
    {
        var scaler = new StaticLossScaler();
        var tensor1 = scaler.GetInverseScaleTensor();
        var tensor2 = scaler.GetInverseScaleTensor();

        Assert.NotSame(tensor1, tensor2);
    }

    #endregion

    #region Helper Methods

    private Tensor CreateTensor(float[] values)
    {
        var tensor = new Tensor(new[] { values.Length });
        for (int i = 0; i < values.Length; i++)
        {
            tensor[i] = values[i];
        }
        return tensor;
    }

    #endregion
}
