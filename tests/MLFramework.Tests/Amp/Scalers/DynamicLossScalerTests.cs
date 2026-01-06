using System;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Amp.Scalers;

/// <summary>
/// Tests for DynamicLossScaler class
/// </summary>
public class DynamicLossScalerTests
{
    [Fact]
    public void Constructor_Default_CreatesScalerWithDefaults()
    {
        var scaler = new DynamicLossScaler();

        Assert.NotNull(scaler);
        Assert.True(scaler.Enabled);
        Assert.Equal(65536.0f, scaler.Scale);
        Assert.Equal(2.0f, scaler.GrowthFactor);
        Assert.Equal(0.5f, scaler.BackoffFactor);
        Assert.Equal(2000, scaler.GrowthInterval);
        Assert.Equal(1.0f, scaler.MinScale);
        Assert.Equal(16777216.0f, scaler.MaxScale);
    }

    [Fact]
    public void Constructor_CustomParameters_SetsCorrectValues()
    {
        var scaler = new DynamicLossScaler(
            initialScale: 1000.0f,
            growthFactor: 3.0f,
            backoffFactor: 0.25f,
            growthInterval: 1000,
            minScale: 0.5f,
            maxScale: 10000.0f,
            enabled: true
        );

        Assert.Equal(1000.0f, scaler.Scale);
        Assert.Equal(3.0f, scaler.GrowthFactor);
        Assert.Equal(0.25f, scaler.BackoffFactor);
        Assert.Equal(1000, scaler.GrowthInterval);
        Assert.Equal(0.5f, scaler.MinScale);
        Assert.Equal(10000.0f, scaler.MaxScale);
        Assert.True(scaler.Enabled);
    }

    [Fact]
    public void Constructor_InvalidInitialScale_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new DynamicLossScaler(initialScale: -1.0f));
    }

    [Fact]
    public void Constructor_InvalidGrowthFactor_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new DynamicLossScaler(growthFactor: 0.5f));
    }

    [Fact]
    public void Constructor_InvalidBackoffFactor_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new DynamicLossScaler(backoffFactor: 1.5f));
    }

    [Fact]
    public void Constructor_InvalidGrowthInterval_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new DynamicLossScaler(growthInterval: -1));
    }

    [Fact]
    public void Constructor_InvalidMinScale_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new DynamicLossScaler(minScale: -1.0f));
    }

    [Fact]
    public void Constructor_MaxScaleLessThanMinScale_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new DynamicLossScaler(minScale: 100.0f, maxScale: 50.0f));
    }

    [Fact]
    public void Constructor_InitialScaleOutsideBounds_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new DynamicLossScaler(initialScale: 200.0f, minScale: 100.0f, maxScale: 150.0f));
    }

    [Fact]
    public void Constructor_Disabled_CreatesDisabledScaler()
    {
        var scaler = new DynamicLossScaler(enabled: false);

        Assert.False(scaler.Enabled);
    }

    [Fact]
    public void ScaleLoss_WithEnabled_ScalesLoss()
    {
        var scaler = new DynamicLossScaler(initialScale: 2.0f);
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = 10.0f;

        var scaled = scaler.ScaleLoss(tensor);

        Assert.Equal(20.0f, scaled[0]);
    }

    [Fact]
    public void ScaleLoss_WithDisabled_ReturnsOriginal()
    {
        var scaler = new DynamicLossScaler(initialScale: 2.0f, enabled: false);
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = 10.0f;

        var scaled = scaler.ScaleLoss(tensor);

        Assert.Equal(10.0f, scaled[0]);
    }

    [Fact]
    public void ScaleLoss_WithUnitScale_ReturnsOriginal()
    {
        var scaler = new DynamicLossScaler(initialScale: 1.0f);
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = 10.0f;

        var scaled = scaler.ScaleLoss(tensor);

        Assert.Equal(10.0f, scaled[0]);
    }

    [Fact]
    public void ScaleLoss_WithNullTensor_ThrowsArgumentNullException()
    {
        var scaler = new DynamicLossScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.ScaleLoss(null!));
    }

    [Fact]
    public void UnscaleGradient_WithEnabled_UnscalesGradient()
    {
        var scaler = new DynamicLossScaler(initialScale: 2.0f);
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = 10.0f;

        var unscaled = scaler.UnscaleGradient(tensor);

        Assert.Equal(5.0f, unscaled[0]);
    }

    [Fact]
    public void UnscaleGradient_WithDisabled_ReturnsOriginal()
    {
        var scaler = new DynamicLossScaler(initialScale: 2.0f, enabled: false);
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = 10.0f;

        var unscaled = scaler.UnscaleGradient(tensor);

        Assert.Equal(10.0f, unscaled[0]);
    }

    [Fact]
    public void UnscaleGradient_WithNullTensor_ThrowsArgumentNullException()
    {
        var scaler = new DynamicLossScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.UnscaleGradient(null!));
    }

    [Fact]
    public void UnscaleGradients_WithEnabled_UnscalesAllGradients()
    {
        var scaler = new DynamicLossScaler(initialScale: 2.0f);
        var tensor1 = new Tensor(new[] { 1 });
        tensor1[0] = 10.0f;
        var tensor2 = new Tensor(new[] { 1 });
        tensor2[0] = 20.0f;

        var gradients = new System.Collections.Generic.Dictionary<string, Tensor>
        {
            ["param1"] = tensor1,
            ["param2"] = tensor2
        };

        var unscaled = scaler.UnscaleGradients(gradients);

        Assert.Equal(5.0f, unscaled["param1"][0]);
        Assert.Equal(10.0f, unscaled["param2"][0]);
    }

    [Fact]
    public void UnscaleGradients_WithNullGradients_ThrowsArgumentNullException()
    {
        var scaler = new DynamicLossScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.UnscaleGradients(null!));
    }

    [Fact]
    public void CheckOverflow_WithInf_ReturnsTrue()
    {
        var scaler = new DynamicLossScaler();
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = float.PositiveInfinity;

        Assert.True(scaler.CheckOverflow(tensor));
    }

    [Fact]
    public void CheckOverflow_WithNaN_ReturnsTrue()
    {
        var scaler = new DynamicLossScaler();
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = float.NaN;

        Assert.True(scaler.CheckOverflow(tensor));
    }

    [Fact]
    public void CheckOverflow_WithNormalValues_ReturnsFalse()
    {
        var scaler = new DynamicLossScaler();
        var tensor = new Tensor(new[] { 2 });
        tensor[0] = 1.0f;
        tensor[1] = 2.0f;

        Assert.False(scaler.CheckOverflow(tensor));
    }

    [Fact]
    public void CheckOverflow_WithMultipleGradients_EarlyExitOnOverflow()
    {
        var scaler = new DynamicLossScaler();
        var tensor1 = new Tensor(new[] { 1 });
        tensor1[0] = 1.0f;
        var tensor2 = new Tensor(new[] { 1 });
        tensor2[0] = float.PositiveInfinity;

        var gradients = new System.Collections.Generic.Dictionary<string, Tensor>
        {
            ["param1"] = tensor1,
            ["param2"] = tensor2
        };

        Assert.True(scaler.CheckOverflow(gradients));
    }

    [Fact]
    public void CheckOverflow_WithNullTensor_ThrowsArgumentNullException()
    {
        var scaler = new DynamicLossScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.CheckOverflow((Tensor)null!));
    }

    [Fact]
    public void CheckOverflow_WithNullGradients_ThrowsArgumentNullException()
    {
        var scaler = new DynamicLossScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.CheckOverflow((System.Collections.Generic.Dictionary<string, Tensor>)null!));
    }

    [Fact]
    public void CheckOverflow_WithDisabled_ReturnsFalse()
    {
        var scaler = new DynamicLossScaler(enabled: false);
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = float.PositiveInfinity;

        Assert.False(scaler.CheckOverflow(tensor));
    }

    [Fact]
    public void UpdateScale_Overflow_DecreasesScale()
    {
        var scaler = new DynamicLossScaler(
            initialScale: 1000.0f,
            backoffFactor: 0.5f
        );

        var skipStep = scaler.UpdateScale(overflow: true);

        Assert.True(skipStep);
        Assert.Equal(500.0f, scaler.Scale);
        Assert.Equal(1, scaler.TotalOverflows);
        Assert.Equal(0, scaler.GrowthCounter);
    }

    [Fact]
    public void UpdateScale_NoOverflow_IncrementsCounter()
    {
        var scaler = new DynamicLossScaler(
            initialScale: 1000.0f,
            growthInterval: 2000
        );
        var initialScale = scaler.Scale;

        for (int i = 0; i < 1999; i++)
        {
            scaler.UpdateScale(overflow: false);
        }

        Assert.Equal(initialScale, scaler.Scale);
        Assert.Equal(1999, scaler.GrowthCounter);
    }

    [Fact]
    public void UpdateScale_AfterGrowthInterval_IncreasesScale()
    {
        var scaler = new DynamicLossScaler(
            initialScale: 1000.0f,
            growthInterval: 2,
            growthFactor: 2.0f
        );

        scaler.UpdateScale(overflow: false);
        scaler.UpdateScale(overflow: false);

        Assert.Equal(2000.0f, scaler.Scale);
        Assert.Equal(0, scaler.GrowthCounter);
    }

    [Fact]
    public void UpdateScale_RespectsMinScale()
    {
        var scaler = new DynamicLossScaler(
            initialScale: 100.0f,
            minScale: 10.0f,
            backoffFactor: 0.5f
        );

        scaler.UpdateScale(overflow: true);
        scaler.UpdateScale(overflow: true);
        scaler.UpdateScale(overflow: true);
        scaler.UpdateScale(overflow: true);

        Assert.Equal(10.0f, scaler.Scale);
    }

    [Fact]
    public void UpdateScale_RespectsMaxScale()
    {
        var scaler = new DynamicLossScaler(
            initialScale: 100.0f,
            maxScale: 200.0f,
            growthInterval: 1,
            growthFactor: 10.0f
        );

        scaler.UpdateScale(overflow: false);

        Assert.Equal(200.0f, scaler.Scale);
    }

    [Fact]
    public void UpdateScale_WithDisabled_ReturnsFalse()
    {
        var scaler = new DynamicLossScaler(
            initialScale: 1000.0f,
            enabled: false
        );

        var skipStep = scaler.UpdateScale(overflow: true);

        Assert.False(skipStep);
        Assert.Equal(1000.0f, scaler.Scale);
    }

    [Fact]
    public void GrowthCounter_IncreasesOnNoOverflow()
    {
        var scaler = new DynamicLossScaler();

        scaler.UpdateScale(overflow: false);

        Assert.Equal(1, scaler.GrowthCounter);
    }

    [Fact]
    public void GrowthCounter_ResetsOnOverflow()
    {
        var scaler = new DynamicLossScaler(growthInterval: 10);

        for (int i = 0; i < 5; i++)
        {
            scaler.UpdateScale(overflow: false);
        }
        Assert.Equal(5, scaler.GrowthCounter);

        scaler.UpdateScale(overflow: true);

        Assert.Equal(0, scaler.GrowthCounter);
    }

    [Fact]
    public void Reset_RestoresInitialState()
    {
        var scaler = new DynamicLossScaler(initialScale: 1000.0f);

        scaler.UpdateScale(overflow: true);
        scaler.UpdateScale(overflow: false);
        scaler.UpdateScale(overflow: false);

        Assert.Equal(1, scaler.TotalOverflows);
        Assert.Equal(2, scaler.GrowthCounter);

        scaler.Reset();

        Assert.Equal(1000.0f, scaler.Scale);
        Assert.Equal(0, scaler.TotalOverflows);
        Assert.Equal(0, scaler.GrowthCounter);
    }

    [Fact]
    public void GetStats_ReturnsCorrectStatistics()
    {
        var scaler = new DynamicLossScaler(initialScale: 1000.0f);

        scaler.UpdateScale(overflow: false);
        scaler.UpdateScale(overflow: false);

        var stats = scaler.GetStats();

        Assert.Equal(1000.0f, stats.CurrentScale);
        Assert.Equal(0, stats.TotalOverflows);
        Assert.Equal(2, stats.TotalSuccessfulIterations);
        Assert.Equal(0, stats.ScaleIncreaseCount);
        Assert.Equal(0, stats.ScaleDecreaseCount);
        Assert.Equal(1.0f, stats.SuccessRate);
    }

    [Fact]
    public void GetStats_WithOverflow_TracksCorrectly()
    {
        var scaler = new DynamicLossScaler(
            initialScale: 1000.0f,
            backoffFactor: 0.5f
        );

        scaler.UpdateScale(overflow: true);
        scaler.UpdateScale(overflow: false);

        var stats = scaler.GetStats();

        Assert.Equal(500.0f, stats.CurrentScale);
        Assert.Equal(1, stats.TotalOverflows);
        Assert.Equal(1, stats.TotalSuccessfulIterations);
        Assert.Equal(0, stats.ScaleIncreaseCount);
        Assert.Equal(1, stats.ScaleDecreaseCount);
        Assert.Equal(0.5f, stats.SuccessRate);
    }

    [Fact]
    public void GetStats_TracksMinScaleReached()
    {
        var scaler = new DynamicLossScaler(
            initialScale: 100.0f,
            backoffFactor: 0.5f
        );

        scaler.UpdateScale(overflow: true);
        scaler.UpdateScale(overflow: true);

        var stats = scaler.GetStats();

        Assert.Equal(25.0f, stats.MinScaleReached);
    }

    [Fact]
    public void GetStats_TracksMaxScaleReached()
    {
        var scaler = new DynamicLossScaler(
            initialScale: 100.0f,
            growthInterval: 1,
            growthFactor: 2.0f
        );

        scaler.UpdateScale(overflow: false);
        scaler.UpdateScale(overflow: false);

        var stats = scaler.GetStats();

        Assert.Equal(400.0f, stats.MaxScaleReached);
    }

    [Fact]
    public void GetStats_ToString_ReturnsFormattedString()
    {
        var scaler = new DynamicLossScaler(initialScale: 1000.0f);
        scaler.UpdateScale(overflow: false);
        scaler.UpdateScale(overflow: true);

        var stats = scaler.GetStats();
        var statsString = stats.ToString();

        Assert.Contains("Scale:", statsString);
        Assert.Contains("Overflows:", statsString);
        Assert.Contains("SuccessIterations:", statsString);
    }

    [Fact]
    public void GetScaleTensor_ReturnsCorrectTensor()
    {
        var scaler = new DynamicLossScaler(initialScale: 1000.0f);

        var scaleTensor = scaler.GetScaleTensor();

        Assert.NotNull(scaleTensor);
        Assert.Equal(1, scaleTensor.Shape.Length);
        Assert.Equal(1000.0f, scaleTensor[0]);
    }

    [Fact]
    public void GetScaleTensor_WithDisabled_ThrowsInvalidOperationException()
    {
        var scaler = new DynamicLossScaler(enabled: false);

        Assert.Throws<InvalidOperationException>(() =>
            scaler.GetScaleTensor());
    }

    [Fact]
    public void GetInverseScaleTensor_ReturnsCorrectTensor()
    {
        var scaler = new DynamicLossScaler(initialScale: 1000.0f);

        var inverseTensor = scaler.GetInverseScaleTensor();

        Assert.NotNull(inverseTensor);
        Assert.Equal(1, inverseTensor.Shape.Length);
        Assert.Equal(0.001f, inverseTensor[0], precision: 6);
    }

    [Fact]
    public void GetInverseScaleTensor_WithDisabled_ThrowsInvalidOperationException()
    {
        var scaler = new DynamicLossScaler(enabled: false);

        Assert.Throws<InvalidOperationException>(() =>
            scaler.GetInverseScaleTensor());
    }

    [Fact]
    public void ContinuousOverflow_MultipleDecreases_ScaleClampedToMin()
    {
        var scaler = new DynamicLossScaler(
            initialScale: 1000.0f,
            minScale: 1.0f,
            backoffFactor: 0.5f
        );

        // Trigger many overflows
        for (int i = 0; i < 20; i++)
        {
            scaler.UpdateScale(overflow: true);
        }

        Assert.Equal(1.0f, scaler.Scale);
    }

    [Fact]
    public void ContinuousSuccess_MultipleIncreases_ScaleClampedToMax()
    {
        var scaler = new DynamicLossScaler(
            initialScale: 100.0f,
            maxScale: 10000.0f,
            growthInterval: 1,
            growthFactor: 2.0f
        );

        // Trigger many successful iterations
        for (int i = 0; i < 20; i++)
        {
            scaler.UpdateScale(overflow: false);
        }

        Assert.Equal(10000.0f, scaler.Scale);
    }
}
