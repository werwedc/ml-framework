using MLFramework.Amp;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using Xunit;

namespace MLFramework.Tests.Amp.Utils;

/// <summary>
/// Tests for GradientUtils class
/// </summary>
public class GradientUtilsTests
{
    #region Unscale Tests

    [Fact]
    public void Unscale_WithScale1_ReturnsOriginalTensor()
    {
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var tensor = new Tensor(data, new int[] { 2, 2 });

        var result = GradientUtils.Unscale(tensor, 1.0f);

        Assert.NotNull(result);
        Assert.Equal(2, result.Shape[0]);
        Assert.Equal(2, result.Shape[1]);
        Assert.Equal(data, result.Data);
    }

    [Fact]
    public void Unscale_WithScale2_DividesBy2()
    {
        var data = new float[] { 2.0f, 4.0f, 6.0f, 8.0f };
        var tensor = new Tensor(data, new int[] { 2, 2 });

        var result = GradientUtils.Unscale(tensor, 2.0f);

        Assert.Equal(1.0f, result.Data[0]);
        Assert.Equal(2.0f, result.Data[1]);
        Assert.Equal(3.0f, result.Data[2]);
        Assert.Equal(4.0f, result.Data[3]);
    }

    [Fact]
    public void Unscale_WithLargeScale_DividesCorrectly()
    {
        var data = new float[] { 65536.0f, 131072.0f };
        var tensor = new Tensor(data, new int[] { 2 });

        var result = GradientUtils.Unscale(tensor, 65536.0f);

        Assert.Equal(1.0f, result.Data[0]);
        Assert.Equal(2.0f, result.Data[1]);
    }

    [Fact]
    public void Unscale_WithNullTensor_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            GradientUtils.Unscale(null, 2.0f));
    }

    [Fact]
    public void Unscale_WithZeroScale_ThrowsArgumentException()
    {
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        Assert.Throws<ArgumentException>(() =>
            GradientUtils.Unscale(tensor, 0.0f));
    }

    [Fact]
    public void Unscale_WithNegativeScale_ThrowsArgumentException()
    {
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        Assert.Throws<ArgumentException>(() =>
            GradientUtils.Unscale(tensor, -1.0f));
    }

    [Fact]
    public void Unscale_WithMultipleGradients_ScalesAllCorrectly()
    {
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 2.0f, 4.0f }, new int[] { 2 }) },
            { "param2", new Tensor(new float[] { 6.0f, 8.0f }, new int[] { 2 }) }
        };

        var result = GradientUtils.Unscale(gradients, 2.0f);

        Assert.Equal(2, result.Count);
        Assert.Equal(1.0f, result["param1"].Data[0]);
        Assert.Equal(2.0f, result["param1"].Data[1]);
        Assert.Equal(3.0f, result["param2"].Data[0]);
        Assert.Equal(4.0f, result["param2"].Data[1]);
    }

    [Fact]
    public void UnscaleInPlace_ModifiesTensorDirectly()
    {
        var data = new float[] { 2.0f, 4.0f, 6.0f, 8.0f };
        var tensor = new Tensor(data, new int[] { 2, 2 });

        GradientUtils.UnscaleInPlace(tensor, 2.0f);

        Assert.Equal(1.0f, tensor.Data[0]);
        Assert.Equal(2.0f, tensor.Data[1]);
        Assert.Equal(3.0f, tensor.Data[2]);
        Assert.Equal(4.0f, tensor.Data[3]);
    }

    #endregion

    #region CheckOverflow Tests

    [Fact]
    public void CheckOverflow_WithValidTensor_ReturnsFalse()
    {
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var tensor = new Tensor(data, new int[] { 2, 2 });

        var result = GradientUtils.CheckOverflow(tensor);

        Assert.False(result);
    }

    [Fact]
    public void CheckOverflow_WithNaNValue_ReturnsTrue()
    {
        var data = new float[] { 1.0f, float.NaN, 3.0f, 4.0f };
        var tensor = new Tensor(data, new int[] { 2, 2 });

        var result = GradientUtils.CheckOverflow(tensor);

        Assert.True(result);
    }

    [Fact]
    public void CheckOverflow_WithPositiveInfinity_ReturnsTrue()
    {
        var data = new float[] { 1.0f, float.PositiveInfinity, 3.0f, 4.0f };
        var tensor = new Tensor(data, new int[] { 2, 2 });

        var result = GradientUtils.CheckOverflow(tensor);

        Assert.True(result);
    }

    [Fact]
    public void CheckOverflow_WithNegativeInfinity_ReturnsTrue()
    {
        var data = new float[] { 1.0f, float.NegativeInfinity, 3.0f, 4.0f };
        var tensor = new Tensor(data, new int[] { 2, 2 });

        var result = GradientUtils.CheckOverflow(tensor);

        Assert.True(result);
    }

    [Fact]
    public void CheckOverflow_WithMultipleGradients_ReturnsTrueOnFirstOverflow()
    {
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }) },
            { "param2", new Tensor(new float[] { float.NaN, 4.0f }, new int[] { 2 }) }
        };

        var result = GradientUtils.CheckOverflow(gradients);

        Assert.True(result);
    }

    [Fact]
    public void CheckOverflow_WithMultipleValidGradients_ReturnsFalse()
    {
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }) },
            { "param2", new Tensor(new float[] { 3.0f, 4.0f }, new int[] { 2 }) }
        };

        var result = GradientUtils.CheckOverflow(gradients);

        Assert.False(result);
    }

    [Fact]
    public void CheckOverflow_WithNullGradients_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            GradientUtils.CheckOverflow((Dictionary<string, Tensor>)null));
    }

    [Fact]
    public void CheckOverflow_WithTensorArray_ReturnsCorrectResult()
    {
        var tensors = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }),
            new Tensor(new float[] { float.NaN, 4.0f }, new int[] { 2 })
        };

        var result = GradientUtils.CheckOverflow(tensors);

        Assert.True(result);
    }

    #endregion

    #region IsInf / IsNaN Tests

    [Fact]
    public void IsInf_WithValidValues_ReturnsFalse()
    {
        var tensor = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });

        var result = GradientUtils.IsInf(tensor);

        Assert.False(result);
    }

    [Fact]
    public void IsInf_WithPositiveInfinity_ReturnsTrue()
    {
        var tensor = new Tensor(new float[] { 1.0f, float.PositiveInfinity, 3.0f }, new int[] { 3 });

        var result = GradientUtils.IsInf(tensor);

        Assert.True(result);
    }

    [Fact]
    public void IsInf_WithNegativeInfinity_ReturnsTrue()
    {
        var tensor = new Tensor(new float[] { 1.0f, float.NegativeInfinity, 3.0f }, new int[] { 3 });

        var result = GradientUtils.IsInf(tensor);

        Assert.True(result);
    }

    [Fact]
    public void IsInf_WithNaN_ReturnsFalse()
    {
        var tensor = new Tensor(new float[] { 1.0f, float.NaN, 3.0f }, new int[] { 3 });

        var result = GradientUtils.IsInf(tensor);

        Assert.False(result);
    }

    [Fact]
    public void IsNaN_WithValidValues_ReturnsFalse()
    {
        var tensor = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });

        var result = GradientUtils.IsNaN(tensor);

        Assert.False(result);
    }

    [Fact]
    public void IsNaN_WithNaNValue_ReturnsTrue()
    {
        var tensor = new Tensor(new float[] { 1.0f, float.NaN, 3.0f }, new int[] { 3 });

        var result = GradientUtils.IsNaN(tensor);

        Assert.True(result);
    }

    [Fact]
    public void IsNaN_WithInfinity_ReturnsFalse()
    {
        var tensor = new Tensor(new float[] { 1.0f, float.PositiveInfinity, 3.0f }, new int[] { 3 });

        var result = GradientUtils.IsNaN(tensor);

        Assert.False(result);
    }

    [Fact]
    public void IsInfOrNaN_WithValidValues_ReturnsFalse()
    {
        var tensor = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });

        var result = GradientUtils.IsInfOrNaN(tensor);

        Assert.False(result);
    }

    [Fact]
    public void IsInfOrNaN_WithNaN_ReturnsTrue()
    {
        var tensor = new Tensor(new float[] { 1.0f, float.NaN, 3.0f }, new int[] { 3 });

        var result = GradientUtils.IsInfOrNaN(tensor);

        Assert.True(result);
    }

    [Fact]
    public void IsInfOrNaN_WithInfinity_ReturnsTrue()
    {
        var tensor = new Tensor(new float[] { 1.0f, float.PositiveInfinity, 3.0f }, new int[] { 3 });

        var result = GradientUtils.IsInfOrNaN(tensor);

        Assert.True(result);
    }

    #endregion

    #region FindOverflowGradients Tests

    [Fact]
    public void FindOverflowGradients_WithValidGradients_ReturnsEmptyList()
    {
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }) },
            { "param2", new Tensor(new float[] { 3.0f, 4.0f }, new int[] { 2 }) }
        };

        var result = GradientUtils.FindOverflowGradients(gradients);

        Assert.Empty(result);
    }

    [Fact]
    public void FindOverflowGradients_WithOverflow_ReturnsCorrectParameters()
    {
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f, float.NaN }, new int[] { 2 }) },
            { "param2", new Tensor(new float[] { 3.0f, 4.0f }, new int[] { 2 }) },
            { "param3", new Tensor(new float[] { float.PositiveInfinity, 4.0f }, new int[] { 2 }) }
        };

        var result = GradientUtils.FindOverflowGradients(gradients);

        Assert.Equal(2, result.Count);
        Assert.Contains("param1", result);
        Assert.Contains("param3", result);
        Assert.DoesNotContain("param2", result);
    }

    #endregion

    #region GetOverflowStats Tests

    [Fact]
    public void GetOverflowStats_WithValidGradients_ReturnsCorrectStats()
    {
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }) },
            { "param2", new Tensor(new float[] { 3.0f, 4.0f }, new int[] { 2 }) }
        };

        var stats = GradientUtils.GetOverflowStats(gradients);

        Assert.Equal(2, stats.TotalGradients);
        Assert.Equal(0, stats.OverflowCount);
        Assert.Equal(0.0f, stats.OverflowRate);
        Assert.False(stats.HasOverflow);
    }

    [Fact]
    public void GetOverflowStats_WithOverflow_ReturnsCorrectStats()
    {
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f, float.NaN }, new int[] { 2 }) },
            { "param2", new Tensor(new float[] { 3.0f, 4.0f }, new int[] { 2 }) },
            { "param3", new Tensor(new float[] { float.PositiveInfinity, 4.0f }, new int[] { 2 }) }
        };

        var stats = GradientUtils.GetOverflowStats(gradients);

        Assert.Equal(3, stats.TotalGradients);
        Assert.Equal(2, stats.OverflowCount);
        Assert.Equal(2.0f / 3.0f, stats.OverflowRate, precision: 4);
        Assert.True(stats.HasOverflow);
    }

    [Fact]
    public void GetOverflowStats_WithEmptyGradients_ReturnsZeroStats()
    {
        var gradients = new Dictionary<string, Tensor>();

        var stats = GradientUtils.GetOverflowStats(gradients);

        Assert.Equal(0, stats.TotalGradients);
        Assert.Equal(0, stats.OverflowCount);
        Assert.Equal(0.0f, stats.OverflowRate);
        Assert.False(stats.HasOverflow);
    }

    #endregion
}
