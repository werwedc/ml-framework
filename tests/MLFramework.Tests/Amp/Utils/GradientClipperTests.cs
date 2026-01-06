using MLFramework.Amp;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using Xunit;

namespace MLFramework.Tests.Amp.Utils;

/// <summary>
/// Tests for GradientClipper class
/// </summary>
public class GradientClipperTests
{
    #region ClipByValue Tests

    [Fact]
    public void ClipByValue_WithValuesWithinBounds_ReturnsOriginalValues()
    {
        var data = new float[] { -0.5f, 0.0f, 0.5f, 1.0f };
        var tensor = new Tensor(data, new int[] { 2, 2 });

        var result = GradientClipper.ClipByValue(tensor, 1.0f);

        Assert.Equal(-0.5f, result.Data[0]);
        Assert.Equal(0.0f, result.Data[1]);
        Assert.Equal(0.5f, result.Data[2]);
        Assert.Equal(1.0f, result.Data[3]);
    }

    [Fact]
    public void ClipByValue_WithValuesAboveClip_ClipsToClipValue()
    {
        var data = new float[] { 0.5f, 1.5f, 2.0f, 3.0f };
        var tensor = new Tensor(data, new int[] { 2, 2 });

        var result = GradientClipper.ClipByValue(tensor, 1.0f);

        Assert.Equal(0.5f, result.Data[0]);
        Assert.Equal(1.0f, result.Data[1]);
        Assert.Equal(1.0f, result.Data[2]);
        Assert.Equal(1.0f, result.Data[3]);
    }

    [Fact]
    public void ClipByValue_WithValuesBelowNegativeClip_ClipsToNegativeClipValue()
    {
        var data = new float[] { -0.5f, -1.5f, -2.0f, -3.0f };
        var tensor = new Tensor(data, new int[] { 2, 2 });

        var result = GradientClipper.ClipByValue(tensor, 1.0f);

        Assert.Equal(-0.5f, result.Data[0]);
        Assert.Equal(-1.0f, result.Data[1]);
        Assert.Equal(-1.0f, result.Data[2]);
        Assert.Equal(-1.0f, result.Data[3]);
    }

    [Fact]
    public void ClipByValue_WithMixedValues_ClipsCorrectly()
    {
        var data = new float[] { -2.0f, -0.5f, 0.5f, 2.0f };
        var tensor = new Tensor(data, new int[] { 2, 2 });

        var result = GradientClipper.ClipByValue(tensor, 1.0f);

        Assert.Equal(-1.0f, result.Data[0]);
        Assert.Equal(-0.5f, result.Data[1]);
        Assert.Equal(0.5f, result.Data[2]);
        Assert.Equal(1.0f, result.Data[3]);
    }

    [Fact]
    public void ClipByValue_WithNullTensor_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            GradientClipper.ClipByValue(null, 1.0f));
    }

    [Fact]
    public void ClipByValue_WithZeroClipValue_ThrowsArgumentException()
    {
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        Assert.Throws<ArgumentException>(() =>
            GradientClipper.ClipByValue(tensor, 0.0f));
    }

    [Fact]
    public void ClipByValue_WithNegativeClipValue_ThrowsArgumentException()
    {
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        Assert.Throws<ArgumentException>(() =>
            GradientClipper.ClipByValue(tensor, -1.0f));
    }

    #endregion

    #region ClipByNorm Tests (Single Tensor)

    [Fact]
    public void ClipByNorm_WithNormWithinBounds_ReturnsOriginalTensor()
    {
        var data = new float[] { 3.0f, 4.0f }; // Norm = 5
        var tensor = new Tensor(data, new int[] { 2 });

        var result = GradientClipper.ClipByNorm(tensor, maxNorm: 10.0f, normType: 2.0f);

        Assert.Equal(3.0f, result.Data[0]);
        Assert.Equal(4.0f, result.Data[1]);
    }

    [Fact]
    public void ClipByNorm_WithNormAboveBounds_ClipsCorrectly()
    {
        var data = new float[] { 3.0f, 4.0f }; // Norm = 5
        var tensor = new Tensor(data, new int[] { 2 });

        var result = GradientClipper.ClipByNorm(tensor, maxNorm: 2.5f, normType: 2.0f);

        // Scale = 2.5 / 5 = 0.5
        // New values: 3 * 0.5 = 1.5, 4 * 0.5 = 2.0
        Assert.Equal(1.5f, result.Data[0], precision: 4);
        Assert.Equal(2.0f, result.Data[1], precision: 4);
    }

    [Fact]
    public void ClipByNorm_WithL1Norm_ClipsCorrectly()
    {
        var data = new float[] { 3.0f, 4.0f }; // L1 norm = 7
        var tensor = new Tensor(data, new int[] { 2 });

        var result = GradientClipper.ClipByNorm(tensor, maxNorm: 3.5f, normType: 1.0f);

        // Scale = 3.5 / 7 = 0.5
        Assert.Equal(1.5f, result.Data[0], precision: 4);
        Assert.Equal(2.0f, result.Data[1], precision: 4);
    }

    [Fact]
    public void ClipByNorm_WithNullTensor_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            GradientClipper.ClipByNorm(null, maxNorm: 1.0f));
    }

    [Fact]
    public void ClipByNorm_WithZeroMaxNorm_ThrowsArgumentException()
    {
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        Assert.Throws<ArgumentException>(() =>
            GradientClipper.ClipByNorm(tensor, maxNorm: 0.0f));
    }

    [Fact]
    public void ClipByNorm_WithNegativeMaxNorm_ThrowsArgumentException()
    {
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        Assert.Throws<ArgumentException>(() =>
            GradientClipper.ClipByNorm(tensor, maxNorm: -1.0f));
    }

    [Fact]
    public void ClipByNorm_WithNegativeNormType_ThrowsArgumentException()
    {
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        Assert.Throws<ArgumentException>(() =>
            GradientClipper.ClipByNorm(tensor, maxNorm: 1.0f, normType: -1.0f));
    }

    #endregion

    #region ClipByNorm Tests (Multiple Gradients)

    [Fact]
    public void ClipByNorm_WithMultipleGradientsAndNormWithinBounds_ReturnsOriginalGradients()
    {
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 3.0f, 0.0f }, new int[] { 2 }) }, // Norm = 3
            { "param2", new Tensor(new float[] { 0.0f, 4.0f }, new int[] { 2 }) }  // Norm = 4
        };
        // Total norm = sqrt(3^2 + 4^2) = 5

        var result = GradientClipper.ClipByNorm(gradients, maxNorm: 10.0f, normType: 2.0f);

        Assert.Equal(3.0f, result["param1"].Data[0]);
        Assert.Equal(0.0f, result["param1"].Data[1]);
        Assert.Equal(0.0f, result["param2"].Data[0]);
        Assert.Equal(4.0f, result["param2"].Data[1]);
    }

    [Fact]
    public void ClipByNorm_WithMultipleGradientsAndNormAboveBounds_ClipsCorrectly()
    {
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 3.0f, 0.0f }, new int[] { 2 }) }, // Norm = 3
            { "param2", new Tensor(new float[] { 0.0f, 4.0f }, new int[] { 2 }) }  // Norm = 4
        };
        // Total norm = sqrt(3^2 + 4^2) = 5

        var result = GradientClipper.ClipByNorm(gradients, maxNorm: 2.5f, normType: 2.0f);

        // Scale = 2.5 / 5 = 0.5
        Assert.Equal(1.5f, result["param1"].Data[0], precision: 4);
        Assert.Equal(0.0f, result["param1"].Data[1], precision: 4);
        Assert.Equal(0.0f, result["param2"].Data[0], precision: 4);
        Assert.Equal(2.0f, result["param2"].Data[1], precision: 4);
    }

    [Fact]
    public void ClipByNorm_WithMultipleGradientsAndNullGradients_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            GradientClipper.ClipByNorm(null, maxNorm: 1.0f));
    }

    #endregion

    #region ComputeNorm Tests (Multiple Gradients)

    [Fact]
    public void ComputeNorm_WithMultipleGradients_CalculatesL2NormCorrectly()
    {
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 3.0f, 0.0f }, new int[] { 2 }) }, // Norm = 3
            { "param2", new Tensor(new float[] { 0.0f, 4.0f }, new int[] { 2 }) }  // Norm = 4
        };

        var norm = GradientClipper.ComputeNorm(gradients, normType: 2.0f);

        // Total norm = sqrt(3^2 + 4^2) = 5
        Assert.Equal(5.0f, norm, precision: 4);
    }

    [Fact]
    public void ComputeNorm_WithMultipleGradientsAndL1Norm_CalculatesCorrectly()
    {
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 3.0f, 0.0f }, new int[] { 2 }) }, // L1 norm = 3
            { "param2", new Tensor(new float[] { 0.0f, 4.0f }, new int[] { 2 }) }  // L1 norm = 4
        };

        var norm = GradientClipper.ComputeNorm(gradients, normType: 1.0f);

        // Total L1 norm = 3 + 4 = 7
        Assert.Equal(7.0f, norm, precision: 4);
    }

    [Fact]
    public void ComputeNorm_WithSingleGradient_CalculatesCorrectly()
    {
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 3.0f, 4.0f }, new int[] { 2 }) } // Norm = 5
        };

        var norm = GradientClipper.ComputeNorm(gradients, normType: 2.0f);

        Assert.Equal(5.0f, norm, precision: 4);
    }

    [Fact]
    public void ComputeNorm_WithEmptyGradients_ReturnsZero()
    {
        var gradients = new Dictionary<string, Tensor>();

        var norm = GradientClipper.ComputeNorm(gradients, normType: 2.0f);

        Assert.Equal(0.0f, norm);
    }

    [Fact]
    public void ComputeNorm_WithNegativeNormType_ThrowsArgumentException()
    {
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f }, new int[] { 1 }) }
        };

        Assert.Throws<ArgumentException>(() =>
            GradientClipper.ComputeNorm(gradients, normType: -1.0f));
    }

    [Fact]
    public void ComputeNorm_WithNullGradients_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            GradientClipper.ComputeNorm(null, normType: 2.0f));
    }

    #endregion
}
