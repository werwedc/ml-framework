using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using MLFramework.Autograd.Functions;
using Xunit;
using System;

namespace MLFramework.Tests.Autograd.Functions;

/// <summary>
/// Unit tests for ClampedMSELoss custom function.
/// </summary>
public class ClampedMSELossTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void Constructor_ValidParameters_CreatesInstance()
    {
        // Arrange & Act
        var loss = new ClampedMSELoss(-1.0, 1.0);

        // Assert
        Assert.NotNull(loss);
    }

    [Fact]
    public void Constructor_ClampMinGreaterThanClampMax_ThrowsException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentException>(() => new ClampedMSELoss(1.0, -1.0));
    }

    [Fact]
    public void Constructor_ClampMinEqualsClampMax_ThrowsException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentException>(() => new ClampedMSELoss(1.0, 1.0));
    }

    [Fact]
    public void Constructor_InvalidReduction_ThrowsException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentException>(() => new ClampedMSELoss(-1.0, 1.0, "invalid"));
    }

    [Fact]
    public void Forward_NoClamping_MatchesStandardMSE()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 1.5f, 2.5f, 3.5f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-10.0, 10.0, "mean");

        // Act
        var result = loss.Apply(predictions, targets);

        // Assert
        // Standard MSE: ((-0.5)^2 + (-0.5)^2 + (-0.5)^2) / 3 = 0.75 / 3 = 0.25
        Assert.Equal(0.25f, result.Data[0], Tolerance);
    }

    [Fact]
    public void Forward_PartialClamping_ClampsCorrectly()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 3.0f, 2.5f, 3.5f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-1.0, 1.0, "mean");

        // Act
        var result = loss.Apply(predictions, targets);

        // Assert
        // Differences: [-2.0, -0.5, -0.5]
        // Clamped: [-1.0, -0.5, -0.5]
        // Squared: [1.0, 0.25, 0.25]
        // Mean: 1.5 / 3 = 0.5
        Assert.Equal(0.5f, result.Data[0], Tolerance);
    }

    [Fact]
    public void Forward_AllClamping_AllDifferencesClamped()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 10.0f, 20.0f, 30.0f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-1.0, 1.0, "mean");

        // Act
        var result = loss.Apply(predictions, targets);

        // Assert
        // All differences clamped to -1.0
        // All squared differences are 1.0
        // Mean: 3.0 / 3 = 1.0
        Assert.Equal(1.0f, result.Data[0], Tolerance);
    }

    [Fact]
    public void Forward_SumReduction_ComputesSum()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 1.5f, 2.5f, 3.5f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-10.0, 10.0, "sum");

        // Act
        var result = loss.Apply(predictions, targets);

        // Assert
        // Standard MSE sum: 0.25 + 0.25 + 0.25 = 0.75
        Assert.Equal(0.75f, result.Data[0], Tolerance);
    }

    [Fact]
    public void Forward_NoneReduction_ReturnsPerElementLosses()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 1.5f, 2.5f, 3.5f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-10.0, 10.0, "none");

        // Act
        var result = loss.Apply(predictions, targets);

        // Assert
        Assert.Equal(3, result.Size);
        Assert.Equal(new[] { 3 }, result.Shape);
        Assert.Equal(0.25f, result.Data[0], Tolerance);
        Assert.Equal(0.25f, result.Data[1], Tolerance);
        Assert.Equal(0.25f, result.Data[2], Tolerance);
    }

    [Fact]
    public void Forward_PerfectPredictions_ReturnsZero()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-1.0, 1.0, "mean");

        // Act
        var result = loss.Apply(predictions, targets);

        // Assert
        Assert.Equal(0.0f, result.Data[0], Tolerance);
    }

    [Fact]
    public void Forward_MultiDimensionalTensor_ComputesCorrectly()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 2, 2 }, true);
        var targets = new Tensor(new[] { 1.5f, 2.5f, 3.5f, 4.5f }, new[] { 2, 2 }, true);
        var loss = new ClampedMSELoss(-10.0, 10.0, "mean");

        // Act
        var result = loss.Apply(predictions, targets);

        // Assert
        // All squared differences are 0.25
        // Mean: 1.0 / 4 = 0.25
        Assert.Equal(0.25f, result.Data[0], Tolerance);
    }

    [Fact]
    public void Forward_NullPredictions_ThrowsException()
    {
        // Arrange
        var targets = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-1.0, 1.0);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => loss.Apply(null!, targets));
    }

    [Fact]
    public void Forward_NullTargets_ThrowsException()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-1.0, 1.0);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => loss.Apply(predictions, null!));
    }

    [Fact]
    public void Forward_DifferentShapes_ThrowsException()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 1.0f, 2.0f }, new[] { 2 }, true);
        var loss = new ClampedMSELoss(-1.0, 1.0);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => loss.Apply(predictions, targets));
    }

    [Fact]
    public void Backward_NoClamping_MatchesStandardMSEGradient()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 1.5f, 2.5f, 3.5f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-10.0, 10.0, "mean");

        // Act
        var lossTensor = loss.Apply(predictions, targets);
        lossTensor.Backward();

        // Assert
        // Gradient should be: 2 * diff / n = 2 * (-0.5) / 3 = -0.333...
        var grad = predictions.Gradient;
        Assert.NotNull(grad);
        Assert.Equal(-0.333333f, grad.Data[0], 0.001f);
        Assert.Equal(-0.333333f, grad.Data[1], 0.001f);
        Assert.Equal(-0.333333f, grad.Data[2], 0.001f);
    }

    [Fact]
    public void Backward_AllClamped_GradientIsZero()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 10.0f, 20.0f, 30.0f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-1.0, 1.0, "mean");

        // Act
        var lossTensor = loss.Apply(predictions, targets);
        lossTensor.Backward();

        // Assert
        // All differences are clamped, so gradient should be zero
        var grad = predictions.Gradient;
        Assert.NotNull(grad);
        Assert.Equal(0.0f, grad.Data[0], Tolerance);
        Assert.Equal(0.0f, grad.Data[1], Tolerance);
        Assert.Equal(0.0f, grad.Data[2], Tolerance);
    }

    [Fact]
    public void Backward_PartialClamping_GradientOnlyForUnclamped()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 3.0f, 2.5f, 3.5f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-1.0, 1.0, "mean");

        // Act
        var lossTensor = loss.Apply(predictions, targets);
        lossTensor.Backward();

        // Assert
        // Differences: [-2.0, -0.5, -0.5]
        // Clamped: [-1.0, -0.5, -0.5]
        // Mask: [0, 1, 1] (first element is clamped)
        // Gradient: [0, -0.333, -0.333]
        var grad = predictions.Gradient;
        Assert.NotNull(grad);
        Assert.Equal(0.0f, grad.Data[0], Tolerance);
        Assert.Equal(-0.333333f, grad.Data[1], 0.001f);
        Assert.Equal(-0.333333f, grad.Data[2], 0.001f);
    }

    [Fact]
    public void Backward_SumReduction_ComputesCorrectGradient()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 1.5f, 2.5f, 3.5f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-10.0, 10.0, "sum");

        // Act
        var lossTensor = loss.Apply(predictions, targets);
        lossTensor.Backward();

        // Assert
        // Gradient for sum: 2 * diff = 2 * (-0.5) = -1.0
        var grad = predictions.Gradient;
        Assert.NotNull(grad);
        Assert.Equal(-1.0f, grad.Data[0], Tolerance);
        Assert.Equal(-1.0f, grad.Data[1], Tolerance);
        Assert.Equal(-1.0f, grad.Data[2], Tolerance);
    }

    [Fact]
    public void Backward_GradTargetsIsNegativeOfGradPreds()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 1.5f, 2.5f, 3.5f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-10.0, 10.0, "mean");

        // Act
        var lossTensor = loss.Apply(predictions, targets);
        lossTensor.Backward();

        // Assert
        var gradPreds = predictions.Gradient;
        var gradTargets = targets.Gradient;

        Assert.NotNull(gradPreds);
        Assert.NotNull(gradTargets);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(-gradPreds.Data[i], gradTargets.Data[i], Tolerance);
        }
    }

    [Fact]
    public void Backward_GradientShapesMatchInputShapes()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 2, 2 }, true);
        var targets = new Tensor(new[] { 1.5f, 2.5f, 3.5f, 4.5f }, new[] { 2, 2 }, true);
        var loss = new ClampedMSELoss(-10.0, 10.0, "mean");

        // Act
        var lossTensor = loss.Apply(predictions, targets);
        lossTensor.Backward();

        // Assert
        var gradPreds = predictions.Gradient;
        var gradTargets = targets.Gradient;

        Assert.NotNull(gradPreds);
        Assert.NotNull(gradTargets);
        Assert.True(gradPreds.HasSameShape(predictions));
        Assert.True(gradTargets.HasSameShape(targets));
    }

    [Fact]
    public void Backward_SingleElementTensor_ComputesCorrectly()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f }, new[] { 1 }, true);
        var targets = new Tensor(new[] { 1.5f }, new[] { 1 }, true);
        var loss = new ClampedMSELoss(-10.0, 10.0, "mean");

        // Act
        var lossTensor = loss.Apply(predictions, targets);
        lossTensor.Backward();

        // Assert
        // Gradient: 2 * (-0.5) / 1 = -1.0
        var grad = predictions.Gradient;
        Assert.NotNull(grad);
        Assert.Equal(-1.0f, grad.Data[0], Tolerance);
    }

    [Fact]
    public void Backward_AllDifferencesClamped_TargetGradientIsZero()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 10.0f, 20.0f, 30.0f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-1.0, 1.0, "mean");

        // Act
        var lossTensor = loss.Apply(predictions, targets);
        lossTensor.Backward();

        // Assert
        var grad = targets.Gradient;
        Assert.NotNull(grad);
        Assert.Equal(0.0f, grad.Data[0], Tolerance);
        Assert.Equal(0.0f, grad.Data[1], Tolerance);
        Assert.Equal(0.0f, grad.Data[2], Tolerance);
    }

    [Fact]
    public void Integration_AutogradIntegration_WorksCorrectly()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 1.5f, 2.5f, 3.5f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-1.0, 1.0, "mean");

        // Act
        var lossTensor = loss.Apply(predictions, targets);

        // Verify forward pass
        Assert.Equal(0.25f, lossTensor.Data[0], Tolerance);

        // Verify backward pass
        lossTensor.Backward();
        Assert.NotNull(predictions.Gradient);
        Assert.NotNull(targets.Gradient);
    }

    [Fact]
    public void Integration_SimpleNetwork_ComputesLossAndGradients()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 1.5f, 2.5f, 3.5f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-1.0, 1.0, "mean");

        // Act
        var lossTensor = loss.Apply(predictions, targets);
        lossTensor.Backward();

        // Assert - verify loss is computed
        Assert.Equal(0.25f, lossTensor.Data[0], Tolerance);

        // Assert - verify gradients are computed
        Assert.NotNull(predictions.Gradient);
        Assert.Equal(3, predictions.Gradient.Size);
    }

    [Fact]
    public void EdgeCase_ClampMinEqualsClampMaxMinusEpsilon_WorksCorrectly()
    {
        // Arrange
        var predictions = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var targets = new Tensor(new[] { 1.5f, 2.5f, 3.5f }, new[] { 3 }, true);
        var loss = new ClampedMSELoss(-0.5f, -0.49f, "mean");

        // Act
        var result = loss.Apply(predictions, targets);

        // Assert - loss should be computed (though with extreme clamping)
        Assert.NotNull(result);
        Assert.True(result.Data[0] >= 0);
    }
}
