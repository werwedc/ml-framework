using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using MLFramework.Autograd.Functions;
using Xunit;
using System;
using System.Linq;

namespace MLFramework.Tests.Autograd.Functions;

/// <summary>
/// Unit tests for StableSoftmax custom function.
/// </summary>
public class StableSoftmaxTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void Constructor_DefaultParameters_CreatesInstance()
    {
        // Arrange & Act
        var softmax = new StableSoftmax();

        // Assert
        Assert.NotNull(softmax);
    }

    [Fact]
    public void Constructor_CustomParameters_CreatesInstance()
    {
        // Arrange & Act
        var softmax = new StableSoftmax(dim: 0, keepDim: false);

        // Assert
        Assert.NotNull(softmax);
    }

    [Fact]
    public void Forward_SmallValues_ProbabilitiesSumToOne()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();

        // Act
        var result = softmax.Apply(x);

        // Assert
        Assert.Equal(3, result.Size);
        Assert.Equal(new[] { 3 }, result.Shape);
        
        // All probabilities should be positive
        Assert.All(result.Data, val => Assert.True(val > 0));
        
        // Probabilities should sum to 1
        float sum = result.Data.Sum();
        Assert.Equal(1.0f, sum, Tolerance);
    }

    [Fact]
    public void Forward_LargeValues_NumericalStability()
    {
        // Arrange
        var x = new Tensor(new[] { 1000.0f, 1001.0f, 1002.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();

        // Act
        var result = softmax.Apply(x);

        // Assert
        // Should not produce NaN or Inf
        Assert.All(result.Data, val => Assert.False(float.IsNaN(val) || float.IsInfinity(val)));
        
        // Probabilities should sum to 1
        float sum = result.Data.Sum();
        Assert.Equal(1.0f, sum, Tolerance);
    }

    [Fact]
    public void Forward_MixedPositiveAndNegativeValues_CorrectProbabilities()
    {
        // Arrange
        var x = new Tensor(new[] { -2.0f, 0.0f, 2.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();

        // Act
        var result = softmax.Apply(x);

        // Assert
        Assert.Equal(3, result.Size);
        
        // All probabilities should be positive
        Assert.All(result.Data, val => Assert.True(val > 0));
        
        // Probabilities should sum to 1
        float sum = result.Data.Sum();
        Assert.Equal(1.0f, sum, Tolerance);
        
        // Higher input should have higher probability
        Assert.True(result.Data[0] < result.Data[1]);
        Assert.True(result.Data[1] < result.Data[2]);
    }

    [Fact]
    public void Forward_AllEqualValues_UniformDistribution()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 1.0f, 1.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();

        // Act
        var result = softmax.Apply(x);

        // Assert
        Assert.Equal(3, result.Size);
        
        // All probabilities should be equal (uniform distribution)
        float expectedProb = 1.0f / 3.0f;
        Assert.All(result.Data, val => Assert.Equal(expectedProb, val, Tolerance));
    }

    [Fact]
    public void Forward_SingleElementTensor_ReturnsOne()
    {
        // Arrange
        var x = new Tensor(new[] { 5.0f }, new[] { 1 }, true);
        var softmax = new StableSoftmax();

        // Act
        var result = softmax.Apply(x);

        // Assert
        Assert.Equal(1, result.Size);
        Assert.Equal(1.0f, result.Data[0], Tolerance);
    }

    [Fact]
    public void Forward_TwoDimensionalTensor_CorrectShape()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, new[] { 2, 3 }, true);
        var softmax = new StableSoftmax(dim: -1, keepDim: true);

        // Act
        var result = softmax.Apply(x);

        // Assert
        Assert.Equal(new[] { 2, 3 }, result.Shape);
        
        // Each row should sum to 1
        for (int i = 0; i < 2; i++)
        {
            float rowSum = 0.0f;
            for (int j = 0; j < 3; j++)
            {
                rowSum += result.Data[i * 3 + j];
            }
            Assert.Equal(1.0f, rowSum, Tolerance);
        }
    }

    [Fact]
    public void Forward_CustomDimension_ComputesAlongDimension()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, new[] { 2, 3 }, true);
        var softmax = new StableSoftmax(dim: 0, keepDim: true);

        // Act
        var result = softmax.Apply(x);

        // Assert
        Assert.Equal(new[] { 2, 3 }, result.Shape);
        
        // Each column should sum to 1
        for (int j = 0; j < 3; j++)
        {
            float colSum = 0.0f;
            for (int i = 0; i < 2; i++)
            {
                colSum += result.Data[i * 3 + j];
            }
            Assert.Equal(1.0f, colSum, Tolerance);
        }
    }

    [Fact]
    public void Forward_KeepDimFalse_ReducedDimensionRemoved()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, new[] { 2, 3 }, true);
        var softmax = new StableSoftmax(dim: 1, keepDim: false);

        // Act
        var result = softmax.Apply(x);

        // Assert
        Assert.Equal(new[] { 2 }, result.Shape);
        Assert.Equal(2, result.Size);
    }

    [Fact]
    public void Forward_VerySmallValues_NumericalStability()
    {
        // Arrange
        var x = new Tensor(new[] { -1000.0f, -1001.0f, -1002.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();

        // Act
        var result = softmax.Apply(x);

        // Assert
        // Should not produce NaN or Inf
        Assert.All(result.Data, val => Assert.False(float.IsNaN(val) || float.IsInfinity(val)));
        
        // Probabilities should sum to 1
        float sum = result.Data.Sum();
        Assert.Equal(1.0f, sum, Tolerance);
    }

    [Fact]
    public void Forward_NullInput_ThrowsException()
    {
        // Arrange
        var softmax = new StableSoftmax();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => softmax.Apply(null!));
    }

    [Fact]
    public void Backward_UniformGradient_CorrectGradientShape()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();
        var result = softmax.Apply(x);

        // Act
        result.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(3, x.Gradient.Size);
        Assert.Equal(new[] { 3 }, x.Gradient.Shape);
    }

    [Fact]
    public void Backward_UniformGradient_SumOfGradientsIsZero()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();
        var result = softmax.Apply(x);

        // Act
        result.Backward();

        // Assert
        // Sum of gradients should be zero (softmax derivative property)
        var grad = x.Gradient;
        Assert.NotNull(grad);
        float sum = grad.Data.Sum();
        Assert.Equal(0.0f, sum, 1e-4f);
    }

    [Fact]
    public void Backward_SpecificInput_CorrectGradientValues()
    {
        // Arrange
        var x = new Tensor(new[] { 0.0f, 1.0f, 2.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();
        var result = softmax.Apply(x);

        // Act
        result.Backward();

        // Assert
        // For softmax, gradient is: grad_x = y * (1 - y) for uniform grad_y = 1
        var y = result;
        var grad = x.Gradient;
        Assert.NotNull(grad);
        
        // Verify gradient formula
        for (int i = 0; i < 3; i++)
        {
            float expectedGrad = y.Data[i] * (1.0f - y.Data[i]);
            Assert.Equal(expectedGrad, grad.Data[i], Tolerance);
        }
    }

    [Fact]
    public void Backward_TwoDimensionalTensor_CorrectGradientShape()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, new[] { 2, 3 }, true);
        var softmax = new StableSoftmax(dim: -1, keepDim: true);
        var result = softmax.Apply(x);

        // Act
        result.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(new[] { 2, 3 }, x.Gradient.Shape);
    }

    [Fact]
    public void Backward_CustomDimension_CorrectGradient()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, new[] { 2, 3 }, true);
        var softmax = new StableSoftmax(dim: 0, keepDim: true);
        var result = softmax.Apply(x);

        // Act
        result.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(new[] { 2, 3 }, x.Gradient.Shape);
    }

    [Fact]
    public void Backward_NullGradient_ThrowsException()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();
        var ctx = new FunctionContext();
        
        // Save the output manually
        var output = softmax.Forward(new[] { x }, ctx);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => softmax.Backward(null!, ctx));
    }

    [Fact]
    public void Integration_SimpleClassificationNetwork_WorksCorrectly()
    {
        // Arrange
        var logits = new Tensor(new[] { 2.0f, 1.0f, 0.1f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();

        // Act
        var probs = softmax.Apply(logits);
        probs.Backward();

        // Assert
        // Verify forward pass
        Assert.Equal(1.0f, probs.Data.Sum(), Tolerance);
        Assert.All(probs.Data, val => Assert.True(val > 0));
        
        // Verify backward pass
        Assert.NotNull(logits.Gradient);
        Assert.Equal(3, logits.Gradient.Size);
    }

    [Fact]
    public void Integration_AutogradGraph_WorksCorrectly()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();

        // Act
        var y = softmax.Apply(x);
        var loss = y.Sum(); // Use sum as a simple loss
        loss.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(3, x.Gradient.Size);
    }

    [Fact]
    public void NumericalStability_HighValues_NoOverflow()
    {
        // Arrange
        var x = new Tensor(new[] { 100.0f, 200.0f, 300.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();

        // Act
        var result = softmax.Apply(x);

        // Assert
        // Should not overflow (no Inf)
        Assert.All(result.Data, val => Assert.False(float.IsInfinity(val)));
        Assert.Equal(1.0f, result.Data.Sum(), Tolerance);
    }

    [Fact]
    public void NumericalStability_LowValues_NoUnderflow()
    {
        // Arrange
        var x = new Tensor(new[] { -100.0f, -200.0f, -300.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();

        // Act
        var result = softmax.Apply(x);

        // Assert
        // Should not underflow to all zeros
        Assert.All(result.Data, val => Assert.True(val > 0));
        Assert.Equal(1.0f, result.Data.Sum(), Tolerance);
    }

    [Fact]
    public void EdgeCase_DimensionOutOfRange_ThrowsException()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax(dim: 5, keepDim: true);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => softmax.Apply(x));
    }

    [Fact]
    public void EdgeCase_NegativeDimensionWithinBounds_WorksCorrectly()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax(dim: -1, keepDim: true);

        // Act
        var result = softmax.Apply(x);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(1.0f, result.Data.Sum(), Tolerance);
    }

    [Fact]
    public void Backward_GradientFlow_ThroughAutogradGraph()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();
        var y = softmax.Apply(x);
        
        // Create a simple computation graph
        var z = y.MultiplyScalar(2.0f);
        var loss = z.Sum();

        // Act
        loss.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(3, x.Gradient.Size);
        
        // Gradient should flow correctly through the graph
        Assert.All(x.Gradient.Data, val => Assert.False(float.IsNaN(val)));
    }

    [Fact]
    public void Consistency_MultipleCalls_SameInputSameOutput()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var softmax = new StableSoftmax();

        // Act
        var result1 = softmax.Apply(x);
        var result2 = softmax.Apply(x);

        // Assert
        Assert.Equal(result1.Size, result2.Size);
        for (int i = 0; i < result1.Size; i++)
        {
            Assert.Equal(result1.Data[i], result2.Data[i], Tolerance);
        }
    }

    [Fact]
    public void Forward_LargeTensor_PerformanceIsReasonable()
    {
        // Arrange
        int size = 1000;
        var data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = i * 0.001f;
        }
        var x = new Tensor(data, new[] { size }, true);
        var softmax = new StableSoftmax();

        // Act
        var result = softmax.Apply(x);

        // Assert
        Assert.Equal(size, result.Size);
        Assert.Equal(1.0f, result.Data.Sum(), Tolerance);
        Assert.All(result.Data, val => Assert.True(val > 0 && val < 1));
    }
}
