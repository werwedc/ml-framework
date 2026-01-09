using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using MLFramework.Autograd.Functions;
using Xunit;
using System.Linq;
using RitterFramework.Core;

namespace MLFramework.Tests.Autograd.Functions;

/// <summary>
/// Unit tests for STEBinary custom function.
/// </summary>
public class STEBinaryTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void Constructor_DefaultParameters_CreatesInstance()
    {
        // Arrange & Act
        var steBinary = new STEBinary();

        // Assert
        Assert.NotNull(steBinary);
    }

    [Fact]
    public void Constructor_CustomZeroValue_CreatesInstance()
    {
        // Arrange & Act
        var steBinary = new STEBinary(zeroValue: 1.0);

        // Assert
        Assert.NotNull(steBinary);
    }

    [Fact]
    public void Forward_PositiveValues_ReturnsOnes()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.Equal(3, result.Size);
        Assert.Equal(new[] { 3 }, result.Shape);
        Assert.All(result.Data, val => Assert.Equal(1.0f, val));
    }

    [Fact]
    public void Forward_NegativeValues_ReturnsMinusOnes()
    {
        // Arrange
        var x = new Tensor(new[] { -1.0f, -2.0f, -3.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.Equal(3, result.Size);
        Assert.Equal(new[] { 3 }, result.Shape);
        Assert.All(result.Data, val => Assert.Equal(-1.0f, val));
    }

    [Fact]
    public void Forward_ZeroValues_ReturnsZero()
    {
        // Arrange
        var x = new Tensor(new[] { 0.0f, 0.0f, 0.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.Equal(3, result.Size);
        Assert.Equal(new[] { 3 }, result.Shape);
        Assert.All(result.Data, val => Assert.Equal(0.0f, val));
    }

    [Fact]
    public void Forward_CustomZeroValue_ReturnsConfiguredValue()
    {
        // Arrange
        var x = new Tensor(new[] { 0.0f, 0.0f, 0.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary(zeroValue: 1.0);

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.Equal(3, result.Size);
        Assert.All(result.Data, val => Assert.Equal(1.0f, val));
    }

    [Fact]
    public void Forward_MixedValues_CorrectBinaryOutput()
    {
        // Arrange
        var x = new Tensor(new[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f }, new[] { 5 }, true);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.Equal(5, result.Size);
        Assert.Equal(new[] { -1.0f, -1.0f, 0.0f, 1.0f, 1.0f }, result.Data);
    }

    [Fact]
    public void Forward_TwoDimensionalTensor_CorrectShape()
    {
        // Arrange
        var x = new Tensor(new[] { -1.0f, 2.0f, -3.0f, 4.0f }, new[] { 2, 2 }, true);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.Equal(new[] { 2, 2 }, result.Shape);
        Assert.Equal(new[] { -1.0f, 1.0f, -1.0f, 1.0f }, result.Data);
    }

    [Fact]
    public void Forward_ThreeDimensionalTensor_CorrectShape()
    {
        // Arrange
        var x = new Tensor(new[] { -1.0f, 2.0f, -3.0f, 4.0f, 0.0f, 6.0f }, new[] { 2, 1, 3 }, true);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.Equal(new[] { 2, 1, 3 }, result.Shape);
        Assert.Equal(new[] { -1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 1.0f }, result.Data);
    }

    [Fact]
    public void Forward_SingleElementTensor_ReturnsCorrectValue()
    {
        // Arrange
        var x = new Tensor(new[] { 5.0f }, new[] { 1 }, true);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.Equal(1, result.Size);
        Assert.Equal(1.0f, result.Data[0]);
    }

    [Fact]
    public void Forward_NaNValues_PropagatesNaN()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, float.NaN, 3.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.Equal(3, result.Size);
        Assert.Equal(1.0f, result.Data[0]);
        Assert.True(float.IsNaN(result.Data[1]));
        Assert.Equal(1.0f, result.Data[2]);
    }

    [Fact]
    public void Forward_PositiveInfinity_ReturnsOne()
    {
        // Arrange
        var x = new Tensor(new[] { float.PositiveInfinity, 0.0f, float.NegativeInfinity }, new[] { 3 }, true);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.Equal(3, result.Size);
        Assert.Equal(1.0f, result.Data[0]);
        Assert.Equal(0.0f, result.Data[1]);
        Assert.Equal(-1.0f, result.Data[2]);
    }

    [Fact]
    public void Forward_NullInput_ThrowsException()
    {
        // Arrange
        var steBinary = new STEBinary();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => steBinary.Apply(null!));
    }

    [Fact]
    public void Backward_UnityGradient_PassesThroughUnchanged()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, -1.0f, 0.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary();
        var result = steBinary.Apply(x);

        // Act - need to provide gradient for non-scalar tensor
        result.Backward(Tensor.Ones(result.Shape));

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(3, x.Gradient.Size);
        // Gradient should be 1.0 for all elements (identity backward)
        Assert.All(x.Gradient.Data, val => Assert.Equal(1.0f, val));
    }

    [Fact]
    public void Backward_GradientShapeMatchesInputShape()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, -1.0f, 0.0f, 2.0f }, new[] { 2, 2 }, true);
        var steBinary = new STEBinary();
        var result = steBinary.Apply(x);

        // Act - need to provide gradient for non-scalar tensor
        result.Backward(Tensor.Ones(result.Shape));

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(new[] { 2, 2 }, x.Gradient.Shape);
    }

    [Fact]
    public void Backward_MultiDimensionalTensor_GradientFlowsCorrectly()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, -1.0f, 0.0f, 2.0f, -3.0f, 4.0f }, new[] { 2, 3 }, true);
        var steBinary = new STEBinary();
        var result = steBinary.Apply(x);

        // Act - need to provide gradient for non-scalar tensor
        result.Backward(Tensor.Ones(result.Shape));

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(new[] { 2, 3 }, x.Gradient.Shape);
        Assert.Equal(6, x.Gradient.Size);
    }

    [Fact]
    public void Backward_NullGradient_ThrowsException()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary();
        var ctx = new FunctionContext();

        // Save the output manually
        var output = steBinary.Forward(new[] { x }, ctx);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => steBinary.Backward(null!, ctx));
    }

    [Fact]
    public void Integration_SimpleNetwork_WorksCorrectly()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, -1.0f, 0.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary();

        // Act
        var y = steBinary.Apply(x);
        var loss = y.Sum();
        loss.Backward();

        // Assert
        // Verify forward pass
        Assert.Equal(new[] { 1.0f, -1.0f, 0.0f }, y.Data);

        // Verify backward pass - gradient should flow
        Assert.NotNull(x.Gradient);
        Assert.Equal(3, x.Gradient.Size);
    }

    [Fact]
    public void Integration_AutogradGraph_WorksCorrectly()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, -1.0f, 0.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary();

        // Act - create binary output
        var y = steBinary.Apply(x);

        // Verify gradient is passed through STEBinary when we call backward with explicit gradient
        y.Backward(Tensor.Ones(y.Shape));

        // Assert
        // Verify gradient flows through the STEBinary function
        Assert.NotNull(x.Gradient);
        Assert.Equal(3, x.Gradient.Size);
        // In STE, gradient passes through unchanged (1.0 for all elements)
        Assert.All(x.Gradient.Data, val => Assert.Equal(1.0f, val));
    }

    [Fact]
    public void GradientChecking_STE_PassesGradient()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, -1.0f, 0.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary();
        var y = steBinary.Apply(x);

        // Act - provide gradient to backward for non-scalar tensor
        y.Backward(Tensor.Ones(y.Shape));

        // Assert
        // In STE, gradient should be identical to the upstream gradient
        // We passed gradient of 1.0 for each element
        Assert.NotNull(x.Gradient);
        Assert.All(x.Gradient.Data, val => Assert.Equal(1.0f, val, Tolerance));
    }

    [Fact]
    public void EdgeCase_AllZeros_WithDefaultZeroValue()
    {
        // Arrange
        var x = new Tensor(new[] { 0.0f, 0.0f, 0.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.All(result.Data, val => Assert.Equal(0.0f, val));
    }

    [Fact]
    public void EdgeCase_AllZeros_WithCustomZeroValue()
    {
        // Arrange
        var x = new Tensor(new[] { 0.0f, 0.0f, 0.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary(zeroValue: -1.0);

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.All(result.Data, val => Assert.Equal(-1.0f, val));
    }

    [Fact]
    public void EdgeCase_VerySmallPositiveValues_ReturnsOnes()
    {
        // Arrange
        var x = new Tensor(new[] { 0.0001f, 1e-10f, 1e-20f }, new[] { 3 }, true);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.All(result.Data, val => Assert.Equal(1.0f, val));
    }

    [Fact]
    public void EdgeCase_VerySmallNegativeValues_ReturnsMinusOnes()
    {
        // Arrange
        var x = new Tensor(new[] { -0.0001f, -1e-10f, -1e-20f }, new[] { 3 }, true);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.All(result.Data, val => Assert.Equal(-1.0f, val));
    }

    [Fact]
    public void Backward_GradientWithNaN_PropagatesNaN()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, -1.0f, 0.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary();
        var y = steBinary.Apply(x);

        // Manually set gradient to include NaN
        var gradWithNaN = new Tensor(new[] { 1.0f, float.NaN, 1.0f }, new[] { 3 }, false);

        // Act
        y.Backward(gradWithNaN);

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.True(float.IsNaN(x.Gradient.Data[1]));
    }

    [Fact]
    public void Consistency_MultipleCalls_SameInputSameOutput()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, -1.0f, 0.0f }, new[] { 3 }, true);
        var steBinary = new STEBinary();

        // Act
        var result1 = steBinary.Apply(x);
        var result2 = steBinary.Apply(x);

        // Assert
        Assert.Equal(result1.Size, result2.Size);
        Assert.Equal(result1.Shape, result2.Shape);
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
            data[i] = (i % 3) - 1.0f; // -1, 0, 1 pattern
        }
        var x = new Tensor(data, new[] { size }, true);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.Equal(size, result.Size);

        // Verify pattern: -1, 0, 1 repeated
        for (int i = 0; i < size; i++)
        {
            int mod = i % 3;
            float expected = mod == 0 ? -1.0f : (mod == 1 ? 0.0f : 1.0f);
            Assert.Equal(expected, result.Data[i]);
        }
    }

    [Fact]
    public void TrainingLoopContext_SimpleTrainingIteration_Works()
    {
        // Arrange
        var x = new Tensor(new[] { 0.5f, -0.5f, 0.0f }, new[] { 3 }, true);
        var target = new Tensor(new[] { 1.0f, -1.0f, 0.0f }, new[] { 3 }, false);
        var steBinary = new STEBinary();

        // Act
        // Forward pass
        var y = steBinary.Apply(x);

        // Simple MSE loss
        var diff = y.Subtract(target);
        var loss = diff.Multiply(diff).Sum();

        // Backward pass
        loss.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(3, x.Gradient.Size);

        // Verify forward pass produced binary values
        Assert.All(y.Data, val =>
        {
            Assert.True(val == 1.0f || val == -1.0f || val == 0.0f);
        });
    }

    [Fact]
    public void RequiresGrad_False_StillWorksCorrectly()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, -1.0f, 0.0f }, new[] { 3 }, false);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.Equal(new[] { 1.0f, -1.0f, 0.0f }, result.Data);
        Assert.False(result.RequiresGrad);
    }

    [Fact]
    public void DtypePreservation_Float32_MaintainsDtype()
    {
        // Arrange
        var x = new Tensor(new[] { 1.0f, -1.0f, 0.0f }, new[] { 3 }, false, DataType.Float32);
        var steBinary = new STEBinary();

        // Act
        var result = steBinary.Apply(x);

        // Assert
        Assert.Equal(DataType.Float32, result.Dtype);
    }
}
