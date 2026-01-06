using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using MLFramework.Autograd.Operations;

namespace MLFramework.Tests.Autograd.Operations;

/// <summary>
/// Tests for gradient computation of basic tensor operations.
/// </summary>
[TestFixture]
public class OperationGradTests
{
    private const float Tolerance = 1e-5f;

    /// <summary>
    /// Creates a test tensor with specified data.
    /// </summary>
    private static Tensor CreateTensor(float[] data, int[] shape, bool requiresGrad = true)
    {
        return new Tensor(data, shape, requiresGrad);
    }

    #region Addition Gradient Tests

    [Fact]
    public void TestAddGradient_SameShapes()
    {
        // Arrange
        var xData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var yData = new float[] { 5.0f, 6.0f, 7.0f, 8.0f };
        var gradOutputData = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };

        var x = CreateTensor(xData, new[] { 2, 2 });
        var y = CreateTensor(yData, new[] { 2, 2 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 2, 2 }, false);

        var addGrad = new AddGrad();
        var context = new OperationContext("Add", g => new Tensor[0]);

        // Act
        var gradients = addGrad.ComputeGrad(gradOutput, new[] { x, y }, context);

        // Assert
        Assert.AreEqual(2, gradients.Length);

        // Gradient for both inputs should be the same as gradOutput
        for (int i = 0; i < gradOutputData.Length; i++)
        {
            Assert.AreEqual(gradOutputData[i], gradients[0].Data[i], Tolerance, "gradX mismatch at index " + i);
            Assert.AreEqual(gradOutputData[i], gradients[1].Data[i], Tolerance, "gradY mismatch at index " + i);
        }
    }

    [Fact]
    public void TestAddGradient_WithBroadcasting()
    {
        // Arrange
        var xData = new float[] { 1.0f, 2.0f };  // Shape: [2]
        var yData = new float[] { 3.0f, 4.0f, 5.0f, 6.0f };  // Shape: [2, 2]
        var gradOutputData = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };

        var x = CreateTensor(xData, new[] { 2 });
        var y = CreateTensor(yData, new[] { 2, 2 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 2, 2 }, false);

        var addGrad = new AddGrad();
        var context = new OperationContext("Add", g => new Tensor[0]);

        // Act
        var gradients = addGrad.ComputeGrad(gradOutput, new[] { x, y }, context);

        // Assert
        // gradX should be summed over the broadcasted dimension
        Assert.AreEqual(2, gradients.Length);
        Assert.AreEqual(new[] { 2 }, gradients[0].Shape);
        Assert.AreEqual(new[] { 2, 2 }, gradients[1].Shape);
    }

    #endregion

    #region Multiplication Gradient Tests

    [Fact]
    public void TestMulGradient()
    {
        // Arrange
        var xData = new float[] { 2.0f, 3.0f };
        var yData = new float[] { 4.0f, 5.0f };
        var gradOutputData = new float[] { 1.0f, 1.0f };

        var x = CreateTensor(xData, new[] { 2 });
        var y = CreateTensor(yData, new[] { 2 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 2 }, false);

        var mulGrad = new MulGrad();
        var context = new OperationContext("Mul", g => new Tensor[0]);

        // Act
        var gradients = mulGrad.ComputeGrad(gradOutput, new[] { x, y }, context);

        // Assert
        // gradX = gradOutput * y = [4, 5]
        // gradY = gradOutput * x = [2, 3]
        Assert.AreEqual(2, gradients.Length);
        Assert.AreEqual(4.0f, gradients[0].Data[0], Tolerance);
        Assert.AreEqual(5.0f, gradients[0].Data[1], Tolerance);
        Assert.AreEqual(2.0f, gradients[1].Data[0], Tolerance);
        Assert.AreEqual(3.0f, gradients[1].Data[1], Tolerance);
    }

    #endregion

    #region Subtraction Gradient Tests

    [Fact]
    public void TestSubGradient()
    {
        // Arrange
        var xData = new float[] { 5.0f, 6.0f };
        var yData = new float[] { 1.0f, 2.0f };
        var gradOutputData = new float[] { 1.0f, 1.0f };

        var x = CreateTensor(xData, new[] { 2 });
        var y = CreateTensor(yData, new[] { 2 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 2 }, false);

        var subGrad = new SubGrad();
        var context = new OperationContext("Sub", g => new Tensor[0]);

        // Act
        var gradients = subGrad.ComputeGrad(gradOutput, new[] { x, y }, context);

        // Assert
        // gradX = gradOutput * 1 = [1, 1]
        // gradY = gradOutput * -1 = [-1, -1]
        Assert.AreEqual(2, gradients.Length);
        Assert.AreEqual(1.0f, gradients[0].Data[0], Tolerance);
        Assert.AreEqual(1.0f, gradients[0].Data[1], Tolerance);
        Assert.AreEqual(-1.0f, gradients[1].Data[0], Tolerance);
        Assert.AreEqual(-1.0f, gradients[1].Data[1], Tolerance);
    }

    #endregion

    #region Division Gradient Tests

    [Fact]
    public void TestDivGradient()
    {
        // Arrange
        var xData = new float[] { 4.0f, 9.0f };
        var yData = new float[] { 2.0f, 3.0f };
        var gradOutputData = new float[] { 1.0f, 1.0f };

        var x = CreateTensor(xData, new[] { 2 });
        var y = CreateTensor(yData, new[] { 2 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 2 }, false);

        var divGrad = new DivGrad();
        var context = new OperationContext("Div", g => new Tensor[0]);

        // Act
        var gradients = divGrad.ComputeGrad(gradOutput, new[] { x, y }, context);

        // Assert
        // gradX = gradOutput * (1/y) = [1/2, 1/3] = [0.5, 0.333...]
        // gradY = gradOutput * (-x/y²) = [-4/4, -9/9] = [-1, -1]
        Assert.AreEqual(2, gradients.Length);
        Assert.AreEqual(0.5f, gradients[0].Data[0], Tolerance);
        Assert.AreEqual(1.0f / 3.0f, gradients[0].Data[1], Tolerance);
        Assert.AreEqual(-1.0f, gradients[1].Data[0], Tolerance);
        Assert.AreEqual(-1.0f, gradients[1].Data[1], Tolerance);
    }

    #endregion

    #region Power Gradient Tests

    [Fact]
    public void TestPowGradient_Square()
    {
        // Arrange
        var xData = new float[] { 2.0f, 3.0f };
        var nData = new float[] { 2.0f };  // Power is 2
        var gradOutputData = new float[] { 1.0f, 1.0f };

        var x = CreateTensor(xData, new[] { 2 });
        var n = CreateTensor(nData, new[] { 1 }, false);
        var gradOutput = CreateTensor(gradOutputData, new[] { 2 }, false);

        var powGrad = new PowGrad();
        var context = new OperationContext("Pow", g => new Tensor[0]);

        // Act
        var gradients = powGrad.ComputeGrad(gradOutput, new[] { x, n }, context);

        // Assert
        // d(x²)/dx = 2*x = [4, 6]
        Assert.AreEqual(2, gradients.Length);
        Assert.AreEqual(4.0f, gradients[0].Data[0], Tolerance);
        Assert.AreEqual(6.0f, gradients[0].Data[1], Tolerance);
    }

    #endregion

    #region Activation Function Gradient Tests

    [Fact]
    public void TestReluGradient_Positive()
    {
        // Arrange
        var xData = new float[] { 1.0f, 2.0f, -1.0f, 0.0f };
        var gradOutputData = new float[] { 1.0f, 1.0f, 1.0f, 1.0f };

        var x = CreateTensor(xData, new[] { 4 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 4 }, false);

        var reluGrad = new ReluGrad();
        var context = new OperationContext("Relu", g => new Tensor[0]);

        // Act
        var gradients = reluGrad.ComputeGrad(gradOutput, new[] { x }, context);

        // Assert
        // grad = [1, 1, 0, 0] (1 where x > 0, else 0)
        Assert.AreEqual(1.0f, gradients[0].Data[0], Tolerance);  // x = 1
        Assert.AreEqual(1.0f, gradients[0].Data[1], Tolerance);  // x = 2
        Assert.AreEqual(0.0f, gradients[0].Data[2], Tolerance);  // x = -1
        Assert.AreEqual(0.0f, gradients[0].Data[3], Tolerance);  // x = 0
    }

    [Fact]
    public void TestSigmoidGradient()
    {
        // Arrange
        var xData = new float[] { 0.0f };
        var gradOutputData = new float[] { 1.0f };

        var x = CreateTensor(xData, new[] { 1 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 1 }, false);

        // Save sigmoid output in context
        var sigmoidX = new float[] { 0.5f };  // sigmoid(0) = 0.5
        var sigmoidOutput = CreateTensor(sigmoidX, new[] { 1 }, false);
        var context = new OperationContext("Sigmoid", g => new Tensor[0]);
        context.SaveTensor("output", sigmoidOutput);

        var sigmoidGrad = new SigmoidGrad();

        // Act
        var gradients = sigmoidGrad.ComputeGrad(gradOutput, new[] { x }, context);

        // Assert
        // d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x)) = 0.5 * 0.5 = 0.25
        Assert.AreEqual(1, gradients.Length);
        Assert.AreEqual(0.25f, gradients[0].Data[0], Tolerance);
    }

    [Fact]
    public void TestTanhGradient()
    {
        // Arrange
        var xData = new float[] { 0.0f };
        var gradOutputData = new float[] { 1.0f };

        var x = CreateTensor(xData, new[] { 1 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 1 }, false);

        // Save tanh output in context
        var tanhX = new float[] { 0.0f };  // tanh(0) = 0
        var tanhOutput = CreateTensor(tanhX, new[] { 1 }, false);
        var context = new OperationContext("Tanh", g => new Tensor[0]);
        context.SaveTensor("output", tanhOutput);

        var tanhGrad = new TanhGrad();

        // Act
        var gradients = tanhGrad.ComputeGrad(gradOutput, new[] { x }, context);

        // Assert
        // d(tanh(x))/dx = 1 - tanh²(x) = 1 - 0 = 1
        Assert.AreEqual(1, gradients.Length);
        Assert.AreEqual(1.0f, gradients[0].Data[0], Tolerance);
    }

    #endregion

    #region Math Function Gradient Tests

    [Fact]
    public void TestExpGradient()
    {
        // Arrange
        var xData = new float[] { 0.0f, 1.0f };
        var gradOutputData = new float[] { 1.0f, 1.0f };

        var x = CreateTensor(xData, new[] { 2 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 2 }, false);

        // Save exp output in context
        var expX = new float[] { 1.0f, (float)Math.E };  // exp(0) = 1, exp(1) = e
        var expOutput = CreateTensor(expX, new[] { 2 }, false);
        var context = new OperationContext("Exp", g => new Tensor[0]);
        context.SaveTensor("output", expOutput);

        var expGrad = new ExpGrad();

        // Act
        var gradients = expGrad.ComputeGrad(gradOutput, new[] { x }, context);

        // Assert
        // d(exp(x))/dx = exp(x)
        Assert.AreEqual(1, gradients.Length);
        Assert.AreEqual(1.0f, gradients[0].Data[0], Tolerance);
        Assert.AreEqual((float)Math.E, gradients[0].Data[1], Tolerance);
    }

    [Fact]
    public void TestLogGradient()
    {
        // Arrange
        var xData = new float[] { 1.0f, 2.0f };
        var gradOutputData = new float[] { 1.0f, 1.0f };

        var x = CreateTensor(xData, new[] { 2 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 2 }, false);

        var logGrad = new LogGrad();
        var context = new OperationContext("Log", g => new Tensor[0]);

        // Act
        var gradients = logGrad.ComputeGrad(gradOutput, new[] { x }, context);

        // Assert
        // d(log(x))/dx = 1/x = [1, 0.5]
        Assert.AreEqual(1, gradients.Length);
        Assert.AreEqual(1.0f, gradients[0].Data[0], Tolerance);
        Assert.AreEqual(0.5f, gradients[0].Data[1], Tolerance);
    }

    [Fact]
    public void TestSqrtGradient()
    {
        // Arrange
        var xData = new float[] { 1.0f, 4.0f };
        var gradOutputData = new float[] { 1.0f, 1.0f };

        var x = CreateTensor(xData, new[] { 2 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 2 }, false);

        var sqrtGrad = new SqrtGrad();
        var context = new OperationContext("Sqrt", g => new Tensor[0]);

        // Act
        var gradients = sqrtGrad.ComputeGrad(gradOutput, new[] { x }, context);

        // Assert
        // d(sqrt(x))/dx = 1/(2*sqrt(x)) = [1/(2*1), 1/(2*2)] = [0.5, 0.25]
        Assert.AreEqual(1, gradients.Length);
        Assert.AreEqual(0.5f, gradients[0].Data[0], Tolerance);
        Assert.AreEqual(0.25f, gradients[0].Data[1], Tolerance);
    }

    #endregion

    #region Reduction Operation Gradient Tests

    [Fact]
    public void TestSumGradient()
    {
        // Arrange
        var xData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var gradOutputData = new float[] { 1.0f };  // Scalar

        var x = CreateTensor(xData, new[] { 2, 2 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 1 }, false);

        var sumGrad = new SumGrad();
        var context = new OperationContext("Sum", g => new Tensor[0]);

        // Act
        var gradients = sumGrad.ComputeGrad(gradOutput, new[] { x }, context);

        // Assert
        // All gradients should be 1 (broadcasted from gradOutput)
        Assert.AreEqual(1, gradients.Length);
        Assert.AreEqual(new[] { 2, 2 }, gradients[0].Shape);
        for (int i = 0; i < 4; i++)
        {
            Assert.AreEqual(1.0f, gradients[0].Data[i], Tolerance);
        }
    }

    [Fact]
    public void TestMeanGradient()
    {
        // Arrange
        var xData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var gradOutputData = new float[] { 1.0f };  // Scalar

        var x = CreateTensor(xData, new[] { 2, 2 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 1 }, false);

        var meanGrad = new MeanGrad();
        var context = new OperationContext("Mean", g => new Tensor[0]);

        // Act
        var gradients = meanGrad.ComputeGrad(gradOutput, new[] { x }, context);

        // Assert
        // All gradients should be 1/4 = 0.25 (broadcasted from gradOutput and scaled)
        Assert.AreEqual(1, gradients.Length);
        Assert.AreEqual(new[] { 2, 2 }, gradients[0].Shape);
        for (int i = 0; i < 4; i++)
        {
            Assert.AreEqual(0.25f, gradients[0].Data[i], Tolerance);
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void TestMulGradient_ZeroValues()
    {
        // Arrange
        var xData = new float[] { 0.0f, 1.0f };
        var yData = new float[] { 1.0f, 0.0f };
        var gradOutputData = new float[] { 1.0f, 1.0f };

        var x = CreateTensor(xData, new[] { 2 });
        var y = CreateTensor(yData, new[] { 2 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 2 }, false);

        var mulGrad = new MulGrad();
        var context = new OperationContext("Mul", g => new Tensor[0]);

        // Act
        var gradients = mulGrad.ComputeGrad(gradOutput, new[] { x, y }, context);

        // Assert
        // gradX = gradOutput * y = [1, 0]
        // gradY = gradOutput * x = [0, 1]
        Assert.AreEqual(0.0f, gradients[0].Data[0], Tolerance);
        Assert.AreEqual(0.0f, gradients[1].Data[1], Tolerance);
    }

    [Fact]
    public void TestLogGradient_NegativeInput()
    {
        // Arrange
        var xData = new float[] { -1.0f };
        var gradOutputData = new float[] { 1.0f };

        var x = CreateTensor(xData, new[] { 1 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 1 }, false);

        var logGrad = new LogGrad();
        var context = new OperationContext("Log", g => new Tensor[0]);

        // Act
        var gradients = logGrad.ComputeGrad(gradOutput, new[] { x }, context);

        // Assert
        // Gradient for log(-1) should be NaN
        Assert.AreEqual(1, gradients.Length);
        Assert.IsTrue(float.IsNaN(gradients[0].Data[0]));
    }

    [Fact]
    public void TestSqrtGradient_NegativeInput()
    {
        // Arrange
        var xData = new float[] { -1.0f };
        var gradOutputData = new float[] { 1.0f };

        var x = CreateTensor(xData, new[] { 1 });
        var gradOutput = CreateTensor(gradOutputData, new[] { 1 }, false);

        var sqrtGrad = new SqrtGrad();
        var context = new OperationContext("Sqrt", g => new Tensor[0]);

        // Act
        var gradients = sqrtGrad.ComputeGrad(gradOutput, new[] { x }, context);

        // Assert
        // Gradient for sqrt(-1) should be NaN
        Assert.AreEqual(1, gradients.Length);
        Assert.IsTrue(float.IsNaN(gradients[0].Data[0]));
    }

    #endregion

    #region OperationName Tests

    [Fact]
    public void TestOperationNames()
    {
        // Arrange
        var operations = new IOperationGrad[]
        {
            new AddGrad(),
            new MulGrad(),
            new SubGrad(),
            new DivGrad(),
            new PowGrad(),
            new ExpGrad(),
            new LogGrad(),
            new SqrtGrad(),
            new ReluGrad(),
            new SigmoidGrad(),
            new TanhGrad(),
            new SumGrad(),
            new MeanGrad()
        };

        var expectedNames = new[]
        {
            "Add", "Mul", "Sub", "Div", "Pow",
            "Exp", "Log", "Sqrt", "Relu", "Sigmoid", "Tanh",
            "Sum", "Mean"
        };

        // Act & Assert
        for (int i = 0; i < operations.Length; i++)
        {
            Assert.AreEqual(expectedNames[i], operations[i].OperationName);
        }
    }

    #endregion
}
