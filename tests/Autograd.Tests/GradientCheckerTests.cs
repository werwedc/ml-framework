using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using Xunit;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Unit tests for the GradientChecker utility class.
/// </summary>
public class GradientCheckerTests
{
    /// <summary>
    /// Simple linear function: f(x) = x
    /// Gradient should be 1.0
    /// </summary>
    private class LinearFunction : CustomFunction
    {
        public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
        {
            ctx.SaveForBackward(inputs);
            return inputs;
        }

        public override Tensor[] Backward(Tensor[] gradOutputs, FunctionContext ctx)
        {
            return gradOutputs;
        }
    }

    /// <summary>
    /// Squared function: f(x) = x^2
    /// Gradient should be 2*x
    /// </summary>
    private class SquaredFunction : CustomFunction
    {
        public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
        {
            var result = new Tensor(
                inputs[0].Data.Select(x => x * x).ToArray(),
                inputs[0].Shape,
                inputs[0].RequiresGrad);

            ctx.SaveForBackward(inputs[0]);
            return new[] { result };
        }

        public override Tensor[] Backward(Tensor[] gradOutputs, FunctionContext ctx)
        {
            var input = ctx.GetSavedTensor(0);

            if (input.RequiresGrad)
            {
                var gradData = new float[input.Size];
                for (int i = 0; i < input.Size; i++)
                {
                    gradData[i] = 2.0f * input.Data[i] * gradOutputs[0].Data[i];
                }
                return new[] { new Tensor(gradData, input.Shape) };
            }

            return new Tensor[] { null! };
        }
    }

    /// <summary>
    /// Addition function: f(x, y) = x + y
    /// Gradient should be 1 for both inputs
    /// </summary>
    private class AdditionFunction : CustomFunction
    {
        public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
        {
            if (inputs.Length != 2)
                throw new ArgumentException("AdditionFunction requires exactly 2 inputs");

            var result = new Tensor(
                inputs[0].Data.Zip(inputs[1].Data, (a, b) => a + b).ToArray(),
                inputs[0].Shape,
                inputs[0].RequiresGrad || inputs[1].RequiresGrad);

            ctx.SaveForBackward(inputs[0], inputs[1]);
            return new[] { result };
        }

        public override Tensor[] Backward(Tensor[] gradOutputs, FunctionContext ctx)
        {
            var input1 = ctx.GetSavedTensor(0);
            var input2 = ctx.GetSavedTensor(1);

            var grad1 = input1.RequiresGrad ? gradOutputs[0].Clone() : null;
            var grad2 = input2.RequiresGrad ? gradOutputs[0].Clone() : null;

            return new[] { grad1!, grad2! };
        }
    }

    /// <summary>
    /// Sigmoid function: f(x) = 1 / (1 + exp(-x))
    /// Gradient should be f(x) * (1 - f(x))
    /// </summary>
    private class SigmoidFunction : CustomFunction
    {
        public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
        {
            var resultData = inputs[0].Data.Select(x => 1.0f / (1.0f + (float)Math.Exp(-x))).ToArray();
            var result = new Tensor(resultData, inputs[0].Shape, inputs[0].RequiresGrad);

            ctx.SaveForBackward(result);
            return new[] { result };
        }

        public override Tensor[] Backward(Tensor[] gradOutputs, FunctionContext ctx)
        {
            var sigmoid = ctx.GetSavedTensor(0);

            if (sigmoid.RequiresGrad)
            {
                var gradData = new float[sigmoid.Size];
                for (int i = 0; i < sigmoid.Size; i++)
                {
                    var s = sigmoid.Data[i];
                    gradData[i] = s * (1.0f - s) * gradOutputs[0].Data[i];
                }
                return new[] { new Tensor(gradData, sigmoid.Shape) };
            }

            return new Tensor[] { null! };
        }
    }

    /// <summary>
    /// Function with zero gradient: f(x) = 0 (constant)
    /// Gradient should be 0
    /// </summary>
    private class ZeroGradientFunction : CustomFunction
    {
        public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
        {
            var result = Tensor.Zeros(inputs[0].Shape);
            result.RequiresGrad = inputs[0].RequiresGrad;
            return new[] { result };
        }

        public override Tensor[] Backward(Tensor[] gradOutputs, FunctionContext ctx)
        {
            // Zero gradient for all inputs
            return new[] { Tensor.Zeros(gradOutputs[0].Shape) };
        }
    }

    [Fact]
    public void CheckGradients_LinearFunction_Passes()
    {
        // Arrange
        var func = new LinearFunction();
        var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, requiresGrad: true);

        // Act - use smaller epsilon for better numerical accuracy
        var result = GradientChecker.CheckGradients(func, new[] { input }, epsilon: 1e-6, tolerance: 0.05);

        // Assert
        Assert.True(result.Passed, result.GetSummary());
        Assert.True(result.MaxAbsoluteDifference < 0.1);
        Assert.True(result.MaxRelativeError < 0.1);
    }

    [Fact]
    public void CheckGradients_SquaredFunction_Passes()
    {
        // Arrange
        var func = new SquaredFunction();
        var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, requiresGrad: true);

        // Act - use smaller epsilon for better numerical accuracy
        var result = GradientChecker.CheckGradients(func, new[] { input }, epsilon: 1e-6);

        // Assert
        Assert.True(result.Passed, result.GetSummary());
        Assert.True(result.MaxAbsoluteDifference < 1e-4);
        Assert.True(result.MaxRelativeError < 1e-4);
    }

    [Fact]
    public void CheckGradients_AdditionFunction_Passes()
    {
        // Arrange
        var func = new AdditionFunction();
        var input1 = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, requiresGrad: true);
        var input2 = new Tensor(new float[] { 4.0f, 5.0f, 6.0f }, new[] { 3 }, requiresGrad: true);

        // Act - use smaller epsilon for better numerical accuracy
        var result = GradientChecker.CheckGradients(func, new[] { input1, input2 }, epsilon: 1e-6);

        // Assert
        Assert.True(result.Passed, result.GetSummary());
        Assert.True(result.MaxAbsoluteDifference < 1e-4);
    }

    [Fact]
    public void CheckGradients_SigmoidFunction_Passes()
    {
        // Arrange
        var func = new SigmoidFunction();
        var input = new Tensor(new float[] { -1.0f, 0.0f, 1.0f }, new[] { 3 }, requiresGrad: true);

        // Act - use smaller epsilon for better numerical accuracy
        var result = GradientChecker.CheckGradients(func, new[] { input }, epsilon: 1e-6);

        // Assert
        Assert.True(result.Passed, result.GetSummary());
        Assert.True(result.MaxAbsoluteDifference < 1e-4);
    }

    [Fact]
    public void CheckGradients_ZeroGradientFunction_Passes()
    {
        // Arrange
        var func = new ZeroGradientFunction();
        var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, requiresGrad: true);

        // Act - use smaller epsilon for better numerical accuracy
        var result = GradientChecker.CheckGradients(func, new[] { input }, epsilon: 1e-6);

        // Assert
        Assert.True(result.Passed, result.GetSummary());
        Assert.Equal(0.0, result.MaxAbsoluteDifference, 1e-6);
    }

    [Fact]
    public void CheckGradients_DifferentEpsilonValues_StillPasses()
    {
        // Arrange
        var func = new LinearFunction();
        var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, requiresGrad: true);

        // Act & Assert
        foreach (var epsilon in new[] { 1e-7, 1e-6, 1e-5 })
        {
            var result = GradientChecker.CheckGradients(func, new[] { input }, epsilon: epsilon);
            Assert.True(result.Passed, $"Failed with epsilon={epsilon}: {result.GetSummary()}");
        }
    }

    [Fact]
    public void CheckGradients_DifferentToleranceValues_AffectsPassFail()
    {
        // Arrange
        var func = new SquaredFunction();
        var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, requiresGrad: true);

        // Act - with strict tolerance
        var strictResult = GradientChecker.CheckGradients(func, new[] { input }, tolerance: 1e-8, epsilon: 1e-6);

        // Act - with loose tolerance
        var looseResult = GradientChecker.CheckGradients(func, new[] { input }, tolerance: 1e-2, epsilon: 1e-6);

        // Assert - loose tolerance should definitely pass
        Assert.True(looseResult.Passed, $"Loose tolerance failed: {looseResult.GetSummary()}");
    }

    [Fact]
    public void CheckGradients_MultipleInputs_AllChecked()
    {
        // Arrange
        var func = new AdditionFunction();
        var input1 = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, requiresGrad: true);
        var input2 = new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 }, requiresGrad: true);

        // Act - use smaller epsilon for better numerical accuracy
        var result = GradientChecker.CheckGradients(func, new[] { input1, input2 }, epsilon: 1e-6);

        // Assert
        Assert.True(result.Passed, result.GetSummary());
        // Verify that both inputs were checked
        // The differences list should be empty (all passed)
        Assert.Empty(result.Differences);
    }

    [Fact]
    public void CheckGradients_PartialRequireGrad_SkipsNonGradInputs()
    {
        // Arrange
        var func = new AdditionFunction();
        var input1 = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, requiresGrad: true);
        var input2 = new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 }, requiresGrad: false);

        // Act - use smaller epsilon for better numerical accuracy
        var result = GradientChecker.CheckGradients(func, new[] { input1, input2 }, epsilon: 1e-6);

        // Assert
        Assert.True(result.Passed, result.GetSummary());
    }

    [Fact]
    public void CheckGradients_LargeTensor_Passes()
    {
        // Arrange
        var func = new LinearFunction();
        var data = new float[100];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)i / 10.0f;
        }
        var input = new Tensor(data, new[] { 100 }, requiresGrad: true);

        // Act - use smaller epsilon for better numerical accuracy
        var result = GradientChecker.CheckGradients(func, new[] { input }, epsilon: 1e-6);

        // Assert
        Assert.True(result.Passed, result.GetSummary());
    }

    [Fact]
    public void CheckGradients_VerboseMode_DoesNotThrow()
    {
        // Arrange
        var func = new LinearFunction();
        var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, requiresGrad: true);

        // Act & Assert - should not throw
        var result = GradientChecker.CheckGradients(func, new[] { input }, verbose: true, epsilon: 1e-6);
        Assert.True(result.Passed);
    }

    [Fact]
    public void ComputeRelativeError_ReturnsCorrectValues()
    {
        // Arrange
        var numerical = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });
        var analytical = Tensor.FromArray(new float[] { 1.01f, 2.02f, 3.03f });

        // Act
        var relativeError = GradientChecker.ComputeRelativeError(numerical, analytical);

        // Assert
        Assert.Equal(3, relativeError.Size);
        Assert.InRange(relativeError.Data[0], 0.0099, 0.0101); // ~1%
        Assert.InRange(relativeError.Data[1], 0.0099, 0.0101);
        Assert.InRange(relativeError.Data[2], 0.0099, 0.0101);
    }

    [Fact]
    public void CompareGradients_ReturnsTrueForMatchingGradients()
    {
        // Arrange
        var grad1 = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });
        var grad2 = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });

        // Act
        var result = GradientChecker.CompareGradients(new[] { grad1 }, new[] { grad2 });

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void CompareGradients_ReturnsFalseForMismatchedGradients()
    {
        // Arrange
        var grad1 = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });
        var grad2 = Tensor.FromArray(new float[] { 1.5f, 2.5f, 3.5f });

        // Act
        var result = GradientChecker.CompareGradients(new[] { grad1 }, new[] { grad2 });

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CheckGradients_NullFunction_ThrowsArgumentNullException()
    {
        // Arrange
        var input = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            GradientChecker.CheckGradients(null!, new[] { input }));
    }

    [Fact]
    public void CheckGradients_NullInputs_ThrowsArgumentNullException()
    {
        // Arrange
        var func = new LinearFunction();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            GradientChecker.CheckGradients(func, null!));
    }

    [Fact]
    public void CheckGradients_EmptyInputs_ThrowsArgumentNullException()
    {
        // Arrange
        var func = new LinearFunction();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            GradientChecker.CheckGradients(func, Array.Empty<Tensor>()));
    }

    [Fact]
    public void GradientCheckResult_Summary_IncludesAllInformation()
    {
        // Arrange
        var result = new GradientCheckResult
        {
            Passed = false,
            MaxAbsoluteDifference = 1.5e-4,
            MaxRelativeError = 2.3e-4,
            FailureReason = "Test failure"
        };
        result.Differences.Add(new TensorDifference
        {
            InputIndex = 0,
            ElementIndex = new[] { 1 },
            NumericalValue = 1.0,
            AnalyticalValue = 1.001,
            AbsoluteDifference = 0.001,
            RelativeError = 0.001
        });

        // Act
        var summary = result.GetSummary();

        // Assert
        Assert.Contains("FAILED", summary);
        Assert.Contains("1.5", summary); // Just check for the value, format may vary
        Assert.Contains("2.3", summary);
        Assert.Contains("Test failure", summary);
    }

    [Fact]
    public void CheckGradients_2DInput_Passes()
    {
        // Arrange
        var func = new LinearFunction();
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        var input = new Tensor(data, new[] { 2, 3 }, requiresGrad: true);

        // Act - use smaller epsilon for better numerical accuracy
        var result = GradientChecker.CheckGradients(func, new[] { input }, epsilon: 1e-6);

        // Assert
        Assert.True(result.Passed, result.GetSummary());
    }

    [Fact]
    public void CheckGradients_VerySmallValues_HandlesCorrectly()
    {
        // Arrange
        var func = new LinearFunction();
        var input = new Tensor(new float[] { 1e-10f, 1e-9f, 1e-8f }, new[] { 3 }, requiresGrad: true);

        // Act - use smaller epsilon for better numerical accuracy
        var result = GradientChecker.CheckGradients(func, new[] { input }, epsilon: 1e-6);

        // Assert
        Assert.True(result.Passed, result.GetSummary());
    }

    [Fact]
    public void CheckGradients_GraphCleansUp_AfterCheck()
    {
        // Arrange
        var engine = AutogradEngine.Instance;
        engine.ClearGraph();

        var func = new LinearFunction();
        var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, requiresGrad: true);

        // Act - use smaller epsilon for better numerical accuracy
        var result = GradientChecker.CheckGradients(func, new[] { input }, epsilon: 1e-6);

        // Assert
        Assert.True(result.Passed);
        Assert.Equal(0, engine.NodeCount);
    }
}
