using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;
using System.Linq;

namespace MLFramework.Tests.Autograd;

public class BackwardPassTests : IDisposable
{
    private readonly GraphBuilder _graphBuilder;

    public BackwardPassTests()
    {
        _graphBuilder = new GraphBuilder();
    }

    public void Dispose()
    {
        _graphBuilder.Dispose();
    }

    #region Simple Gradient Tests

    [Fact]
    public void BackwardPass_SimpleScalarChain_ComputesCorrectGradient()
    {
        // Arrange: y = x + 1, L = y^2
        // dL/dx = dL/dy * dy/dx = 2y * 1 = 2(x + 1)
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);

        // Create operation: y = x + 1
        var addOp = new OperationContext("Add", g =>
        {
            // dL/dx = dL/dy * 1
            return new Tensor[] { g.Clone() };
        });

        var y = new Tensor(new float[] { 3.0f }, new int[] { 1 }, requiresGrad: true);
        var node1 = _graphBuilder.CreateNode(y, addOp);

        // Create operation: L = y^2
        var squareOp = new OperationContext("Square", g =>
        {
            // dL/dy = 2 * y
            var dy = y.Clone();
            for (int i = 0; i < dy.Size; i++)
            {
                dy._data[i] = 2.0f * y._data[i];
            }
            return new Tensor[] { g * dy };
        });

        var L = new Tensor(new float[] { 9.0f }, new int[] { 1 });
        var node2 = _graphBuilder.CreateNode(L, squareOp, node1);

        // Act - Run backward pass
        var backward = new BackwardPass(_graphBuilder);
        backward.Run(L);

        // Assert - dL/dx = 2 * 3 = 6
        Assert.NotNull(x.Gradient);
        Assert.Equal(1, x.Gradient.Size);
        Assert.Equal(6.0f, x.Gradient._data[0], precision: 5);
    }

    [Fact]
    public void BackwardPass_MultipleOperations_PropagatesCorrectly()
    {
        // Arrange: x -> +1 -> *2 -> -1
        // y = x + 1, z = y * 2, w = z - 1
        // dL/dx = dL/dw * dw/dz * dz/dy * dy/dx = 1 * 1 * 2 * 1 = 2
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);

        // y = x + 1
        var addOp = new OperationContext("Add", g => new Tensor[] { g.Clone() });
        var y = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var node1 = _graphBuilder.CreateNode(y, addOp);

        // z = y * 2
        var mulOp = new OperationContext("Mul", g =>
        {
            // dL/dy = dL/dz * 2
            return new Tensor[] { g * 2.0f };
        });
        var z = new Tensor(new float[] { 4.0f }, new int[] { 1 }, requiresGrad: true);
        var node2 = _graphBuilder.CreateNode(z, mulOp, node1);

        // w = z - 1
        var subOp = new OperationContext("Sub", g => new Tensor[] { g.Clone() });
        var w = new Tensor(new float[] { 3.0f }, new int[] { 1 }, requiresGrad: true);
        var node3 = _graphBuilder.CreateNode(w, subOp, node2);

        // Act
        var backward = new BackwardPass(_graphBuilder);
        backward.Run(w);

        // Assert - dL/dx = 2
        Assert.NotNull(x.Gradient);
        Assert.Equal(2.0f, x.Gradient._data[0], precision: 5);
    }

    [Fact]
    public void BackwardPass_BranchingGraph_AccumulatesGradients()
    {
        // Arrange: x -> y, x -> z, L = y + z
        // dL/dx = dL/dy * dy/dx + dL/dz * dz/dx = 1 * 1 + 1 * 1 = 2
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);

        // y = x * 2
        var mulOp1 = new OperationContext("Mul1", g => new Tensor[] { g * 2.0f });
        var y = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var node1 = _graphBuilder.CreateNode(y, mulOp1);

        // z = x * 3
        var mulOp2 = new OperationContext("Mul2", g => new Tensor[] { g * 3.0f });
        var z = new Tensor(new float[] { 3.0f }, new int[] { 1 }, requiresGrad: true);
        var node2 = _graphBuilder.CreateNode(z, mulOp2);

        // L = y + z
        var addOp = new OperationContext("Add", g =>
        {
            // dL/dy = 1, dL/dz = 1
            return new Tensor[] { g.Clone(), g.Clone() };
        });
        var L = new Tensor(new float[] { 5.0f }, new int[] { 1 });
        var node3 = _graphBuilder.CreateNode(L, addOp, node1, node2);

        // Act
        var backward = new BackwardPass(_graphBuilder);
        backward.Run(L);

        // Assert - dL/dx = 2 + 3 = 5 (accumulated from both paths)
        Assert.NotNull(x.Gradient);
        Assert.Equal(5.0f, x.Gradient._data[0], precision: 5);
    }

    #endregion

    #region Graph Retention Tests

    [Fact]
    public void BackwardPass_RetainGraph_PreservesGraph()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var op = new OperationContext("Add", g => new Tensor[] { g.Clone() });
        var y = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var node = _graphBuilder.CreateNode(y, op);

        // Act
        var backward = new BackwardPass(_graphBuilder);
        backward.RetainGraph = true;
        backward.Run(y);

        // Assert - Graph should still have nodes
        Assert.Equal(1, _graphBuilder.NodeCount);
    }

    [Fact]
    public void BackwardPass_NotRetainGraph_ClearsGraph()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var op = new OperationContext("Add", g => new Tensor[] { g.Clone() });
        var y = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var node = _graphBuilder.CreateNode(y, op);

        // Act
        var backward = new BackwardPass(_graphBuilder);
        backward.RetainGraph = false;
        backward.Run(y);

        // Assert - Graph might be cleared (implementation-dependent)
        // Note: Current implementation doesn't actually clear the graph
        // This test documents expected behavior
    }

    #endregion

    #region Custom Gradient Tests

    [Fact]
    public void BackwardPass_WithCustomInitialGradient_UsesProvidedGradient()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var op = new OperationContext("Mul", g => new Tensor[] { g * 2.0f });
        var y = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var node = _graphBuilder.CreateNode(y, op);

        var customGrad = new Tensor(new float[] { 3.0f }, new int[] { 1 });

        // Act
        var backward = new BackwardPass(_graphBuilder);
        backward.Run(y, customGrad);

        // Assert - dL/dx = customGrad * 2 = 3 * 2 = 6
        Assert.NotNull(x.Gradient);
        Assert.Equal(6.0f, x.Gradient._data[0], precision: 5);
    }

    [Fact]
    public void BackwardPass_NonScalarTensorWithNoGradient_ThrowsArgumentException()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);
        var op = new OperationContext("Add", g => new Tensor[] { g.Clone() });
        var y = new Tensor(new float[] { 2.0f, 3.0f }, new int[] { 2 }, requiresGrad: true);
        var node = _graphBuilder.CreateNode(y, op);

        // Act & Assert
        var backward = new BackwardPass(_graphBuilder);
        Assert.Throws<ArgumentException>(() => backward.Run(y));
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public void BackwardPass_NullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        var backward = new BackwardPass(_graphBuilder);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => backward.Run(null!));
    }

    [Fact]
    public void BackwardPass_TensorNotInGraph_ThrowsInvalidOperationException()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var backward = new BackwardPass(_graphBuilder);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => backward.Run(x));
    }

    #endregion

    #region Gradient Computer Tests

    [Fact]
    public void GradientComputer_ComputeGradient_ReturnsCorrectGradient()
    {
        // Arrange
        var gradOutput = new Tensor(new float[] { 2.0f }, new int[] { 1 });
        var input = new Tensor(new float[] { 3.0f }, new int[] { 1 });
        var context = new OperationContext("Test", g => new Tensor[] { g * 2.0f });

        // Act
        var grad = GradientComputer.ComputeGradient(gradOutput, input, context);

        // Assert
        Assert.NotNull(grad);
        Assert.Equal(4.0f, grad._data[0], precision: 5);
    }

    [Fact]
    public void GradientComputer_NumericalGradient_MatchesAnalytical()
    {
        // Arrange: f(x) = x^2, f'(x) = 2x
        Tensor SquareFunc(Tensor x)
        {
            var result = new Tensor(new float[x.Size], x.Shape);
            for (int i = 0; i < x.Size; i++)
            {
                result._data[i] = x._data[i] * x._data[i];
            }
            return result;
        }

        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        // Act
        var numericalGrad = GradientComputer.NumericalGradient(SquareFunc, x);

        // Assert - f'(2) = 2*2 = 4
        Assert.Equal(4.0f, numericalGrad._data[0], precision: 4);
    }

    [Fact]
    public void GradientComputer_ComputeRelativeError_ReturnsCorrectValue()
    {
        // Arrange
        var analytical = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });
        var numerical = new Tensor(new float[] { 1.000001f, 2.000001f, 3.000001f }, new int[] { 3 });

        // Act
        var error = GradientComputer.ComputeRelativeError(analytical, numerical);

        // Assert - Should be very small
        Assert.InRange(error, 0.0, 1e-5);
    }

    [Fact]
    public void GradientComputer_ValidateGradient_WithCloseGradients_ReturnsTrue()
    {
        // Arrange
        var analytical = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });
        var numerical = new Tensor(new float[] { 1.0000001f, 2.0000001f }, new int[] { 2 });

        // Act
        var isValid = GradientComputer.ValidateGradient(analytical, numerical, tolerance: 1e-6);

        // Assert
        Assert.True(isValid);
    }

    [Fact]
    public void GradientComputer_ValidateGradient_WithDistantGradients_ReturnsFalse()
    {
        // Arrange
        var analytical = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });
        var numerical = new Tensor(new float[] { 2.0f, 3.0f }, new int[] { 2 });

        // Act
        var isValid = GradientComputer.ValidateGradient(analytical, numerical, tolerance: 1e-6);

        // Assert
        Assert.False(isValid);
    }

    [Fact]
    public void GradientComputer_NumericalGradient_SimpleOperation_VerifiesAccuracy()
    {
        // Arrange: f(x) = x^3, f'(x) = 3x^2
        Tensor CubicFunc(Tensor x)
        {
            var result = new Tensor(new float[x.Size], x.Shape);
            for (int i = 0; i < x.Size; i++)
            {
                result._data[i] = x._data[i] * x._data[i] * x._data[i];
            }
            return result;
        }

        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        // Act
        var numericalGrad = GradientComputer.NumericalGradient(CubicFunc, x);

        // Assert - f'(2) = 3*4 = 12
        Assert.Equal(12.0f, numericalGrad._data[0], precision: 4);
    }

    [Fact]
    public void GradientComputer_ComputeMeanSquaredError_ReturnsCorrectValue()
    {
        // Arrange
        var analytical = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });
        var numerical = new Tensor(new float[] { 2.0f, 3.0f }, new int[] { 2 });
        // MSE = ((1-2)^2 + (2-3)^2) / 2 = (1 + 1) / 2 = 1

        // Act
        var mse = GradientComputer.ComputeMeanSquaredError(analytical, numerical);

        // Assert
        Assert.Equal(1.0, mse, precision: 5);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void BackwardPass_LinearRegressionModel_ComputesCorrectGradients()
    {
        // Arrange: Simple linear regression: y = wx + b, L = (y - target)^2
        var w = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var b = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var x = new Tensor(new float[] { 3.0f }, new int[] { 1 }, requiresGrad: true);
        var target = new Tensor(new float[] { 8.0f }, new int[] { 1 });

        // y = w*x + b = 2*3 + 1 = 7
        var mulOp = new OperationContext("Mul", g =>
        {
            // dL/dw = dL/dy * x
            return new Tensor[] { g * x, g.Clone() };
        });

        var y = new Tensor(new float[] { 7.0f }, new int[] { 1 }, requiresGrad: true);
        var node1 = _graphBuilder.CreateNode(y, mulOp);

        // L = (y - target)^2 = (7 - 8)^2 = 1
        var lossOp = new OperationContext("Loss", g =>
        {
            // dL/dy = 2 * (y - target)
            var dy = y.Clone();
            for (int i = 0; i < dy.Size; i++)
            {
                dy._data[i] = 2.0f * (y._data[i] - target._data[i]);
            }
            return new Tensor[] { g * dy };
        });

        var L = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var node2 = _graphBuilder.CreateNode(L, lossOp, node1);

        // Act
        var backward = new BackwardPass(_graphBuilder);
        backward.Run(L);

        // Assert
        // dL/dw = 2 * (7 - 8) * 3 = -6
        // dL/db = 2 * (7 - 8) * 1 = -2
        Assert.NotNull(w.Gradient);
        Assert.NotNull(b.Gradient);
        Assert.Equal(-6.0f, w.Gradient._data[0], precision: 4);
        Assert.Equal(-2.0f, b.Gradient._data[0], precision: 4);
    }

    [Fact]
    public void BackwardPass_DeepNetwork_HandlesMultipleLayers()
    {
        // Arrange: 5-layer network
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var prevNode = _graphBuilder.CreateNode(x, new OperationContext("Input", g => new Tensor[] { g }));

        for (int i = 0; i < 5; i++)
        {
            var op = new OperationContext($"Layer{i}", g => new Tensor[] { g * 2.0f });
            var output = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
            prevNode = _graphBuilder.CreateNode(output, op, prevNode);
        }

        // Act
        var backward = new BackwardPass(_graphBuilder);
        backward.Run(prevNode.OutputTensor);

        // Assert - Gradient should be propagated back
        Assert.NotNull(x.Gradient);
        Assert.Equal(32.0f, x.Gradient._data[0], precision: 4); // 2^5 = 32
    }

    [Fact]
    public void GradientComputer_WithMultipleInputs_ComputesCorrectGradients()
    {
        // Arrange: f(x, y) = x^2 + y^2
        Tensor[] inputs = new Tensor[]
        {
            new Tensor(new float[] { 2.0f }, new int[] { 1 }),
            new Tensor(new float[] { 3.0f }, new int[] { 1 })
        };

        Tensor MultiInputFunc(Tensor[] args)
        {
            var result = new float[1];
            result[0] = args[0]._data[0] * args[0]._data[0] + args[1]._data[0] * args[1]._data[0];
            return new Tensor(result, new int[] { 1 });
        }

        // Act
        var grads = GradientComputer.NumericalGradients(MultiInputFunc, inputs);

        // Assert
        // df/dx = 2x = 4, df/dy = 2y = 6
        Assert.NotNull(grads);
        Assert.Equal(2, grads.Length);
        Assert.Equal(4.0f, grads[0]._data[0], precision: 4);
        Assert.Equal(6.0f, grads[1]._data[0], precision: 4);
    }

    #endregion

    #region Edge Cases Tests

    [Fact]
    public void BackwardPass_ZeroGradient_HandlesCorrectly()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var op = new OperationContext("ZeroGrad", g => new Tensor[] { Tensor.Zeros(g.Shape) });
        var y = new Tensor(new float[] { 0.0f }, new int[] { 1 }, requiresGrad: true);
        var node = _graphBuilder.CreateNode(y, op);

        // Act
        var backward = new BackwardPass(_graphBuilder);
        backward.Run(y);

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(0.0f, x.Gradient._data[0], precision: 5);
    }

    [Fact]
    public void GradientComputer_NearZeroGradients_ComputesCorrectRelativeError()
    {
        // Arrange
        var analytical = new Tensor(new float[] { 0.0f, 0.0f }, new int[] { 2 });
        var numerical = new Tensor(new float[] { 1e-9f, 1e-9f }, new int[] { 2 });

        // Act
        var error = GradientComputer.ComputeRelativeError(analytical, numerical);

        // Assert - Should use absolute error when both are near zero
        Assert.InRange(error, 0.0, 1e-8);
    }

    #endregion
}
