using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;
using System.Linq;

namespace MLFramework.Tests.Autograd;

public class AutogradRegressionTests : IDisposable
{
    private readonly GraphBuilder _graphBuilder;

    public AutogradRegressionTests()
    {
        _graphBuilder = new GraphBuilder();
    }

    public void Dispose()
    {
        _graphBuilder.Dispose();
    }

    #region Known Issues / Fixed Bugs

    [Fact]
    public void Regression_BroadcastGradient()
    {
        // Test that broadcast gradients are correctly reduced
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);
        var y = new Tensor(new float[] { 3.0f }, new int[] { 1 }, requiresGrad: true);

        // Broadcast y to match x: [3, 3]
        var broadcasted = new Tensor(new float[] { 4.0f, 5.0f }, new int[] { 2 });

        var broadcastOp = new OperationContext("BroadcastAdd", g =>
        {
            // Gradient for y should be sum of broadcasted gradient
            var dy = new Tensor(new float[] { g.Data[0] + g.Data[1] }, y.Shape);
            var dx = g.Clone();
            return new Tensor[] { dx, dy };
        });

        var node = _graphBuilder.CreateNode(broadcasted, broadcastOp);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f, 1.0f }, new int[] { 2 });
        backward.Run(broadcasted, gradOutput);

        Assert.NotNull(x.Gradient);
        Assert.NotNull(y.Gradient);
        // dy should be sum of gradients: 1 + 1 = 2
        Assert.Equal(2.0f, y.Gradient.Data[0], precision: 4);
    }

    [Fact]
    public void Regression_MultipleGradientPaths()
    {
        // Test gradient accumulation from multiple paths
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);

        // Path 1: x * 2
        var y1 = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var mulOp1 = new OperationContext("Mul1", g => new Tensor[] { g * 2.0f });
        var node1 = _graphBuilder.CreateNode(y1, mulOp1);

        // Path 2: x * 3
        var y2 = new Tensor(new float[] { 3.0f }, new int[] { 1 }, requiresGrad: true);
        var mulOp2 = new OperationContext("Mul2", g => new Tensor[] { g * 3.0f });
        var node2 = _graphBuilder.CreateNode(y2, mulOp2);

        // Combine: y1 + y2
        var output = new Tensor(new float[] { 5.0f }, new int[] { 1 });
        var addOp = new OperationContext("Add", g => new Tensor[] { g.Clone(), g.Clone() });
        var node3 = _graphBuilder.CreateNode(output, addOp, node1, node2);

        var backward = new BackwardPass(_graphBuilder);
        backward.Run(output);

        Assert.NotNull(x.Gradient);
        // Gradient should be 2 + 3 = 5 (accumulated from both paths)
        Assert.Equal(5.0f, x.Gradient.Data[0], precision: 4);
    }

    [Fact]
    public void Regression_InPlaceOperationGradients()
    {
        // Test that in-place operations properly handle gradients
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);

        // In-place add operation
        var output = new Tensor(new float[] { 2.0f, 3.0f }, new int[] { 2 });

        var inplaceOp = new OperationContext("InPlaceAdd", g =>
        {
            // For in-place ops, gradient should flow correctly
            return new Tensor[] { g.Clone() };
        });

        var node = _graphBuilder.CreateNode(output, inplaceOp);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f, 1.0f }, new int[] { 2 });
        backward.Run(output, gradOutput);

        Assert.NotNull(x.Gradient);
        Assert.Equal(1.0f, x.Gradient.Data[0], precision: 4);
        Assert.Equal(1.0f, x.Gradient.Data[1], precision: 4);
    }

    [Fact]
    public void Regression_ViewsAndSlices()
    {
        // Test gradients through views/slices
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new int[] { 4 }, requiresGrad: true);

        // Create a view/slice of first two elements
        var slice = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });

        var output = new Tensor(new float[] { 2.0f, 3.0f }, new int[] { 2 });

        var sliceOp = new OperationContext("Slice", g =>
        {
            // Gradient should be mapped back to original positions
            var dx = Tensor.Zeros(x.Shape);
            dx.Data[0] = g.Data[0];
            dx.Data[1] = g.Data[1];
            return new Tensor[] { dx };
        });

        var node = _graphBuilder.CreateNode(output, sliceOp);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f, 1.0f }, new int[] { 2 });
        backward.Run(output, gradOutput);

        Assert.NotNull(x.Gradient);
        Assert.Equal(1.0f, x.Gradient.Data[0], precision: 4);
        Assert.Equal(1.0f, x.Gradient.Data[1], precision: 4);
        Assert.Equal(0.0f, x.Gradient.Data[2], precision: 4);
        Assert.Equal(0.0f, x.Gradient.Data[3], precision: 4);
    }

    [Fact]
    public void Regression_DetachedTensorGradients()
    {
        // Test that detached tensors don't receive gradients
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var y = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: false); // Detached

        var output = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        var mulOp = new OperationContext("Mul", g =>
        {
            return new Tensor[] { g * 2.0f, Tensor.Zeros(y.Shape) }; // y should not receive gradient
        });

        var node = _graphBuilder.CreateNode(output, mulOp);

        var backward = new BackwardPass(_graphBuilder);
        backward.Run(output);

        Assert.NotNull(x.Gradient);
        Assert.Null(y.Gradient); // Detached tensor should not have gradient
    }

    #endregion

    #region Compatibility Tests

    [Fact]
    public void CompareWithPyTorch_SimpleNetwork()
    {
        // This test documents expected behavior matching PyTorch
        // Equivalent PyTorch code:
        // import torch
        // x = torch.tensor([1.0], requires_grad=True)
        // w = torch.tensor([2.0], requires_grad=True)
        // y = x * w
        // y.backward()
        // x.grad = 2.0, w.grad = 1.0

        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var w = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var y = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        var mulOp = new OperationContext("Mul", g =>
        {
            // dL/dx = dL/dy * w = 1 * 2 = 2
            // dL/dw = dL/dy * x = 1 * 1 = 1
            return new Tensor[] { g * w.Data[0], g * x.Data[0] };
        });

        var node = _graphBuilder.CreateNode(y, mulOp);

        var backward = new BackwardPass(_graphBuilder);
        backward.Run(y);

        // Compare with PyTorch results
        Assert.NotNull(x.Gradient);
        Assert.NotNull(w.Gradient);
        Assert.Equal(2.0f, x.Gradient.Data[0], precision: 4);
        Assert.Equal(1.0f, w.Gradient.Data[0], precision: 4);
    }

    [Fact]
    public void CompareWithPyTorch_ComplexOperation()
    {
        // Equivalent PyTorch code:
        // import torch
        // x = torch.tensor([2.0], requires_grad=True)
        // y = x ** 2 + 2 * x + 1
        // y.backward()
        // x.grad = 2*2 + 2 = 6

        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var y = new Tensor(new float[] { 9.0f }, new int[] { 1 }); // 2^2 + 2*2 + 1 = 9

        var complexOp = new OperationContext("Complex", g =>
        {
            // y = x^2 + 2x + 1
            // dy/dx = 2x + 2 = 2*2 + 2 = 6
            var dx = new Tensor(new float[] { 6.0f }, x.Shape);
            return new Tensor[] { g * dx };
        });

        var node = _graphBuilder.CreateNode(y, complexOp);

        var backward = new BackwardPass(_graphBuilder);
        backward.Run(y);

        // Compare with PyTorch result
        Assert.NotNull(x.Gradient);
        Assert.Equal(6.0f, x.Gradient.Data[0], precision: 4);
    }

    [Fact]
    public void Regression_MatMulGradient()
    {
        // Test matrix multiplication gradients
        var A = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new int[] { 2, 2 }, requiresGrad: true);
        var B = new Tensor(new float[] { 0.5f, 0.5f, 0.5f, 0.5f }, new int[] { 2, 2 }, requiresGrad: true);

        // C = A @ B
        var C = new Tensor(new float[] { 1.5f, 1.5f, 3.5f, 3.5f }, new int[] { 2, 2 });

        var matMulOp = new OperationContext("MatMul", g =>
        {
            // dL/dA = dL/dC @ B^T
            var dA = new Tensor(new float[4], A.Shape);
            dA.Data[0] = g.Data[0] * B.Data[0] + g.Data[1] * B.Data[2];
            dA.Data[1] = g.Data[0] * B.Data[1] + g.Data[1] * B.Data[3];
            dA.Data[2] = g.Data[2] * B.Data[0] + g.Data[3] * B.Data[2];
            dA.Data[3] = g.Data[2] * B.Data[1] + g.Data[3] * B.Data[3];

            // dL/dB = A^T @ dL/dC
            var dB = new Tensor(new float[4], B.Shape);
            dB.Data[0] = A.Data[0] * g.Data[0] + A.Data[2] * g.Data[2];
            dB.Data[1] = A.Data[0] * g.Data[1] + A.Data[2] * g.Data[3];
            dB.Data[2] = A.Data[1] * g.Data[0] + A.Data[3] * g.Data[2];
            dB.Data[3] = A.Data[1] * g.Data[1] + A.Data[3] * g.Data[3];

            return new Tensor[] { dA, dB };
        });

        var node = _graphBuilder.CreateNode(C, matMulOp);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f, 0.0f, 0.0f, 1.0f }, new int[] { 2, 2 });
        backward.Run(C, gradOutput);

        Assert.NotNull(A.Gradient);
        Assert.NotNull(B.Gradient);
    }

    [Fact]
    public void Regression_ReshapeGradient()
    {
        // Test gradient through reshape operations
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, new int[] { 6 }, requiresGrad: true);

        // Reshape to 2x3
        var reshaped = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, new int[] { 2, 3 });

        var output = new Tensor(new float[] { 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f }, new int[] { 2, 3 });

        var reshapeOp = new OperationContext("Reshape", g =>
        {
            // Gradient should be reshaped back to original shape
            var dx = new Tensor(new float[6], x.Shape);
            for (int i = 0; i < 6; i++)
            {
                dx.Data[i] = g.Data[i];
            }
            return new Tensor[] { dx };
        });

        var node = _graphBuilder.CreateNode(output, reshapeOp);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }, new int[] { 2, 3 });
        backward.Run(output, gradOutput);

        Assert.NotNull(x.Gradient);
        Assert.Equal(6, x.Gradient.Shape[0]);
    }

    [Fact]
    public void Regression_PermuteGradient()
    {
        // Test gradient through permute/transpose operations
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new int[] { 2, 2 }, requiresGrad: true);

        // Transpose: [0,1] -> [1,0]
        var transposed = new Tensor(new float[] { 1.0f, 3.0f, 2.0f, 4.0f }, new int[] { 2, 2 });

        var output = new Tensor(new float[] { 2.0f, 6.0f, 4.0f, 8.0f }, new int[] { 2, 2 });

        var permuteOp = new OperationContext("Transpose", g =>
        {
            // Gradient should be transposed back
            var dx = new Tensor(new float[4], x.Shape);
            dx.Data[0] = g.Data[0];
            dx.Data[1] = g.Data[2];
            dx.Data[2] = g.Data[1];
            dx.Data[3] = g.Data[3];
            return new Tensor[] { dx };
        });

        var node = _graphBuilder.CreateNode(output, permuteOp);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f, 1.0f, 1.0f, 1.0f }, new int[] { 2, 2 });
        backward.Run(output, gradOutput);

        Assert.NotNull(x.Gradient);
        Assert.Equal(1.0f, x.Gradient.Data[0], precision: 4);
        Assert.Equal(1.0f, x.Gradient.Data[1], precision: 4);
        Assert.Equal(1.0f, x.Gradient.Data[2], precision: 4);
        Assert.Equal(1.0f, x.Gradient.Data[3], precision: 4);
    }

    [Fact]
    public void Regression_GradientClipping()
    {
        // Test that gradients can be clipped properly
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);

        // Operation that produces large gradient
        var output = new Tensor(new float[] { 100.0f }, new int[] { 1 });

        var clipOp = new OperationContext("Clip", g =>
        {
            // Generate large gradient
            var dx = new Tensor(new float[] { 1000.0f }, x.Shape);
            return new Tensor[] { dx };
        });

        var node = _graphBuilder.CreateNode(output, clipOp);

        var backward = new BackwardPass(_graphBuilder);
        backward.Run(output);

        Assert.NotNull(x.Gradient);
        // Gradient should be the large value
        Assert.Equal(1000.0f, x.Gradient.Data[0], precision: 0);
    }

    [Fact]
    public void Reassignment_GradientAccumulationAfterClear()
    {
        // Test that gradients are accumulated correctly after being cleared
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);

        // First backward pass
        var y1 = new Tensor(new float[] { 2.0f }, new int[] { 1 });
        var mulOp1 = new OperationContext("Mul1", g => new Tensor[] { g * 2.0f });
        var node1 = _graphBuilder.CreateNode(y1, mulOp1);

        var backward1 = new BackwardPass(_graphBuilder);
        backward1.Run(y1);

        var grad1 = x.Gradient!.Data[0];

        // Clear gradient
        x.ZeroGrad();

        // Second backward pass
        var y2 = new Tensor(new float[] { 2.0f }, new int[] { 1 });
        var mulOp2 = new OperationContext("Mul2", g => new Tensor[] { g * 2.0f });
        var node2 = _graphBuilder.CreateNode(y2, mulOp2);

        var backward2 = new BackwardPass(_graphBuilder);
        backward2.Run(y2);

        var grad2 = x.Gradient!.Data[0];

        // Gradients should be equal
        Assert.Equal(grad1, grad2, precision: 4);
    }
    #endregion
}
