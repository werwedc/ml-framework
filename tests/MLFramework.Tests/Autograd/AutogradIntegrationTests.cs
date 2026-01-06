using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;
using System.Linq;

namespace MLFramework.Tests.Autograd;

public class AutogradIntegrationTests : IDisposable
{
    private readonly GraphBuilder _graphBuilder;

    public AutogradIntegrationTests()
    {
        _graphBuilder = new GraphBuilder();
    }

    public void Dispose()
    {
        _graphBuilder.Dispose();
    }

    #region Neural Network Components

    [Fact]
    public void TestLinearLayerGradients()
    {
        // Simple linear layer: y = Wx + b
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);
        var W = new Tensor(new float[] { 0.5f, -0.5f, 0.3f, 0.7f }, new int[] { 2, 2 }, requiresGrad: true);
        var b = new Tensor(new float[] { 0.1f, -0.1f }, new int[] { 2 }, requiresGrad: true);

        // Manual forward pass
        var y = new Tensor(new float[] { 1.4f, 1.7f }, new int[] { 2 });

        // Create operation for backward pass
        var linearOp = new OperationContext("Linear", g =>
        {
            // dL/dW = g * x^T (simplified for this test)
            var dW = new Tensor(new float[4], W.Shape);
            dW.Data[0] = g.Data[0] * x.Data[0];
            dW.Data[1] = g.Data[1] * x.Data[0];
            dW.Data[2] = g.Data[0] * x.Data[1];
            dW.Data[3] = g.Data[1] * x.Data[1];

            // dL/db = g
            var db = g.Clone();

            // dL/dx = W^T * g (simplified)
            var dx = new Tensor(new float[2], x.Shape);
            dx.Data[0] = W.Data[0] * g.Data[0] + W.Data[2] * g.Data[1];
            dx.Data[1] = W.Data[1] * g.Data[0] + W.Data[3] * g.Data[1];

            return new Tensor[] { dW, db, dx };
        });

        var node = _graphBuilder.CreateNode(y, linearOp);

        // Run backward
        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f, 1.0f }, new int[] { 2 });
        backward.Run(y, gradOutput);

        // Verify gradients exist
        Assert.NotNull(W.Gradient);
        Assert.NotNull(b.Gradient);
        Assert.NotNull(x.Gradient);
    }

    [Fact]
    public void TestConv2DLayerGradients()
    {
        // Simple convolution test (simplified for unit testing)
        var input = new Tensor(new float[4], new int[] { 1, 2, 2 }, requiresGrad: true); // 1x2x2 input
        var kernel = new Tensor(new float[4], new int[] { 1, 2, 2 }, requiresGrad: true); // 1x2x2 kernel

        // Output shape would be 1x1x1 for this configuration
        var output = new Tensor(new float[] { 0.0f }, new int[] { 1 });

        var convOp = new OperationContext("Conv2D", g =>
        {
            // Simplified gradient computation for conv2d
            var dInput = Tensor.Zeros(input.Shape);
            var dKernel = Tensor.Zeros(kernel.Shape);

            // In real conv2d, this would involve proper convolution of gradient with input/kernel
            for (int i = 0; i < input.Size; i++)
            {
                dInput.Data[i] = g.Data[0] * kernel.Data[i];
                dKernel.Data[i] = g.Data[0] * input.Data[i];
            }

            return new Tensor[] { dInput, dKernel };
        });

        var node = _graphBuilder.CreateNode(output, convOp);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        backward.Run(output, gradOutput);

        Assert.NotNull(input.Gradient);
        Assert.NotNull(kernel.Gradient);
    }

    [Fact]
    public void TestMaxPoolGradients()
    {
        // 2x2 input with 2x2 max pool -> 1x1 output
        var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new int[] { 2, 2 }, requiresGrad: true);
        var output = new Tensor(new float[] { 4.0f }, new int[] { 1 });

        var maxPoolOp = new OperationContext("MaxPool2D", g =>
        {
            // Gradient only flows through the maximum element (index 3)
            var dInput = Tensor.Zeros(input.Shape);
            dInput.Data[3] = g.Data[0];
            return new Tensor[] { dInput };
        });

        var node = _graphBuilder.CreateNode(output, maxPoolOp);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        backward.Run(output, gradOutput);

        Assert.NotNull(input.Gradient);
        Assert.Equal(0.0f, input.Gradient.Data[0]);
        Assert.Equal(0.0f, input.Gradient.Data[1]);
        Assert.Equal(0.0f, input.Gradient.Data[2]);
        Assert.Equal(1.0f, input.Gradient.Data[3]);
    }

    [Fact]
    public void TestBatchNormGradients()
    {
        // Simplified batch normalization
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new int[] { 4 }, requiresGrad: true);
        var gamma = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var beta = new Tensor(new float[] { 0.0f }, new int[] { 1 }, requiresGrad: true);

        // Normalized output (mean=2.5, variance=1.25)
        var output = new Tensor(new float[] { -1.3416f, -0.4472f, 0.4472f, 1.3416f }, new int[] { 4 });

        var batchNormOp = new OperationContext("BatchNorm", g =>
        {
            // Simplified gradient computation
            var dx = Tensor.Zeros(x.Shape);
            var dGamma = Tensor.Zeros(gamma.Shape);
            var dBeta = Tensor.Zeros(beta.Shape);

            // dL/dbeta = sum(g)
            dBeta.Data[0] = g.Data.Sum();

            // dL/dgamma = sum(g * normalized)
            for (int i = 0; i < x.Size; i++)
            {
                dGamma.Data[0] += g.Data[i] * output.Data[i];
            }

            // dL/dx = simplified (actual batch norm gradient is more complex)
            for (int i = 0; i < x.Size; i++)
            {
                dx.Data[i] = gamma.Data[0] * g.Data[i] / 4.0f; // Simplified
            }

            return new Tensor[] { dx, dGamma, dBeta };
        });

        var node = _graphBuilder.CreateNode(output, batchNormOp);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f, 1.0f, 1.0f, 1.0f }, new int[] { 4 });
        backward.Run(output, gradOutput);

        Assert.NotNull(x.Gradient);
        Assert.NotNull(gamma.Gradient);
        Assert.NotNull(beta.Gradient);
    }

    #endregion

    #region Training Scenarios

    [Fact]
    public void TestSimpleRegressionTraining()
    {
        // Simple linear regression: y = wx + b, L = (y - target)^2
        var w = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var b = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var x = new Tensor(new float[] { 3.0f }, new int[] { 1 }, requiresGrad: false);
        var target = new Tensor(new float[] { 8.0f }, new int[] { 1 });

        // y = wx + b = 2*3 + 1 = 7
        var y = new Tensor(new float[] { 7.0f }, new int[] { 1 });

        var lossOp = new OperationContext("MSELoss", g =>
        {
            // dL/dy = 2 * (y - target)
            var dy = y.Clone();
            for (int i = 0; i < dy.Size; i++)
            {
                dy.Data[i] = 2.0f * (y.Data[i] - target.Data[i]);
            }
            return new Tensor[] { g * dy };
        });

        var L = new Tensor(new float[] { 1.0f }, new int[] { 1 }); // (7-8)^2 = 1
        var node1 = _graphBuilder.CreateNode(y, new OperationContext("Linear", g => new Tensor[] { }));
        var node2 = _graphBuilder.CreateNode(L, lossOp, node1);

        var backward = new BackwardPass(_graphBuilder);
        backward.Run(L);

        // Verify gradients
        Assert.NotNull(w.Gradient);
        Assert.NotNull(b.Gradient);
    }

    [Fact]
    public void TestBinaryClassificationTraining()
    {
        // Binary classification with sigmoid
        var w = new Tensor(new float[] { 0.5f }, new int[] { 1 }, requiresGrad: true);
        var b = new Tensor(new float[] { 0.0f }, new int[] { 1 }, requiresGrad: true);
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 });
        var target = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        // Logits and sigmoid
        var logits = new Tensor(new float[] { 1.0f }, new int[] { 1 }); // 0.5*2 + 0 = 1
        var prob = new Tensor(new float[] { 0.7311f }, new int[] { 1 }); // sigmoid(1)

        // Binary cross-entropy loss: -[target*log(p) + (1-target)*log(1-p)]
        var loss = new Tensor(new float[] { 0.3133f }, new int[] { 1 });

        var bceOp = new OperationContext("BCELoss", g =>
        {
            // dL/dp = -(target/p - (1-target)/(1-p))
            var dp = prob.Clone();
            dp.Data[0] = -(target.Data[0] / prob.Data[0] - (1 - target.Data[0]) / (1 - prob.Data[0]));
            return new Tensor[] { g * dp };
        });

        var node = _graphBuilder.CreateNode(loss, bceOp);

        var backward = new BackwardPass(_graphBuilder);
        backward.Run(loss);

        Assert.NotNull(w.Gradient);
        Assert.NotNull(b.Gradient);
    }

    [Fact]
    public void TestMultiClassClassificationTraining()
    {
        // Multi-class with softmax and cross-entropy
        var W = new Tensor(new float[] { 0.5f, 0.5f, 0.5f }, new int[] { 3 }, requiresGrad: true);
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 });
        var target = new Tensor(new float[] { 0, 1, 0 }, new int[] { 3 });

        // Logits and softmax
        var logits = new Tensor(new float[] { 1.0f, 1.0f, 1.0f }, new int[] { 3 });
        var softmax = new Tensor(new float[] { 0.333f, 0.333f, 0.333f }, new int[] { 3 });

        // Cross-entropy loss: -sum(target * log(softmax))
        var loss = new Tensor(new float[] { 1.099f }, new int[] { 1 });

        var ceOp = new OperationContext("CrossEntropyLoss", g =>
        {
            // dL/dlogits = softmax - target
            var dLogits = new Tensor(new float[3], logits.Shape);
            for (int i = 0; i < 3; i++)
            {
                dLogits.Data[i] = softmax.Data[i] - target.Data[i];
            }
            return new Tensor[] { g * dLogits };
        });

        var node = _graphBuilder.CreateNode(loss, ceOp);

        var backward = new BackwardPass(_graphBuilder);
        backward.Run(loss);

        Assert.NotNull(W.Gradient);
    }

    [Fact]
    public void TestResidualNetworkGradients()
    {
        // Simple residual block: output = input + conv(input)
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new int[] { 4 }, requiresGrad: true);
        var W = new Tensor(new float[] { 0.1f, 0.1f, 0.1f, 0.1f }, new int[] { 4 }, requiresGrad: true);

        // Forward: conv = W*x, output = x + conv
        var conv = new Tensor(new float[] { 0.4f, 0.8f, 1.2f, 1.6f }, new int[] { 4 });
        var output = new Tensor(new float[] { 1.4f, 2.8f, 4.2f, 5.6f }, new int[] { 4 });

        var residualOp = new OperationContext("Residual", g =>
        {
            // dL/dx = g + g*W^T (simplified)
            var dx = new Tensor(new float[4], x.Shape);
            for (int i = 0; i < 4; i++)
            {
                dx.Data[i] = g.Data[i] + g.Data[i] * W.Data[i];
            }

            // dL/dW = g*x
            var dW = new Tensor(new float[4], W.Shape);
            for (int i = 0; i < 4; i++)
            {
                dW.Data[i] = g.Data[i] * x.Data[i];
            }

            return new Tensor[] { dx, dW };
        });

        var node = _graphBuilder.CreateNode(output, residualOp);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f, 1.0f, 1.0f, 1.0f }, new int[] { 4 });
        backward.Run(output, gradOutput);

        Assert.NotNull(x.Gradient);
        Assert.NotNull(W.Gradient);
    }

    [Fact]
    public void TestLSTMGradients()
    {
        // Simplified LSTM cell
        var input = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);
        var hidden = new Tensor(new float[] { 0.5f, 0.5f }, new int[] { 2 }, requiresGrad: true);
        var Wih = new Tensor(new float[] { 0.1f, 0.1f, 0.1f, 0.1f }, new int[] { 2, 2 }, requiresGrad: true);
        var Whh = new Tensor(new float[] { 0.1f, 0.1f, 0.1f, 0.1f }, new int[] { 2, 2 }, requiresGrad: true);

        // LSTM cell output (simplified)
        var output = new Tensor(new float[] { 0.6f, 0.7f }, new int[] { 2 });

        var lstmOp = new OperationContext("LSTMCell", g =>
        {
            // Simplified LSTM gradients
            var dInput = g.Clone();
            var dHidden = g.Clone();
            var dWih = Tensor.Zeros(Wih.Shape);
            var dWhh = Tensor.Zeros(Whh.Shape);

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    dWih.Data[i * 2 + j] = g.Data[i] * input.Data[j];
                    dWhh.Data[i * 2 + j] = g.Data[i] * hidden.Data[j];
                }
            }

            return new Tensor[] { dInput, dHidden, dWih, dWhh };
        });

        var node = _graphBuilder.CreateNode(output, lstmOp);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f, 1.0f }, new int[] { 2 });
        backward.Run(output, gradOutput);

        Assert.NotNull(input.Gradient);
        Assert.NotNull(hidden.Gradient);
        Assert.NotNull(Wih.Gradient);
        Assert.NotNull(Whh.Gradient);
    }

    #endregion

    #region Gradient Accumulation

    [Fact]
    public void TestGradientAccumulation4Steps()
    {
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var w = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);

        // Accumulate gradients over 4 steps
        for (int step = 0; step < 4; step++)
        {
            // y = w * x
            var y = new Tensor(new float[] { 2.0f }, new int[] { 1 });

            var mulOp = new OperationContext("Mul", g =>
            {
                return new Tensor[] { g * x.Data[0], g * w.Data[0] };
            });

            var node = _graphBuilder.CreateNode(y, mulOp);

            var backward = new BackwardPass(_graphBuilder);
            backward.RetainGraph = true;
            backward.Run(y);

            // Gradient should accumulate
            if (step == 3)
            {
                Assert.NotNull(w.Gradient);
                // After 4 steps, gradient should be 4 * x = 4
            }
        }
    }

    [Fact]
    public void TestGradientAccumulationScaling()
    {
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var w = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);

        // Accumulate with scaling
        for (int step = 0; step < 4; step++)
        {
            var y = new Tensor(new float[] { 2.0f }, new int[] { 1 });

            var mulOp = new OperationContext("Mul", g =>
            {
                return new Tensor[] { g * x.Data[0] / 4.0f, g * w.Data[0] / 4.0f };
            });

            var node = _graphBuilder.CreateNode(y, mulOp);

            var backward = new BackwardPass(_graphBuilder);
            backward.RetainGraph = true;
            backward.Run(y);
        }

        Assert.NotNull(w.Gradient);
    }

    #endregion

    #region Gradient Checkpointing

    [Fact]
    public void TestManualCheckpointing()
    {
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new int[] { 4 }, requiresGrad: true);

        // Create a 3-layer network with checkpoints
        var layer1 = new Tensor(new float[] { 2.0f, 4.0f, 6.0f, 8.0f }, new int[] { 4 }, requiresGrad: true);
        var layer2 = new Tensor(new float[] { 4.0f, 8.0f, 12.0f, 16.0f }, new int[] { 4 }, requiresGrad: true);
        var layer3 = new Tensor(new float[] { 8.0f, 16.0f, 24.0f, 32.0f }, new int[] { 4 });

        var checkpointManager = new CheckpointManager();
        checkpointManager.SaveCheckpoint(layer1);

        var op1 = new OperationContext("Layer1", g => new Tensor[] { g.Clone() });
        var node1 = _graphBuilder.CreateNode(layer1, op1);

        var op2 = new OperationContext("Layer2", g => new Tensor[] { g.Clone() });
        var node2 = _graphBuilder.CreateNode(layer2, op2, node1);

        var op3 = new OperationContext("Layer3", g => new Tensor[] { g.Clone() });
        var node3 = _graphBuilder.CreateNode(layer3, op3, node2);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f, 1.0f, 1.0f, 1.0f }, new int[] { 4 });
        backward.Run(layer3, gradOutput);

        Assert.NotNull(x.Gradient);
    }

    [Fact]
    public void TestAutoCheckpointing()
    {
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);

        // Create deep network with automatic checkpointing
        var output = x.Clone();
        GraphNode? prevNode = null;

        for (int i = 0; i < 5; i++)
        {
            output = output * 2.0f;
            var op = new OperationContext($"Layer{i}", g => new Tensor[] { g.Clone() });
            prevNode = _graphBuilder.CreateNode(output, op, prevNode);
        }

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f, 1.0f }, new int[] { 2 });
        backward.Run(output, gradOutput);

        Assert.NotNull(x.Gradient);
    }

    [Fact]
    public void TestCheckpointMemorySavings()
    {
        // This is more of a documentation test - actual memory measurement
        // would require more sophisticated infrastructure
        var x = new Tensor(new float[1000], new int[] { 1000 }, requiresGrad: true);

        // With checkpointing, we should save intermediate activations
        var checkpointManager = new CheckpointManager();
        var layer1 = x * 2.0f;
        checkpointManager.SaveCheckpoint(layer1);

        var layer2 = layer1 * 2.0f;
        var output = layer2 * 2.0f;

        var op1 = new OperationContext("Layer1", g => new Tensor[] { g.Clone() });
        var node1 = _graphBuilder.CreateNode(layer1, op1);

        var op2 = new OperationContext("Layer2", g => new Tensor[] { g.Clone() });
        var node2 = _graphBuilder.CreateNode(layer2, op2, node1);

        var op3 = new OperationContext("Layer3", g => new Tensor[] { g.Clone() });
        var node3 = _graphBuilder.CreateNode(output, op3, node2);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[1000], new int[] { 1000 });
        gradOutput.Fill(1.0f);
        backward.Run(output, gradOutput);

        Assert.NotNull(x.Gradient);
    }

    #endregion

    #region Higher-Order Derivatives

    [Fact]
    public void TestJacobianComputation()
    {
        // f(x) = [x^2, x^3]
        Tensor VectorFunc(Tensor input)
        {
            var result = new Tensor(new float[2], new int[] { 2 });
            result.Data[0] = input.Data[0] * input.Data[0];
            result.Data[1] = input.Data[0] * input.Data[0] * input.Data[0];
            return result;
        }

        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        var jacobian = GradientComputer.Jacobian(VectorFunc, x);

        // Expected Jacobian: [[2x], [3x^2]] at x=2: [[4], [12]]
        Assert.NotNull(jacobian);
        Assert.Equal(2, jacobian.Shape[0]);
        Assert.Equal(1, jacobian.Shape[1]);
        Assert.Equal(4.0f, jacobian.Data[0], precision: 4);
        Assert.Equal(12.0f, jacobian.Data[1], precision: 4);
    }

    [Fact]
    public void TestHessianComputation()
    {
        // f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x
        Tensor CubicFunc(Tensor input)
        {
            var result = new Tensor(new float[1], new int[] { 1 });
            result.Data[0] = input.Data[0] * input.Data[0] * input.Data[0];
            return result;
        }

        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        // Compute Hessian (second derivative)
        var grad1 = GradientComputer.NumericalGradient(CubicFunc, x);
        var hessian = GradientComputer.NumericalGradient(
            input => GradientComputer.NumericalGradient(CubicFunc, input),
            x
        );

        // f''(2) = 6*2 = 12
        Assert.NotNull(hessian);
        Assert.Equal(12.0f, hessian.Data[0], precision: 3);
    }

    [Fact]
    public void TestGradientOfGradient()
    {
        // Test computing gradient of gradient (higher-order derivatives)
        Tensor QuarticFunc(Tensor input)
        {
            var result = new Tensor(new float[1], new int[] { 1 });
            result.Data[0] = input.Data[0] * input.Data[0] * input.Data[0] * input.Data[0];
            return result;
        }

        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        // First gradient
        var grad1 = GradientComputer.NumericalGradient(QuarticFunc, x);
        // f'(x) = 4x^3, f'(2) = 32

        // Second gradient
        var grad2 = GradientComputer.NumericalGradient(
            input => GradientComputer.NumericalGradient(QuarticFunc, input),
            x
        );
        // f''(x) = 12x^2, f''(2) = 48

        Assert.Equal(32.0f, grad1.Data[0], precision: 3);
        Assert.Equal(48.0f, grad2.Data[0], precision: 3);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void TestZeroGradient()
    {
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);
        var output = new Tensor(new float[] { 0.0f, 0.0f }, new int[] { 2 });

        var zeroOp = new OperationContext("Zero", g =>
        {
            return new Tensor[] { Tensor.Zeros(x.Shape) };
        });

        var node = _graphBuilder.CreateNode(output, zeroOp);

        var backward = new BackwardPass(_graphBuilder);
        backward.Run(output);

        Assert.NotNull(x.Gradient);
        Assert.Equal(0.0f, x.Gradient.Data[0]);
        Assert.Equal(0.0f, x.Gradient.Data[1]);
    }

    [Fact]
    public void TestNaNPropagation()
    {
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var output = new Tensor(new float[] { float.NaN }, new int[] { 1 });

        var nanOp = new OperationContext("NaN", g =>
        {
            return new Tensor[] { new Tensor(new float[] { float.NaN }, new int[] { 1 }) };
        });

        var node = _graphBuilder.CreateNode(output, nanOp);

        var backward = new BackwardPass(_graphBuilder);
        backward.Run(output);

        Assert.NotNull(x.Gradient);
        Assert.True(float.IsNaN(x.Gradient.Data[0]));
    }

    [Fact]
    public void TestInfPropagation()
    {
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var output = new Tensor(new float[] { float.PositiveInfinity }, new int[] { 1 });

        var infOp = new OperationContext("Inf", g =>
        {
            return new Tensor[] { new Tensor(new float[] { float.PositiveInfinity }, new int[] { 1 }) };
        });

        var node = _graphBuilder.CreateNode(output, infOp);

        var backward = new BackwardPass(_graphBuilder);
        backward.Run(output);

        Assert.NotNull(x.Gradient);
        Assert.True(float.IsPositiveInfinity(x.Gradient.Data[0]));
    }

    [Fact]
    public void TestNumericalStability()
    {
        // Test with very small numbers to check for numerical stability
        var x = new Tensor(new float[] { 1e-10f }, new int[] { 1 }, requiresGrad: true);
        var output = new Tensor(new float[] { 1e-20f }, new int[] { 1 });

        var stableOp = new OperationContext("Stable", g =>
        {
            return new Tensor[] { g * 2e-10f };
        });

        var node = _graphBuilder.CreateNode(output, stableOp);

        var backward = new BackwardPass(_graphBuilder);
        var gradOutput = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        backward.Run(output, gradOutput);

        Assert.NotNull(x.Gradient);
        // Gradient should not overflow or underflow
        Assert.False(float.IsNaN(x.Gradient.Data[0]));
        Assert.False(float.IsInfinity(x.Gradient.Data[0]));
    }

    #endregion
}
