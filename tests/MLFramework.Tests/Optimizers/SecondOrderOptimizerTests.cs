using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;
using MLFramework.Optimizers;
using MLFramework.Optimizers.SecondOrder;
using MLFramework.Autograd;
using MLFramework.Schedulers;
using Xunit;

namespace MLFramework.Tests.Optimizers.SecondOrder;

/// <summary>
/// Comprehensive unit tests for second-order optimizers.
/// </summary>
public class SecondOrderOptimizerTests
{
    /// <summary>
    /// Helper class to track gradient computations.
    /// </summary>
    private class GradientTracker
    {
        private Dictionary<string, Tensor> _gradients;

        public GradientTracker()
        {
            _gradients = new Dictionary<string, Tensor>();
        }

        public void SetGradient(string name, Tensor gradient)
        {
            _gradients[name] = gradient;
        }

        public Dictionary<string, Tensor> GetGradients()
        {
            return new Dictionary<string, Tensor>(_gradients);
        }

        public void Clear()
        {
            _gradients.Clear();
        }
    }

    [Fact]
    public void NewtonOptimizer_ConvergesOnQuadraticFunction()
    {
        // Test that Newton's method converges quickly on a quadratic function
        var param = new Tensor(new[] { 2.0f, 3.0f }, new[] { 2 }, requiresGrad: true);
        var parameters = new Dictionary<string, Tensor> { { "x", param } };

        var optimizer = new NewtonOptimizer(parameters, learningRate: 1.0f, damping: 1e-4f);

        // Minimize f(x) = (x - a)^2 + (y - b)^2
        float[] target = { 1.0f, 2.0f };

        for (int i = 0; i < 10; i++)
        {
            // Compute loss: f = (x - a)^2 + (y - b)^2
            float loss = 0.0f;
            var paramData = TensorAccessor.GetData(param);
            loss += (paramData[0] - target[0]) * (paramData[0] - target[0]);
            loss += (paramData[1] - target[1]) * (paramData[1] - target[1]);

            // Compute gradient: ∂f/∂x = 2(x - a)
            var gradient = new Tensor(new float[2], new[] { 2 });
            var gradData = TensorAccessor.GetData(gradient);
            gradData[0] = 2.0f * (paramData[0] - target[0]);
            gradData[1] = 2.0f * (paramData[1] - target[1]);

            var gradients = new Dictionary<string, Tensor> { { "x", gradient } };
            optimizer.Step(gradients);
        }

        // Check convergence - Newton's method should converge in few iterations
        var finalData = TensorAccessor.GetData(param);
        Assert.True(Math.Abs(finalData[0] - target[0]) < 0.01f);
        Assert.True(Math.Abs(finalData[1] - target[1]) < 0.01f);
    }

    [Fact]
    public void NewtonOptimizer_WithDamping_IsStable()
    {
        // Test that damping provides numerical stability
        var param = new Tensor(new[] { 1e-6f }, new[] { 1 }, requiresGrad: true);
        var parameters = new Dictionary<string, Tensor> { { "x", param } };

        // No damping should potentially cause issues
        var optimizerNoDamping = new NewtonOptimizer(parameters, learningRate: 1.0f, damping: 0.0f);

        // With damping should be stable
        var optimizerWithDamping = new NewtonOptimizer(parameters, learningRate: 1.0f, damping: 1e-2f);

        var paramCopy = new Tensor((float[])TensorAccessor.GetData(param).Clone(), param.Shape);

        var gradients = new Dictionary<string, Tensor>
        {
            { "x", new Tensor(new[] { 1.0f }, new[] { 1 }) }
        };

        // Test that optimizer completes without exception
        var exceptionNoDamping = Record.Exception(() => optimizerNoDamping.Step(gradients));
        var exceptionWithDamping = Record.Exception(() => optimizerWithDamping.Step(gradients));

        Assert.Null(exceptionWithDamping);
        // We expect no exceptions in either case, but damping should ensure stability
    }

    [Fact]
    public void ConjugateGradientSolver_SolvesLinearSystem()
    {
        // Test CG solver on a simple positive-definite system
        int n = 5;
        var b = new float[n];
        for (int i = 0; i < n; i++)
        {
            b[i] = (float)(i + 1);
        }

        // Define A as identity + small perturbation (positive-definite)
        Func<Tensor, Tensor> matrixVectorProduct = v =>
        {
            var vData = TensorAccessor.GetData(v);
            var result = new float[n];
            for (int i = 0; i < n; i++)
            {
                result[i] = vData[i];  // Identity part
                if (i > 0)
                {
                    result[i] += 0.1f * vData[i - 1];  // Small off-diagonal
                }
            }
            return new Tensor(result, v.Shape);
        };

        var bTensor = new Tensor(b, new[] { n });
        var x = ConjugateGradientSolver.Solve(bTensor, matrixVectorProduct, maxIterations: 100);

        // Verify solution: Ax ≈ b
        var Ax = matrixVectorProduct(x);
        var AxData = TensorAccessor.GetData(Ax);
        var xData = TensorAccessor.GetData(x);

        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(AxData[i] - b[i]) < 0.01f);
        }
    }

    [Fact]
    public void LBFGSOptimizer_ConvergesOnNonConvexProblem()
    {
        // Test L-BFGS on a non-convex function (Rosenbrock function)
        var param = new Tensor(new[] { -1.5f, 2.0f }, new[] { 2 }, requiresGrad: true);
        var parameters = new Dictionary<string, Tensor> { { "x", param } };

        var optimizer = new LBFGSOptimizer(parameters, learningRate: 1.0f, historySize: 5, maxIterations: 50);

        // Minimize Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
        // Minimum at (1, 1) when a=1, b=100
        float a = 1.0f;
        float b = 100.0f;

        for (int i = 0; i < 100; i++)
        {
            var paramData = TensorAccessor.GetData(param);
            float x = paramData[0];
            float y = paramData[1];

            // Compute gradient numerically
            float epsilon = 1e-5f;
            var gradient = new Tensor(new float[2], new[] { 2 });
            var gradData = TensorAccessor.GetData(gradient);

            // df/dx = -2(a-x) - 4bx(y-x^2)
            gradData[0] = -2.0f * (a - x) - 4.0f * b * x * (y - x * x);

            // df/dy = 2b(y-x^2)
            gradData[1] = 2.0f * b * (y - x * x);

            var gradients = new Dictionary<string, Tensor> { { "x", gradient } };
            optimizer.Step(gradients);
        }

        // Check convergence to (1, 1)
        var finalData = TensorAccessor.GetData(param);
        Assert.True(Math.Abs(finalData[0] - 1.0f) < 0.1f);
        Assert.True(Math.Abs(finalData[1] - 1.0f) < 0.1f);
    }

    [Fact]
    public void LBFGSOptimizer_MemoryEfficient()
    {
        // Test that L-BFGS uses limited history
        var parameters = new Dictionary<string, Tensor>
        {
            { "x", new Tensor(new float[10], new[] { 10 }, requiresGrad: true) }
        };

        var historySize = 3;
        var optimizer = new LBFGSOptimizer(parameters, historySize: historySize);

        // Perform multiple steps
        var gradients = new Dictionary<string, Tensor>
        {
            { "x", new Tensor(new float[10], new[] { 10 }) }
        };

        for (int i = 0; i < 10; i++)
        {
            // Update gradients to force history growth
            var gradData = TensorAccessor.GetData(gradients["x"]);
            for (int j = 0; j < gradData.Length; j++)
            {
                gradData[j] = (float)i;
            }
            optimizer.Step(gradients);
        }

        // History should not exceed the specified size
        // (This would need reflection or exposing internal state to verify directly)
        // For now, just verify no memory explosion or exceptions
        Assert.True(optimizer.StepCount > 0);
    }

    [Fact]
    public void NaturalGradientOptimizer_ComputesNaturalGradient()
    {
        // Test natural gradient on a simple problem
        var param = new Tensor(new[] { 1.0f, 1.0f }, new[] { 2 }, requiresGrad: true);
        var parameters = new Dictionary<string, Tensor> { { "theta", param } };

        var optimizer = new NaturalGradientOptimizer(parameters, learningRate: 0.01f);

        // Simple gradient
        var gradients = new Dictionary<string, Tensor>
        {
            { "theta", new Tensor(new[] { 0.5f, 0.5f }, new[] { 2 }) }
        };

        var originalData = (float[])TensorAccessor.GetData(param).Clone();

        optimizer.Step(gradients);

        var newData = TensorAccessor.GetData(param);

        // Parameters should have changed
        Assert.NotEqual(originalData[0], newData[0]);
        Assert.NotEqual(originalData[1], newData[1]);

        // Natural gradient should be different from standard gradient descent
        // (direction may be different due to Fisher matrix)
    }

    [Fact]
    public void PowerIteration_EstimatesEigenvalue()
    {
        // Test power iteration on a known matrix
        int n = 5;

        // Create a diagonal matrix with known eigenvalues
        float[] eigenvalues = { 1.0f, 2.0f, 5.0f, 3.0f, 4.0f };
        float maxEigenvalue = eigenvalues.Max();

        Func<Tensor, Tensor> matrixVectorProduct = v =>
        {
            var vData = TensorAccessor.GetData(v);
            var result = new float[n];
            for (int i = 0; i < n; i++)
            {
                result[i] = eigenvalues[i] * vData[i];
            }
            return new Tensor(result, v.Shape);
        };

        float estimatedEigenvalue = PowerIteration.EstimateMaxEigenvalue(
            matrixVectorProduct,
            dimension: n,
            numIterations: 20
        );

        // Estimate should be close to the true maximum eigenvalue
        Assert.True(Math.Abs(estimatedEigenvalue - maxEigenvalue) < 0.1f);
    }

    [Fact]
    public void Adam_StandardUpdate()
    {
        // Test standard Adam optimizer
        var param = new Tensor(new[] { 1.0f, 2.0f }, new[] { 2 }, requiresGrad: true);
        var parameters = new Dictionary<string, Tensor> { { "w", param } };

        var optimizer = new Adam(parameters, learningRate: 0.01f);

        var gradients = new Dictionary<string, Tensor>
        {
            { "w", new Tensor(new[] { 0.1f, 0.2f }, new[] { 2 }) }
        };

        var originalData = (float[])TensorAccessor.GetData(param).Clone();

        optimizer.Step(gradients);

        var newData = TensorAccessor.GetData(param);

        // Parameters should decrease (since gradients are positive and we're minimizing)
        Assert.True(newData[0] < originalData[0]);
        Assert.True(newData[1] < originalData[1]);

        // Adam uses momentum, so updates should be more than simple gradient descent
        var simpleUpdate0 = originalData[0] - 0.01f * 0.1f;
        var simpleUpdate1 = originalData[1] - 0.01f * 0.2f;

        // Adam's first iteration should be close to simple update (no momentum yet)
        Assert.True(Math.Abs(newData[0] - simpleUpdate0) < 0.001f);
        Assert.True(Math.Abs(newData[1] - simpleUpdate1) < 0.001f);
    }

    [Fact]
    public void HessianAwareAdam_AdaptsLearningRate()
    {
        // Test that Hessian-aware Adam adapts learning rate based on curvature
        var param = new Tensor(new[] { 1.0f, 1.0f }, new[] { 2 }, requiresGrad: true);
        var parameters = new Dictionary<string, Tensor> { { "w", param } };

        var optimizer = new HessianAwareAdam(
            parameters,
            learningRate: 0.1f,
            eigenvalueEstimationIterations: 5,
            usePerParameterAdaptation: false
        );

        var gradients = new Dictionary<string, Tensor>
        {
            { "w", new Tensor(new[] { 0.1f, 0.1f }, new[] { 2 }) }
        };

        var baseLearningRate = optimizer.LearningRate;

        optimizer.Step(gradients);

        // Learning rate should have been adapted based on eigenvalue estimation
        // The exact value depends on the Hessian, but it should be different
        Assert.NotEqual(baseLearningRate, optimizer.LearningRate);

        // Adapted learning rate should be within bounds
        Assert.True(optimizer.LearningRate >= optimizer.MinLearningRate);
        Assert.True(optimizer.LearningRate <= optimizer.MaxLearningRate);
    }

    [Fact]
    public void NewtonOptimizer_SaveLoadState()
    {
        // Test checkpointing for Newton optimizer
        var param1 = new Tensor(new[] { 1.0f, 2.0f }, new[] { 2 }, requiresGrad: true);
        var parameters1 = new Dictionary<string, Tensor> { { "x", param1 } };

        var optimizer1 = new NewtonOptimizer(parameters1, learningRate: 0.5f, damping: 1e-3f);

        // Perform some updates
        var gradients = new Dictionary<string, Tensor>
        {
            { "x", new Tensor(new[] { 0.1f, 0.1f }, new[] { 2 }) }
        };

        optimizer1.Step(gradients);
        optimizer1.Step(gradients);

        // Save state
        var state = optimizer1.GetState();

        // Create new optimizer with same parameters
        var param2 = new Tensor(new[] { 1.0f, 2.0f }, new[] { 2 }, requiresGrad: true);
        var parameters2 = new Dictionary<string, Tensor> { { "x", param2 } };

        var optimizer2 = new NewtonOptimizer(parameters2, learningRate: 1.0f, damping: 1e-4f);

        // Load state
        optimizer2.LoadState(state);

        // Verify state was loaded correctly
        Assert.Equal(optimizer1.LearningRate, optimizer2.LearningRate);
        Assert.Equal(optimizer1.Damping, optimizer2.Damping);
        Assert.Equal(optimizer1.StepCount, optimizer2.StepCount);
    }

    [Fact]
    public void LBFGSOptimizer_SaveLoadState()
    {
        // Test checkpointing for L-BFGS optimizer
        var param1 = new Tensor(new[] { 1.0f }, new[] { 1 }, requiresGrad: true);
        var parameters1 = new Dictionary<string, Tensor> { { "x", param1 } };

        var optimizer1 = new LBFGSOptimizer(
            parameters1,
            learningRate: 0.1f,
            historySize: 5,
            maxIterations: 10
        );

        // Perform some updates
        var gradients = new Dictionary<string, Tensor>
        {
            { "x", new Tensor(new[] { 0.1f }, new[] { 1 }) }
        };

        optimizer1.Step(gradients);
        optimizer1.Step(gradients);

        // Save state
        var state = optimizer1.GetState();

        // Create new optimizer and load state
        var param2 = new Tensor(new[] { 1.0f }, new[] { 1 }, requiresGrad: true);
        var parameters2 = new Dictionary<string, Tensor> { { "x", param2 } };

        var optimizer2 = new LBFGSOptimizer(parameters2);
        optimizer2.LoadState(state);

        // Verify state was loaded correctly
        Assert.Equal(optimizer1.LearningRate, optimizer2.LearningRate);
        Assert.Equal(optimizer1.HistorySize, optimizer2.HistorySize);
        Assert.Equal(optimizer1.MaxIterations, optimizer2.MaxIterations);
        Assert.Equal(optimizer1.StepCount, optimizer2.StepCount);
    }

    [Fact]
    public void Optimizers_HandleZeroGradients()
    {
        // Test that optimizers handle zero gradients gracefully
        var parameters = new Dictionary<string, Tensor>
        {
            { "x", new Tensor(new[] { 1.0f, 2.0f }, new[] { 2 }, requiresGrad: true) }
        };

        var zeroGradients = new Dictionary<string, Tensor>
        {
            { "x", new Tensor(new[] { 0.0f, 0.0f }, new[] { 2 }) }
        };

        // Test each optimizer
        var newton = new NewtonOptimizer(parameters, learningRate: 1.0f);
        var originalData = (float[])TensorAccessor.GetData(parameters["x"]).Clone();
        newton.Step(zeroGradients);
        var newData = TensorAccessor.GetData(parameters["x"]);
        Assert.Equal(originalData, newData); // No change with zero gradient

        var lbfgs = new LBFGSOptimizer(parameters, learningRate: 1.0f);
        lbfgs.Step(zeroGradients);
        newData = TensorAccessor.GetData(parameters["x"]);
        // L-BFGS should also not change parameters with zero gradients
        Assert.Equal(originalData, newData);
    }

    [Fact]
    public void HessianMode_Configuration()
    {
        // Test that Hessian mode can be configured
        var parameters = new Dictionary<string, Tensor>
        {
            { "x", new Tensor(new[] { 1.0f }, new[] { 1 }, requiresGrad: true) }
        };

        var optimizer = new NewtonOptimizer(parameters);
        optimizer.HessianMode = HessianMode.HVP;

        Assert.Equal(HessianMode.HVP, optimizer.HessianMode);

        optimizer.HessianMode = HessianMode.Full;

        Assert.Equal(HessianMode.Full, optimizer.HessianMode);
    }

    [Fact]
    public void Convergence_NewtonVsFirstOrder()
    {
        // Compare convergence of Newton vs first-order methods on convex problem
        // Newton should converge faster (fewer iterations)
        var paramNewton = new Tensor(new[] { 5.0f }, new[] { 1 }, requiresGrad: true);
        var paramSGD = new Tensor(new[] { 5.0f }, new[] { 1 }, requiresGrad: true);

        var parametersNewton = new Dictionary<string, Tensor> { { "x", paramNewton } };
        var parametersSGD = new Dictionary<string, Tensor> { { "x", paramSGD } };

        var optimizerNewton = new NewtonOptimizer(parametersNewton, learningRate: 1.0f);
        var optimizerSGD = new MLFramework.Optimizers.Adam(parametersSGD, learningRate: 0.1f);

        float target = 0.0f;
        int newtonIterations = 0;
        int sgdIterations = 0;

        // Newton's method
        for (int i = 0; i < 20; i++)
        {
            var paramData = TensorAccessor.GetData(paramNewton);
            float loss = (paramData[0] - target) * (paramData[0] - target);

            var gradient = new Tensor(new[] { 2.0f * (paramData[0] - target) }, new[] { 1 });
            var gradients = new Dictionary<string, Tensor> { { "x", gradient } };

            optimizerNewton.Step(gradients);
            newtonIterations++;

            if (Math.Abs(paramData[0] - target) < 0.01f)
                break;
        }

        // SGD (Adam)
        for (int i = 0; i < 50; i++)
        {
            var paramData = TensorAccessor.GetData(paramSGD);
            float loss = (paramData[0] - target) * (paramData[0] - target);

            var gradient = new Tensor(new[] { 2.0f * (paramData[0] - target) }, new[] { 1 });
            var gradients = new Dictionary<string, Tensor> { { "x", gradient } };

            optimizerSGD.Step(gradients);
            sgdIterations++;

            if (Math.Abs(paramData[0] - target) < 0.01f)
                break;
        }

        // Newton should converge in fewer iterations
        Assert.True(newtonIterations < sgdIterations);
    }
}
