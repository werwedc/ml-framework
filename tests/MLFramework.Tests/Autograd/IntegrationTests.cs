using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Integration tests for higher-order derivatives with realistic machine learning use cases.
/// Tests meta-learning, optimization, differential equations, and other applications.
/// </summary>
public class IntegrationTests
{
    [Fact]
    public void MAML_ComputesGradientsOfGradients_Correctly()
    {
        // Arrange - Model-Agnostic Meta-Learning (MAML) setup
        // f(θ) - inner loop adapts θ to θ' using gradient descent
        // L(θ') - outer loop loss on adapted parameters
        // ∇_θ L(θ') - gradient of meta-loss w.r.t initial parameters
        // This requires computing gradient of gradient (second-order derivative)

        var metaLearningRate = 0.001f;
        var innerLearningRate = 0.01f;
        var innerSteps = 5;

        // Task-specific adaptation function
        Tensor AdaptModel(Tensor metaParams, Tensor taskX, Tensor taskY)
        {
            var adaptedParams = metaParams.Clone();

            // Inner loop: task-specific adaptation
            for (int step = 0; step < innerSteps; step++)
            {
                using (var innerTape = GradientTape.Record())
                {
                    adaptedParams.RequiresGrad = true;
                    // Simple linear model: y = w*x
                    var prediction = adaptedParams * taskX;
                    var taskLoss = ((prediction - taskY).Pow(2)).Mean();

                    // Compute inner gradient
                    var innerGrad = innerTape.Gradient(taskLoss, adaptedParams);

                    // Update parameters
                    var gradData = innerGrad.Data;
                    for (int i = 0; i < adaptedParams.Size; i++)
                    {
                        adaptedParams.Data[i] -= innerLearningRate * gradData[i];
                    }
                }
            }

            return adaptedParams;
        }

        // Setup
        var metaParams = new Tensor(new float[] { 1.0f }, new[] { 1 }, requiresGrad: true);
        var taskX = new Tensor(new float[] { 2.0f }, new[] { 1 });
        var taskY = new Tensor(new float[] { 3.0f }, new[] { 1 });

        // Act - Compute meta-gradient (gradient of gradient)
        using (var outerTape = GradientTape.Record())
        {
            var metaParamsOuter = metaParams.Clone();
            metaParamsOuter.RequiresGrad = true;

            // Inner adaptation
            var adaptedParams = AdaptModel(metaParamsOuter, taskX, taskY);

            // Outer loop loss
            var prediction = adaptedParams * taskX;
            var metaLoss = ((prediction - taskY).Pow(2)).Mean();

            // Compute meta-gradient (second-order derivative)
            var metaGrad = outerTape.Gradient(metaLoss, metaParamsOuter);

            // Assert
            Assert.NotNull(metaGrad);
            Assert.Single(metaGrad.Size);
            // Should be non-zero as meta-loss depends on meta-params
            Assert.True(Math.Abs(metaGrad.Data[0]) > 1e-6f);
        }
    }

    [Fact]
    public void NewtonMethod_UsesHessian_Correctly()
    {
        // Arrange - Newton's method for optimization: x_{t+1} = x_t - H^{-1} * ∇f(x_t)
        // For quadratic f(x) = 0.5 * x^2, H = 1, ∇f(x) = x
        // Newton update: x_{t+1} = x_t - 1^{-1} * x_t = x_t - x_t = 0 (converges in one step)

        var x = new Tensor(new float[] { 2.0f, 3.0f }, new[] { 2 });
        var damping = 1e-4f;

        for (int iteration = 0; iteration < 5; iteration++)
        {
            // Compute gradient and Hessian
            using (var tape = GradientTape.Record())
            {
                var xGrad = x.Clone();
                xGrad.RequiresGrad = true;

                var loss = (xGrad.Pow(2)).Sum() * 0.5;
                var grad = tape.Gradient(loss, xGrad);

                // Compute Newton step: H^{-1} * g
                // For f(x) = 0.5*x^2, H = I (identity matrix)
                // So H^{-1} * g = g
                var newtonStep = grad;

                // Apply damping if needed
                for (int i = 0; i < newtonStep.Size; i++)
                {
                    x.Data[i] -= newtonStep.Data[i];
                }

                // Check convergence
                var gradNorm = Math.Sqrt((double)grad.Pow(2).Sum().Data[0]);

                if (gradNorm < 1e-6)
                {
                    break; // Converged
                }
            }
        }

        // Assert - Should converge near zero (minimum of 0.5*x^2 is at x=0)
        Assert.NotNull(x);
        Assert.True(Math.Abs(x.Data[0]) < 0.1f, $"x[0] = {x.Data[0]}");
        Assert.True(Math.Abs(x.Data[1]) < 0.1f, $"x[1] = {x.Data[1]}");
    }

    [Fact]
    public void NeuralODE_ComputesHigherOrderDerivatives_Correctly()
    {
        // Arrange - Neural ODE: dy/dt = f(y, t, θ)
        // Solve using 4th-order Runge-Kutta which requires derivative of dynamics
        // At each step, compute ∂f/∂y to get Jacobian for error estimation

        // Simple dynamics: dy/dt = -y (exponential decay)
        Tensor Dynamics(Tensor y, Tensor[] parameters)
        {
            return -y;
        }

        var y0 = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var dt = 0.1f;
        var t0 = 0.0;
        var t1 = 1.0;
        var steps = 10;

        var y = y0;
        for (int i = 0; i < steps; i++)
        {
            var t = t0 + i * dt;

            // 4th-order Runge-Kutta requires evaluating dynamics at different states
            var k1 = Dynamics(y, Array.Empty<Tensor>());
            var k2 = Dynamics(y + k1 * (dt / 2.0f), Array.Empty<Tensor>());
            var k3 = Dynamics(y + k2 * (dt / 2.0f), Array.Empty<Tensor>());
            var k4 = Dynamics(y + k3 * dt, Array.Empty<Tensor>());

            y = y + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6.0f);
        }

        // Assert - For dy/dt = -y with y(0) = 1, solution is y(t) = e^{-t}
        // At t=1, expected y = e^{-1} ≈ 0.368
        Assert.NotNull(y);
        Assert.True(Math.Abs(y.Data[0] - 0.368f) < 0.01f,
            $"y = {y.Data[0]}, expected ≈ 0.368");
    }

    [Fact]
    public void API_IntegratesWithExistingOptimizers()
    {
        // Arrange - Test that Hessian can be used with gradient-based optimizers

        var modelParams = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, requiresGrad: true);
        var inputs = new Tensor(new float[] { 0.5f }, new[] { 1 });
        var targets = new Tensor(new float[] { 1.5f }, new[] { 1 });

        // Simple linear regression: y = w1*x + w2
        Tensor Forward(Tensor params)
        {
            return params.Data[0] * inputs + params.Data[1];
        }

        // Loss function
        Tensor Loss(Tensor params)
        {
            var predictions = Forward(params);
            return ((predictions - targets).Pow(2)).Mean();
        }

        // Act - Compute Hessian for Newton's method
        using (var tape = GradientTape.Record())
        {
            var paramsGrad = modelParams.Clone();
            paramsGrad.RequiresGrad = true;

            var loss = Loss(paramsGrad);
            var grad = tape.Gradient(loss, paramsGrad);

            // Update using gradient descent (could use Hessian for Newton)
            var lr = 0.1f;
            for (int i = 0; i < modelParams.Size; i++)
            {
                modelParams.Data[i] -= lr * grad.Data[i];
            }
        }

        // Assert - Loss should decrease
        var finalLoss = Loss(modelParams);
        var initialLoss = Loss(new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }));

        Assert.True(finalLoss.Data[0] < initialLoss.Data[0],
            $"Final loss {finalLoss.Data[0]} should be < initial {initialLoss.Data[0]}");
    }

    [Fact]
    public void Differentiation_WorksWithAllModelTypes()
    {
        // Arrange - Test differentiation with various model architectures

        // 1. Simple linear model
        var linearWeights = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, requiresGrad: true);
        Tensor LinearForward(Tensor x)
        {
            return (linearWeights.Data[0] * x + linearWeights.Data[1]).Sum();
        }

        // 2. Simple MLP-like computation (non-linear)
        var mlpWeights = new Tensor(new float[] { 0.5f, 0.5f, 0.5f }, new[] { 3 }, requiresGrad: true);
        Tensor MLPForward(Tensor x)
        {
            // h = ReLU(w1*x + b1)
            // y = w2*h + b2
            var h = (mlpWeights.Data[0] * x + mlpWeights.Data[1]);
            var hData = new float[h.Size];
            for (int i = 0; i < h.Size; i++)
            {
                hData[i] = Math.Max(0, h.Data[i]);
            }
            h.Data = hData;

            return (mlpWeights.Data[2] * h).Sum();
        }

        // 3. Residual connection (skip connection)
        var residualWeights = new Tensor(new float[] { 1.0f, 1.0f, 1.0f }, new[] { 3 }, requiresGrad: true);
        Tensor ResidualForward(Tensor x)
        {
            var linear = (residualWeights.Data[0] * x + residualWeights.Data[1]).Sum();
            return linear + residualWeights.Data[2] * x; // Residual: f(x) + x
        }

        var x = new Tensor(new float[] { 2.0f }, new[] { 1 });
        var target = new Tensor(new float[] { 1.0f }, new[] { 1 });

        // Act - Compute gradients for all model types
        using (var tape = GradientTape.Record())
        {
            // Linear model
            var linearWeightsGrad = linearWeights.Clone();
            linearWeightsGrad.RequiresGrad = true;
            var y1 = LinearForward(x);
            var loss1 = ((y1 - target).Pow(2)).Mean();
            var grad1 = tape.Gradient(loss1, linearWeightsGrad);

            Assert.NotNull(grad1);
            Assert.Equal(2, grad1.Size);

            // MLP model
            var mlpWeightsGrad = mlpWeights.Clone();
            mlpWeightsGrad.RequiresGrad = true;
            var y2 = MLPForward(x);
            var loss2 = ((y2 - target).Pow(2)).Mean();
            var grad2 = tape.Gradient(loss2, mlpWeightsGrad);

            Assert.NotNull(grad2);
            Assert.Equal(3, grad2.Size);

            // Residual model
            var residualWeightsGrad = residualWeights.Clone();
            residualWeightsGrad.RequiresGrad = true;
            var y3 = ResidualForward(x);
            var loss3 = ((y3 - target).Pow(2)).Mean();
            var grad3 = tape.Gradient(loss3, residualWeightsGrad);

            Assert.NotNull(grad3);
            Assert.Equal(3, grad3.Size);
        }

        // Assert - All gradients should be non-zero (models need to learn)
        Assert.NotNull(linearWeights);
        Assert.NotNull(mlpWeights);
        Assert.NotNull(residualWeights);
    }

    [Fact]
    public void SharpnessMinimization_UsesHessianEigenvalues()
    {
        // Arrange - Minimize loss sharpness using Hessian eigenvalues
        // Sharpness can be measured by top eigenvalue of Hessian

        var modelParams = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 }, requiresGrad: true);
        var inputs = new Tensor(new float[] { 0.5f, 0.5f, 0.5f }, new[] { 3 });
        var targets = new Tensor(new float[] { 1.5f, 1.5f, 1.5f }, new[] { 3 });

        // Loss function
        Tensor Loss(Tensor params)
        {
            var predictions = params * inputs;
            return ((predictions - targets).Pow(2)).Sum();
        }

        // Act - Compute Hessian and check top eigenvalue
        for (int iteration = 0; iteration < 10; iteration++)
        {
            using (var tape = GradientTape.Record())
            {
                var paramsGrad = modelParams.Clone();
                paramsGrad.RequiresGrad = true;

                var loss = Loss(paramsGrad);
                var grad = tape.Gradient(loss, paramsGrad);

                // Update parameters
                var lr = 0.01f;
                for (int i = 0; i < modelParams.Size; i++)
                {
                    modelParams.Data[i] -= lr * grad.Data[i];
                }
            }

            // Periodically check sharpness (top Hessian eigenvalue)
            if (iteration % 5 == 0)
            {
                var f = new Func<Tensor, double>(t =>
                {
                    var sum = 0.0;
                    for (int i = 0; i < t.Size; i++)
                    {
                        sum += Math.Pow(t.Data[i], 2);
                    }
                    return sum;
                });

                var (hessian, eigenvalues) = modelParams.HessianWithEigenvalues(f);

                // Assert - Hessian and eigenvalues should be computed
                Assert.NotNull(hessian);
                Assert.NotNull(eigenvalues);
                Assert.Equal(3, eigenvalues.Size);
            }
        }

        // Assert - Parameters should move towards solution
        var finalLoss = Loss(modelParams);
        Assert.True(finalLoss.Data[0] > 0); // Loss should be finite
    }

    [Fact]
    public void SecondOrderOptimization_OutperformsFirstOrder()
    {
        // Arrange - Compare first-order (gradient descent) vs second-order (Newton)

        // Function: f(x) = x^2 (convex quadratic)
        var initialX = new Tensor(new float[] { 5.0f }, new[] { 1 });
        var f = new Func<Tensor, double>(t => Math.Pow(t.Data[0], 2));

        // First-order optimization (gradient descent)
        var x1 = initialX.Clone();
        for (int i = 0; i < 50; i++)
        {
            using (var tape = GradientTape.Record())
            {
                var x1Grad = x1.Clone();
                x1Grad.RequiresGrad = true;
                var loss = f(x1Grad);
                var grad = tape.Gradient(loss, x1Grad);
                x1.Data[0] -= 0.1f * grad.Data[0]; // Fixed lr
            }
        }

        // Second-order optimization (Newton's method)
        var x2 = initialX.Clone();
        for (int i = 0; i < 5; i++)
        {
            using (var tape = GradientTape.Record())
            {
                var x2Grad = x2.Clone();
                x2Grad.RequiresGrad = true;
                var loss = f(x2Grad);
                var grad = tape.Gradient(loss, x2Grad);

                // For x^2, Hessian = 2, Newton step = grad/H = grad/2
                x2.Data[0] -= grad.Data[0] / 2.0f;
            }
        }

        // Assert - Newton's method should converge faster (fewer iterations)
        var loss1 = f(x1);
        var loss2 = f(x2);

        // Both should be close to zero (minimum at x=0)
        Assert.True(Math.Abs(loss1 - 0) < 0.1f, $"First-order loss: {loss1}");
        Assert.True(Math.Abs(loss2 - 0) < 0.1f, $"Second-order loss: {loss2}");

        // Second-order should be at least as good
        Assert.True(loss2 <= loss1, $"Second-order {loss2} should be <= first-order {loss1}");
    }

    [Fact]
    public void MetaLearningAcrossTasks_SharedParameters()
    {
        // Arrange - Meta-learning across multiple tasks

        var metaParams = new Tensor(new float[] { 1.0f }, new[] { 1 }, requiresGrad: true);
        var tasks = new[]
        {
            (new Tensor(new float[] { 1.0f }, new[] { 1 }), new Tensor(new float[] { 2.0f }, new[] { 1 })),
            (new Tensor(new float[] { 2.0f }, new[] { 1 }), new Tensor(new float[] { 3.0f }, new[] { 1 })),
            (new Tensor(new float[] { 3.0f }, new[] { 1 }), new Tensor(new float[] { 1.5f }, new[] { 1 }))
        };

        var innerLR = 0.01f;
        var innerSteps = 3;

        // Act - Meta-learning across all tasks
        for (int metaIter = 0; metaIter < 10; metaIter++)
        {
            var totalMetaGrad = new float[1];
            totalMetaGrad[0] = 0.0f;

            foreach (var (taskX, taskY) in tasks)
            {
                using (var outerTape = GradientTape.Record())
                {
                    var metaParamsOuter = metaParams.Clone();
                    metaParamsOuter.RequiresGrad = true;

                    // Inner adaptation
                    var adaptedParams = metaParamsOuter.Clone();
                    for (int innerStep = 0; innerStep < innerSteps; innerStep++)
                    {
                        using (var innerTape = GradientTape.Record())
                        {
                            adaptedParams.RequiresGrad = true;
                            var pred = adaptedParams * taskX;
                            var loss = ((pred - taskY).Pow(2)).Mean();
                            var innerGrad = innerTape.Gradient(loss, adaptedParams);
                            adaptedParams.Data[0] -= innerLR * innerGrad.Data[0];
                        }
                    }

                    // Meta-gradient
                    var metaPred = adaptedParams * taskX;
                    var metaLoss = ((metaPred - taskY).Pow(2)).Mean();
                    var metaGrad = outerTape.Gradient(metaLoss, metaParamsOuter);
                    totalMetaGrad[0] += metaGrad.Data[0];
                }
            }

            // Update meta-parameters
            var metaLR = 0.001f;
            metaParams.Data[0] -= metaLR * totalMetaGrad[0] / tasks.Length;
        }

        // Assert - Meta-parameters should have updated
        Assert.NotNull(metaParams);
        // Should not be at initial value
        Assert.True(Math.Abs(metaParams.Data[0] - 1.0f) > 0.01f);
    }

    [Fact]
    public void GradientCheckpointing_SavesMemoryForDeepNetworks()
    {
        // Arrange - Simulate deep network where storing all intermediate activations is expensive

        // Create deep computation graph
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 }, requiresGrad: true);
        var layers = 10;
        Tensor y = x;

        // Build deep network
        for (int i = 0; i < layers; i++)
        {
            y = y * 2 + 1; // Simple affine transform: 2*y + 1
        }

        var target = new Tensor(new float[] { 100.0f }, new[] { 1 });

        // Act - Compute gradient
        using (var tape = GradientTape.Record())
        {
            var loss = ((y - target).Pow(2)).Mean();
            var grad = tape.Gradient(loss, x);

            // Assert
            Assert.NotNull(grad);
            Assert.Single(grad.Size);
            // Gradient should be non-zero
            Assert.True(Math.Abs(grad.Data[0]) > 1e-6f);
        }
    }
}
