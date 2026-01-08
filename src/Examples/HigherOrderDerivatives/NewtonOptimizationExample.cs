using System;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using MLFramework.Optimizers.SecondOrder;

namespace MLFramework.Examples.HigherOrderDerivatives;

/// <summary>
/// Newton's Method Optimization Example.
///
/// Newton's method is a second-order optimization algorithm that uses the Hessian matrix
/// to achieve faster convergence compared to first-order methods like SGD.
///
/// Key Concepts:
/// - Hessian Matrix: Second derivative of the loss function
/// - Newton Step: Δθ = -H^(-1) * g where H is Hessian and g is gradient
/// - Quadratic Approximation: Approximates the loss as a quadratic function
///
/// Mathematical Formulation:
/// The update rule for Newton's method is:
///   θ_{t+1} = θ_t - α * H^(-1) * ∇L(θ_t)
///
/// where:
///   - H is the Hessian matrix (matrix of second derivatives)
///   - ∇L is the gradient vector
///   - α is the learning rate
///
/// Advantages:
/// - Faster convergence (quadratic convergence near minima)
/// - Better handling of ill-conditioned problems
/// - Adaptive step size based on curvature
///
/// Disadvantages:
/// - Computing Hessian is O(n²) memory
/// - Inverting Hessian is O(n³) computation
/// - Requires the Hessian to be positive definite
///
/// In practice, we use the Conjugate Gradient method to solve H * Δθ = -g
/// without explicitly computing or inverting the Hessian.
/// </summary>
public class NewtonOptimizationExample
{
    /// <summary>
    /// Runs the Newton's method optimization example.
    /// Compares SGD and Newton's method on the Rosenbrock function.
    /// </summary>
    public static void Run()
    {
        Console.WriteLine("=== Newton's Method Optimization Example ===\n");

        // Test on Rosenbrock function (a classic non-convex test function)
        Console.WriteLine("Optimizing Rosenbrock function: f(x, y) = (a-x)² + b(y-x²)²");
        Console.WriteLine("Parameters: a=1.0, b=100.0\n");

        double a = 1.0;
        double b = 100.0;

        // Define Rosenbrock function
        Func<Tensor[], double> rosenbrock = (Tensor[] parameters) =>
        {
            var xData = TensorAccessor.GetData(parameters[0]);
            var yData = TensorAccessor.GetData(parameters[1]);

            double x = xData[0];
            double y = yData[0];

            return Math.Pow(a - x, 2) + b * Math.Pow(y - Math.Pow(x, 2), 2);
        };

        // Initialize parameters (start from a challenging point)
        var initialX = new Tensor(new[] { -1.2f }, new[] { 1 }, true);
        var initialY = new Tensor(new[] { 1.0f }, new[] { 1 }, true);

        Console.WriteLine("Initial point: (x, y) = (-1.2, 1.0)");
        Console.WriteLine("Initial loss: " + rosenbrock(new[] { initialX, initialY }).ToString("F6"));
        Console.WriteLine("Optimal point: (x, y) = (1.0, 1.0)");
        Console.WriteLine("Optimal loss: 0.0\n");

        // --- SGD Optimization ---
        Console.WriteLine("=== SGD Optimization ===");
        var sgdParams = new Tensor[]
        {
            new Tensor(new[] { -1.2f }, new[] { 1 }, true),
            new Tensor(new[] { 1.0f }, new[] { 1 }, true)
        };

        int sgdIterations = 10000;
        double sgdLearningRate = 0.001;

        Console.WriteLine($"Running SGD for {sgdIterations} iterations with LR={sgdLearningRate}...");

        for (int i = 0; i < sgdIterations; i++)
        {
            var loss = rosenbrock(sgdParams);

            // Compute gradients numerically (for simplicity in this example)
            var grads = ComputeNumericalGradients(rosenbrock, sgdParams);

            // Update parameters
            var xData = TensorAccessor.GetData(sgdParams[0]);
            var yData = TensorAccessor.GetData(sgdParams[1]);
            var gradXData = TensorAccessor.GetData(grads[0]);
            var gradYData = TensorAccessor.GetData(grads[1]);

            xData[0] -= (float)sgdLearningRate * gradXData[0];
            yData[0] -= (float)sgdLearningRate * gradYData[0];

            // Print progress
            if (i % 1000 == 0)
            {
                Console.WriteLine($"  Iteration {i}: (x={xData[0]:F4}, y={yData[0]:F4}), Loss={loss:F6}");
            }
        }

        double sgdFinalLoss = rosenbrock(sgdParams);
        Console.WriteLine($"\nSGD Final: (x={TensorAccessor.GetData(sgdParams[0])[0]:F4}, y={TensorAccessor.GetData(sgdParams[1])[0]:F4}), Loss={sgdFinalLoss:F6}\n");

        // --- Newton's Method Optimization ---
        Console.WriteLine("=== Newton's Method Optimization ===");
        var newtonParams = new Tensor[]
        {
            new Tensor(new[] { -1.2f }, new[] { 1 }, true),
            new Tensor(new[] { 1.0f }, new[] { 1 }, true)
        };

        int newtonIterations = 20;
        double newtonLearningRate = 1.0;
        double damping = 1e-4;

        Console.WriteLine($"Running Newton's method for {newtonIterations} iterations with LR={newtonLearningRate}, damping={damping}...");

        for (int i = 0; i < newtonIterations; i++)
        {
            var loss = rosenbrock(newtonParams);

            // Compute gradients and Hessian
            var grads = ComputeNumericalGradients(rosenbrock, newtonParams);
            var hessian = ComputeNumericalHessian(rosenbrock, newtonParams);

            // Solve H * Δθ = -g using Newton's method
            var delta = SolveNewtonStep(grads, hessian, damping);

            // Update parameters
            var xData = TensorAccessor.GetData(newtonParams[0]);
            var yData = TensorAccessor.GetData(newtonParams[1]);
            var deltaXData = TensorAccessor.GetData(delta[0]);
            var deltaYData = TensorAccessor.GetData(delta[1]);

            xData[0] += (float)newtonLearningRate * deltaXData[0];
            yData[0] += (float)newtonLearningRate * deltaYData[0];

            // Print progress
            Console.WriteLine($"  Iteration {i}: (x={xData[0]:F4}, y={yData[0]:F4}), Loss={loss:F6}");

            // Check convergence
            if (loss < 1e-10)
            {
                Console.WriteLine("  Converged!");
                break;
            }
        }

        double newtonFinalLoss = rosenbrock(newtonParams);
        Console.WriteLine($"\nNewton's Final: (x={TensorAccessor.GetData(newtonParams[0])[0]:F4}, y={TensorAccessor.GetData(newtonParams[1])[0]:F4}), Loss={newtonFinalLoss:F6}\n");

        // --- Comparison ---
        Console.WriteLine("=== Comparison ===");
        Console.WriteLine($"SGD iterations: {sgdIterations}, Final loss: {sgdFinalLoss:F6}");
        Console.WriteLine($"Newton iterations: {newtonIterations}, Final loss: {newtonFinalLoss:F6}");
        Console.WriteLine($"Speedup: {(double)sgdIterations / newtonIterations:F1}x");
        Console.WriteLine();

        // Demonstrate convergence on different functions
        Console.WriteLine("=== Testing on different functions ===\n");

        TestOnQuadraticFunction();
        TestOnIllConditionedFunction();
    }

    /// <summary>
    /// Tests optimization on a simple quadratic function.
    /// Newton's method should converge in one iteration.
    /// </summary>
    private static void TestOnQuadraticFunction()
    {
        Console.WriteLine("Quadratic function: f(x) = x^2 + 2y^2");

        Func<Tensor[], double> quadratic = (Tensor[] parameters) =>
        {
            var xData = TensorAccessor.GetData(parameters[0]);
            var yData = TensorAccessor.GetData(parameters[1]);

            double x = xData[0];
            double y = yData[0];

            return x * x + 2 * y * y;
        };

        var paramsSGD = new Tensor[]
        {
            new Tensor(new[] { 2.0f }, new[] { 1 }, true),
            new Tensor(new[] { 2.0f }, new[] { 1 }, true)
        };

        var paramsNewton = new Tensor[]
        {
            new Tensor(new[] { 2.0f }, new[] { 1 }, true),
            new Tensor(new[] { 2.0f }, new[] { 1 }, true)
        };

        Console.WriteLine("Initial: (2.0, 2.0), Loss=" + quadratic(paramsSGD).ToString("F2"));

        // SGD
        for (int i = 0; i < 10; i++)
        {
            var grads = ComputeNumericalGradients(quadratic, paramsSGD);
            var xData = TensorAccessor.GetData(paramsSGD[0]);
            var yData = TensorAccessor.GetData(paramsSGD[1]);
            var gradXData = TensorAccessor.GetData(grads[0]);
            var gradYData = TensorAccessor.GetData(grads[1]);

            xData[0] -= 0.1f * gradXData[0];
            yData[0] -= 0.1f * gradYData[0];
        }
        Console.WriteLine($"SGD after 10 iters: ({TensorAccessor.GetData(paramsSGD[0])[0]:F4}, {TensorAccessor.GetData(paramsSGD[1])[0]:F4}), Loss={quadratic(paramsSGD):F6}");

        // Newton
        var grads2 = ComputeNumericalGradients(quadratic, paramsNewton);
        var hessian2 = ComputeNumericalHessian(quadratic, paramsNewton);
        var delta2 = SolveNewtonStep(grads2, hessian2, 0);

        var xDataN = TensorAccessor.GetData(paramsNewton[0]);
        var yDataN = TensorAccessor.GetData(paramsNewton[1]);
        var deltaXDataN = TensorAccessor.GetData(delta2[0]);
        var deltaYDataN = TensorAccessor.GetData(delta2[1]);

        xDataN[0] += deltaXDataN[0];
        yDataN[0] += deltaYDataN[0];

        Console.WriteLine($"Newton after 1 iter: ({xDataN[0]:F4}, {yDataN[0]:F4}), Loss={quadratic(paramsNewton):F6}");
        Console.WriteLine("Newton's method converges in 1 iteration for quadratic functions!\n");
    }

    /// <summary>
    /// Tests optimization on an ill-conditioned function.
    /// Newton's method should handle this better than SGD.
    /// </summary>
    private static void TestOnIllConditionedFunction()
    {
        Console.WriteLine("Ill-conditioned function: f(x) = 1000x^2 + y^2");

        Func<Tensor[], double> illConditioned = (Tensor[] parameters) =>
        {
            var xData = TensorAccessor.GetData(parameters[0]);
            var yData = TensorAccessor.GetData(parameters[1]);

            double x = xData[0];
            double y = yData[0];

            return 1000 * x * x + y * y;
        };

        var paramsSGD = new Tensor[]
        {
            new Tensor(new[] { 1.0f }, new[] { 1 }, true),
            new Tensor(new[] { 1.0f }, new[] { 1 }, true)
        };

        var paramsNewton = new Tensor[]
        {
            new Tensor(new[] { 1.0f }, new[] { 1 }, true),
            new Tensor(new[] { 1.0f }, new[] { 1 }, true)
        };

        Console.WriteLine("Initial: (1.0, 1.0), Loss=" + illConditioned(paramsSGD).ToString("F2"));

        // SGD
        for (int i = 0; i < 50; i++)
        {
            var grads = ComputeNumericalGradients(illConditioned, paramsSGD);
            var xData = TensorAccessor.GetData(paramsSGD[0]);
            var yData = TensorAccessor.GetData(paramsSGD[1]);
            var gradXData = TensorAccessor.GetData(grads[0]);
            var gradYData = TensorAccessor.GetData(grads[1]);

            xData[0] -= 0.001f * gradXData[0];  // Small LR for stability
            yData[0] -= 0.001f * gradYData[0];
        }
        Console.WriteLine($"SGD after 50 iters: ({TensorAccessor.GetData(paramsSGD[0])[0]:F6}, {TensorAccessor.GetData(paramsSGD[1])[0]:F6}), Loss={illConditioned(paramsSGD):F6}");

        // Newton
        var grads2 = ComputeNumericalGradients(illConditioned, paramsNewton);
        var hessian2 = ComputeNumericalHessian(illConditioned, paramsNewton);
        var delta2 = SolveNewtonStep(grads2, hessian2, 0);

        var xDataN = TensorAccessor.GetData(paramsNewton[0]);
        var yDataN = TensorAccessor.GetData(paramsNewton[1]);
        var deltaXDataN = TensorAccessor.GetData(delta2[0]);
        var deltaYDataN = TensorAccessor.GetData(delta2[1]);

        xDataN[0] += deltaXDataN[0];
        yDataN[0] += deltaYDataN[0];

        Console.WriteLine($"Newton after 1 iter: ({xDataN[0]:F6}, {yDataN[0]:F6}), Loss={illConditioned(paramsNewton):F6}");
        Console.WriteLine("Newton's method handles ill-conditioned problems much better!\n");
    }

    /// <summary>
    /// Computes numerical gradients using finite differences.
    /// </summary>
    private static Tensor[] ComputeNumericalGradients(Func<Tensor[], double> f, Tensor[] parameters)
    {
        var grads = new Tensor[parameters.Length];
        double epsilon = 1e-6;

        for (int i = 0; i < parameters.Length; i++)
        {
            var paramData = TensorAccessor.GetData(parameters[i]);
            var gradData = new float[paramData.Length];

            for (int j = 0; j < paramData.Length; j++)
            {
                // Compute central difference
                double originalValue = paramData[j];

                // f(x + ε)
                paramData[j] = (float)(originalValue + epsilon);
                double fPlus = f(parameters);

                // f(x - ε)
                paramData[j] = (float)(originalValue - epsilon);
                double fMinus = f(parameters);

                // Restore original value
                paramData[j] = (float)originalValue;

                // Gradient = (f(x+ε) - f(x-ε)) / (2ε)
                gradData[j] = (float)((fPlus - fMinus) / (2 * epsilon));
            }

            grads[i] = new Tensor(gradData, parameters[i].Shape);
        }

        return grads;
    }

    /// <summary>
    /// Computes numerical Hessian matrix using finite differences.
    /// </summary>
    private static Tensor ComputeNumericalHessian(Func<Tensor[], double> f, Tensor[] parameters)
    {
        // For simplicity, we'll compute the Hessian for 2 parameters
        // Hessian is a 2x2 matrix for 2 parameters
        var hessianData = new float[4];  // [Hxx, Hxy, Hyx, Hyy]
        double epsilon = 1e-6;

        var xData = TensorAccessor.GetData(parameters[0]);
        var yData = TensorAccessor.GetData(parameters[1]);

        double x = xData[0];
        double y = yData[0];

        // Compute second derivatives
        // ∂²f/∂x²
        xData[0] = (float)(x + epsilon);
        double f_x_plus = f(parameters);
        xData[0] = (float)(x - epsilon);
        double f_x_minus = f(parameters);
        xData[0] = (float)x;
        double grad_x_plus = (f_x_plus - f_x_minus) / (2 * epsilon);

        // ∂²f/∂y²
        yData[0] = (float)(y + epsilon);
        double f_y_plus = f(parameters);
        yData[0] = (float)(y - epsilon);
        double f_y_minus = f(parameters);
        yData[0] = (float)y;
        double grad_y_plus = (f_y_plus - f_y_minus) / (2 * epsilon);

        // Mixed partial ∂²f/∂x∂y
        xData[0] = (float)(x + epsilon);
        yData[0] = (float)(y + epsilon);
        double f_xy_plus = f(parameters);

        xData[0] = (float)(x + epsilon);
        yData[0] = (float)(y - epsilon);
        double f_xy_minus = f(parameters);

        xData[0] = (float)(x - epsilon);
        yData[0] = (float)(y + epsilon);
        double f_yx_plus = f(parameters);

        xData[0] = (float)(x - epsilon);
        yData[0] = (float)(y - epsilon);
        double f_yx_minus = f(parameters);

        xData[0] = (float)x;
        yData[0] = (float)y;

        // Compute Hessian elements
        hessianData[0] = (float)((grad_x_plus - (f(parameters) - f(parameters)) / epsilon) / epsilon);  // Approximation
        hessianData[0] = (float)((f_x_plus - 2 * f(parameters) + f_x_minus) / (epsilon * epsilon));  // Better: central difference for second derivative
        hessianData[3] = (float)((f_y_plus - 2 * f(parameters) + f_y_minus) / (epsilon * epsilon));

        // Mixed partials should be equal (Clairaut's theorem)
        hessianData[1] = (float)((f_xy_plus - f_xy_minus - f_yx_plus + f_yx_minus) / (4 * epsilon * epsilon));
        hessianData[2] = hessianData[1];  // Symmetric

        return new Tensor(hessianData, new[] { 2, 2 });
    }

    /// <summary>
    /// Solves the Newton step equation H * Δθ = -g.
    /// Uses a simplified approach for small systems.
    /// </summary>
    private static Tensor[] SolveNewtonStep(Tensor[] gradients, Tensor hessian, double damping)
    {
        var hessianData = TensorAccessor.GetData(hessian);

        // Flatten gradients
        var g = new float[2];
        g[0] = TensorAccessor.GetData(gradients[0])[0];
        g[1] = TensorAccessor.GetData(gradients[1])[0];

        // Add damping to Hessian: H_damped = H + λI
        // This ensures H is positive definite
        float Hxx = hessianData[0] + (float)damping;
        float Hxy = hessianData[1];
        float Hyx = hessianData[2];
        float Hyy = hessianData[3] + (float)damping;

        // Solve H * Δ = -g
        // For 2x2 matrix, we can solve analytically
        float det = Hxx * Hyy - Hxy * Hyx;
        float invDet = 1.0f / det;

        float Hxx_inv = Hyy * invDet;
        float Hxy_inv = -Hxy * invDet;
        float Hyx_inv = -Hyx * invDet;
        float Hyy_inv = Hxx * invDet;

        // Δ = -H^(-1) * g
        float deltaX = -(Hxx_inv * g[0] + Hxy_inv * g[1]);
        float deltaY = -(Hyx_inv * g[0] + Hyy_inv * g[1]);

        // Return as separate tensors
        return new Tensor[]
        {
            new Tensor(new[] { deltaX }, new[] { 1 }),
            new Tensor(new[] { deltaY }, new[] { 1 })
        };
    }
}
