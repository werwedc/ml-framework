using System;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;

namespace MLFramework.Examples.HigherOrderDerivatives;

/// <summary>
/// Neural Ordinary Differential Equations (Neural ODEs) Example.
///
/// Neural ODEs parameterize the derivative of a hidden state with a neural network:
///   dh/dt = f(h(t), t, θ)
///
/// The dynamics are solved using ODE solvers, and the network parameters θ are learned
/// by backpropagating through the solver using the adjoint sensitivity method.
///
/// Key Concepts:
/// - ODE Solver: Numerical integration (e.g., Runge-Kutta, Euler)
/// - Adjoint Sensitivity Method: Efficient gradient computation for ODEs
/// - Higher-order derivatives: Solver accuracy depends on derivative approximations
///
/// Mathematical Formulation:
///   Forward pass: h(t1) = h(t0) + ∫_{t0}^{t1} f(h(t), t, θ) dt
///   Loss: L = ||h(t1) - target||²
///   Gradient: ∂L/∂θ = -∫_{t1}^{t0} a(t)ᵀ ∂f/∂θ dt
///   where a(t) = ∂L/∂h(t) is the adjoint state
///
/// Advantages:
/// - Memory efficient: No need to store intermediate activations
/// - Adaptive computation: ODE solver can adjust step size
/// - Continuous depth models: Not tied to discrete layers
///
/// Disadvantages:
/// - Slower forward pass: Numerical integration is expensive
/// - Complex gradient computation: Requires adjoint method
/// - Solver accuracy affects model quality
/// </summary>
public class NeuralODEExample
{
    /// <summary>
    /// Simple neural network to parameterize the dynamics.
    /// In practice, this would be a more complex network.
    /// </summary>
    private class DynamicsNetwork
    {
        private Tensor _weight1;
        private Tensor _bias1;
        private Tensor _weight2;
        private Tensor _bias2;

        public DynamicsNetwork(int inputDim, int hiddenDim)
        {
            // Initialize network parameters
            var rand = new Random(42);
            _weight1 = RandomTensor(inputDim, hiddenDim, rand);
            _bias1 = new Tensor(new float[hiddenDim], new[] { hiddenDim }, true);
            _weight2 = RandomTensor(hiddenDim, inputDim, rand);
            _bias2 = new Tensor(new float[inputDim], new[] { inputDim }, true);
        }

        /// <summary>
        /// Forward pass through the dynamics network.
        /// dh/dt = f(h, t, θ)
        /// </summary>
        public Tensor Forward(Tensor state, double time)
        {
            // Simple MLP: h -> W1 -> ReLU -> W2 -> output
            var hidden = LinearLayer(state, _weight1, _bias1);
            var activated = ReLU(hidden);
            var output = LinearLayer(activated, _weight2, _bias2);

            return output;
        }

        /// <summary>
        /// Returns trainable parameters.
        /// </summary>
        public Tensor[] GetParameters()
        {
            return new Tensor[] { _weight1, _bias1, _weight2, _bias2 };
        }
    }

    /// <summary>
    /// Runs the Neural ODE example.
    /// </summary>
    public static void Run()
    {
        Console.WriteLine("=== Neural ODE Example ===\n");

        // Configuration
        int stateDim = 2;  // 2D state
        int hiddenDim = 10;  // Hidden layer size
        double t0 = 0.0;  // Start time
        double t1 = 1.0;  // End time
        int solverSteps = 100;  // Number of solver steps

        Console.WriteLine($"Problem: Learn dynamics to transform state from t={t0} to t={t1}");
        Console.WriteLine($"State dimension: {stateDim}, Hidden dimension: {hiddenDim}\n");

        // Initialize dynamics network
        var dynamics = new DynamicsNetwork(stateDim, hiddenDim);
        Console.WriteLine("Initialized dynamics network\n");

        // Generate training data: Simple oscillatory dynamics
        Console.WriteLine("Generating training data...");
        var initialState = Tensor.FromArray(new[] { 1.0f, 0.0f });
        var targetState = GenerateTargetTrajectory(initialState, t0, t1, solverSteps);
        Console.WriteLine($"Initial state: [{TensorAccessor.GetData(initialState)[0]:F3}, {TensorAccessor.GetData(initialState)[1]:F3}]");
        Console.WriteLine($"Target state: [{TensorAccessor.GetData(targetState)[0]:F3}, {TensorAccessor.GetData(targetState)[1]:F3}]\n");

        // --- ODE Solver Examples ---
        Console.WriteLine("=== Testing ODE Solvers ===\n");

        // Test different ODE solvers
        Console.WriteLine("1. Euler's Method (1st order):");
        var eulerFinal = SolveODE(dynamics, initialState, t0, t1, solverSteps, "euler");
        Console.WriteLine($"   Final state: [{TensorAccessor.GetData(eulerFinal)[0]:F3}, {TensorAccessor.GetData(eulerFinal)[1]:F3}]\n");

        Console.WriteLine("2. Runge-Kutta 4th Order (RK4):");
        var rk4Final = SolveODE(dynamics, initialState, t0, t1, solverSteps, "rk4");
        Console.WriteLine($"   Final state: [{TensorAccessor.GetData(rk4Final)[0]:F3}, {TensorAccessor.GetData(rk4Final)[1]:F3}]\n");

        Console.WriteLine("3. Adaptive Step Size RK4:");
        var adaptiveFinal = SolveODEAdaptive(dynamics, initialState, t0, t1, 1e-6, 1e-3);
        Console.WriteLine($"   Final state: [{TensorAccessor.GetData(adaptiveFinal)[0]:F3}, {TensorAccessor.GetData(adaptiveFinal)[1]:F3}]\n");

        // --- Training the Neural ODE ---
        Console.WriteLine("=== Training Neural ODE ===\n");

        int numIterations = 100;
        double learningRate = 0.01;

        Console.WriteLine($"Training for {numIterations} iterations with LR={learningRate}...\n");

        for (int iter = 0; iter < numIterations; iter++)
        {
            // Forward pass: Solve ODE
            var predictedState = SolveODE(dynamics, initialState, t0, t1, solverSteps, "rk4");

            // Compute loss (MSE)
            var loss = ComputeMSE(predictedState, targetState);

            // Compute gradients (simplified - in practice, use adjoint method)
            var grads = ComputeGradientsNumerically(dynamics, initialState, t0, t1, solverSteps, targetState);

            // Update parameters
            UpdateParameters(dynamics, grads, learningRate);

            // Print progress
            if (iter % 10 == 0)
            {
                Console.WriteLine($"Iteration {iter}: Loss = {loss:F6}, State = [{TensorAccessor.GetData(predictedState)[0]:F3}, {TensorAccessor.GetData(predictedState)[1]:F3}]");
            }
        }

        // Final evaluation
        var finalPredicted = SolveODE(dynamics, initialState, t0, t1, solverSteps, "rk4");
        var finalLoss = ComputeMSE(finalPredicted, targetState);

        Console.WriteLine($"\nFinal Loss: {finalLoss:F6}");
        Console.WriteLine($"Final State: [{TensorAccessor.GetData(finalPredicted)[0]:F3}, {TensorAccessor.GetData(finalPredicted)[1]:F3}]");
        Console.WriteLine($"Target State: [{TensorAccessor.GetData(targetState)[0]:F3}, {TensorAccessor.GetData(targetState)[1]:F3}]\n");

        Console.WriteLine("=== Neural ODE Example completed successfully! ===\n");
    }

    /// <summary>
    /// Solves the ODE using the specified solver.
    /// </summary>
    private static Tensor SolveODE(DynamicsNetwork dynamics, Tensor initialState, double t0, double t1, int steps, string method)
    {
        var dt = (t1 - t0) / steps;
        var state = initialState.Clone();

        for (int i = 0; i < steps; i++)
        {
            var t = t0 + i * dt;

            switch (method.ToLower())
            {
                case "euler":
                    state = EulerStep(dynamics, state, t, dt);
                    break;
                case "rk4":
                    state = RK4Step(dynamics, state, t, dt);
                    break;
                default:
                    throw new ArgumentException($"Unknown solver: {method}");
            }
        }

        return state;
    }

    /// <summary>
    /// Solves the ODE using adaptive step size.
    /// </summary>
    private static Tensor SolveODEAdaptive(DynamicsNetwork dynamics, Tensor initialState, double t0, double t1, double tol, double maxStep)
    {
        var state = initialState.Clone();
        var t = t0;
        double dt = maxStep;

        int steps = 0;

        while (t < t1 && steps < 10000)
        {
            // Take two half steps
            var halfStep1 = RK4Step(dynamics, state, t, dt / 2.0);
            var halfStep2 = RK4Step(dynamics, halfStep1, t + dt / 2.0, dt / 2.0);

            // Take one full step
            var fullStep = RK4Step(dynamics, state, t, dt);

            // Estimate error
            var error = ComputeError(halfStep2, fullStep);

            // Adjust step size
            if (error < tol)
            {
                // Accept step
                state = halfStep2;
                t += dt;
                // Increase step size
                dt = Math.Min(dt * 1.5, maxStep);
            }
            else
            {
                // Reject step, decrease step size
                dt *= 0.5;
            }

            steps++;
        }

        Console.WriteLine($"   Used {steps} adaptive steps");
        return state;
    }

    /// <summary>
    /// Euler's method step (1st order).
    /// y_{n+1} = y_n + h * f(y_n, t_n)
    /// </summary>
    private static Tensor EulerStep(DynamicsNetwork dynamics, Tensor state, double t, double dt)
    {
        var derivative = dynamics.Forward(state, t);
        return Add(state, Scale(derivative, (float)dt));
    }

    /// <summary>
    /// Runge-Kutta 4th order step (RK4).
    /// k1 = f(y_n, t_n)
    /// k2 = f(y_n + h/2*k1, t_n + h/2)
    /// k3 = f(y_n + h/2*k2, t_n + h/2)
    /// k4 = f(y_n + h*k3, t_n + h)
    /// y_{n+1} = y_n + h/6*(k1 + 2*k2 + 2*k3 + k4)
    /// </summary>
    private static Tensor RK4Step(DynamicsNetwork dynamics, Tensor state, double t, double dt)
    {
        // k1
        var k1 = dynamics.Forward(state, t);

        // k2
        var y2 = Add(state, Scale(k1, (float)(dt / 2.0)));
        var k2 = dynamics.Forward(y2, t + dt / 2.0);

        // k3
        var y3 = Add(state, Scale(k2, (float)(dt / 2.0)));
        var k3 = dynamics.Forward(y3, t + dt / 2.0);

        // k4
        var y4 = Add(state, Scale(k3, (float)dt));
        var k4 = dynamics.Forward(y4, t + dt);

        // Combine
        var increment = Add(
            Add(
                k1,
                Scale(k2, 2.0f)
            ),
            Add(
                Scale(k3, 2.0f),
                k4
            )
        );

        return Add(state, Scale(increment, (float)(dt / 6.0)));
    }

    /// <summary>
    /// Computes gradients numerically (simplified).
    /// In practice, use the adjoint sensitivity method for efficiency.
    /// </summary>
    private static Dictionary<string, Tensor> ComputeGradientsNumerically(
        DynamicsNetwork dynamics,
        Tensor initialState,
        double t0,
        double t1,
        int steps,
        Tensor targetState)
    {
        var grads = new Dictionary<string, Tensor>();
        double epsilon = 1e-4;

        var paramsDict = new Dictionary<string, Tensor>
        {
            { "weight1", dynamics.GetParameters()[0] },
            { "bias1", dynamics.GetParameters()[1] },
            { "weight2", dynamics.GetParameters()[2] },
            { "bias2", dynamics.GetParameters()[3] }
        };

        // Compute gradient for each parameter
        foreach (var kvp in paramsDict)
        {
            var paramName = kvp.Key;
            var param = kvp.Value;
            var paramData = TensorAccessor.GetData(param);
            var gradData = new float[paramData.Length];

            for (int i = 0; i < paramData.Length; i++)
            {
                // Save original value
                float original = paramData[i];

                // f(x + ε)
                paramData[i] = original + (float)epsilon;
                var predictionPlus = SolveODE(dynamics, initialState, t0, t1, steps, "rk4");
                double lossPlus = ComputeMSE(predictionPlus, targetState);

                // f(x - ε)
                paramData[i] = original - (float)epsilon;
                var predictionMinus = SolveODE(dynamics, initialState, t0, t1, steps, "rk4");
                double lossMinus = ComputeMSE(predictionMinus, targetState);

                // Restore original value
                paramData[i] = original;

                // Gradient = (f(x+ε) - f(x-ε)) / (2ε)
                gradData[i] = (float)((lossPlus - lossMinus) / (2 * epsilon));
            }

            grads[paramName] = new Tensor(gradData, param.Shape);
        }

        return grads;
    }

    /// <summary>
    /// Updates parameters using gradient descent.
    /// </summary>
    private static void UpdateParameters(DynamicsNetwork dynamics, Dictionary<string, Tensor> grads, double learningRate)
    {
        var paramsDict = new Dictionary<string, Tensor>
        {
            { "weight1", dynamics.GetParameters()[0] },
            { "bias1", dynamics.GetParameters()[1] },
            { "weight2", dynamics.GetParameters()[2] },
            { "bias2", dynamics.GetParameters()[3] }
        };

        foreach (var kvp in paramsDict)
        {
            var paramName = kvp.Key;
            var param = kvp.Value;

            if (grads.ContainsKey(paramName))
            {
                var paramData = TensorAccessor.GetData(param);
                var gradData = TensorAccessor.GetData(grads[paramName]);

                for (int i = 0; i < paramData.Length; i++)
                {
                    paramData[i] -= (float)learningRate * gradData[i];
                }
            }
        }
    }

    /// <summary>
    /// Generates a target trajectory using simple harmonic motion.
    /// h(t) = [cos(t), sin(t)]
    /// </summary>
    private static Tensor GenerateTargetTrajectory(Tensor initialState, double t0, double t1, int steps)
    {
        // Simple harmonic motion: [cos(t), sin(t)]
        var x = Math.Cos(t1);
        var y = Math.Sin(t1);
        return Tensor.FromArray(new[] { (float)x, (float)y });
    }

    /// <summary>
    /// Computes Mean Squared Error.
    /// </summary>
    private static double ComputeMSE(Tensor prediction, Tensor target)
    {
        var predData = TensorAccessor.GetData(prediction);
        var targetData = TensorAccessor.GetData(target);

        double sumSquaredError = 0;
        int count = Math.Min(predData.Length, targetData.Length);

        for (int i = 0; i < count; i++)
        {
            double error = predData[i] - targetData[i];
            sumSquaredError += error * error;
        }

        return sumSquaredError / count;
    }

    /// <summary>
    /// Computes error between two states.
    /// </summary>
    private static double ComputeError(Tensor state1, Tensor state2)
    {
        var data1 = TensorAccessor.GetData(state1);
        var data2 = TensorAccessor.GetData(state2);

        double sum = 0;
        int count = Math.Min(data1.Length, data2.Length);

        for (int i = 0; i < count; i++)
        {
            double diff = data1[i] - data2[i];
            sum += diff * diff;
        }

        return Math.Sqrt(sum);
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    private static Tensor Add(Tensor a, Tensor b)
    {
        var aData = TensorAccessor.GetData(a);
        var bData = TensorAccessor.GetData(b);
        var result = new float[Math.Min(aData.Length, bData.Length)];

        for (int i = 0; i < result.Length; i++)
        {
            result[i] = aData[i] + bData[i];
        }

        return new Tensor(result, a.Shape);
    }

    /// <summary>
    /// Scales a tensor by a scalar.
    /// </summary>
    private static Tensor Scale(Tensor tensor, float scalar)
    {
        var data = TensorAccessor.GetData(tensor);
        var result = new float[data.Length];

        for (int i = 0; i < data.Length; i++)
        {
            result[i] = data[i] * scalar;
        }

        return new Tensor(result, tensor.Shape);
    }

    /// <summary>
    /// Linear layer operation.
    /// </summary>
    private static Tensor LinearLayer(Tensor input, Tensor weight, Tensor bias)
    {
        var inputData = TensorAccessor.GetData(input);
        var weightData = TensorAccessor.GetData(weight);
        var biasData = TensorAccessor.GetData(bias);

        int inputDim = inputData.Length;
        int outputDim = biasData.Length;
        var result = new float[outputDim];

        for (int i = 0; i < outputDim; i++)
        {
            result[i] = biasData[i];
            for (int j = 0; j < inputDim; j++)
            {
                result[i] += inputData[j] * weightData[j * outputDim + i];
            }
        }

        return new Tensor(result, new[] { outputDim });
    }

    /// <summary>
    /// ReLU activation function.
    /// </summary>
    private static Tensor ReLU(Tensor input)
    {
        var inputData = TensorAccessor.GetData(input);
        var result = new float[inputData.Length];

        for (int i = 0; i < inputData.Length; i++)
        {
            result[i] = Math.Max(0, inputData[i]);
        }

        return new Tensor(result, input.Shape);
    }

    /// <summary>
    /// Creates a random tensor.
    /// </summary>
    private static Tensor RandomTensor(int rows, int cols, Random rand)
    {
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)(rand.NextDouble() * 2 - 1);  // Random in [-1, 1]
        }
        return new Tensor(data, new[] { rows, cols }, true);
    }
}
