using System;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;

namespace MLFramework.Examples.HigherOrderDerivatives;

/// <summary>
/// Sharpness Minimization Example.
///
/// Sharpness of minima refers to how sensitive the loss is to parameter perturbations.
/// Sharp minima (high curvature) generalize poorly, while flat minima generalize well.
///
/// Key Concepts:
/// - Hessian Eigenvalues: Large eigenvalues indicate sharp minima
/// - Sharpness-Aware Training (SAM): Explicitly minimizes sharpness
/// - Second-order information: Hessian reveals curvature of loss landscape
///
/// Mathematical Formulation:
/// Sharpness at parameters θ is measured by:
///   L_s(θ) = max_{||ε|| ≤ ρ} L(θ + ε)
///
/// The optimal perturbation is approximately:
///   ε* ≈ ρ * v / ||v||
/// where v is the eigenvector of the Hessian with the largest eigenvalue.
///
/// SAM training step:
///   1. Compute ascent direction: ε = argmax L(θ + ε)
///   2. Update θ ← θ - η ∇L(θ + ε*)
///
/// Advantages:
/// - Better generalization performance
/// - More robust to distribution shift
/// - Improved training stability
///
/// Disadvantages:
/// - Double the computational cost (two forward-backward passes per step)
/// - Requires Hessian information (or approximation)
/// - Hyperparameter tuning (ρ)
/// </summary>
public class SharpnessMinimizationExample
{
    /// <summary>
    /// Simple linear model for demonstration.
    /// </summary>
    private class LinearModel
    {
        private Tensor _weight;
        private Tensor _bias;

        public LinearModel(int inputDim)
        {
            var rand = new Random(42);
            _weight = RandomTensor(inputDim, rand);
            _bias = new Tensor(new[] { 0.0f }, new[] { 1 }, true);
        }

        public Tensor Forward(Tensor input)
        {
            var result = LinearLayer(input, _weight, _bias);
            return result;
        }

        public Tensor[] GetParameters()
        {
            return new Tensor[] { _weight, _bias };
        }

        public void SetParameters(Tensor[] newParams)
        {
            var srcData = TensorAccessor.GetData(newParams[0]);
            var dstData = TensorAccessor.GetData(_weight);
            Array.Copy(srcData, dstData, srcData.Length);

            var biasData = TensorAccessor.GetData(newParams[1]);
            var biasDstData = TensorAccessor.GetData(_bias);
            Array.Copy(biasData, biasDstData, biasData.Length);
        }
    }

    /// <summary>
    /// Runs the sharpness minimization example.
    /// </summary>
    public static void Run()
    {
        Console.WriteLine("=== Sharpness Minimization Example ===\n");

        // Configuration
        int inputDim = 2;
        int numSamples = 100;
        int numIterations = 100;
        double learningRate = 0.01;
        double sharpnessRadius = 0.1;  // ρ for SAM

        Console.WriteLine($"Model: Linear regression with {inputDim}D input");
        Console.WriteLine($"Training samples: {numSamples}");
        Console.WriteLine($"Iterations: {numIterations}");
        Console.WriteLine($"Learning rate: {learningRate}");
        Console.WriteLine($"Sharpness radius (ρ): {sharpnessRadius}\n");

        // Generate synthetic data
        Console.WriteLine("Generating synthetic data...");
        var (inputs, targets) = GenerateData(numSamples, inputDim);
        Console.WriteLine("Data generated\n");

        // --- Standard Training ---
        Console.WriteLine("=== Standard Training ===");
        var standardModel = new LinearModel(inputDim);
        TrainStandard(standardModel, inputs, targets, numIterations, learningRate);

        // --- Sharpness-Aware Training (SAM) ---
        Console.WriteLine("\n=== Sharpness-Aware Training (SAM) ===");
        var samModel = new LinearModel(inputDim);
        TrainSAM(samModel, inputs, targets, numIterations, learningRate, sharpnessRadius);

        // --- Comparison ---
        Console.WriteLine("\n=== Comparison ===");
        var testInputs = Tensor.FromArray(new float[] { 1.0f, 1.0f });
        var testTarget = Tensor.FromArray(new float[] { 3.0f });  // Target: 1 + 2*1 = 3

        var standardPred = standardModel.Forward(testInputs);
        var samPred = samModel.Forward(testInputs);

        var standardLoss = ComputeLoss(standardPred, testTarget);
        var samLoss = ComputeLoss(samPred, testTarget);

        Console.WriteLine($"Standard model prediction: {TensorAccessor.GetData(standardPred)[0]:F4} (loss: {standardLoss:F4})");
        Console.WriteLine($"SAM model prediction: {TensorAccessor.GetData(samPred)[0]:F4} (loss: {samLoss:F4})");

        // Evaluate sharpness
        var standardSharpness = EvaluateSharpness(standardModel, inputs, targets, sharpnessRadius);
        var samSharpness = EvaluateSharpness(samModel, inputs, targets, sharpnessRadius);

        Console.WriteLine($"\nSharpness analysis:");
        Console.WriteLine($"Standard model sharpness: {standardSharpness:F6}");
        Console.WriteLine($"SAM model sharpness: {samSharpness:F6}");
        Console.WriteLine($"Sharpness reduction: {(1 - samSharpness / standardSharpness) * 100:F2}%\n");

        Console.WriteLine("=== Sharpness Minimization Example completed successfully! ===\n");
    }

    /// <summary>
    /// Standard training without sharpness awareness.
    /// </summary>
    private static void TrainStandard(LinearModel model, Tensor inputs, Tensor targets, int iterations, double lr)
    {
        Console.WriteLine("Training standard model...");

        for (int iter = 0; iter < iterations; iter++)
        {
            // Forward pass
            var predictions = model.Forward(inputs);
            var loss = ComputeLoss(predictions, targets);

            // Compute gradients
            var grads = ComputeGradients(model, inputs, targets);

            // Update parameters
            UpdateParameters(model, grads, lr);

            // Print progress
            if (iter % 20 == 0)
            {
                Console.WriteLine($"  Iteration {iter}: Loss = {loss:F6}");
            }
        }

        var finalPred = model.Forward(inputs);
        var finalLoss = ComputeLoss(finalPred, targets);
        Console.WriteLine($"  Final loss: {finalLoss:F6}\n");
    }

    /// <summary>
    /// Sharpness-Aware Training (SAM).
    /// </summary>
    private static void TrainSAM(LinearModel model, Tensor inputs, Tensor targets, int iterations, double lr, double rho)
    {
        Console.WriteLine("Training SAM model...");

        for (int iter = 0; iter < iterations; iter++)
        {
            // Step 1: Find perturbation that maximizes loss (sharpness direction)
            var epsilon = FindSharpnessDirection(model, inputs, targets, rho);

            // Step 2: Perturb parameters
            var paramsCopy = CloneParameters(model.GetParameters());
            PerturbParameters(model, epsilon, rho);

            // Step 3: Compute gradients at perturbed parameters
            var gradsAtPerturbed = ComputeGradients(model, inputs, targets);

            // Step 4: Restore original parameters
            model.SetParameters(paramsCopy);

            // Step 5: Update original parameters using gradients from perturbed position
            UpdateParameters(model, gradsAtPerturbed, lr);

            // Print progress
            var loss = ComputeLoss(model.Forward(inputs), targets);
            if (iter % 20 == 0)
            {
                Console.WriteLine($"  Iteration {iter}: Loss = {loss:F6}");
            }
        }

        var finalPred = model.Forward(inputs);
        var finalLoss = ComputeLoss(finalPred, targets);
        Console.WriteLine($"  Final loss: {finalLoss:F6}\n");
    }

    /// <summary>
    /// Finds the direction of maximum loss (sharpness direction).
    /// Uses power iteration to approximate the top Hessian eigenvector.
    /// </summary>
    private static Tensor FindSharpnessDirection(LinearModel model, Tensor inputs, Tensor targets, double rho)
    {
        // Simplified: Use gradient direction as approximation
        // In practice, we would compute the top Hessian eigenvector
        var grads = ComputeGradients(model, inputs, targets);

        // Flatten and normalize
        var weightGrad = grads[0];
        var biasGrad = grads[1];

        var weightGradData = TensorAccessor.GetData(weightGrad);
        var biasGradData = TensorAccessor.GetData(biasGrad);

        // Compute norm
        float norm = 0;
        foreach (var val in weightGradData)
        {
            norm += val * val;
        }
        norm += biasGradData[0] * biasGradData[0];
        norm = (float)Math.Sqrt(norm);

        // Normalize to unit length, then scale by rho
        if (norm > 1e-10)
        {
            for (int i = 0; i < weightGradData.Length; i++)
            {
                weightGradData[i] = weightGradData[i] / norm * (float)rho;
            }
            biasGradData[0] = biasGradData[0] / norm * (float)rho;
        }

        return weightGrad;
    }

    /// <summary>
    /// Perturbs parameters in the specified direction.
    /// </summary>
    private static void PerturbParameters(LinearModel model, Tensor epsilon, double rho)
    {
        var paramsList = model.GetParameters();
        var weightParams = paramsList[0];
        var weightData = TensorAccessor.GetData(weightParams);
        var epsilonData = TensorAccessor.GetData(epsilon);

        for (int i = 0; i < weightData.Length; i++)
        {
            weightData[i] += epsilonData[i];
        }
    }

    /// <summary>
    /// Clones parameters.
    /// </summary>
    private static Tensor[] CloneParameters(Tensor[] parameters)
    {
        var clones = new Tensor[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            var data = TensorAccessor.GetData(parameters[i]);
            var copyData = new float[data.Length];
            Array.Copy(data, copyData, data.Length);
            clones[i] = new Tensor(copyData, parameters[i].Shape);
        }
        return clones;
    }

    /// <summary>
    /// Evaluates sharpness of a model's parameters.
    /// Measures maximum loss increase under perturbation of size ρ.
    /// </summary>
    private static double EvaluateSharpness(LinearModel model, Tensor inputs, Tensor targets, double rho)
    {
        // Save original parameters
        var originalParams = CloneParameters(model.GetParameters());

        // Compute loss at current parameters
        var currentLoss = ComputeLoss(model.Forward(inputs), targets);

        // Find direction of maximum increase
        var epsilon = FindSharpnessDirection(model, inputs, targets, rho);

        // Perturb parameters
        PerturbParameters(model, epsilon, rho);

        // Compute loss at perturbed parameters
        var perturbedLoss = ComputeLoss(model.Forward(inputs), targets);

        // Restore original parameters
        model.SetParameters(originalParams);

        // Sharpness = maximum loss increase
        return perturbedLoss - currentLoss;
    }

    /// <summary>
    /// Computes loss (Mean Squared Error).
    /// </summary>
    private static double ComputeLoss(Tensor predictions, Tensor targets)
    {
        var predData = TensorAccessor.GetData(predictions);
        var targetData = TensorAccessor.GetData(targets);

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
    /// Computes gradients using numerical differentiation.
    /// </summary>
    private static Tensor[] ComputeGradients(LinearModel model, Tensor inputs, Tensor targets)
    {
        var parameters = model.GetParameters();
        var grads = new Tensor[parameters.Length];
        double epsilon = 1e-6;

        for (int i = 0; i < parameters.Length; i++)
        {
            var param = parameters[i];
            var paramData = TensorAccessor.GetData(param);
            var gradData = new float[paramData.Length];

            for (int j = 0; j < paramData.Length; j++)
            {
                // Save original value
                float original = paramData[j];

                // f(x + ε)
                paramData[j] = original + (float)epsilon;
                var predPlus = model.Forward(inputs);
                double lossPlus = ComputeLoss(predPlus, targets);

                // f(x - ε)
                paramData[j] = original - (float)epsilon;
                var predMinus = model.Forward(inputs);
                double lossMinus = ComputeLoss(predMinus, targets);

                // Restore original value
                paramData[j] = original;

                // Gradient
                gradData[j] = (float)((lossPlus - lossMinus) / (2 * epsilon));
            }

            grads[i] = new Tensor(gradData, param.Shape);
        }

        return grads;
    }

    /// <summary>
    /// Updates parameters using gradient descent.
    /// </summary>
    private static void UpdateParameters(LinearModel model, Tensor[] grads, double lr)
    {
        var parameters = model.GetParameters();

        for (int i = 0; i < parameters.Length; i++)
        {
            var paramData = TensorAccessor.GetData(parameters[i]);
            var gradData = TensorAccessor.GetData(grads[i]);

            for (int j = 0; j < paramData.Length; j++)
            {
                paramData[j] -= (float)lr * gradData[j];
            }
        }
    }

    /// <summary>
    /// Generates synthetic linear regression data.
    /// </summary>
    private static (Tensor inputs, Tensor targets) GenerateData(int numSamples, int inputDim)
    {
        var rand = new Random(42);
        var inputData = new float[numSamples * inputDim];
        var targetData = new float[numSamples];

        // True weights: [1.0, 2.0] and bias: 0.0
        float[] trueWeights = { 1.0f, 2.0f };
        float trueBias = 0.0f;

        for (int i = 0; i < numSamples; i++)
        {
            // Random inputs
            for (int j = 0; j < inputDim; j++)
            {
                inputData[i * inputDim + j] = (float)(rand.NextDouble() * 2 - 1);
            }

            // Compute target with noise
            float sum = trueBias;
            for (int j = 0; j < inputDim; j++)
            {
                sum += inputData[i * inputDim + j] * trueWeights[j];
            }
            sum += (float)(rand.NextDouble() * 0.1 - 0.05);  // Add small noise

            targetData[i] = sum;
        }

        var inputs = new Tensor(inputData, new[] { numSamples, inputDim });
        var targets = new Tensor(targetData, new[] { numSamples });

        return (inputs, targets);
    }

    /// <summary>
    /// Linear layer operation.
    /// </summary>
    private static Tensor LinearLayer(Tensor input, Tensor weight, Tensor bias)
    {
        var inputData = TensorAccessor.GetData(input);
        var weightData = TensorAccessor.GetData(weight);
        var biasData = TensorAccessor.GetData(bias);

        int numSamples = input.Shape[0];
        int outputDim = biasData.Length;
        var result = new float[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            result[i] = biasData[0];
            for (int j = 0; j < weightData.Length; j++)
            {
                result[i] += inputData[i * weightData.Length + j] * weightData[j];
            }
        }

        return new Tensor(result, new[] { numSamples });
    }

    /// <summary>
    /// Creates a random tensor.
    /// </summary>
    private static Tensor RandomTensor(int size, Random rand)
    {
        var data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = (float)(rand.NextDouble() * 0.2 - 0.1);
        }
        return new Tensor(data, new[] { size }, true);
    }
}
