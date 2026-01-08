using System;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;

namespace MLFramework.Examples.HigherOrderDerivatives;

/// <summary>
/// Adversarial Robustness Example.
///
/// Adversarial examples are inputs with small, imperceptible perturbations that cause
/// models to make incorrect predictions. Understanding curvature (Hessian) helps
/// generate and defend against such attacks.
///
/// Key Concepts:
/// - Adversarial Attacks: Crafting inputs to maximize loss
/// - Curvature: Hessian reveals input sensitivity directions
/// - PGD (Projected Gradient Descent): Iterative attack method
/// - Fast Gradient Sign Method (FGSM): Single-step attack
///
/// Mathematical Formulation:
/// Adversarial attack solves:
///   max_{||δ|| ≤ ε} L(f(x+δ), y)
///
/// where:
///   - δ is the perturbation
///   - ε is the maximum allowed perturbation (e.g., 0.01 for pixel values)
///   - L is the loss function
///   - f is the model
///   - x is the input, y is the true label
///
/// Curvature-based attack:
///   Use Hessian eigenvectors to find most sensitive directions
///   δ = ε * v_top  (where v_top is top Hessian eigenvector)
///
/// Advantages of using Hessian:
/// - More effective attacks (higher success rate)
/// - Fewer iterations needed
/// - Better understanding of model vulnerabilities
///
/// Disadvantages:
/// - Computing Hessian is expensive for large inputs
/// - Requires more complex implementation
/// - May be slower than gradient-based methods
/// </summary>
public class AdversarialRobustnessExample
{
    /// <summary>
    /// Simple neural network for binary classification.
    /// </summary>
    private class BinaryClassifier
    {
        private Tensor _weight1;
        private Tensor _bias1;
        private Tensor _weight2;
        private Tensor _bias2;

        public BinaryClassifier(int inputDim, int hiddenDim)
        {
            var rand = new Random(42);
            _weight1 = RandomTensor(inputDim, hiddenDim, rand);
            _bias1 = new Tensor(new float[hiddenDim], new[] { hiddenDim }, true);
            _weight2 = RandomTensor(hiddenDim, 1, rand);
            _bias2 = new Tensor(new[] { 0.0f }, new[] { 1 }, true);
        }

        public Tensor Forward(Tensor input)
        {
            var hidden = LinearLayer(input, _weight1, _bias1);
            var activated = ReLU(hidden);
            var output = LinearLayer(activated, _weight2, _bias2);
            var probs = Sigmoid(output);
            return probs;
        }

        public Tensor[] GetParameters()
        {
            return new Tensor[] { _weight1, _bias1, _weight2, _bias2 };
        }
    }

    /// <summary>
    /// Runs the adversarial robustness example.
    /// </summary>
    public static void Run()
    {
        Console.WriteLine("=== Adversarial Robustness Example ===\n");

        // Configuration
        int inputDim = 10;
        int hiddenDim = 5;
        int numSamples = 100;
        int trainingIterations = 200;
        double trainingLR = 0.01;

        Console.WriteLine($"Model: Binary classifier ({inputDim} -> {hiddenDim} -> 1)");
        Console.WriteLine($"Training samples: {numSamples}");
        Console.WriteLine($"Training iterations: {trainingIterations}\n");

        // Generate synthetic data
        Console.WriteLine("Generating synthetic data...");
        var (inputs, labels) = GenerateData(numSamples, inputDim);
        Console.WriteLine("Data generated\n");

        // Train model
        Console.WriteLine("=== Training Model ===");
        var model = new BinaryClassifier(inputDim, hiddenDim);
        TrainModel(model, inputs, labels, trainingIterations, trainingLR);

        // Select a test sample
        var testInput = Tensor.FromArray(new float[] { 1.0f, 0.5f, -0.3f, 0.8f, -0.2f, 0.1f, -0.5f, 0.9f, 0.3f, -0.7f });
        var testLabel = Tensor.FromArray(new[] { 1.0f });

        // Clean prediction
        Console.WriteLine("=== Clean Prediction ===");
        var cleanPred = model.Forward(testInput);
        var cleanProb = TensorAccessor.GetData(cleanPred)[0];
        var cleanLoss = ComputeBinaryCrossEntropy(cleanPred, testLabel);
        var cleanPrediction = cleanProb > 0.5 ? "Class 1" : "Class 0";

        Console.WriteLine($"Input: [{string.Join(", ", TensorAccessor.GetData(testInput).Select(x => x.ToString("F2")))}]");
        Console.WriteLine($"True label: {TensorAccessor.GetData(testLabel)[0]} ({(TensorAccessor.GetData(testLabel)[0] > 0.5 ? "Class 1" : "Class 0")})");
        Console.WriteLine($"Predicted probability: {cleanProb:F4}");
        Console.WriteLine($"Prediction: {cleanPrediction}");
        Console.WriteLine($"Loss: {cleanLoss:F6}\n");

        // --- Adversarial Attacks ---
        Console.WriteLine("=== Adversarial Attacks ===\n");

        // FGSM Attack
        Console.WriteLine("1. Fast Gradient Sign Method (FGSM):");
        var fgsmEpsilon = 0.1;
        var fgsmAttack = FGSMAttack(model, testInput, testLabel, fgsmEpsilon);
        var fgsmPred = model.Forward(fgsmAttack);
        var fgsmProb = TensorAccessor.GetData(fgsmPred)[0];
        var fgsmPrediction = fgsmProb > 0.5 ? "Class 1" : "Class 0";

        Console.WriteLine($"Epsilon: {fgsmEpsilon}");
        Console.WriteLine($"Perturbation: [{string.Join(", ", TensorAccessor.GetData(Subtract(fgsmAttack, testInput)).Select(x => x.ToString("F3")))}]");
        Console.WriteLine($"Perturbation norm: {ComputeNorm(Subtract(fgsmAttack, testInput)):F4}");
        Console.WriteLine($"Predicted probability: {fgsmProb:F4}");
        Console.WriteLine($"Prediction: {fgsmPrediction}");
        Console.WriteLine($"Attack successful: {cleanPrediction != fgsmPrediction}\n");

        // PGD Attack
        Console.WriteLine("2. Projected Gradient Descent (PGD):");
        var pgdEpsilon = 0.1;
        var pgdSteps = 10;
        var pgdStepSize = 0.01;
        var pgdAttack = PGDAttack(model, testInput, testLabel, pgdEpsilon, pgdSteps, pgdStepSize);
        var pgdPred = model.Forward(pgdAttack);
        var pgdProb = TensorAccessor.GetData(pgdPred)[0];
        var pgdPrediction = pgdProb > 0.5 ? "Class 1" : "Class 0";

        Console.WriteLine($"Epsilon: {pgdEpsilon}, Steps: {pgdSteps}, Step size: {pgdStepSize}");
        Console.WriteLine($"Perturbation norm: {ComputeNorm(Subtract(pgdAttack, testInput)):F4}");
        Console.WriteLine($"Predicted probability: {pgdProb:F4}");
        Console.WriteLine($"Prediction: {pgdPrediction}");
        Console.WriteLine($"Attack successful: {cleanPrediction != pgdPrediction}\n");

        // Curvature-based Attack (using Hessian)
        Console.WriteLine("3. Curvature-based Attack (Hessian):");
        var hessianEpsilon = 0.1;
        var hessianAttack = HessianAttack(model, testInput, testLabel, hessianEpsilon);
        var hessianPred = model.Forward(hessianAttack);
        var hessianProb = TensorAccessor.GetData(hessianPred)[0];
        var hessianPrediction = hessianProb > 0.5 ? "Class 1" : "Class 0";

        Console.WriteLine($"Epsilon: {hessianEpsilon}");
        Console.WriteLine($"Perturbation norm: {ComputeNorm(Subtract(hessianAttack, testInput)):F4}");
        Console.WriteLine($"Predicted probability: {hessianProb:F4}");
        Console.WriteLine($"Prediction: {hessianPrediction}");
        Console.WriteLine($"Attack successful: {cleanPrediction != hessianPrediction}\n");

        // --- Comparison ---
        Console.WriteLine("=== Attack Comparison ===");
        Console.WriteLine($"Clean prediction: {cleanPrediction} (prob: {cleanProb:F4})");
        Console.WriteLine($"FGSM prediction: {fgsmPrediction} (prob: {fgsmProb:F4})");
        Console.WriteLine($"PGD prediction: {pgdPrediction} (prob: {pgdProb:F4})");
        Console.WriteLine($"Hessian prediction: {hessianPrediction} (prob: {hessianProb:F4})\n");

        Console.WriteLine("=== Adversarial Robustness Example completed successfully! ===\n");
    }

    /// <summary>
    /// Trains the model using gradient descent.
    /// </summary>
    private static void TrainModel(BinaryClassifier model, Tensor inputs, Tensor labels, int iterations, double lr)
    {
        for (int iter = 0; iter < iterations; iter++)
        {
            // Forward pass
            var predictions = model.Forward(inputs);
            var loss = ComputeBinaryCrossEntropy(predictions, labels);

            // Compute gradients (numerical for simplicity)
            var grads = ComputeGradients(model, inputs, labels);

            // Update parameters
            UpdateParameters(model, grads, lr);

            // Print progress
            if (iter % 50 == 0)
            {
                var accuracy = ComputeAccuracy(model, inputs, labels);
                Console.WriteLine($"  Iteration {iter}: Loss = {loss:F6}, Accuracy = {accuracy * 100:F1}%");
            }
        }

        var finalPred = model.Forward(inputs);
        var finalLoss = ComputeBinaryCrossEntropy(finalPred, labels);
        var finalAccuracy = ComputeAccuracy(model, inputs, labels);
        Console.WriteLine($"  Final: Loss = {finalLoss:F6}, Accuracy = {finalAccuracy * 100:F1}%\n");
    }

    /// <summary>
    /// Fast Gradient Sign Method (FGSM) attack.
    /// δ = ε * sign(∇_x L(f(x), y))
    /// </summary>
    private static Tensor FGSMAttack(BinaryClassifier model, Tensor input, Tensor label, double epsilon)
    {
        // Compute gradient w.r.t. input
        var grad = ComputeInputGradient(model, input, label);
        var gradData = TensorAccessor.GetData(grad);

        // Apply sign and scale by epsilon
        var perturbation = new float[gradData.Length];
        for (int i = 0; i < perturbation.Length; i++)
        {
            perturbation[i] = (float)(epsilon * Math.Sign(gradData[i]));
        }

        // Add perturbation to input
        var perturbed = Add(input, new Tensor(perturbation, input.Shape));
        return perturbed;
    }

    /// <summary>
    /// Projected Gradient Descent (PGD) attack.
    /// Iteratively applies FGSM steps with projection.
    /// </summary>
    private static Tensor PGDAttack(BinaryClassifier model, Tensor input, Tensor label, double epsilon, int steps, double stepSize)
    {
        var x = input.Clone();

        for (int s = 0; s < steps; s++)
        {
            // Compute FGSM step
            var grad = ComputeInputGradient(model, x, label);
            var gradData = TensorAccessor.GetData(grad);

            var perturbation = new float[gradData.Length];
            for (int i = 0; i < perturbation.Length; i++)
            {
                perturbation[i] = (float)(stepSize * Math.Sign(gradData[i]));
            }

            // Update and project back to epsilon ball
            var xNew = Add(x, new Tensor(perturbation, x.Shape));
            var delta = Subtract(xNew, input);

            // Project to ensure ||delta|| ≤ ε
            var deltaData = TensorAccessor.GetData(delta);
            var deltaNorm = ComputeNorm(delta);
            if (deltaNorm > epsilon)
            {
                var scale = (float)(epsilon / deltaNorm);
                for (int i = 0; i < deltaData.Length; i++)
                {
                    deltaData[i] *= scale;
                }
                xNew = Add(input, new Tensor(deltaData, delta.Shape));
            }

            x = xNew;
        }

        return x;
    }

    /// <summary>
    /// Curvature-based attack using Hessian.
    /// Uses top Hessian eigenvector to find most sensitive direction.
    /// </summary>
    private static Tensor HessianAttack(BinaryClassifier model, Tensor input, Tensor label, double epsilon)
    {
        // Compute Hessian w.r.t. input
        var hessian = ComputeInputHessian(model, input, label);

        // Compute top eigenvector (using power iteration)
        var topEigenvector = ComputeTopEigenvector(hessian, 100);
        var topEigenvalue = ComputeRayleighQuotient(hessian, topEigenvector);

        Console.WriteLine($"  Top eigenvalue: {topEigenvalue:F6}");

        // Perturb in direction of top eigenvector
        var eigenvectorData = TensorAccessor.GetData(topEigenvector);
        var perturbation = new float[eigenvectorData.Length];

        // Normalize and scale by epsilon
        float norm = 0;
        foreach (var val in eigenvectorData)
        {
            norm += val * val;
        }
        norm = (float)Math.Sqrt(norm);

        if (norm > 1e-10)
        {
            for (int i = 0; i < perturbation.Length; i++)
            {
                perturbation[i] = eigenvectorData[i] / norm * (float)epsilon;
            }
        }

        // Add perturbation
        var perturbed = Add(input, new Tensor(perturbation, input.Shape));
        return perturbed;
    }

    /// <summary>
    /// Computes gradient w.r.t. input numerically.
    /// </summary>
    private static Tensor ComputeInputGradient(BinaryClassifier model, Tensor input, Tensor label)
    {
        var inputData = TensorAccessor.GetData(input);
        var gradData = new float[inputData.Length];
        double epsilon = 1e-6;

        for (int i = 0; i < inputData.Length; i++)
        {
            // Save original value
            float original = inputData[i];

            // f(x + ε)
            inputData[i] = original + (float)epsilon;
            var predPlus = model.Forward(input);
            double lossPlus = ComputeBinaryCrossEntropy(predPlus, label);

            // f(x - ε)
            inputData[i] = original - (float)epsilon;
            var predMinus = model.Forward(input);
            double lossMinus = ComputeBinaryCrossEntropy(predMinus, label);

            // Restore original value
            inputData[i] = original;

            // Gradient
            gradData[i] = (float)((lossPlus - lossMinus) / (2 * epsilon));
        }

        return new Tensor(gradData, input.Shape);
    }

    /// <summary>
    /// Computes Hessian w.r.t. input numerically.
    /// </summary>
    private static Tensor ComputeInputHessian(BinaryClassifier model, Tensor input, Tensor label)
    {
        var inputData = TensorAccessor.GetData(input);
        int dim = inputData.Length;
        var hessianData = new float[dim * dim];
        double epsilon = 1e-6;

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                float originalI = inputData[i];
                float originalJ = inputData[j];

                // Compute second partial derivative
                inputData[i] = originalI + (float)epsilon;
                inputData[j] = originalJ + (float)epsilon;
                var f_pp = model.Forward(input);
                double loss_pp = ComputeBinaryCrossEntropy(f_pp, label);

                inputData[i] = originalI + (float)epsilon;
                inputData[j] = originalJ - (float)epsilon;
                var f_pm = model.Forward(input);
                double loss_pm = ComputeBinaryCrossEntropy(f_pm, label);

                inputData[i] = originalI - (float)epsilon;
                inputData[j] = originalJ + (float)epsilon;
                var f_mp = model.Forward(input);
                double loss_mp = ComputeBinaryCrossEntropy(f_mp, label);

                inputData[i] = originalI - (float)epsilon;
                inputData[j] = originalJ - (float)epsilon;
                var f_mm = model.Forward(input);
                double loss_mm = ComputeBinaryCrossEntropy(f_mm, label);

                // Restore
                inputData[i] = originalI;
                inputData[j] = originalJ;

                // Second derivative
                hessianData[i * dim + j] = (float)((loss_pp - loss_pm - loss_mp + loss_mm) / (4 * epsilon * epsilon));
            }
        }

        return new Tensor(hessianData, new[] { dim, dim });
    }

    /// <summary>
    /// Computes top eigenvector using power iteration.
    /// </summary>
    private static Tensor ComputeTopEigenvector(Tensor hessian, int iterations)
    {
        var hessianData = TensorAccessor.GetData(hessian);
        int dim = hessian.Shape[0];

        // Initialize with random vector
        var rand = new Random(42);
        var vData = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            vData[i] = (float)(rand.NextDouble() * 2 - 1);
        }

        // Power iteration
        for (int iter = 0; iter < iterations; iter++)
        {
            // Compute H * v
            var hv = new float[dim];
            for (int i = 0; i < dim; i++)
            {
                hv[i] = 0;
                for (int j = 0; j < dim; j++)
                {
                    hv[i] += hessianData[i * dim + j] * vData[j];
                }
            }

            // Normalize
            float norm = 0;
            foreach (var val in hv)
            {
                norm += val * val;
            }
            norm = (float)Math.Sqrt(norm);

            if (norm > 1e-10)
            {
                for (int i = 0; i < dim; i++)
                {
                    vData[i] = hv[i] / norm;
                }
            }
        }

        return new Tensor(vData, new[] { dim });
    }

    /// <summary>
    /// Computes Rayleigh quotient (approximate eigenvalue).
    /// </summary>
    private static double ComputeRayleighQuotient(Tensor hessian, Tensor v)
    {
        var hessianData = TensorAccessor.GetData(hessian);
        var vData = TensorAccessor.GetData(v);
        int dim = vData.Length;

        // Compute H * v
        var hv = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            hv[i] = 0;
            for (int j = 0; j < dim; j++)
            {
                hv[i] += hessianData[i * dim + j] * vData[j];
            }
        }

        // Compute v^T * H * v
        double numerator = 0;
        double denominator = 0;
        for (int i = 0; i < dim; i++)
        {
            numerator += vData[i] * hv[i];
            denominator += vData[i] * vData[i];
        }

        return numerator / denominator;
    }

    /// <summary>
    /// Computes binary cross-entropy loss.
    /// </summary>
    private static double ComputeBinaryCrossEntropy(Tensor predictions, Tensor labels)
    {
        var predData = TensorAccessor.GetData(predictions);
        var labelData = TensorAccessor.GetData(labels);

        double sum = 0;
        int count = Math.Min(predData.Length, labelData.Length);

        for (int i = 0; i < count; i++)
        {
            double p = Math.Max(1e-10, Math.Min(1 - 1e-10, predData[i]));
            double y = labelData[i];
            sum -= y * Math.Log(p) + (1 - y) * Math.Log(1 - p);
        }

        return sum / count;
    }

    /// <summary>
    /// Computes model accuracy.
    /// </summary>
    private static double ComputeAccuracy(BinaryClassifier model, Tensor inputs, Tensor labels)
    {
        var predictions = model.Forward(inputs);
        var predData = TensorAccessor.GetData(predictions);
        var labelData = TensorAccessor.GetData(labels);

        int correct = 0;
        int count = Math.Min(predData.Length, labelData.Length);

        for (int i = 0; i < count; i++)
        {
            bool predictedClass = predData[i] > 0.5;
            bool trueClass = labelData[i] > 0.5;
            if (predictedClass == trueClass)
            {
                correct++;
            }
        }

        return (double)correct / count;
    }

    /// <summary>
    /// Computes gradients (simplified numerical computation).
    /// </summary>
    private static Tensor[] ComputeGradients(BinaryClassifier model, Tensor inputs, Tensor labels)
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
                float original = paramData[j];

                // f(x + ε)
                paramData[j] = original + (float)epsilon;
                var predPlus = model.Forward(inputs);
                double lossPlus = ComputeBinaryCrossEntropy(predPlus, labels);

                // f(x - ε)
                paramData[j] = original - (float)epsilon;
                var predMinus = model.Forward(inputs);
                double lossMinus = ComputeBinaryCrossEntropy(predMinus, labels);

                // Restore
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
    private static void UpdateParameters(BinaryClassifier model, Tensor[] grads, double lr)
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
    /// Generates synthetic binary classification data.
    /// </summary>
    private static (Tensor inputs, Tensor labels) GenerateData(int numSamples, int inputDim)
    {
        var rand = new Random(42);
        var inputData = new float[numSamples * inputDim];
        var labelData = new float[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            // Random inputs
            for (int j = 0; j < inputDim; j++)
            {
                inputData[i * inputDim + j] = (float)(rand.NextDouble() * 2 - 1);
            }

            // Label based on sum of first 2 features (simple decision boundary)
            float sum = inputData[i * inputDim + 0] + inputData[i * inputDim + 1];
            labelData[i] = sum > 0 ? 1.0f : 0.0f;
        }

        var inputs = new Tensor(inputData, new[] { numSamples, inputDim });
        var labels = new Tensor(labelData, new[] { numSamples });

        return (inputs, labels);
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
        var result = new float[numSamples * outputDim];

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < outputDim; j++)
            {
                result[i * outputDim + j] = biasData[j];
                for (int k = 0; k < inputData.Length / numSamples; k++)
                {
                    result[i * outputDim + j] += inputData[i * (inputData.Length / numSamples) + k] * weightData[k * outputDim + j];
                }
            }
        }

        return new Tensor(result, new[] { numSamples, outputDim });
    }

    /// <summary>
    /// ReLU activation.
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
    /// Sigmoid activation.
    /// </summary>
    private static Tensor Sigmoid(Tensor input)
    {
        var inputData = TensorAccessor.GetData(input);
        var result = new float[inputData.Length];

        for (int i = 0; i < inputData.Length; i++)
        {
            result[i] = 1.0f / (1.0f + (float)Math.Exp(-inputData[i]));
        }

        return new Tensor(result, input.Shape);
    }

    /// <summary>
    /// Subtracts two tensors.
    /// </summary>
    private static Tensor Subtract(Tensor a, Tensor b)
    {
        var aData = TensorAccessor.GetData(a);
        var bData = TensorAccessor.GetData(b);
        var result = new float[Math.Min(aData.Length, bData.Length)];

        for (int i = 0; i < result.Length; i++)
        {
            result[i] = aData[i] - bData[i];
        }

        return new Tensor(result, a.Shape);
    }

    /// <summary>
    /// Adds two tensors.
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
    /// Computes L2 norm.
    /// </summary>
    private static float ComputeNorm(Tensor tensor)
    {
        var data = TensorAccessor.GetData(tensor);
        float sum = 0;
        foreach (var val in data)
        {
            sum += val * val;
        }
        return (float)Math.Sqrt(sum);
    }

    /// <summary>
    /// Creates a random tensor.
    /// </summary>
    private static Tensor RandomTensor(int rows, int cols, Random rand)
    {
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)(rand.NextDouble() * 2 - 1);
        }
        return new Tensor(data, new[] { rows, cols }, true);
    }
}
