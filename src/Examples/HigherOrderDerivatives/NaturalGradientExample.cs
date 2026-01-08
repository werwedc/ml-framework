using System;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using MLFramework.Optimizers.SecondOrder;

namespace MLFramework.Examples.HigherOrderDerivatives;

/// <summary>
/// Natural Gradient Descent Example.
///
/// Natural gradient descent uses the Fisher Information Matrix (FIM) to perform
/// updates in the Riemannian geometry of the parameter space, leading to more
/// efficient optimization.
///
/// Key Concepts:
/// - Fisher Information Matrix: Measures curvature in probability space
/// - Natural Gradient: ∇̃L = F^(-1) ∇L where F is the FIM
/// - Riemannian Geometry: Parameters live on a manifold, not Euclidean space
/// - Second-order Information: Like Newton's method, but for stochastic objectives
///
/// Mathematical Formulation:
/// Standard gradient descent: θ_{t+1} = θ_t - η ∇L(θ_t)
/// Natural gradient descent: θ_{t+1} = θ_t - η F^(-1) ∇L(θ_t)
///
/// where the Fisher Information Matrix F is:
///   F = E[∇_θ log p(y|x,θ) ∇_θ log p(y|x,θ)ᵀ]
///
/// For neural networks with softmax output:
///   F ≈ E[∇_θ z * ∇_θ zᵀ] where z are logits
///
/// Advantages:
/// - Invariant to reparameterization
/// - Better conditioning and faster convergence
/// - Handles ill-conditioned problems well
/// - Theoretically optimal update direction
///
/// Disadvantages:
/// - Computing F is O(n²) memory
/// - Inverting F is O(n³) computation
/// - Stochastic estimation can be noisy
/// - Requires approximations for large models
///
/// Practical implementation uses:
/// - Conjugate Gradient to solve F * d = g without materializing F
/// - Low-rank approximations (e.g., K-FAC)
/// - Diagonal approximation (AdaGrad, RMSProp)
/// </summary>
public class NaturalGradientExample
{
    /// <summary>
    /// Simple neural network for demonstration.
    /// </summary>
    private class NeuralNetwork
    {
        private Tensor _weight1;
        private Tensor _bias1;
        private Tensor _weight2;
        private Tensor _bias2;

        public NeuralNetwork(int inputDim, int hiddenDim, int outputDim)
        {
            var rand = new Random(42);
            _weight1 = RandomTensor(inputDim, hiddenDim, rand);
            _bias1 = new Tensor(new float[hiddenDim], new[] { hiddenDim }, true);
            _weight2 = RandomTensor(hiddenDim, outputDim, rand);
            _bias2 = new Tensor(new float[outputDim], new[] { outputDim }, true);
        }

        public Tensor Forward(Tensor input)
        {
            var hidden = LinearLayer(input, _weight1, _bias1);
            var activated = ReLU(hidden);
            var output = LinearLayer(activated, _weight2, _bias2);
            var probs = Softmax(output);
            return probs;
        }

        public Tensor GetLogits(Tensor input)
        {
            var hidden = LinearLayer(input, _weight1, _bias1);
            var activated = ReLU(hidden);
            var output = LinearLayer(activated, _weight2, _bias2);
            return output;
        }

        public Tensor[] GetParameters()
        {
            return new Tensor[] { _weight1, _bias1, _weight2, _bias2 };
        }

        public int GetParameterCount()
        {
            int count = 0;
            foreach (var param in GetParameters())
            {
                count += param.Size;
            }
            return count;
        }
    }

    /// <summary>
    /// Runs the natural gradient descent example.
    /// </summary>
    public static void Run()
    {
        Console.WriteLine("=== Natural Gradient Descent Example ===\n");

        // Configuration
        int inputDim = 4;
        int hiddenDim = 8;
        int outputDim = 3;  // 3 classes
        int numSamples = 100;
        int numIterations = 100;
        double learningRate = 0.01;
        double damping = 1e-4;  // For numerical stability

        Console.WriteLine($"Model: Neural network ({inputDim} -> {hiddenDim} -> {outputDim})");
        Console.WriteLine($"Problem: Multi-class classification");
        Console.WriteLine($"Training samples: {numSamples}");
        Console.WriteLine($"Iterations: {numIterations}\n");

        // Generate synthetic data
        Console.WriteLine("Generating synthetic data...");
        var (inputs, labels) = GenerateData(numSamples, inputDim, outputDim);
        Console.WriteLine("Data generated\n");

        // --- SGD Training ---
        Console.WriteLine("=== Standard SGD Training ===");
        var sgdModel = new NeuralNetwork(inputDim, hiddenDim, outputDim);
        TrainSGD(sgdModel, inputs, labels, numIterations, learningRate);

        // --- Natural Gradient Training ---
        Console.WriteLine("\n=== Natural Gradient Training ===");
        var ngModel = new NeuralNetwork(inputDim, hiddenDim, outputDim);
        TrainNaturalGradient(ngModel, inputs, labels, numIterations, learningRate, damping);

        // --- Comparison ---
        Console.WriteLine("\n=== Comparison ===");
        var sgdLoss = ComputeLoss(sgdModel, inputs, labels);
        var ngLoss = ComputeLoss(ngModel, inputs, labels);

        var sgdAccuracy = ComputeAccuracy(sgdModel, inputs, labels);
        var ngAccuracy = ComputeAccuracy(ngModel, inputs, labels);

        Console.WriteLine($"SGD - Loss: {sgdLoss:F6}, Accuracy: {sgdAccuracy * 100:F1}%");
        Console.WriteLine($"Natural Gradient - Loss: {ngLoss:F6}, Accuracy: {ngAccuracy * 100:F1}%");
        Console.WriteLine($"Improvement: {((sgdLoss - ngLoss) / sgdLoss * 100):F2}% loss reduction\n");

        // --- Fisher Matrix Analysis ---
        Console.WriteLine("=== Fisher Information Matrix Analysis ===");
        var fisher = ComputeFisherMatrix(ngModel, inputs, labels);
        var fisherEigenvalues = ComputeEigenvalues(fisher);

        Console.WriteLine($"Fisher matrix size: {fisher.Shape[0]} x {fisher.Shape[1]}");
        Console.WriteLine($"Number of parameters: {ngModel.GetParameterCount()}");
        Console.WriteLine($"Max eigenvalue: {fisherEigenvalues[0]:F6}");
        Console.WriteLine($"Min eigenvalue: {fisherEigenvalues[fisherEigenvalues.Length - 1]:F6}");
        Console.WriteLine($"Condition number: {fisherEigenvalues[0] / fisherEigenvalues[fisherEigenvalues.Length - 1]:F2}\n");

        Console.WriteLine("=== Natural Gradient Example completed successfully! ===\n");
    }

    /// <summary>
    /// Trains model using standard SGD.
    /// </summary>
    private static void TrainSGD(NeuralNetwork model, Tensor inputs, Tensor labels, int iterations, double lr)
    {
        Console.WriteLine("Training with SGD...");

        for (int iter = 0; iter < iterations; iter++)
        {
            // Compute loss and gradients
            var loss = ComputeLoss(model, inputs, labels);
            var grads = ComputeGradients(model, inputs, labels);

            // Update parameters
            UpdateParameters(model, grads, lr);

            // Print progress
            if (iter % 20 == 0)
            {
                var accuracy = ComputeAccuracy(model, inputs, labels);
                Console.WriteLine($"  Iteration {iter}: Loss = {loss:F6}, Accuracy = {accuracy * 100:F1}%");
            }
        }

        var finalLoss = ComputeLoss(model, inputs, labels);
        var finalAccuracy = ComputeAccuracy(model, inputs, labels);
        Console.WriteLine($"  Final: Loss = {finalLoss:F6}, Accuracy = {finalAccuracy * 100:F1}%\n");
    }

    /// <summary>
    /// Trains model using natural gradient descent.
    /// </summary>
    private static void TrainNaturalGradient(NeuralNetwork model, Tensor inputs, Tensor labels, int iterations, double lr, double damping)
    {
        Console.WriteLine("Training with Natural Gradient...");

        for (int iter = 0; iter < iterations; iter++)
        {
            // Compute standard gradients
            var grads = ComputeGradients(model, inputs, labels);

            // Compute Fisher Information Matrix
            var fisher = ComputeFisherMatrix(model, inputs, labels);

            // Solve F * natural_grad = grad using conjugate gradient
            var naturalGrads = SolveConjugateGradient(fisher, grads, damping);

            // Update parameters using natural gradients
            UpdateParameters(model, naturalGrads, lr);

            // Print progress
            var loss = ComputeLoss(model, inputs, labels);
            if (iter % 20 == 0)
            {
                var accuracy = ComputeAccuracy(model, inputs, labels);
                Console.WriteLine($"  Iteration {iter}: Loss = {loss:F6}, Accuracy = {accuracy * 100:F1}%");
            }
        }

        var finalLoss = ComputeLoss(model, inputs, labels);
        var finalAccuracy = ComputeAccuracy(model, inputs, labels);
        Console.WriteLine($"  Final: Loss = {finalLoss:F6}, Accuracy = {finalAccuracy * 100:F1}%\n");
    }

    /// <summary>
    /// Computes Fisher Information Matrix.
    /// F = E[∇_θ z * ∇_θ zᵀ] where z are logits
    /// </summary>
    private static Tensor ComputeFisherMatrix(NeuralNetwork model, Tensor inputs, Tensor labels)
    {
        var parameters = model.GetParameters();
        int paramCount = 0;
        foreach (var param in parameters)
        {
            paramCount += param.Size;
        }

        // Flatten parameters
        var flatParams = FlattenParameters(parameters);

        // Compute gradients for each sample
        var inputData = TensorAccessor.GetData(inputs);
        int numSamples = inputs.Shape[0];

        var fisherSum = new float[paramCount * paramCount];
        var rand = new Random();

        // Use subset of samples for efficiency
        int subsetSize = Math.Min(10, numSamples);

        for (int s = 0; s < subsetSize; s++)
        {
            // Select random sample
            int sampleIdx = rand.Next(numSamples);

            // Extract single sample
            var sampleData = new float[inputData.Length / numSamples];
            Array.Copy(inputData, sampleIdx * (inputData.Length / numSamples), sampleData, 0, sampleData.Length);
            var sample = new Tensor(sampleData, new[] { inputData.Length / numSamples });

            // Compute logits
            var logits = model.GetLogits(sample);

            // Compute gradients w.r.t. logits
            var logitsGrad = ComputeLogitsGradient(logits, labels, sampleIdx);

            // Compute gradients w.r.t. parameters
            var paramGrads = ComputeParameterGradientsNumerically(model, inputs, labels);

            // Flatten gradients
            var flatGrads = FlattenParameters(paramGrads);

            // Add outer product to Fisher: F += ∇θ * ∇θᵀ
            for (int i = 0; i < paramCount; i++)
            {
                for (int j = 0; j < paramCount; j++)
                {
                    fisherSum[i * paramCount + j] += flatGrads[i] * flatGrads[j];
                }
            }
        }

        // Average
        for (int i = 0; i < fisherSum.Length; i++)
        {
            fisherSum[i] /= subsetSize;
        }

        return new Tensor(fisherSum, new[] { paramCount, paramCount });
    }

    /// <summary>
    /// Computes gradients w.r.t. logits (simplified).
    /// </summary>
    private static Tensor ComputeLogitsGradient(Tensor logits, Tensor labels, int sampleIdx)
    {
        var logitsData = TensorAccessor.GetData(logits);
        var labelsData = TensorAccessor.GetData(labels);
        int numClasses = logitsData.Length;

        var gradData = new float[numClasses];
        var probs = Softmax(logits);

        for (int i = 0; i < numClasses; i++)
        {
            int labelIdx = sampleIdx * numClasses + i;
            float target = labelsData[labelIdx];
            gradData[i] = target - probs.Data[i];
        }

        return new Tensor(gradData, logits.Shape);
    }

    /// <summary>
    /// Solves F * x = b using conjugate gradient method.
    /// </summary>
    private static Tensor[] SolveConjugateGradient(Tensor fisher, Tensor[] b, double damping)
    {
        var fisherData = TensorAccessor.GetData(fisher);
        int n = fisher.Shape[0];

        // Flatten b
        var flatB = FlattenParameters(b);
        var bData = TensorAccessor.GetData(flatB);

        // Add damping: F_damped = F + λI
        var fisherDamped = new float[n * n];
        Array.Copy(fisherData, fisherDamped, fisherData.Length);
        for (int i = 0; i < n; i++)
        {
            fisherDamped[i * n + i] += (float)damping;
        }

        // Conjugate gradient
        var x = new float[n];  // Initial guess (zeros)

        // r = b - Ax
        var r = new float[n];
        for (int i = 0; i < n; i++)
        {
            r[i] = bData[i];
            for (int j = 0; j < n; j++)
            {
                r[i] -= fisherDamped[i * n + j] * x[j];
            }
        }

        var p = new float[n];
        Array.Copy(r, p, n);

        var rOld = Dot(r, r);

        for (int iter = 0; iter < 100; iter++)
        {
            // Ap = A * p
            var Ap = new float[n];
            for (int i = 0; i < n; i++)
            {
                Ap[i] = 0;
                for (int j = 0; j < n; j++)
                {
                    Ap[i] += fisherDamped[i * n + j] * p[j];
                }
            }

            // α = rᵀr / pᵀAp
            var alpha = rOld / Dot(p, Ap);

            // x = x + αp
            for (int i = 0; i < n; i++)
            {
                x[i] += (float)alpha * p[i];
            }

            // r = r - αAp
            for (int i = 0; i < n; i++)
            {
                r[i] -= (float)alpha * Ap[i];
            }

            var rNew = Dot(r, r);

            // Check convergence
            if (rNew < 1e-10)
            {
                break;
            }

            // β = r_newᵀr_new / r_oldᵀr_old
            var beta = rNew / rOld;

            // p = r + βp
            for (int i = 0; i < n; i++)
            {
                p[i] = r[i] + (float)beta * p[i];
            }

            rOld = rNew;
        }

        // Unflatten x to match parameter shapes
        return UnflattenParameters(new Tensor(x, new[] { n }), b);
    }

    /// <summary>
    /// Computes eigenvalues (simplified power iteration for top eigenvalue).
    /// </summary>
    private static float[] ComputeEigenvalues(Tensor matrix)
    {
        var matData = TensorAccessor.GetData(matrix);
        int n = matrix.Shape[0];

        // Simplified: compute just top and bottom eigenvalues
        var eigenvalues = new float[2];

        // Top eigenvalue (power iteration)
        var v = new float[n];
        var rand = new Random(42);
        for (int i = 0; i < n; i++)
        {
            v[i] = (float)(rand.NextDouble() * 2 - 1);
        }

        for (int iter = 0; iter < 100; iter++)
        {
            var Av = new float[n];
            for (int i = 0; i < n; i++)
            {
                Av[i] = 0;
                for (int j = 0; j < n; j++)
                {
                    Av[i] += matData[i * n + j] * v[j];
                }
            }

            // Normalize
            float norm = 0;
            foreach (var val in Av)
            {
                norm += val * val;
            }
            norm = (float)Math.Sqrt(norm);

            if (norm > 1e-10)
            {
                for (int i = 0; i < n; i++)
                {
                    v[i] = Av[i] / norm;
                }
            }
        }

        // Compute Rayleigh quotient for top eigenvalue
        float topEigenvalue = 0;
        for (int i = 0; i < n; i++)
        {
            float Av_i = 0;
            for (int j = 0; j < n; j++)
            {
                Av_i += matData[i * n + j] * v[j];
            }
            topEigenvalue += v[i] * Av_i;
        }

        eigenvalues[0] = topEigenvalue;

        // Bottom eigenvalue (inverse power iteration) - approximate as 1/top of inverse
        // Simplified: assume positive definite and use minimum diagonal element
        float minDiagonal = float.MaxValue;
        for (int i = 0; i < n; i++)
        {
            minDiagonal = Math.Min(minDiagonal, matData[i * n + i]);
        }
        eigenvalues[1] = minDiagonal;

        return eigenvalues;
    }

    /// <summary>
    /// Computes dot product.
    /// </summary>
    private static float Dot(float[] a, float[] b)
    {
        float sum = 0;
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /// <summary>
    /// Computes loss (cross-entropy).
    /// </summary>
    private static double ComputeLoss(NeuralNetwork model, Tensor inputs, Tensor labels)
    {
        var predictions = model.Forward(inputs);
        var predData = TensorAccessor.GetData(predictions);
        var labelData = TensorAccessor.GetData(labels);

        double sum = 0;
        int count = Math.Min(predData.Length, labelData.Length);

        for (int i = 0; i < count; i++)
        {
            double p = Math.Max(1e-10, predData[i]);
            sum -= labelData[i] * Math.Log(p);
        }

        return sum / inputs.Shape[0];  // Average over samples
    }

    /// <summary>
    /// Computes accuracy.
    /// </summary>
    private static double ComputeAccuracy(NeuralNetwork model, Tensor inputs, Tensor labels)
    {
        var predictions = model.Forward(inputs);
        var predData = TensorAccessor.GetData(predictions);
        var labelData = TensorAccessor.GetData(labels);

        int correct = 0;
        int numSamples = inputs.Shape[0];
        int numClasses = predData.Length / numSamples;

        for (int i = 0; i < numSamples; i++)
        {
            // Find predicted class
            int predClass = 0;
            float maxProb = predData[i * numClasses];
            for (int j = 1; j < numClasses; j++)
            {
                if (predData[i * numClasses + j] > maxProb)
                {
                    maxProb = predData[i * numClasses + j];
                    predClass = j;
                }
            }

            // Find true class
            int trueClass = 0;
            for (int j = 1; j < numClasses; j++)
            {
                if (labelData[i * numClasses + j] > labelData[i * numClasses + trueClass])
                {
                    trueClass = j;
                }
            }

            if (predClass == trueClass)
            {
                correct++;
            }
        }

        return (double)correct / numSamples;
    }

    /// <summary>
    /// Computes gradients numerically (simplified).
    /// </summary>
    private static Tensor[] ComputeGradients(NeuralNetwork model, Tensor inputs, Tensor labels)
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
                double lossPlus = ComputeLossFromPredictions(predPlus, labels);

                // f(x - ε)
                paramData[j] = original - (float)epsilon;
                var predMinus = model.Forward(inputs);
                double lossMinus = ComputeLossFromPredictions(predMinus, labels);

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
    /// Computes parameter gradients numerically for Fisher matrix.
    /// </summary>
    private static Tensor[] ComputeParameterGradientsNumerically(NeuralNetwork model, Tensor inputs, Tensor labels)
    {
        // Simplified: use same gradient computation
        return ComputeGradients(model, inputs, labels);
    }

    /// <summary>
    /// Computes loss from predictions (helper for numerical gradient).
    /// </summary>
    private static double ComputeLossFromPredictions(Tensor predictions, Tensor labels)
    {
        var predData = TensorAccessor.GetData(predictions);
        var labelData = TensorAccessor.GetData(labels);

        double sum = 0;
        int count = Math.Min(predData.Length, labelData.Length);

        for (int i = 0; i < count; i++)
        {
            double p = Math.Max(1e-10, predData[i]);
            sum -= labelData[i] * Math.Log(p);
        }

        return sum / predictions.Shape[0];
    }

    /// <summary>
    /// Updates parameters using gradient descent.
    /// </summary>
    private static void UpdateParameters(NeuralNetwork model, Tensor[] grads, double lr)
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
    /// Flattens parameters to 1D tensor.
    /// </summary>
    private static Tensor FlattenParameters(Tensor[] parameters)
    {
        int totalSize = 0;
        foreach (var param in parameters)
        {
            totalSize += param.Size;
        }

        var flatData = new float[totalSize];
        int offset = 0;

        foreach (var param in parameters)
        {
            var paramData = TensorAccessor.GetData(param);
            Array.Copy(paramData, 0, flatData, offset, param.Size);
            offset += param.Size;
        }

        return new Tensor(flatData, new[] { totalSize });
    }

    /// <summary>
    /// Unflattens parameters from 1D tensor.
    /// </summary>
    private static Tensor[] UnflattenParameters(Tensor flat, Tensor[] reference)
    {
        var result = new Tensor[reference.Length];
        var flatData = TensorAccessor.GetData(flat);
        int offset = 0;

        for (int i = 0; i < reference.Length; i++)
        {
            int size = reference[i].Size;
            var paramData = new float[size];
            Array.Copy(flatData, offset, paramData, 0, size);
            result[i] = new Tensor(paramData, reference[i].Shape);
            offset += size;
        }

        return result;
    }

    /// <summary>
    /// Generates synthetic multi-class data.
    /// </summary>
    private static (Tensor inputs, Tensor labels) GenerateData(int numSamples, int inputDim, int numClasses)
    {
        var rand = new Random(42);
        var inputData = new float[numSamples * inputDim];
        var labelData = new float[numSamples * numClasses];

        for (int i = 0; i < numSamples; i++)
        {
            // Random inputs
            for (int j = 0; j < inputDim; j++)
            {
                inputData[i * inputDim + j] = (float)(rand.NextDouble() * 2 - 1);
            }

            // Random class label
            int labelClass = rand.Next(numClasses);
            for (int j = 0; j < numClasses; j++)
            {
                labelData[i * numClasses + j] = (j == labelClass) ? 1.0f : 0.0f;
            }
        }

        var inputs = new Tensor(inputData, new[] { numSamples, inputDim });
        var labels = new Tensor(labelData, new[] { numSamples, numClasses });

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
    /// Softmax activation.
    /// </summary>
    private static Tensor Softmax(Tensor input)
    {
        var inputData = TensorAccessor.GetData(input);
        var result = new float[inputData.Length];

        // Apply softmax to each row (sample)
        int numSamples = input.Shape[0];
        int numClasses = input.Shape[1];

        for (int i = 0; i < numSamples; i++)
        {
            // Find max for numerical stability
            float maxVal = inputData[i * numClasses];
            for (int j = 1; j < numClasses; j++)
            {
                maxVal = Math.Max(maxVal, inputData[i * numClasses + j]);
            }

            // Compute exp and sum
            float sum = 0;
            for (int j = 0; j < numClasses; j++)
            {
                result[i * numClasses + j] = (float)Math.Exp(inputData[i * numClasses + j] - maxVal);
                sum += result[i * numClasses + j];
            }

            // Normalize
            for (int j = 0; j < numClasses; j++)
            {
                result[i * numClasses + j] /= sum;
            }
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
            data[i] = (float)(rand.NextDouble() * 0.2 - 0.1);
        }
        return new Tensor(data, new[] { rows, cols }, true);
    }
}
