using System;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Provides test data generators for creating various function and tensor combinations
/// used in automatic differentiation tests.
/// </summary>
public static class TestDataGenerator
{
    private static readonly Random _random = new Random(42); // Fixed seed for reproducibility

    /// <summary>
    /// Generates a quadratic function f(x) = x^T * A * x + b^T * x + c
    /// with a random symmetric matrix A, vector b, and scalar c.
    /// </summary>
    /// <param name="dimension">The dimension of the input space.</param>
    /// <returns>A tuple containing the function and a test point.</returns>
    public static (Func<Tensor, Tensor> f, Tensor x) GenerateQuadraticFunction(int dimension = 3)
    {
        if (dimension <= 0)
            throw new ArgumentException("Dimension must be positive", nameof(dimension));

        // Generate random symmetric positive-definite matrix A
        var A = new float[dimension, dimension];
        for (int i = 0; i < dimension; i++)
        {
            for (int j = i; j < dimension; j++)
            {
                var value = (float)(_random.NextDouble() * 2.0 - 1.0);
                A[i, j] = value;
                A[j, i] = value; // Symmetric
            }
            A[i, i] += (float)_random.NextDouble() + 0.1f; // Ensure positive diagonal
        }

        // Generate random vector b
        var b = new float[dimension];
        for (int i = 0; i < dimension; i++)
        {
            b[i] = (float)(_random.NextDouble() * 2.0 - 1.0);
        }

        // Generate random scalar c
        var c = (float)_random.NextDouble();

        // Generate random test point x
        var xData = new float[dimension];
        for (int i = 0; i < dimension; i++)
        {
            xData[i] = (float)(_random.NextDouble() * 4.0 - 2.0);
        }

        // Define the quadratic function
        Func<Tensor, Tensor> f = x =>
        {
            var sum = 0.0;
            var xData = x.Data;

            // x^T * A * x
            for (int i = 0; i < dimension; i++)
            {
                for (int j = 0; j < dimension; j++)
                {
                    sum += xData[i] * A[i, j] * xData[j];
                }
            }

            // + b^T * x
            for (int i = 0; i < dimension; i++)
            {
                sum += b[i] * xData[i];
            }

            // + c
            sum += c;

            return new Tensor(new[] { (float)sum }, new[] { 1 });
        };

        return (f, new Tensor(xData, new[] { dimension }));
    }

    /// <summary>
    /// Generates a sinusoidal function with multiple frequencies.
    /// </summary>
    /// <param name="dimension">The dimension of the input space.</param>
    /// <returns>A tuple containing the function and a test point.</returns>
    public static (Func<Tensor, Tensor> f, Tensor x) GenerateSinusoidalFunction(int dimension = 3)
    {
        if (dimension <= 0)
            throw new ArgumentException("Dimension must be positive", nameof(dimension));

        // Generate random frequencies for each dimension
        var frequencies = new float[dimension];
        var phases = new float[dimension];
        for (int i = 0; i < dimension; i++)
        {
            frequencies[i] = (float)(_random.NextDouble() * 2.0 + 0.5); // 0.5 to 2.5
            phases[i] = (float)(_random.NextDouble() * 2.0 * Math.PI);
        }

        // Generate random test point x in range [0, 2π]
        var xData = new float[dimension];
        for (int i = 0; i < dimension; i++)
        {
            xData[i] = (float)(_random.NextDouble() * 2.0 * Math.PI);
        }

        // Define sinusoidal function
        Func<Tensor, Tensor> f = x =>
        {
            var sum = 0.0;
            var xData = x.Data;

            for (int i = 0; i < dimension; i++)
            {
                sum += Math.Sin(frequencies[i] * xData[i] + phases[i]);
            }

            return new Tensor(new[] { (float)sum }, new[] { 1 });
        };

        return (f, new Tensor(xData, new[] { dimension }));
    }

    /// <summary>
    /// Generates a complex neural network-like function with multiple layers.
    /// This simulates realistic machine learning use cases.
    /// </summary>
    /// <param name="inputDim">Input dimension.</param>
    /// <param name="hiddenDim">Hidden dimension.</param>
    /// <param name="outputDim">Output dimension.</param>
    /// <returns>A tuple containing the function, test input, and parameters.</returns>
    public static (Func<Tensor, Tensor> f, Tensor x, Tensor[] parameters) GenerateComplexNeuralNetwork(
        int inputDim = 2,
        int hiddenDim = 4,
        int outputDim = 1)
    {
        if (inputDim <= 0 || hiddenDim <= 0 || outputDim <= 0)
            throw new ArgumentException("All dimensions must be positive");

        // Generate random parameters for 2-layer network
        // Layer 1: W1 (hiddenDim x inputDim), b1 (hiddenDim)
        var W1Data = new float[hiddenDim * inputDim];
        var b1Data = new float[hiddenDim];

        for (int i = 0; i < W1Data.Length; i++)
        {
            W1Data[i] = (float)(_random.NextDouble() * 2.0 - 1.0);
        }

        for (int i = 0; i < hiddenDim; i++)
        {
            b1Data[i] = (float)(_random.NextDouble() * 2.0 - 1.0);
        }

        // Layer 2: W2 (outputDim x hiddenDim), b2 (outputDim)
        var W2Data = new float[outputDim * hiddenDim];
        var b2Data = new float[outputDim];

        for (int i = 0; i < W2Data.Length; i++)
        {
            W2Data[i] = (float)(_random.NextDouble() * 2.0 - 1.0);
        }

        for (int i = 0; i < outputDim; i++)
        {
            b2Data[i] = (float)(_random.NextDouble() * 2.0 - 1.0);
        }

        var W1 = new Tensor(W1Data, new[] { hiddenDim, inputDim });
        var b1 = new Tensor(b1Data, new[] { hiddenDim });
        var W2 = new Tensor(W2Data, new[] { outputDim, hiddenDim });
        var b2 = new Tensor(b2Data, new[] { outputDim });

        var parameters = new Tensor[] { W1, b1, W2, b2 };

        // Generate random test input
        var xData = new float[inputDim];
        for (int i = 0; i < inputDim; i++)
        {
            xData[i] = (float)(_random.NextDouble() * 2.0 - 1.0);
        }
        var x = new Tensor(xData, new[] { inputDim });

        // Define neural network function
        Func<Tensor, Tensor> f = input =>
        {
            // Layer 1: Linear + ReLU
            var h1 = MatMul(W1, input);
            for (int i = 0; i < hiddenDim; i++)
            {
                var sum = h1.Data[i];
                for (int j = 0; j < inputDim; j++)
                {
                    sum += W1Data[i * inputDim + j] * input.Data[j];
                }
                sum += b1Data[i];
                // ReLU activation
                h1.Data[i] = Math.Max(0, sum);
            }

            // Layer 2: Linear
            var output = MatMul(W2, h1);
            for (int i = 0; i < outputDim; i++)
            {
                var sum = 0.0;
                for (int j = 0; j < hiddenDim; j++)
                {
                    sum += W2Data[i * hiddenDim + j] * h1.Data[j];
                }
                sum += b2Data[i];
                output.Data[i] = (float)sum;
            }

            return output;
        };

        return (f, x, parameters);
    }

    /// <summary>
    /// Generates the Rosenbrock function: f(x) = sum(a(x_i+1 - x_i^2)^2 + b(1 - x_i)^2)
    /// with a=1, b=100 (standard parameters).
    /// This is a common test function for optimization algorithms.
    /// </summary>
    /// <param name="dimension">The dimension of the input space.</param>
    /// <returns>A tuple containing the function and a test point.</returns>
    public static (Func<Tensor, double> f, Tensor x) GenerateRosenbrockFunction(int dimension = 2)
    {
        if (dimension < 2)
            throw new ArgumentException("Dimension must be at least 2 for Rosenbrock", nameof(dimension));

        const double a = 1.0;
        const double b = 100.0;

        // Generate random test point (not at the minimum which is at x_i=1)
        var xData = new float[dimension];
        for (int i = 0; i < dimension; i++)
        {
            xData[i] = (float)(_random.NextDouble() * 4.0 - 1.0); // Range [-1, 3]
        }
        var x = new Tensor(xData, new[] { dimension });

        // Define Rosenbrock function
        Func<Tensor, double> f = tensor =>
        {
            var data = tensor.Data;
            var sum = 0.0;

            for (int i = 0; i < dimension - 1; i++)
            {
                var term1 = a * Math.Pow(data[i + 1] - data[i] * data[i], 2);
                var term2 = b * Math.Pow(1.0 - data[i], 2);
                sum += term1 + term2;
            }

            return sum;
        };

        return (f, x);
    }

    /// <summary>
    /// Generates a test tensor with random values in a specified range.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="min">Minimum value (default: -1.0).</param>
    /// <param name="max">Maximum value (default: 1.0).</param>
    /// <returns>A random tensor.</returns>
    public static Tensor RandomTensor(int[] shape, double min = -1.0, double max = 1.0)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape must not be null or empty", nameof(shape));

        var size = 1;
        foreach (var dim in shape)
        {
            size *= dim;
        }

        var data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = (float)(_random.NextDouble() * (max - min) + min);
        }

        return new Tensor(data, shape);
    }

    /// <summary>
    /// Generates a batch of random test tensors.
    /// </summary>
    /// <param name="batchSize">Number of tensors in the batch.</param>
    /// <param name="shape">Shape of each tensor.</param>
    /// <param name="min">Minimum value.</param>
    /// <param name="max">Maximum value.</param>
    /// <returns>An array of random tensors.</returns>
    public static Tensor[] RandomTensorBatch(int batchSize, int[] shape, double min = -1.0, double max = 1.0)
    {
        if (batchSize <= 0)
            throw new ArgumentException("Batch size must be positive", nameof(batchSize));

        var batch = new Tensor[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            batch[i] = RandomTensor(shape, min, max);
        }

        return batch;
    }

    /// <summary>
    /// Generates a diagonal matrix tensor.
    /// </summary>
    /// <param name="size">Size of the diagonal matrix.</param>
    /// <param name="diagonalValue">Value for diagonal elements.</param>
    /// <returns>A diagonal matrix.</returns>
    public static Tensor DiagonalMatrix(int size, float diagonalValue = 1.0f)
    {
        if (size <= 0)
            throw new ArgumentException("Size must be positive", nameof(size));

        var data = new float[size * size];
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                data[i * size + j] = (i == j) ? diagonalValue : 0.0f;
            }
        }

        return new Tensor(data, new[] { size, size });
    }

    /// <summary>
    /// Helper function for matrix multiplication.
    /// </summary>
    private static Tensor MatMul(Tensor a, Tensor b)
    {
        var aShape = a.Shape;
        var bShape = b.Shape;

        if (aShape.Length != 2 || bShape.Length != 2)
            throw new ArgumentException("Only 2D matrices supported");

        int m = aShape[0];
        int k = aShape[1];
        int n = bShape[1];

        if (k != bShape[0])
            throw new ArgumentException("Inner dimensions must match for matrix multiplication");

        var resultData = new float[m * n];

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                var sum = 0.0;
                for (int l = 0; l < k; l++)
                {
                    sum += a.Data[i * k + l] * b.Data[l * n + j];
                }
                resultData[i * n + j] = (float)sum;
            }
        }

        return new Tensor(resultData, new[] { m, n });
    }
}
