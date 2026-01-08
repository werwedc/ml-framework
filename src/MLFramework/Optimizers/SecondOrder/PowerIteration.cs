using System;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;

namespace MLFramework.Optimizers.SecondOrder;

/// <summary>
/// Power iteration algorithm for estimating the maximum eigenvalue
/// of a symmetric matrix using only matrix-vector products.
/// </summary>
public static class PowerIteration
{
    private static readonly Random _random = new Random();

    /// <summary>
    /// Estimates the maximum eigenvalue of a matrix using power iteration.
    /// </summary>
    /// <param name="matrixVectorProduct">Function that computes Av for any vector v.</param>
    /// <param name="dimension">Dimension of the matrix (size of vectors).</param>
    /// <param name="numIterations">Number of iterations (default: 20).</param>
    /// <returns>Estimated maximum eigenvalue.</returns>
    public static float EstimateMaxEigenvalue(
        Func<Tensor, Tensor> matrixVectorProduct,
        int dimension,
        int numIterations = 20)
    {
        if (matrixVectorProduct == null)
            throw new ArgumentNullException(nameof(matrixVectorProduct));
        if (dimension <= 0)
            throw new ArgumentException("Dimension must be positive.");

        // Initialize with random vector
        var vData = new float[dimension];
        for (int i = 0; i < dimension; i++)
        {
            vData[i] = (float)_random.NextDouble() * 2.0f - 1.0f; // Random in [-1, 1]
        }

        // Normalize initial vector
        Normalize(vData);

        var v = new Tensor(vData, new[] { dimension });

        // Power iteration
        float eigenvalue = 0.0f;
        for (int iter = 0; iter < numIterations; iter++)
        {
            // Compute Av
            var Av = matrixVectorProduct(v);
            var AvData = TensorAccessor.GetData(Av);

            // Rayleigh quotient: λ = (v^T * Av) / (v^T * v)
            float vDotAv = DotProduct(vData, AvData);
            float vDotV = DotProduct(vData, vData);
            eigenvalue = vDotAv / vDotV;

            // Update v = Av / ||Av||
            float normAv = Norm(AvData);
            for (int i = 0; i < dimension; i++)
            {
                vData[i] = AvData[i] / normAv;
            }
        }

        return eigenvalue;
    }

    /// <summary>
    /// Estimates the maximum eigenvalue with an initial guess vector.
    /// </summary>
    /// <param name="matrixVectorProduct">Function that computes Av for any vector v.</param>
    /// <param name="initialVector">Initial guess vector.</param>
    /// <param name="numIterations">Number of iterations (default: 20).</param>
    /// <returns>Estimated maximum eigenvalue.</returns>
    public static float EstimateMaxEigenvalueWithInitial(
        Func<Tensor, Tensor> matrixVectorProduct,
        Tensor initialVector,
        int numIterations = 20)
    {
        if (matrixVectorProduct == null)
            throw new ArgumentNullException(nameof(matrixVectorProduct));
        if (initialVector == null)
            throw new ArgumentNullException(nameof(initialVector));

        int dimension = initialVector.Size;
        var vData = (float[])TensorAccessor.GetData(initialVector).Clone();

        // Normalize initial vector
        Normalize(vData);

        var v = new Tensor(vData, initialVector.Shape);

        // Power iteration
        float eigenvalue = 0.0f;
        for (int iter = 0; iter < numIterations; iter++)
        {
            // Compute Av
            var Av = matrixVectorProduct(v);
            var AvData = TensorAccessor.GetData(Av);

            // Rayleigh quotient: λ = (v^T * Av) / (v^T * v)
            float vDotAv = DotProduct(vData, AvData);
            float vDotV = DotProduct(vData, vData);
            eigenvalue = vDotAv / vDotV;

            // Update v = Av / ||Av||
            float normAv = Norm(AvData);
            for (int i = 0; i < dimension; i++)
            {
                vData[i] = AvData[i] / normAv;
            }
        }

        return eigenvalue;
    }

    /// <summary>
    /// Estimates the maximum eigenvalue and corresponding eigenvector.
    /// </summary>
    /// <param name="matrixVectorProduct">Function that computes Av for any vector v.</param>
    /// <param name="dimension">Dimension of the matrix (size of vectors).</param>
    /// <param name="numIterations">Number of iterations (default: 20).</param>
    /// <returns>Tuple containing the estimated eigenvalue and eigenvector.</returns>
    public static (float eigenvalue, Tensor eigenvector) EstimateEigenpair(
        Func<Tensor, Tensor> matrixVectorProduct,
        int dimension,
        int numIterations = 20)
    {
        if (matrixVectorProduct == null)
            throw new ArgumentNullException(nameof(matrixVectorProduct));
        if (dimension <= 0)
            throw new ArgumentException("Dimension must be positive.");

        // Initialize with random vector
        var vData = new float[dimension];
        for (int i = 0; i < dimension; i++)
        {
            vData[i] = (float)_random.NextDouble() * 2.0f - 1.0f;
        }

        // Normalize initial vector
        Normalize(vData);

        var v = new Tensor(vData, new[] { dimension });

        // Power iteration
        float eigenvalue = 0.0f;
        for (int iter = 0; iter < numIterations; iter++)
        {
            // Compute Av
            var Av = matrixVectorProduct(v);
            var AvData = TensorAccessor.GetData(Av);

            // Rayleigh quotient: λ = (v^T * Av) / (v^T * v)
            float vDotAv = DotProduct(vData, AvData);
            float vDotV = DotProduct(vData, vData);
            eigenvalue = vDotAv / vDotV;

            // Update v = Av / ||Av||
            float normAv = Norm(AvData);
            for (int i = 0; i < dimension; i++)
            {
                vData[i] = AvData[i] / normAv;
            }
        }

        return (eigenvalue, new Tensor(vData, new[] { dimension }));
    }

    /// <summary>
    /// Computes the dot product of two vectors.
    /// </summary>
    private static float DotProduct(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length.");

        float result = 0.0f;
        for (int i = 0; i < a.Length; i++)
        {
            result += a[i] * b[i];
        }

        return result;
    }

    /// <summary>
    /// Computes the L2 norm of a vector.
    /// </summary>
    private static float Norm(float[] v)
    {
        float sum = 0.0f;
        foreach (float val in v)
        {
            sum += val * val;
        }
        return (float)Math.Sqrt(sum);
    }

    /// <summary>
    /// Normalizes a vector in-place.
    /// </summary>
    private static void Normalize(float[] v)
    {
        float norm = Norm(v);
        if (norm > 1e-10f)
        {
            for (int i = 0; i < v.Length; i++)
            {
                v[i] /= norm;
            }
        }
    }
}
