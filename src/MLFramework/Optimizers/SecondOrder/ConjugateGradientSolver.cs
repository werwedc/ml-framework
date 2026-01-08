using System;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;

namespace MLFramework.Optimizers.SecondOrder;

/// <summary>
/// Conjugate Gradient solver for solving linear systems Ax = b
/// where A is positive-definite (or positive-semidefinite with damping).
/// Uses matrix-vector products (HVP) without materializing the full matrix.
/// </summary>
public static class ConjugateGradientSolver
{
    /// <summary>
    /// Solves the linear system Ax = b using the conjugate gradient algorithm.
    /// </summary>
    /// <param name="b">Right-hand side vector.</param>
    /// <param name="matrixVectorProduct">Function that computes Av for any vector v.</param>
    /// <param name="maxIterations">Maximum number of iterations (default: 100).</param>
    /// <param name="tolerance">Convergence tolerance (default: 1e-10).</param>
    /// <param name="initialGuess">Optional initial guess for x (null for zero initialization).</param>
    /// <returns>The solution vector x.</returns>
    public static Tensor Solve(
        Tensor b,
        Func<Tensor, Tensor> matrixVectorProduct,
        int maxIterations = 100,
        float tolerance = 1e-10f,
        Tensor? initialGuess = null)
    {
        if (b == null)
            throw new ArgumentNullException(nameof(b));
        if (matrixVectorProduct == null)
            throw new ArgumentNullException(nameof(matrixVectorProduct));

        int n = b.Size;
        var bData = TensorAccessor.GetData(b);

        // Initialize x (solution vector)
        var xData = initialGuess != null
            ? (float[])TensorAccessor.GetData(initialGuess).Clone()
            : new float[n];

        // Compute initial residual: r = b - Ax
        var Ax = matrixVectorProduct(new Tensor(xData, b.Shape));
        var AxData = TensorAccessor.GetData(Ax);
        var rData = new float[n];

        for (int i = 0; i < n; i++)
        {
            rData[i] = bData[i] - AxData[i];
        }

        // Compute initial residual norm squared
        float rNormSquared = DotProduct(rData, rData);

        // Check for convergence
        if (rNormSquared < tolerance * tolerance)
        {
            return new Tensor(xData, b.Shape);
        }

        // Initialize search direction: p = r
        var pData = (float[])rData.Clone();

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Compute Ap = A * p
            var p = new Tensor(pData, b.Shape);
            var Ap = matrixVectorProduct(p);
            var ApData = TensorAccessor.GetData(Ap);

            // Compute step size: alpha = (r^T * r) / (p^T * Ap)
            float pAp = DotProduct(pData, ApData);
            float alpha = rNormSquared / pAp;

            // Update solution: x = x + alpha * p
            for (int i = 0; i < n; i++)
            {
                xData[i] += alpha * pData[i];
            }

            // Update residual: r = r - alpha * Ap
            var rNormSquaredOld = rNormSquared;
            for (int i = 0; i < n; i++)
            {
                rData[i] -= alpha * ApData[i];
            }

            // Compute new residual norm squared
            rNormSquared = DotProduct(rData, rData);

            // Check for convergence
            if (rNormSquared < tolerance * tolerance)
            {
                break;
            }

            // Compute beta: beta = (r_new^T * r_new) / (r_old^T * r_old)
            float beta = rNormSquared / rNormSquaredOld;

            // Update search direction: p = r + beta * p
            for (int i = 0; i < n; i++)
            {
                pData[i] = rData[i] + beta * pData[i];
            }
        }

        return new Tensor(xData, b.Shape);
    }

    /// <summary>
    /// Computes the dot product of two vectors.
    /// </summary>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector.</param>
    /// <returns>The dot product a · b.</returns>
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
    /// Solves the linear system Ax = b with additional damping: (A + λI)x = b.
    /// This is useful for Newton's method with damped Hessian.
    /// </summary>
    /// <param name="b">Right-hand side vector.</param>
    /// <param name="matrixVectorProduct">Function that computes Av for any vector v.</param>
    /// <param name="damping">Damping parameter λ (default: 0).</param>
    /// <param name="maxIterations">Maximum number of iterations (default: 100).</param>
    /// <param name="tolerance">Convergence tolerance (default: 1e-10).</param>
    /// <returns>The solution vector x.</returns>
    public static Tensor SolveWithDamping(
        Tensor b,
        Func<Tensor, Tensor> matrixVectorProduct,
        float damping = 0.0f,
        int maxIterations = 100,
        float tolerance = 1e-10f)
    {
        // Wrap the matrix-vector product to include damping
        Func<Tensor, Tensor> dampedMVP = v =>
        {
            var Av = matrixVectorProduct(v);
            var vData = TensorAccessor.GetData(v);
            var AvData = TensorAccessor.GetData(Av);
            var resultData = new float[vData.Length];

            for (int i = 0; i < vData.Length; i++)
            {
                resultData[i] = AvData[i] + damping * vData[i];
            }

            return new Tensor(resultData, v.Shape);
        };

        return Solve(b, dampedMVP, maxIterations, tolerance);
    }
}
