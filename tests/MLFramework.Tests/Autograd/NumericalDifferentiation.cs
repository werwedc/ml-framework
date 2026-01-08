using System;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Provides numerical differentiation utilities for validating automatic differentiation computations.
/// Uses finite difference methods to compute derivatives numerically.
/// </summary>
public static class NumericalDifferentiation
{
    /// <summary>
    /// Computes the numerical gradient of a scalar function f(x) using finite differences.
    /// </summary>
    /// <param name="f">The scalar function to differentiate.</param>
    /// <param name="x">The point at which to compute the gradient.</param>
    /// <param name="epsilon">The perturbation size for finite differences (default: 1e-6).</param>
    /// <returns>The numerical gradient as a tensor.</returns>
    public static Tensor Gradient(Func<Tensor, Tensor> f, Tensor x, double epsilon = 1e-6)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));

        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var gradData = new float[x.Size];
        var originalData = (float[])x.Data.Clone();

        // Compute gradient using central differences for each element
        for (int i = 0; i < x.Size; i++)
        {
            // f(x + eps * e_i)
            var xPlus = x.Clone();
            var xPlusData = (float[])xPlus.Data.Clone();
            xPlusData[i] += (float)epsilon;
            var fPlus = f(xPlus);

            // f(x - eps * e_i)
            var xMinus = x.Clone();
            var xMinusData = (float[])xMinus.Data.Clone();
            xMinusData[i] -= (float)epsilon;
            var fMinus = f(xMinus);

            // Central difference: (f(x + eps) - f(x - eps)) / (2 * eps)
            gradData[i] = (float)((fPlus.Data[0] - fMinus.Data[0]) / (2.0 * epsilon));
        }

        return new Tensor(gradData, x.Shape);
    }

    /// <summary>
    /// Computes the numerical Jacobian of a vector function f(x) using finite differences.
    /// </summary>
    /// <param name="f">The vector function to differentiate.</param>
    /// <param name="x">The point at which to compute the Jacobian.</param>
    /// <returns>The numerical Jacobian as a tensor.</returns>
    public static Tensor Jacobian(Func<Tensor, Tensor> f, Tensor x)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));

        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var y = f(x);
        var m = y.Size; // Output dimension
        var n = x.Size; // Input dimension

        var jacobianData = new float[m * n];
        var epsilon = 1e-6;

        // Compute Jacobian column by column using central differences
        for (int j = 0; j < n; j++)
        {
            // f(x + eps * e_j)
            var xPlus = x.Clone();
            var xPlusData = (float[])xPlus.Data.Clone();
            xPlusData[j] += (float)epsilon;
            var yPlus = f(xPlus);

            // f(x - eps * e_j)
            var xMinus = x.Clone();
            var xMinusData = (float[])xMinus.Data.Clone();
            xMinusData[j] -= (float)epsilon;
            var yMinus = f(xMinus);

            // Column j of Jacobian: (y_plus - y_minus) / (2 * eps)
            for (int i = 0; i < m; i++)
            {
                jacobianData[i * n + j] = (float)((yPlus.Data[i] - yMinus.Data[i]) / (2.0 * epsilon));
            }
        }

        return new Tensor(jacobianData, new[] { m, n });
    }

    /// <summary>
    /// Computes the numerical Hessian of a scalar function f(x) using finite differences.
    /// </summary>
    /// <param name="f">The scalar function to differentiate twice.</param>
    /// <param name="x">The point at which to compute the Hessian.</param>
    /// <returns>The numerical Hessian as a tensor.</returns>
    public static Tensor Hessian(Func<Tensor, double> f, Tensor x)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));

        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var n = x.Size;
        var hessianData = new float[n * n];
        var epsilon = 1e-4; // Use larger epsilon for second derivatives

        // Compute Hessian using finite differences of gradients
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double f_ij_pp = FunctionAt(x, i, epsilon, j, epsilon, f);
                double f_ij_pm = FunctionAt(x, i, epsilon, j, -epsilon, f);
                double f_ij_mp = FunctionAt(x, i, -epsilon, j, epsilon, f);
                double f_ij_mm = FunctionAt(x, i, -epsilon, j, -epsilon, f);

                // Second-order central difference formula
                hessianData[i * n + j] = (float)((f_ij_pp - f_ij_pm - f_ij_mp + f_ij_mm) / (4.0 * epsilon * epsilon));
            }
        }

        return new Tensor(hessianData, new[] { n, n });
    }

    /// <summary>
    /// Helper function to evaluate f(x + eps_i * e_i + eps_j * e_j).
    /// </summary>
    private static double FunctionAt(Tensor x, int i, double epsI, int j, double epsJ, Func<Tensor, double> f)
    {
        var xPerturbed = x.Clone();
        var data = (float[])xPerturbed.Data.Clone();
        data[i] += (float)epsI;
        if (j >= 0)
        {
            data[j] += (float)epsJ;
        }
        xPerturbed.Data = data;
        return f(xPerturbed);
    }

    /// <summary>
    /// Computes the numerical Hessian-Vector Product (HVP) without computing the full Hessian.
    /// </summary>
    /// <param name="f">The scalar function to differentiate twice.</param>
    /// <param name="x">The point at which to compute the HVP.</param>
    /// <param name="v">The vector to multiply with the Hessian.</param>
    /// <returns>The numerical HVP as a tensor.</returns>
    public static Tensor HessianVectorProduct(Func<Tensor, double> f, Tensor x, Tensor v)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));

        if (x == null)
            throw new ArgumentNullException(nameof(x));

        if (v == null)
            throw new ArgumentNullException(nameof(v));

        if (x.Size != v.Size)
            throw new ArgumentException("x and v must have the same size");

        var n = x.Size;
        var hvpData = new float[n];
        var epsilon = 1e-5;

        // Compute HVP using directional differences
        // H*v = (g(x + eps*v) - g(x - eps*v)) / (2*eps)
        // where g is the gradient

        for (int i = 0; i < n; i++)
        {
            // g(x + eps*v) using finite differences
            var xPlus = x.Clone();
            var xPlusData = (float[])xPlus.Data.Clone();
            for (int k = 0; k < n; k++)
            {
                xPlusData[k] += (float)(epsilon * v.Data[k]);
            }

            // g(x - eps*v) using finite differences
            var xMinus = x.Clone();
            var xMinusData = (float[])xMinus.Data.Clone();
            for (int k = 0; k < n; k++)
            {
                xMinusData[k] -= (float)(epsilon * v.Data[k]);
            }

            // Compute directional derivative of gradient
            // d(g(x + eps*v) - g(x - eps*v)) / (2*eps)
            // This is an approximation - for exact HVP, we'd need to compute gradients at x+eps*v and x-eps*v

            // For simplicity, use gradient difference
            var gradPlus = Gradient(t => new Tensor(new[] { (float)f(t) }, new[] { 1 }), xPlus);
            var gradMinus = Gradient(t => new Tensor(new[] { (float)f(t) }, new[] { 1 }), xMinus);

            hvpData[i] = (float)((gradPlus.Data[i] - gradMinus.Data[i]) / (2.0 * epsilon));
        }

        return new Tensor(hvpData, x.Shape);
    }

    /// <summary>
    /// Checks if two tensors are approximately equal within a tolerance.
    /// </summary>
    /// <param name="a">First tensor to compare.</param>
    /// <param name="b">Second tensor to compare.</param>
    /// <param name="tolerance">The maximum allowed absolute difference (default: 1e-6).</param>
    /// <returns>True if tensors are approximately equal, false otherwise.</returns>
    public static bool IsEqual(Tensor a, Tensor b, double tolerance = 1e-6)
    {
        if (a == null || b == null)
            return false;

        if (a.Size != b.Size)
            return false;

        for (int i = 0; i < a.Size; i++)
        {
            if (Math.Abs(a.Data[i] - b.Data[i]) > tolerance)
                return false;
        }

        return true;
    }
}
