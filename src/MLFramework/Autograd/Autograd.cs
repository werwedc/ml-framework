using System;
using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

// Type aliases to avoid naming conflicts
using HessianClass = MLFramework.Autograd.Hessian;
using HessianVectorProductClass = MLFramework.Autograd.HessianVectorProduct;
using JacobianClass = MLFramework.Autograd.Jacobian;
using JacobianVectorProductClass = MLFramework.Autograd.JacobianVectorProduct;

/// <summary>
/// Static class providing convenient access to automatic differentiation operations.
/// Includes gradient, Jacobian, Hessian, and higher-order derivative computations.
/// </summary>
public static class Autograd
{
    /// <summary>
    /// Computes the gradient of a scalar-valued function with respect to a tensor.
    /// </summary>
    /// <param name="loss">The scalar loss function to differentiate.</param>
    /// <param name="parameters">The parameters to compute gradients for.</param>
    /// <returns>The gradient tensor with the same shape as parameters.</returns>
    /// <exception cref="ArgumentNullException">Thrown when loss or parameters is null.</exception>
    public static Tensor Gradient(Func<Tensor, double> loss, Tensor parameters)
    {
        if (loss == null)
            throw new ArgumentNullException(nameof(loss));
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        var paramsWithGrad = TensorAccessor.CloneWithGrad(parameters);
        var lossTensor = TensorAccessor.CreateScalar(loss(paramsWithGrad));
        lossTensor.Backward();

        return paramsWithGrad.Gradient!;
    }

    /// <summary>
    /// Computes the full Hessian matrix of a scalar loss function.
    /// </summary>
    /// <param name="loss">The scalar loss function to differentiate.</param>
    /// <param name="parameters">The parameters at which to compute the Hessian.</param>
    /// <returns>A 2D tensor representing the Hessian matrix (n x n where n is the size of parameters).</returns>
    /// <exception cref="ArgumentNullException">Thrown when loss or parameters is null.</exception>
    public static Tensor Hessian(Func<Tensor, double> loss, Tensor parameters)
    {
        return HessianClass.Compute(loss, parameters, new HessianOptions()).Hessian;
    }

    /// <summary>
    /// Computes the Hessian matrix with custom options.
    /// </summary>
    /// <param name="loss">The scalar loss function to differentiate.</param>
    /// <param name="parameters">The parameters at which to compute the Hessian.</param>
    /// <param name="sparse">Whether to compute a sparse Hessian (default: false).</param>
    /// <returns>A 2D tensor representing the Hessian matrix.</returns>
    /// <exception cref="ArgumentNullException">Thrown when loss or parameters is null.</exception>
    public static Tensor HessianSparse(Func<Tensor, double> loss, Tensor parameters, bool sparse)
    {
        var options = new HessianOptions { Sparse = sparse };
        return HessianClass.Compute(loss, parameters, options).Hessian;
    }

    /// <summary>
    /// Computes the Hessian matrix with custom options.
    /// </summary>
    /// <param name="loss">The scalar loss function to differentiate.</param>
    /// <param name="parameters">The parameters at which to compute the Hessian.</param>
    /// <param name="sparse">Whether to compute a sparse Hessian (default: false).</param>
    /// <param name="detectStructure">Whether to detect Hessian structure (default: true).</param>
    /// <returns>A 2D tensor representing the Hessian matrix.</returns>
    /// <exception cref="ArgumentNullException">Thrown when loss or parameters is null.</exception>
    public static Tensor HessianWithDetection(Func<Tensor, double> loss, Tensor parameters, bool sparse, bool detectStructure)
    {
        var options = new HessianOptions
        {
            Sparse = sparse,
            DetectStructure = detectStructure
        };
        return HessianClass.Compute(loss, parameters, options).Hessian;
    }

    /// <summary>
    /// Computes the Hessian matrix with custom options.
    /// </summary>
    /// <param name="loss">The scalar loss function to differentiate.</param>
    /// <param name="parameters">The parameters at which to compute the Hessian.</param>
    /// <param name="options">Options for Hessian computation.</param>
    /// <returns>A HessianResult containing the Hessian and optionally eigenvalues.</returns>
    /// <exception cref="ArgumentNullException">Thrown when loss, parameters, or options is null.</exception>
    public static HessianResult Hessian(Func<Tensor, double> loss, Tensor parameters, HessianOptions options)
    {
        return HessianClass.Compute(loss, parameters, options);
    }

    /// <summary>
    /// Computes the Hessian matrix and its eigenvalues.
    /// </summary>
    /// <param name="loss">The scalar loss function to differentiate.</param>
    /// <param name="parameters">The parameters at which to compute the Hessian.</param>
    /// <returns>A tuple containing (hessian matrix, eigenvalues tensor).</returns>
    /// <exception cref="ArgumentNullException">Thrown when loss or parameters is null.</exception>
    public static (Tensor hessian, Tensor eigenvalues) HessianWithEigenvalues(Func<Tensor, double> loss, Tensor parameters)
    {
        var options = new HessianOptions
        {
            ComputeEigenvalues = true,
            EigenvalueMethod = EigenvalueMethod.PowerIteration
        };

        var result = HessianClass.Compute(loss, parameters, options);

        return (result.Hessian, result.Eigenvalues!);
    }

    /// <summary>
    /// Computes the Hessian matrix and its eigenvalues with custom method.
    /// </summary>
    /// <param name="loss">The scalar loss function to differentiate.</param>
    /// <param name="parameters">The parameters at which to compute the Hessian.</param>
    /// <param name="method">The eigenvalue computation method.</param>
    /// <returns>A tuple containing (hessian matrix, eigenvalues tensor).</returns>
    /// <exception cref="ArgumentNullException">Thrown when loss or parameters is null.</exception>
    public static (Tensor hessian, Tensor eigenvalues) HessianWithEigenvalues(
        Func<Tensor, double> loss,
        Tensor parameters,
        EigenvalueMethod method)
    {
        var options = new HessianOptions
        {
            ComputeEigenvalues = true,
            EigenvalueMethod = method
        };

        var result = HessianClass.Compute(loss, parameters, options);

        return (result.Hessian, result.Eigenvalues!);
    }

    /// <summary>
    /// Computes a partial Hessian for a subset of parameters.
    /// </summary>
    /// <param name="loss">The scalar loss function to differentiate.</param>
    /// <param name="parameters">The full parameter tensor.</param>
    /// <param name="parameterIndices">Indices of the parameter subset to compute the Hessian for.</param>
    /// <returns>A Hessian matrix for the specified parameter subset.</returns>
    /// <exception cref="ArgumentNullException">Thrown when loss, parameters, or parameterIndices is null.</exception>
    public static Tensor Hessian(Func<Tensor, double> loss, Tensor parameters, int[] parameterIndices)
    {
        if (loss == null)
            throw new ArgumentNullException(nameof(loss));
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));
        if (parameterIndices == null)
            throw new ArgumentNullException(nameof(parameterIndices));

        var options = new HessianOptions { ParameterIndices = parameterIndices };
        return HessianClass.Compute(loss, parameters, options).Hessian;
    }

    /// <summary>
    /// Computes the Hessian-Vector Product (HVP) without computing the full Hessian.
    /// More efficient for large models when only H * v is needed.
    /// </summary>
    /// <param name="loss">The scalar loss function to differentiate.</param>
    /// <param name="parameters">The parameters at which to compute the HVP.</param>
    /// <param name="vector">The vector to multiply with the Hessian (must match parameters shape).</param>
    /// <returns>The Hessian-Vector product H * v.</returns>
    /// <exception cref="ArgumentNullException">Thrown when loss, parameters, or vector is null.</exception>
    /// <exception cref="ArgumentException">Thrown when vector shape doesn't match parameters shape.</exception>
    public static Tensor HessianVectorProduct(Func<Tensor, double> loss, Tensor parameters, Tensor vector)
    {
        return HessianVectorProductClass.Compute(loss, parameters, vector);
    }

    /// <summary>
    /// Computes the Jacobian matrix of a function.
    /// For scalar-valued functions, this is equivalent to the gradient.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor.</param>
    /// <returns>The Jacobian matrix.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    public static Tensor Jacobian(Func<Tensor, Tensor> f, Tensor x)
    {
        return JacobianClass.Compute(f, x);
    }

    /// <summary>
    /// Computes the Vector-Jacobian Product (VJP).
    /// VJP computes v^T * J where v is a vector and J is the Jacobian.
    /// Equivalent to the gradient when f is scalar-valued.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor.</param>
    /// <param name="vector">The vector to multiply with the Jacobian.</param>
    /// <returns>The Vector-Jacobian product v^T * J.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or vector is null.</exception>
    public static Tensor VectorJacobianProduct(Func<Tensor, Tensor> f, Tensor x, Tensor vector)
    {
        return JacobianClass.ComputeVectorJacobianProduct(f, x, vector);
    }

    /// <summary>
    /// Computes the Jacobian-Vector Product (JVP).
    /// JVP computes J * v where J is the Jacobian and v is a vector.
    /// Also known as forward-mode differentiation.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor.</param>
    /// <param name="vector">The vector to multiply with the Jacobian.</param>
    /// <returns>The Jacobian-Vector product J * v.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or vector is null.</exception>
    public static Tensor JacobianVectorProduct(Func<Tensor, Tensor> f, Tensor x, Tensor vector)
    {
        return JacobianVectorProductClass.Compute(f, x, vector);
    }
}
