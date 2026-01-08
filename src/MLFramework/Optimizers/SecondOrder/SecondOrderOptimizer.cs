using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;

namespace MLFramework.Optimizers.SecondOrder;

/// <summary>
/// Enumeration specifying the mode of Hessian computation.
/// </summary>
public enum HessianMode
{
    /// <summary>
    /// Compute the full Hessian matrix.
    /// </summary>
    Full,

    /// <summary>
    /// Compute Hessian-Vector products (HVP) only, more memory efficient.
    /// </summary>
    HVP,

    /// <summary>
    /// Compute only the diagonal of the Hessian.
    /// </summary>
    Diagonal
}

/// <summary>
/// Abstract base class for second-order optimizers that utilize Hessian information.
/// Provides common functionality for Hessian computation and management.
/// </summary>
public abstract class SecondOrderOptimizer : Optimizer
{
    /// <summary>
    /// Gets or sets the Hessian computation mode.
    /// </summary>
    public HessianMode HessianMode { get; set; } = HessianMode.HVP;

    /// <summary>
    /// Gets or sets whether to use gradient checkpointing for Hessian computation
    /// to reduce memory at the cost of additional computation.
    /// </summary>
    public bool UseCheckpointing { get; set; } = false;

    /// <summary>
    /// Initializes a new instance of the SecondOrderOptimizer class.
    /// </summary>
    /// <param name="parameters">Dictionary mapping parameter names to tensors.</param>
    protected SecondOrderOptimizer(Dictionary<string, Tensor> parameters)
        : base(parameters)
    {
    }

    /// <summary>
    /// Computes the Hessian-Vector Product (HVP) for a given loss with respect to parameters.
    /// This is more memory-efficient than computing the full Hessian.
    /// </summary>
    /// <param name="loss">The loss tensor.</param>
    /// <param name="parameters">The parameters to compute the HVP for.</param>
    /// <param name="vector">The vector to multiply with the Hessian.</param>
    /// <returns>The Hessian-Vector product.</returns>
    protected Tensor ComputeHVP(Tensor loss, Tensor[] parameters, Tensor vector)
    {
        // Use the Hessian API to compute HVP
        return Hessian.ComputeVectorHessianProduct(
            f: t => loss.Data[0],  // Extract scalar loss value
            x: FlattenParameters(parameters),
            v: vector
        );
    }

    /// <summary>
    /// Computes the full Hessian matrix for a given loss with respect to parameters.
    /// Warning: This can be very memory-intensive for large parameter sets.
    /// </summary>
    /// <param name="loss">The loss tensor.</param>
    /// <param name="parameters">The parameters to compute the Hessian for.</param>
    /// <returns>The Hessian matrix.</returns>
    protected Tensor ComputeHessian(Tensor loss, Tensor[] parameters)
    {
        // Use the Hessian API to compute full Hessian
        return Hessian.Compute(
            f: t => loss.Data[0],  // Extract scalar loss value
            x: FlattenParameters(parameters)
        );
    }

    /// <summary>
    /// Computes the diagonal of the Hessian matrix.
    /// Memory-efficient alternative to full Hessian computation.
    /// </summary>
    /// <param name="loss">The loss tensor.</param>
    /// <param name="parameters">The parameters to compute the diagonal Hessian for.</param>
    /// <returns>A 1D tensor containing the diagonal of the Hessian.</returns>
    protected Tensor ComputeHessianDiagonal(Tensor loss, Tensor[] parameters)
    {
        return Hessian.ComputeDiagonal(
            f: t => loss.Data[0],  // Extract scalar loss value
            x: FlattenParameters(parameters)
        );
    }

    /// <summary>
    /// Flattens a set of parameters into a single 1D tensor.
    /// </summary>
    /// <param name="parameters">The parameters to flatten.</param>
    /// <returns>A flattened 1D tensor containing all parameters.</returns>
    protected Tensor FlattenParameters(Tensor[] parameters)
    {
        int totalSize = 0;
        foreach (var param in parameters)
        {
            totalSize += param.Size;
        }

        var flattened = new float[totalSize];
        int offset = 0;

        foreach (var param in parameters)
        {
            var paramData = TensorAccessor.GetData(param);
            Array.Copy(paramData, 0, flattened, offset, param.Size);
            offset += param.Size;
        }

        return new Tensor(flattened, new[] { totalSize });
    }

    /// <summary>
    /// Reconstructs a set of parameters from a flattened tensor.
    /// </summary>
    /// <param name="flattened">The flattened tensor.</param>
    /// <param name="originalParameters">The original parameter tensors (for shape reference).</param>
    /// <returns>An array of parameter tensors reconstructed from the flattened tensor.</returns>
    protected Tensor[] UnflattenParameters(Tensor flattened, Tensor[] originalParameters)
    {
        var result = new Tensor[originalParameters.Length];
        var flattenedData = TensorAccessor.GetData(flattened);
        int offset = 0;

        for (int i = 0; i < originalParameters.Length; i++)
        {
            int size = originalParameters[i].Size;
            var paramData = new float[size];
            Array.Copy(flattenedData, offset, paramData, 0, size);

            result[i] = new Tensor(paramData, originalParameters[i].Shape);
            offset += size;
        }

        return result;
    }

    /// <summary>
    /// Computes the update step using second-order information.
    /// </summary>
    /// <param name="loss">The loss tensor.</param>
    /// <param name="gradients">The gradients with respect to parameters.</param>
    /// <param name="parameters">The parameters to update.</param>
    /// <returns>The computed update step (delta) for each parameter.</returns>
    protected abstract Tensor[] ComputeUpdateStep(
        Tensor loss,
        Tensor[] gradients,
        Tensor[] parameters);
}
