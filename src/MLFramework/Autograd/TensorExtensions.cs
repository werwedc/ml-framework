using System;
using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Extension methods for Tensor to support higher-order derivative computation.
/// </summary>
public static class TensorExtensions
{
    /// <summary>
    /// Determines if a tensor can be differentiated (has gradient tracking enabled).
    /// </summary>
    /// <param name="tensor">The tensor to check.</param>
    /// <returns>True if the tensor requires gradients, false otherwise.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static bool IsDifferentiable(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return tensor.RequiresGrad && tensor.Gradient != null;
    }

    /// <summary>
    /// Gets or sets whether a tensor's gradient should be retained for higher-order differentiation.
    /// This is used in conjunction with GradientTape to control gradient retention policy.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <param name="retainGrad">Whether to retain the gradient.</param>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static void SetRetainGradient(this Tensor tensor, bool retainGrad)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // In a full implementation, this would set a flag on the tensor
        // For now, we use RequiresGrad as the primary indicator
        // This could be extended with a dedicated property
    }

    /// <summary>
    /// Creates a tensor that requires gradient tracking from an existing tensor.
    /// </summary>
    /// <param name="tensor">The tensor to clone with gradient tracking.</param>
    /// <returns>A new tensor with gradient tracking enabled.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static Tensor RequiresGrad(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return TensorAccessor.CloneWithGrad(tensor);
    }

    /// <summary>
    /// Detaches a tensor from the computation graph.
    /// The returned tensor does not require gradients and has no gradient history.
    /// </summary>
    /// <param name="tensor">The tensor to detach.</param>
    /// <returns>A detached clone of the tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static Tensor Detach(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return TensorAccessor.CloneWithoutGrad(tensor);
    }

    /// <summary>
    /// Creates a deep copy of a tensor with optional gradient tracking.
    /// </summary>
    /// <param name="tensor">The tensor to clone.</param>
    /// <param name="requiresGrad">Whether to enable gradient tracking on the clone (default: false).</param>
    /// <returns>A cloned tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static Tensor Clone(this Tensor tensor, bool requiresGrad = false)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (requiresGrad)
        {
            return TensorAccessor.CloneWithGrad(tensor);
        }
        else
        {
            return TensorAccessor.CloneWithoutGrad(tensor);
        }
    }
}
