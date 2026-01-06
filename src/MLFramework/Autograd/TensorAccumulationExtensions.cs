using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Extension methods for enabling gradient accumulation on Tensor instances.
/// </summary>
public static class TensorAccumulationExtensions
{
    private static readonly Dictionary<Tensor, bool> _accumulationEnabled = new();

    /// <summary>
    /// Enables gradient accumulation for this tensor.
    /// </summary>
    /// <param name="tensor">The tensor to enable accumulation for.</param>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static void EnableGradAccumulation(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!tensor.RequiresGrad)
            tensor.RequiresGrad = true;

        if (tensor.Gradient == null)
            tensor.Gradient = Tensor.Zeros(tensor.Shape);

        _accumulationEnabled[tensor] = true;
    }

    /// <summary>
    /// Disables gradient accumulation for this tensor.
    /// </summary>
    /// <param name="tensor">The tensor to disable accumulation for.</param>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static void DisableGradAccumulation(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        _accumulationEnabled.Remove(tensor);
    }

    /// <summary>
    /// Checks whether gradient accumulation is enabled for this tensor.
    /// </summary>
    /// <param name="tensor">The tensor to check.</param>
    /// <returns>True if accumulation is enabled; otherwise, false.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static bool HasGradAccumulation(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return _accumulationEnabled.ContainsKey(tensor);
    }

    /// <summary>
    /// Clears the accumulation state for all registered tensors.
    /// This should be called when accumulation is no longer needed.
    /// </summary>
    public static void ClearAllAccumulationStates()
    {
        _accumulationEnabled.Clear();
    }
}
