using RitterFramework.Core.Tensor;
using System.Runtime.CompilerServices;

namespace MLFramework.Autograd;

/// <summary>
/// Extension methods for Tensor class to support custom function autograd integration.
/// </summary>
public static class TensorAutogradExtensions
{
    /// <summary>
    /// Gets or sets the backward function for this tensor.
    /// </summary>
    private static readonly ConditionalWeakTable<Tensor, TensorAutogradState> _autogradState =
        new ConditionalWeakTable<Tensor, TensorAutogradState>();

    /// <summary>
    /// Gets the backward function for this tensor.
    /// </summary>
    public static Func<Tensor[], Tensor[]>? GetGradFn(this Tensor tensor)
    {
        if (_autogradState.TryGetValue(tensor, out var state))
        {
            return state.GradFn;
        }
        return null;
    }

    /// <summary>
    /// Sets the backward function for this tensor.
    /// </summary>
    public static void SetGradFn(this Tensor tensor, Func<Tensor[], Tensor[]>? gradFn)
    {
        var state = _autogradState.GetOrCreateValue(tensor);
        state.GradFn = gradFn;
    }

    /// <summary>
    /// Gets the gradient tensor for this tensor.
    /// </summary>
    public static Tensor? GetGradient(this Tensor tensor)
    {
        return tensor.Gradient;
    }

    /// <summary>
    /// Accumulates a gradient to this tensor's gradient.
    /// If the gradient doesn't exist, it's created. If it exists, the new gradient is added.
    /// </summary>
    /// <param name="tensor">The tensor to accumulate gradient on.</param>
    /// <param name="newGrad">The gradient to accumulate.</param>
    /// <exception cref="ArgumentNullException">Thrown when tensor or newGrad is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when gradient shapes don't match.</exception>
    public static void AccumulateGrad(this Tensor tensor, Tensor newGrad)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (newGrad == null)
            throw new ArgumentNullException(nameof(newGrad));

        if (!tensor.HasSameShape(newGrad))
            throw new InvalidOperationException(
                $"Gradient shape {newGrad.GetShapeString()} doesn't match tensor shape {tensor.GetShapeString()}");

        if (tensor.Gradient == null)
        {
            // Create gradient tensor
            tensor.Gradient = newGrad.Clone();
        }
        else
        {
            // Accumulate gradient
            for (int i = 0; i < tensor.Size; i++)
            {
                tensor.Gradient.Data[i] += newGrad.Data[i];
            }
        }
    }

    /// <summary>
    /// Computes gradients backward through the computational graph from this tensor.
    /// </summary>
    /// <param name="tensor">The output tensor to start backward pass from.</param>
    /// <param name="gradOutput">Optional initial gradient (defaults to ones like the tensor).</param>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static void Backward(this Tensor tensor, Tensor? gradOutput = null)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        AutogradEngine.Instance.Backward(tensor, gradOutput);
    }

    /// <summary>
    /// Gets or sets whether this tensor requires gradient computation.
    /// </summary>
    public static bool RequiresGrad(this Tensor tensor)
    {
        return tensor.RequiresGrad;
    }

    /// <summary>
    /// Sets whether this tensor requires gradient computation.
    /// </summary>
    public static void SetRequiresGrad(this Tensor tensor, bool requiresGrad)
    {
        tensor.RequiresGrad = requiresGrad;
    }

    /// <summary>
    /// Creates a tensor filled with ones with the same shape as the input tensor.
    /// </summary>
    /// <param name="tensor">The tensor to get the shape from.</param>
    /// <returns>A new tensor filled with ones.</returns>
    public static Tensor OnesLike(this Tensor tensor)
    {
        return Tensor.Ones(tensor.Shape, tensor.Dtype);
    }

    /// <summary>
    /// Creates a tensor filled with zeros with the same shape as the input tensor.
    /// </summary>
    /// <param name="tensor">The tensor to get the shape from.</param>
    /// <returns>A new tensor filled with zeros.</returns>
    public static Tensor ZerosLike(this Tensor tensor)
    {
        return Tensor.Zeros(tensor.Shape, tensor.Dtype);
    }
}

/// <summary>
/// Internal class to store autograd-specific state for a tensor.
/// </summary>
internal class TensorAutogradState
{
    public Func<Tensor[], Tensor[]>? GradFn { get; set; }
}
