using RitterFramework.Core.Tensor;
using System.Linq;

namespace MLFramework.Autograd;

/// <summary>
/// Abstract base class for user-defined custom autograd functions.
/// Enables manual implementation of forward and backward passes for specialized operations.
/// </summary>
public abstract class CustomFunction
{
    /// <summary>
    /// Computes the forward pass of the operation.
    /// </summary>
    /// <param name="inputs">Input tensors to the operation.</param>
    /// <param name="ctx">Function context for saving state for backward pass.</param>
    /// <returns>Array of output tensors (can be empty).</returns>
    public abstract Tensor[] Forward(Tensor[] inputs, FunctionContext ctx);

    /// <summary>
    /// Computes the backward pass of the operation.
    /// </summary>
    /// <param name="gradOutputs">Gradients with respect to outputs.</param>
    /// <param name="ctx">Function context containing saved state from forward pass.</param>
    /// <returns>Array of gradient tensors (one per input, can be null for inputs that don't require gradients).</returns>
    public abstract Tensor[] Backward(Tensor[] gradOutputs, FunctionContext ctx);

    /// <summary>
    /// Applies this custom function to the given inputs and returns a single output tensor.
    /// Convenience method for single-output functions.
    /// </summary>
    /// <param name="inputs">Input tensors.</param>
    /// <returns>The first output tensor.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the function produces no outputs.</exception>
    public Tensor Apply(params Tensor[] inputs)
    {
        var outputs = ApplyMany(inputs);
        if (outputs.Length == 0)
            throw new InvalidOperationException("Function produced no outputs");

        return outputs[0];
    }

    /// <summary>
    /// Applies this custom function to the given inputs and returns all output tensors.
    /// </summary>
    /// <param name="inputs">Input tensors.</param>
    /// <returns>Array of output tensors (can be empty for side-effect functions).</returns>
    /// <exception cref="ArgumentNullException">Thrown when inputs array is null or contains null tensors.</exception>
    /// <exception cref="InvalidOperationException">Thrown when forward pass returns null.</exception>
    public Tensor[] ApplyMany(params Tensor[] inputs)
    {
        // Validate inputs
        if (inputs == null || inputs.Any(t => t == null))
            throw new ArgumentNullException(nameof(inputs));

        // Create context
        var ctx = new FunctionContext();

        // Call forward
        var outputs = Forward(inputs, ctx);
        if (outputs == null)
            throw new InvalidOperationException("Forward pass returned null");

        // Register with autograd engine
        AutogradEngine.Instance.RegisterCustomFunction(outputs, this, ctx, inputs);

        return outputs;
    }

    /// <summary>
    /// Validates that gradient shapes match the expected input shapes.
    /// </summary>
    /// <param name="grads">Gradient tensors to validate.</param>
    /// <param name="inputs">Input tensors for shape comparison.</param>
    /// <exception cref="ArgumentException">Thrown when gradient shapes don't match input shapes.</exception>
    protected void ValidateGradientShapes(Tensor[] grads, Tensor[] inputs)
    {
        if (grads.Length != inputs.Length)
            throw new ArgumentException($"Expected {inputs.Length} gradients but got {grads.Length}");

        for (int i = 0; i < grads.Length; i++)
        {
            if (grads[i] != null && inputs[i] != null)
            {
                if (!grads[i].HasSameShape(inputs[i]))
                {
                    throw new ArgumentException(
                        $"Gradient at index {i} has shape {grads[i].GetShapeString()} but expected {inputs[i].GetShapeString()}");
                }
            }
        }
    }
}
