using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations;

/// <summary>
/// Interface for gradient computation of operations in the computational graph.
/// Implementations define how to compute gradients with respect to operation inputs.
/// </summary>
public interface IOperationGrad
{
    /// <summary>
    /// Gets the name of the operation this gradient function corresponds to.
    /// </summary>
    string OperationName { get; }

    /// <summary>
    /// Computes gradients for the operation's inputs given the upstream gradient.
    /// </summary>
    /// <param name="gradOutput">The gradient from downstream operations.</param>
    /// <param name="inputs">The input tensors to the original operation.</param>
    /// <param name="context">The operation context containing saved intermediate values.</param>
    /// <returns>An array of gradients, one for each input tensor.</returns>
    Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context);
}
