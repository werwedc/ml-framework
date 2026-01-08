using RitterFramework.Core.Tensor;
using MLFramework.Autograd.Operations;

namespace MLFramework.Autograd
{
    /// <summary>
    /// Interface for gradient functions that support dynamic shapes.
    /// Extends IOperationGrad to add shape-aware gradient computation.
    /// </summary>
    public interface IDynamicGradientFunction : IOperationGrad
    {
        /// <summary>
        /// Computes the output shape of the operation given the input shapes.
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>The shape of the output tensor.</returns>
        Shapes.SymbolicShape GetOutputShape(System.Collections.Generic.List<Shapes.SymbolicShape> inputShapes);

        /// <summary>
        /// Validates that the gradient shape matches the expected shape for a given input index.
        /// </summary>
        /// <param name="gradientShape">The shape of the gradient tensor.</param>
        /// <exception cref="ArgumentException">Thrown when the gradient shape is invalid.</exception>
        void ValidateGradientShape(Shapes.SymbolicShape gradientShape);

        /// <summary>
        /// Computes the output shapes for all outputs of the operation.
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>A list of output shapes.</returns>
        System.Collections.Generic.List<Shapes.SymbolicShape> GetOutputShapes(
            System.Collections.Generic.List<Shapes.SymbolicShape> inputShapes);
    }
}
