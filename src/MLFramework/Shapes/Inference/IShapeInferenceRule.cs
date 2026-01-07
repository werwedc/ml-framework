using MLFramework.Shapes;

namespace MLFramework.Shapes.Inference
{
    /// <summary>
    /// Interface for shape inference rules that determine output shapes for operations.
    /// </summary>
    public interface IShapeInferenceRule
    {
        /// <summary>
        /// Determines whether this rule can infer the output shapes for the given operation and input shapes.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes.</param>
        /// <returns>True if this rule can infer the output shapes; otherwise, false.</returns>
        bool CanInfer(string opName, IReadOnlyList<SymbolicShape> inputs);

        /// <summary>
        /// Infers the output shapes for the given operation and input shapes.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes.</param>
        /// <returns>The inferred output shapes.</returns>
        /// <exception cref="ArgumentException">Thrown when the inputs are invalid for this operation.</exception>
        /// <exception cref="InvalidOperationException">Thrown when inference cannot be performed.</exception>
        List<SymbolicShape> Infer(string opName, IReadOnlyList<SymbolicShape> inputs);
    }
}
