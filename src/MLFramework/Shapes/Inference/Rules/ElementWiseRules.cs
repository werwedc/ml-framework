using MLFramework.Shapes;
using MLFramework.Shapes.Inference;

namespace MLFramework.Shapes.Inference.Rules
{
    /// <summary>
    /// Base class for element-wise operation rules that support broadcasting.
    /// </summary>
    public abstract class ElementWiseBinaryRule : ShapeInferenceRuleBase
    {
        /// <summary>
        /// Gets the supported operations.
        /// </summary>
        protected override string[] SupportedOperations => GetSupportedOps();

        /// <summary>
        /// Must be implemented by derived classes to specify supported operations.
        /// </summary>
        protected abstract string[] GetSupportedOps();

        /// <summary>
        /// Gets the expected input count.
        /// </summary>
        protected override int GetExpectedInputCount(string opName)
        {
            return 2;
        }

        /// <summary>
        /// Infers the output shape for element-wise operations with broadcasting.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes.</param>
        /// <returns>The inferred output shape.</returns>
        protected override List<SymbolicShape> InferOutputShapes(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            var shapeA = inputs[0];
            var shapeB = inputs[1];

            // Use the ShapeComparer to get the broadcast result shape
            var broadcastShape = ShapeComparer.GetBroadcastShape(shapeA, shapeB);

            return new List<SymbolicShape> { broadcastShape };
        }
    }

    /// <summary>
    /// Shape inference rule for element-wise addition operations.
    /// </summary>
    public class AddRule : ElementWiseBinaryRule
    {
        protected override string[] GetSupportedOps() => new[] { "Add", "+" };
    }

    /// <summary>
    /// Shape inference rule for element-wise subtraction operations.
    /// </summary>
    public class SubRule : ElementWiseBinaryRule
    {
        protected override string[] GetSupportedOps() => new[] { "Sub", "-", "Subtract" };
    }

    /// <summary>
    /// Shape inference rule for element-wise multiplication operations.
    /// </summary>
    public class MulRule : ElementWiseBinaryRule
    {
        protected override string[] GetSupportedOps() => new[] { "Mul", "*", "Multiply" };
    }

    /// <summary>
    /// Shape inference rule for element-wise division operations.
    /// </summary>
    public class DivRule : ElementWiseBinaryRule
    {
        protected override string[] GetSupportedOps() => new[] { "Div", "/", "Divide" };
    }

    /// <summary>
    /// Shape inference rule for element-wise comparison operations.
    /// </summary>
    public class CompareRule : ElementWiseBinaryRule
    {
        protected override string[] GetSupportedOps() => new[] { "Equal", "NotEqual", "LessThan", "LessEqual", "GreaterThan", "GreaterEqual" };
    }
}
