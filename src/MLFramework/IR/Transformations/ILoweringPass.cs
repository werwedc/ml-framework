using MLFramework.IR.Operations;

namespace MLFramework.IR.Transformations
{
    /// <summary>
    /// Interface for lowering passes that transform IR from one level to another
    /// </summary>
    public interface ILoweringPass
    {
        /// <summary>
        /// Gets the source IR level (e.g., "HLIR", "MLIR")
        /// </summary>
        string SourceIRLevel { get; }

        /// <summary>
        /// Gets the target IR level (e.g., "MLIR", "LLIR")
        /// </summary>
        string TargetIRLevel { get; }

        /// <summary>
        /// Determines if the given operation can be lowered by this pass
        /// </summary>
        /// <param name="op">The operation to check</param>
        /// <returns>True if the operation can be lowered, false otherwise</returns>
        bool CanLower(IROperation op);

        /// <summary>
        /// Lowers the given operation from source IR to target IR
        /// </summary>
        /// <param name="targetContext">The target IR context</param>
        /// <param name="op">The operation to lower</param>
        /// <returns>The lowered operation in the target IR</returns>
        IROperation Lower(IRContext targetContext, IROperation op);
    }
}
