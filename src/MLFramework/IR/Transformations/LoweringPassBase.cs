using System.Collections.Generic;
using System.Linq;
using MLFramework.IR.Operations;
using MLFramework.IR.Values;

namespace MLFramework.IR.Transformations
{
    /// <summary>
    /// Base class for lowering passes that transform IR from one level to another
    /// </summary>
    public abstract class LoweringPassBase : IRTransformation, ILoweringPass
    {
        /// <summary>
        /// Gets the source IR level (e.g., "HLIR", "MLIR")
        /// </summary>
        public string SourceIRLevel { get; }

        /// <summary>
        /// Gets the target IR level (e.g., "MLIR", "LLIR")
        /// </summary>
        public string TargetIRLevel { get; }

        /// <summary>
        /// Determines if the given operation can be lowered by this pass
        /// </summary>
        /// <param name="op">The operation to check</param>
        /// <returns>True if the operation can be lowered, false otherwise</returns>
        public abstract bool CanLower(IROperation op);

        /// <summary>
        /// Lowers the given operation from source IR to target IR
        /// </summary>
        /// <param name="targetContext">The target IR context</param>
        /// <param name="op">The operation to lower</param>
        /// <returns>The lowered operation in the target IR</returns>
        public abstract IROperation Lower(IRContext targetContext, IROperation op);

        /// <summary>
        /// Runs the lowering pass on the given module
        /// </summary>
        /// <param name="module">The module to lower</param>
        /// <returns>True if any operation was lowered (i.e., the module was modified), false otherwise</returns>
        public override bool Run(HLIRModule module)
        {
            bool changed = false;
            var targetContext = new IRContext();

            // Lower all operations in all functions
            foreach (var function in module.Functions)
            {
                changed |= LowerFunction(function, targetContext);
            }

            return changed;
        }

        /// <summary>
        /// Lowers all operations in a function
        /// </summary>
        /// <param name="function">The function to lower</param>
        /// <param name="targetContext">The target IR context</param>
        /// <returns>True if any operation was lowered, false otherwise</returns>
        protected virtual bool LowerFunction(HIRFunction function, IRContext targetContext)
        {
            bool changed = false;
            var rewriter = new OperationRewriter(function.Context, targetContext);

            // Collect operations to lower to avoid modifying the collection while iterating
            var operations = function.Body.Operations.ToList();
            var loweredOps = new Dictionary<IROperation, IROperation>();

            // First pass: lower all operations
            foreach (var op in operations)
            {
                if (CanLower(op))
                {
                    var lowered = Lower(targetContext, op);
                    loweredOps[op] = lowered;
                    rewriter.SetMapping(op.Results[0], lowered.Results[0]);
                    changed = true;
                }
            }

            // Second pass: replace operations if needed
            if (changed)
            {
                // This is a simplified implementation
                // In a full implementation, we would:
                // 1. Create a new function with lowered operations
                // 2. Replace the old function with the new one
                // 3. Update all references to the old function
            }

            return changed;
        }

        /// <summary>
        /// Initializes a new LoweringPassBase instance
        /// </summary>
        /// <param name="sourceLevel">The source IR level</param>
        /// <param name="targetLevel">The target IR level</param>
        protected LoweringPassBase(string sourceLevel, string targetLevel)
            : base($"{sourceLevel}To{targetLevel}", false)
        {
            SourceIRLevel = sourceLevel ?? throw new System.ArgumentNullException(nameof(sourceLevel));
            TargetIRLevel = targetLevel ?? throw new System.ArgumentNullException(nameof(targetLevel));
        }
    }
}
