using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.IR.Passes
{
    using MLFramework.IR.Transformations;
    using MLFramework.IR.Graph;
    using MLFramework.IR.Operations;
    using MLFramework.IR.Values;

    /// <summary>
    /// Dead code elimination pass that removes operations whose results are not used.
    /// </summary>
    public class DeadCodeEliminationPass : IRTransformation
    {
        public DeadCodeEliminationPass() : base("DeadCodeElimination", false)
        {
        }

        public override bool Run(HLIRModule module)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            bool changed = false;

            foreach (var function in module.Functions)
            {
                changed |= EliminateDeadCode(function);
            }

            return changed;
        }

        private bool EliminateDeadCode(HLIRFunction function)
        {
            // 1. Mark all values in function.Results as live
            var liveValues = new HashSet<IRValue>();
            foreach (var result in function.Results)
            {
                MarkLive(result, liveValues, function);
            }

            // Also mark parameters as live (they're inputs to the function)
            foreach (var param in function.Parameters)
            {
                liveValues.Add(param);
            }

            // 2. Remove all operations that don't produce live values
            bool changed = false;
            foreach (var block in function.GetAllBlocks())
            {
                var deadOps = new List<IROperation>();

                foreach (var op in block.Operations)
                {
                    bool producesLiveValue = false;
                    foreach (var result in op.Results)
                    {
                        if (liveValues.Contains(result))
                        {
                            producesLiveValue = true;
                            break;
                        }
                    }

                    if (!producesLiveValue)
                    {
                        deadOps.Add(op);
                    }
                }

                // Remove dead operations
                foreach (var op in deadOps)
                {
                    block.Operations.Remove(op);
                    changed = true;
                }
            }

            return changed;
        }

        /// <summary>
        /// Marks a value as live and recursively marks all operands of operations that produce it as live.
        /// </summary>
        private void MarkLive(IRValue value, HashSet<IRValue> liveValues, HLIRFunction function)
        {
            if (liveValues.Contains(value))
                return;

            liveValues.Add(value);

            // Find the operation that produces this value
            var definingOp = FindDefiningOperation(value, function);
            if (definingOp != null)
            {
                // Mark all operands as live
                foreach (var operand in definingOp.Operands)
                {
                    MarkLive(operand, liveValues, function);
                }
            }
        }

        /// <summary>
        /// Finds the operation that produces the given value.
        /// </summary>
        private IROperation FindDefiningOperation(IRValue value, HLIRFunction function)
        {
            foreach (var block in function.GetAllBlocks())
            {
                foreach (var op in block.Operations)
                {
                    foreach (var result in op.Results)
                    {
                        if (result.Id == value.Id)
                            return op;
                    }
                }
            }

            // Check if it's a parameter
            if (function.Parameters.Any(p => p.Id == value.Id))
                return null;

            return null;
        }
    }
}
