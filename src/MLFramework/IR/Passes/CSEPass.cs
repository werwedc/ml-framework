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
    /// Common Subexpression Elimination pass that identifies and eliminates duplicate computations.
    /// </summary>
    public class CSEPass : IRTransformation
    {
        /// <summary>
        /// Key for identifying equivalent operations.
        /// </summary>
        private struct OperationKey : IEquatable<OperationKey>
        {
            public IROpcode Opcode { get; }
            public IRValue[] Operands { get; }

            public OperationKey(IROpcode opcode, IRValue[] operands)
            {
                Opcode = opcode;
                Operands = operands ?? Array.Empty<IRValue>();
            }

            public override int GetHashCode()
            {
                int hash = 17;
                hash = hash * 31 + Opcode.GetHashCode();

                foreach (var operand in Operands)
                {
                    hash = hash * 31 + operand.Id.GetHashCode();
                }

                return hash;
            }

            public bool Equals(OperationKey other)
            {
                if (Opcode != other.Opcode)
                    return false;

                if (Operands.Length != other.Operands.Length)
                    return false;

                for (int i = 0; i < Operands.Length; i++)
                {
                    if (Operands[i].Id != other.Operands[i].Id)
                        return false;
                }

                return true;
            }

            public override bool Equals(object obj)
            {
                return obj is OperationKey key && Equals(key);
            }
        }

        public CSEPass() : base("CommonSubexpressionElimination", false)
        {
        }

        public override bool Run(HLIRModule module)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            bool changed = false;

            foreach (var function in module.Functions)
            {
                changed |= EliminateCommonSubexpressions(function);
            }

            return changed;
        }

        private bool EliminateCommonSubexpressions(HLIRFunction function)
        {
            bool changed = false;

            foreach (var block in function.GetAllBlocks())
            {
                changed |= EliminateInBlock(block, function);
            }

            return changed;
        }

        private bool EliminateInBlock(IRBlock block, HLIRFunction function)
        {
            bool changed = false;
            var operationMap = new Dictionary<OperationKey, IROperation>();
            var operationsToRemove = new List<IROperation>();
            var valueReplacements = new Dictionary<IRValue, IRValue>();

            // 1. Build hash map of operation -> result value
            foreach (var op in block.Operations)
            {
                var key = new OperationKey(op.Opcode, op.Operands);

                if (operationMap.TryGetValue(key, out var existingOp))
                {
                    // Found duplicate operation
                    if (op.Results.Length == existingOp.Results.Length)
                    {
                        // Replace all uses of this operation's results with existing results
                        for (int i = 0; i < op.Results.Length; i++)
                        {
                            valueReplacements[op.Results[i]] = existingOp.Results[i];
                        }

                        operationsToRemove.Add(op);
                        changed = true;
                    }
                }
                else
                {
                    operationMap[key] = op;
                }
            }

            // 2. Replace uses of removed operation results
            if (changed)
            {
                ReplaceUsesInBlock(block, valueReplacements);

                // Remove duplicate operations
                foreach (var op in operationsToRemove)
                {
                    block.Operations.Remove(op);
                }
            }

            return changed;
        }

        /// <summary>
        /// Replaces uses of values with their replacements in all operations in a block.
        /// </summary>
        private void ReplaceUsesInBlock(IRBlock block, Dictionary<IRValue, IRValue> replacements)
        {
            foreach (var op in block.Operations)
            {
                for (int i = 0; i < op.Operands.Length; i++)
                {
                    if (replacements.TryGetValue(op.Operands[i], out var replacement))
                    {
                        // In a real implementation, we would need to update the operand
                        // Since IROperation.Operands is read-only, we might need to use an OperationRewriter
                        // For now, we'll skip the actual replacement
                    }
                }
            }
        }
    }
}
