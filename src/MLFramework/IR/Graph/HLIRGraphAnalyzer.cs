using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.IR.Graph
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Values;

    /// <summary>
    /// Provides analysis utilities for HLIR graphs.
    /// Includes topological sorting, use-def chain analysis, and graph validation.
    /// </summary>
    public class HLIRGraphAnalyzer
    {
        /// <summary>
        /// Performs topological sort on operations in a function.
        /// </summary>
        /// <param name="function">The function to sort.</param>
        /// <returns>A list of operations in topological order.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the graph contains cycles.</exception>
        public static List<IROperation> TopologicalSort(HLIRFunction function)
        {
            if (function == null)
            {
                throw new ArgumentNullException(nameof(function));
            }

            var visited = new HashSet<IROperation>();
            var visiting = new HashSet<IROperation>();
            var sortedOps = new List<IROperation>();

            // Collect all operations from all blocks
            var allOperations = function.GetAllOperations().ToList();

            foreach (var op in allOperations)
            {
                if (!visited.Contains(op))
                {
                    VisitOperation(op, visited, visiting, sortedOps);
                }
            }

            // Reverse since we add to list after visiting dependents
            sortedOps.Reverse();
            return sortedOps;
        }

        /// <summary>
        /// DFS-based visitor for topological sort.
        /// </summary>
        private static void VisitOperation(IROperation op,
                                         HashSet<IROperation> visited,
                                         HashSet<IROperation> visiting,
                                         List<IROperation> sortedOps)
        {
            if (visiting.Contains(op))
            {
                throw new InvalidOperationException("Graph contains a cycle - cannot perform topological sort");
            }

            if (visited.Contains(op))
            {
                return;
            }

            visiting.Add(op);

            // Visit all operations that produce the operands
            foreach (var operand in op.Operands)
            {
                var definingOp = FindDefiningOperation(op, operand);
                if (definingOp != null && !visited.Contains(definingOp))
                {
                    VisitOperation(definingOp, visited, visiting, sortedOps);
                }
            }

            visiting.Remove(op);
            visited.Add(op);
            sortedOps.Add(op);
        }

        /// <summary>
        /// Finds the operation that produces a given value.
        /// </summary>
        /// <param name="currentOp">The current operation being visited.</param>
        /// <param name="value">The value to find the producer for.</param>
        /// <returns>The operation that produces the value, or null if it's a parameter.</returns>
        private static IROperation FindDefiningOperation(IROperation currentOp, IRValue value)
        {
            // Check all blocks in the function to find the operation that produces this value
            // This is a simplified version - in practice we might want a more efficient lookup
            var function = FindFunctionForOperation(currentOp);
            if (function == null)
            {
                return null;
            }

            foreach (var block in function.GetAllBlocks())
            {
                foreach (var op in block.Operations)
                {
                    if (op.Results.Contains(value))
                    {
                        return op;
                    }
                }
            }

            // Value might be a parameter
            return null;
        }

        /// <summary>
        /// Finds the function that contains a given operation.
        /// </summary>
        private static HLIRFunction FindFunctionForOperation(IROperation op)
        {
            // This is a placeholder - in a real implementation we would need to track
            // which operation belongs to which function more efficiently
            return null;
        }

        /// <summary>
        /// Finds all uses of a value in a function.
        /// </summary>
        /// <param name="value">The value to find uses for.</param>
        /// <param name="function">The function to search in (optional).</param>
        /// <returns>A dictionary mapping values to the list of operations that use them.</returns>
        public static Dictionary<IRValue, List<IROperation>> FindUses(IRValue value, HLIRFunction function = null)
        {
            var uses = new Dictionary<IRValue, List<IROperation>>();

            if (function == null)
            {
                return uses;
            }

            foreach (var block in function.GetAllBlocks())
            {
                foreach (var op in block.Operations)
                {
                    foreach (var operand in op.Operands)
                    {
                        if (!uses.ContainsKey(operand))
                        {
                            uses[operand] = new List<IROperation>();
                        }
                        uses[operand].Add(op);
                    }
                }
            }

            return uses;
        }

        /// <summary>
        /// Validates that a graph is well-formed.
        /// </summary>
        /// <param name="function">The function to validate.</param>
        /// <returns>True if the graph is valid, false otherwise.</returns>
        public static bool ValidateGraph(HLIRFunction function)
        {
            if (function == null)
            {
                return false;
            }

            // Check all values are defined before use
            var definedValues = new HashSet<IRValue>();

            // Parameters are already defined
            foreach (var param in function.Parameters)
            {
                definedValues.Add(param);
            }

            foreach (var block in function.GetAllBlocks())
            {
                // Block arguments are defined
                foreach (var arg in block.Arguments)
                {
                    if (!definedValues.Contains(arg))
                    {
                        definedValues.Add(arg);
                    }
                }

                // Check each operation
                foreach (var op in block.Operations)
                {
                    // Validate all operands are defined
                    foreach (var operand in op.Operands)
                    {
                        if (!definedValues.Contains(operand))
                        {
                            return false;
                        }
                    }

                    // Define the results
                    foreach (var result in op.Results)
                    {
                        definedValues.Add(result);
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Finds all input values (parameters) of a function.
        /// </summary>
        /// <param name="function">The function to analyze.</param>
        /// <returns>A list of parameter values.</returns>
        public static List<IRValue> FindInputs(HLIRFunction function)
        {
            if (function == null)
            {
                return new List<IRValue>();
            }

            return new List<IRValue>(function.Parameters);
        }

        /// <summary>
        /// Finds all output values (results) of a function.
        /// </summary>
        /// <param name="function">The function to analyze.</param>
        /// <returns>A list of result values.</returns>
        public static List<IRValue> FindOutputs(HLIRFunction function)
        {
            if (function == null)
            {
                return new List<IRValue>();
            }

            return new List<IRValue>(function.Results);
        }

        /// <summary>
        /// Finds all operations that produce values used but not redefined within a region.
        /// </summary>
        /// <param name="function">The function to analyze.</param>
        /// <returns>A dictionary mapping values to their defining operations.</returns>
        public static Dictionary<IRValue, IROperation> BuildUseDefChains(HLIRFunction function)
        {
            var useDefChains = new Dictionary<IRValue, IROperation>();

            if (function == null)
            {
                return useDefChains;
            }

            foreach (var block in function.GetAllBlocks())
            {
                foreach (var op in block.Operations)
                {
                    foreach (var result in op.Results)
                    {
                        useDefChains[result] = op;
                    }
                }
            }

            return useDefChains;
        }

        /// <summary>
        /// Counts the number of operations in a function.
        /// </summary>
        /// <param name="function">The function to analyze.</param>
        /// <returns>The number of operations across all blocks.</returns>
        public static int CountOperations(HLIRFunction function)
        {
            if (function == null)
            {
                return 0;
            }

            return function.GetAllOperations().Count();
        }

        /// <summary>
        /// Counts the number of blocks in a function.
        /// </summary>
        /// <param name="function">The function to analyze.</param>
        /// <returns>The number of blocks.</returns>
        public static int CountBlocks(HLIRFunction function)
        {
            if (function == null)
            {
                return 0;
            }

            return function.GetAllBlocks().Count();
        }

        /// <summary>
        /// Gets operation type statistics for a function.
        /// </summary>
        /// <param name="function">The function to analyze.</param>
        /// <returns>A dictionary mapping operation types to their counts.</returns>
        public static Dictionary<Type, int> GetOperationStats(HLIRFunction function)
        {
            var stats = new Dictionary<Type, int>();

            if (function == null)
            {
                return stats;
            }

            foreach (var op in function.GetAllOperations())
            {
                var opType = op.GetType();
                if (!stats.ContainsKey(opType))
                {
                    stats[opType] = 0;
                }
                stats[opType]++;
            }

            return stats;
        }
    }
}
