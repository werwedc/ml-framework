using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.IR.Graph
{
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    /// <summary>
    /// Represents a function in the High-Level IR.
    /// A function contains parameters, a body (which consists of blocks), and return values.
    /// </summary>
    public class HLIRFunction
    {
        /// <summary>Gets the name of this function.</summary>
        public string Name { get; }

        /// <summary>Gets the parameters (inputs) of this function.</summary>
        public List<IRValue> Parameters { get; }

        /// <summary>Gets the return values (outputs) of this function.</summary>
        public List<IRValue> Results { get; }

        /// <summary>Gets the main body block of this function.</summary>
        public IRBlock Body { get; }

        /// <summary>Gets the IR context this function belongs to.</summary>
        public IRContext Context { get; }

        /// <summary>
        /// Initializes a new instance of the HLIRFunction class.
        /// </summary>
        /// <param name="name">The name of the function.</param>
        /// <param name="ctx">The IR context this function belongs to.</param>
        public HLIRFunction(string name, IRContext ctx = null)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("Function name cannot be null or whitespace.", nameof(name));
            }

            Name = name;
            Context = ctx ?? new IRContext();
            Parameters = new List<IRValue>();
            Results = new List<IRValue>();
            Body = new IRBlock($"{name}_body");
        }

        /// <summary>
        /// Adds a parameter to this function.
        /// </summary>
        /// <param name="type">The type of the parameter.</param>
        /// <param name="name">The optional name of the parameter.</param>
        /// <returns>The created parameter value.</returns>
        public IRValue AddParameter(TensorType type, string name = null)
        {
            if (type == null)
            {
                throw new ArgumentNullException(nameof(type));
            }

            var param = Context.CreateValue(type, name ?? GenerateParameterName(Parameters.Count));
            Parameters.Add(param);
            Body.AddArgument(param);

            return param;
        }

        /// <summary>
        /// Sets the results (outputs) of this function.
        /// </summary>
        /// <param name="results">The result values.</param>
        public void SetResults(params IRValue[] results)
        {
            if (results == null)
            {
                throw new ArgumentNullException(nameof(results));
            }

            Results.Clear();
            foreach (var result in results)
            {
                if (result == null)
                {
                    throw new ArgumentException("Result values cannot be null.", nameof(results));
                }

                Results.Add(result);
                Body.AddReturn(result);
            }
        }

        /// <summary>
        /// Adds a result (output) to this function.
        /// </summary>
        /// <param name="result">The result value.</param>
        public void AddResult(IRValue result)
        {
            if (result == null)
            {
                throw new ArgumentNullException(nameof(result));
            }

            Results.Add(result);
            Body.AddReturn(result);
        }

        /// <summary>
        /// Gets all blocks in this function.
        /// For now, this only returns the main body block.
        /// </summary>
        /// <returns>A collection containing all blocks in this function.</returns>
        public IEnumerable<IRBlock> GetAllBlocks()
        {
            yield return Body;

            // TODO: Add support for nested blocks in control flow operations (IfOp, LoopOp, etc.)
        }

        /// <summary>
        /// Gets all operations in this function across all blocks.
        /// </summary>
        /// <returns>A collection of all operations in this function.</returns>
        public IEnumerable<Operations.IROperation> GetAllOperations()
        {
            return GetAllBlocks().SelectMany(block => block.Operations);
        }

        /// <summary>
        /// Gets all values in this function (parameters and all operation results).
        /// </summary>
        /// <returns>A collection of all values in this function.</returns>
        public IEnumerable<IRValue> GetAllValues()
        {
            return Parameters.Concat(GetAllBlocks().SelectMany(block => block.GetAllDefinedValues()));
        }

        /// <summary>
        /// Generates a default name for a parameter.
        /// </summary>
        /// <param name="index">The parameter index.</param>
        /// <returns>A default parameter name in the format "param{index}".</returns>
        private string GenerateParameterName(int index)
        {
            return $"param{index}";
        }

        public override string ToString()
        {
            return $"Function '{Name}' ({Parameters.Count} params, {Results.Count} results)";
        }
    }
}
