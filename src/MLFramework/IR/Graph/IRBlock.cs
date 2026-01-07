using System.Collections.Generic;
using System.Linq;

namespace MLFramework.IR.Graph
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Values;

    /// <summary>
    /// Represents a basic block in the IR.
    /// A block is a sequence of operations that execute in order.
    /// </summary>
    public class IRBlock
    {
        private static int _nextBlockId = 0;

        /// <summary>Gets the name of this block.</summary>
        public string Name { get; }

        /// <summary>Gets the unique ID of this block.</summary>
        public int Id { get; }

        /// <summary>Gets the operations in this block.</summary>
        public List<IROperation> Operations { get; }

        /// <summary>Gets the arguments (inputs) to this block.</summary>
        public List<IRValue> Arguments { get; }

        /// <summary>Gets the return values (outputs) from this block.</summary>
        public List<IRValue> Returns { get; }

        /// <summary>
        /// Initializes a new instance of the IRBlock class.
        /// </summary>
        /// <param name="name">The name of the block (optional, auto-generated if not provided).</param>
        public IRBlock(string name = null)
        {
            Name = name ?? GenerateDefaultName();
            Id = _nextBlockId++;
            Operations = new List<IROperation>();
            Arguments = new List<IRValue>();
            Returns = new List<IRValue>();
        }

        /// <summary>
        /// Adds an operation to this block.
        /// </summary>
        /// <param name="op">The operation to add.</param>
        public void AddOperation(IROperation op)
        {
            if (op == null)
            {
                throw new System.ArgumentNullException(nameof(op));
            }

            Operations.Add(op);
        }

        /// <summary>
        /// Adds an argument (input) to this block.
        /// </summary>
        /// <param name="arg">The argument value to add.</param>
        public void AddArgument(IRValue arg)
        {
            if (arg == null)
            {
                throw new System.ArgumentNullException(nameof(arg));
            }

            Arguments.Add(arg);
        }

        /// <summary>
        /// Adds a return value (output) to this block.
        /// </summary>
        /// <param name="ret">The return value to add.</param>
        public void AddReturn(IRValue ret)
        {
            if (ret == null)
            {
                throw new System.ArgumentNullException(nameof(ret));
            }

            Returns.Add(ret);
        }

        /// <summary>
        /// Removes an operation from this block.
        /// </summary>
        /// <param name="op">The operation to remove.</param>
        /// <returns>True if the operation was removed, false otherwise.</returns>
        public bool RemoveOperation(IROperation op)
        {
            return Operations.Remove(op);
        }

        /// <summary>
        /// Gets all values defined in this block (operation results and arguments).
        /// </summary>
        /// <returns>A collection of all values defined in this block.</returns>
        public IEnumerable<IRValue> GetAllDefinedValues()
        {
            return Arguments.Concat(Operations.SelectMany(op => op.Results));
        }

        /// <summary>
        /// Gets all values used in this block.
        /// </summary>
        /// <returns>A collection of all values used as operands in this block.</returns>
        public IEnumerable<IRValue> GetAllUsedValues()
        {
            return Operations.SelectMany(op => op.Operands).Distinct();
        }

        /// <summary>
        /// Generates a default name for the block.
        /// </summary>
        /// <returns>A default name in the format "block{Id}".</returns>
        private string GenerateDefaultName()
        {
            return $"block{Id}";
        }

        public override string ToString()
        {
            return $"Block '{Name}' ({Operations.Count} ops)";
        }
    }
}
