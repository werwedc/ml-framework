using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.IR
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Values;

    /// <summary>
    /// Represents a basic block in the IR, containing a sequence of operations.
    /// </summary>
    public class IRBlock
    {
        /// <summary>Gets the name of the block.</summary>
        public string Name { get; }

        /// <summary>Gets the operations in this block.</summary>
        public List<IROperation> Operations { get; }

        /// <summary>Gets the arguments to this block.</summary>
        public List<IRValue> Arguments { get; }

        /// <summary>Gets the return values of this block.</summary>
        public List<IRValue> Returns { get; }

        /// <summary>
        /// Initializes a new instance of the IRBlock class.
        /// </summary>
        /// <param name="name">The name of the block.</param>
        public IRBlock(string name)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
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
                throw new ArgumentNullException(nameof(op));

            Operations.Add(op);
        }

        /// <summary>
        /// Adds an argument to this block.
        /// </summary>
        /// <param name="arg">The argument to add.</param>
        public void AddArgument(IRValue arg)
        {
            if (arg == null)
                throw new ArgumentNullException(nameof(arg));

            Arguments.Add(arg);
        }

        /// <summary>
        /// Adds a return value to this block.
        /// </summary>
        /// <param name="ret">The return value to add.</param>
        public void AddReturn(IRValue ret)
        {
            if (ret == null)
                throw new ArgumentNullException(nameof(ret));

            Returns.Add(ret);
        }

        public override string ToString()
        {
            return $"Block {Name} ({Operations.Count} ops)";
        }
    }
}
