using System.Collections.Generic;
using System.Linq;

namespace MLFramework.IR.LLIR.Operations
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Phi node operation in the Low-Level IR (LLIR).
    /// Implements SSA (Static Single Assignment) form by selecting a value
    /// based on which control flow predecessor block was executed.
    /// </summary>
    public class PhiNode : IROperation
    {
        /// <summary>Gets the result value of this phi node.</summary>
        public LLIRValue Result => Results[0] as LLIRValue;

        /// <summary>Gets the incoming values from different predecessor blocks.</summary>
        public List<(IRBlock IncomingBlock, LLIRValue IncomingValue)> IncomingValues { get; }

        /// <summary>
        /// Initializes a new instance of the PhiNode class.
        /// </summary>
        /// <param name="result">The result value.</param>
        /// <param name="incomingValues">List of incoming block-value pairs.</param>
        public PhiNode(LLIRValue result, List<(IRBlock, LLIRValue)> incomingValues)
            : base("phi", IROpcode.TypeConversion, // Using TypeConversion as a placeholder opcode for Phi
                  incomingValues.Select(iv => iv.Item2).ToArray(),
                  new[] { result.Type }, null)
        {
            Result = result ?? throw new System.ArgumentNullException(nameof(result));

            if (incomingValues == null || incomingValues.Count == 0)
            {
                throw new System.ArgumentException("PhiNode must have at least one incoming value.", nameof(incomingValues));
            }

            IncomingValues = incomingValues;
        }

        /// <summary>
        /// Adds an incoming value to this phi node.
        /// </summary>
        /// <param name="block">The predecessor block.</param>
        /// <param name="value">The value coming from that block.</param>
        public void AddIncoming(IRBlock block, LLIRValue value)
        {
            if (block == null)
            {
                throw new System.ArgumentNullException(nameof(block));
            }

            if (value == null)
            {
                throw new System.ArgumentNullException(nameof(value));
            }

            IncomingValues.Add((block, value));
        }

        public override void Validate()
        {
            // Validate all incoming values have the same type as result
            foreach (var (block, value) in IncomingValues)
            {
                if (value.Type != Result.Type)
                {
                    throw new System.InvalidOperationException(
                        $"PhiNode incoming value type {value.Type} does not match result type {Result.Type}.");
                }
            }
        }

        public override IROperation Clone()
        {
            var newIncomingValues = new List<(IRBlock, LLIRValue)>(IncomingValues);
            return new PhiNode(Result, newIncomingValues);
        }
    }
}
