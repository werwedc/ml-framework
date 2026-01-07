using System;
using System.Linq;

namespace MLFramework.IR.Operations
{
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    /// <summary>
    /// Abstract base class for all IR operations.
    /// </summary>
    public abstract class IROperation
    {
        /// <summary>Gets the name of the operation.</summary>
        public string Name { get; }

        /// <summary>Gets the operands (input values) of the operation.</summary>
        public IRValue[] Operands { get; }

        /// <summary>Gets the results (output values) of the operation.</summary>
        public IRValue[] Results { get; }

        /// <summary>Gets the opcode of the operation.</summary>
        public IROpcode Opcode { get; }

        /// <summary>Gets the result types of the operation.</summary>
        public IIRType[] ResultTypes { get; }

        /// <summary>Gets the IR context this operation belongs to.</summary>
        public IRContext Context { get; }

        /// <summary>
        /// Initializes a new instance of the IROperation class.
        /// </summary>
        /// <param name="name">The name of the operation.</param>
        /// <param name="opcode">The opcode of the operation.</param>
        /// <param name="operands">The operands (input values) of the operation.</param>
        /// <param name="resultTypes">The result types of the operation.</param>
        protected IROperation(string name, IROpcode opcode, IRValue[] operands, IIRType[] resultTypes)
            : this(name, opcode, operands, resultTypes, null)
        {
        }

        /// <summary>
        /// Initializes a new instance of the IROperation class with a context.
        /// </summary>
        /// <param name="name">The name of the operation.</param>
        /// <param name="opcode">The opcode of the operation.</param>
        /// <param name="operands">The operands (input values) of the operation.</param>
        /// <param name="resultTypes">The result types of the operation.</param>
        /// <param name="context">The IR context this operation belongs to.</param>
        protected IROperation(string name, IROpcode opcode, IRValue[] operands, IIRType[] resultTypes, IRContext context)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Opcode = opcode;
            Operands = operands ?? Array.Empty<IRValue>();
            ResultTypes = resultTypes ?? Array.Empty<IIRType>();
            Context = context;

            // Create result values
            Results = new IRValue[resultTypes.Length];
            for (int i = 0; i < resultTypes.Length; i++)
            {
                Results[i] = new IRValue(resultTypes[i]);
            }
        }

        /// <summary>
        /// Validates the operation (checks operand count, types, etc.).
        /// </summary>
        public abstract void Validate();

        /// <summary>
        /// Creates a clone of this operation.
        /// </summary>
        /// <returns>A new operation that is a copy of this one.</returns>
        public abstract IROperation Clone();

        public override string ToString()
        {
            string resultStr = Results.Length > 0
                ? $"{string.Join(", ", Results.Select(r => r.Name))} = "
                : "";

            string operandStr = Operands.Length > 0
                ? $"({string.Join(", ", Operands.Select(o => o.Name))})"
                : "()";

            return $"{resultStr}{Name}{operandStr}";
        }
    }
}
